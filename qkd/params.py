# qkd/params.py
# -*- coding: utf-8 -*-
"""
Parameter handling, validation, and serialization for the QKD framework.
"""
from __future__ import annotations
from .detectors import AfterpulseModel

import copy
import dataclasses
import difflib
import logging
import math
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    Protocol as TypingProtocol,
    get_type_hints,
    get_origin,
    get_args,
)

try:
    import numpy as np
    _NUMPY_GENERIC_TYPES: tuple = (np.generic,)
    _NUMPY_ARRAY_TYPES: tuple = (np.ndarray,)
except Exception:
    np = None  # type: ignore[assignment]
    _NUMPY_GENERIC_TYPES = ()
    _NUMPY_ARRAY_TYPES = ()


from .datatypes import (
    DoubleClickPolicy,
    SecurityProof,
    ConfidenceBoundMethod,
    PulseTypeConfig,
    SourceStatisticsType,
    DecoderArchitecture,
    SourceErrorModel,
    PulseEnsembleConfig,
)
from .exceptions import ParameterValidationError, ConfigurationError
from .utils.validation import parse_bool
from .sources import OpticalSource
from .channel import FiberChannel
from .detectors import ThresholdDetector
from .constants import LP_SOLVER_METHODS, DEFAULT_POISSON_TAIL_THRESHOLD
from .protocols import Protocol as ProtocolFromModule

__all__ = [
    "QKDParams",
    "load_lim2014_dedicated_params",
    "load_lim2014_dwdm_params",
    "SerializableComponent",
    "QKDProtocol",
    "CURRENT_SCHEMA_VERSION",
]

logger = logging.getLogger(__name__)
EnumType = TypeVar("EnumType", bound=Enum)

CURRENT_SCHEMA_VERSION = "1.9"

def _parse_schema_version(v: Any) -> tuple:
    """Parse a schema version string into a comparable tuple of ints.

    Accepts '1.9', '1.9.0', '1.10' → (1, 9), (1, 9), (1, 10).
    Trailing zeros are stripped so 1.9 == 1.9.0 == 1.9.0.0.
    Falls back to (0,) for unparseable inputs.
    """
    if not isinstance(v, str):
        return (0,)
    parts = []
    for p in v.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            return (0,)
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts) if parts else (0,)

class SerializableComponent(TypingProtocol):
    def to_config_dict(self) -> Dict[str, Any]:
        ...

class QKDProtocol(SerializableComponent, TypingProtocol):
    protocol_name: str

MAX_SERIALIZABLE_LIST_LEN = 1000
MAX_SERIALIZABLE_DICT_LEN = 1000
MAX_SERIALIZABLE_RECURSION_DEPTH = 20

def _validate_serializable_object(obj: Any, _depth: int = 0) -> None:
    if _depth > MAX_SERIALIZABLE_RECURSION_DEPTH:
        raise ParameterValidationError("Serialization error: Exceeded max recursion depth.")

    if isinstance(obj, (list, tuple)):
        if len(obj) > MAX_SERIALIZABLE_LIST_LEN:
            raise ParameterValidationError(f"Serialization error: List/tuple length exceeds limit of {MAX_SERIALIZABLE_LIST_LEN}.")
        for item in obj:
            _validate_serializable_object(item, _depth + 1)
    elif isinstance(obj, dict):
        if len(obj) > MAX_SERIALIZABLE_DICT_LEN:
            raise ParameterValidationError(f"Serialization error: Dict length exceeds limit of {MAX_SERIALIZABLE_DICT_LEN}.")
        for key, value in obj.items():
            if not isinstance(key, str):
                raise ParameterValidationError("Serialization error: Dictionary keys must be strings.")
            _validate_serializable_object(value, _depth + 1)
    elif isinstance(obj, float) and not math.isfinite(obj):
        raise ParameterValidationError(f"Non-finite float value '{obj}' found in serialized output.")
    elif not isinstance(obj, (bool, int, float, str, type(None))):
        raise ParameterValidationError(f"Serialization error: Unsupported type '{type(obj).__name__}' in final output.")

_SEED_KEYS = frozenset({
    "master_seed", "seed", "random_seed", "prng_seed", "initial_seed",
})

def _redact_seed_recursive(obj: Any) -> None:
    """Recursively drop any seed-like key from nested dicts/lists.

    Belt-and-suspenders complement to to_summary_dict's redaction: ensures
    that no component's to_config_dict() can leak a seed-like field.
    """
    if isinstance(obj, dict):
        for _k in list(obj):
            if _k.lower() in _SEED_KEYS:
                obj.pop(_k, None)
        for v in obj.values():
            _redact_seed_recursive(v)
    elif isinstance(obj, list):
        for item in obj:
            _redact_seed_recursive(item)

def _to_serializable(o: Any, _depth: int = 0) -> Any:
    if _depth > MAX_SERIALIZABLE_RECURSION_DEPTH:
        raise ParameterValidationError("Serialization error: Exceeded max recursion depth in _to_serializable.")

    # Guard against calling to_config_dict on a class (type), not an instance.
    if (not isinstance(o, type)
            and hasattr(o, "to_config_dict")
            and callable(o.to_config_dict)):
        config_dict = o.to_config_dict()
        serializable_dict = _to_serializable(config_dict, _depth + 1)
        _validate_serializable_object(serializable_dict)
        return serializable_dict

    if np is not None:
        if isinstance(o, _NUMPY_GENERIC_TYPES):
            return o.item()
        if isinstance(o, _NUMPY_ARRAY_TYPES):
            if o.size > MAX_SERIALIZABLE_LIST_LEN:
                raise ParameterValidationError(
                    f"Cannot serialize NumPy array of size {o.size} (limit {MAX_SERIALIZABLE_LIST_LEN})."
                )
            return o.tolist()

    if isinstance(o, Enum):
        return o.value

    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        _dc_result = {f.name: _to_serializable(getattr(o, f.name), _depth + 1) for f in dataclasses.fields(o)}
        _validate_serializable_object(_dc_result, _depth)
        return _dc_result

    if isinstance(o, dict):
        return {k: _to_serializable(v, _depth + 1) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_serializable(i, _depth + 1) for i in o]
    if isinstance(o, (set, frozenset)):
        # Sort for reproducible serialization (config hashing, diffs).
        # Fall back to insertion order if elements are not comparable.
        try:
            _ordered = sorted(o)
        except TypeError:
            _ordered = sorted(o, key=repr)
        return [_to_serializable(i, _depth + 1) for i in _ordered]
    if isinstance(o, (bytes, bytearray)):
        return o.hex()
    if isinstance(o, Decimal):
        return float(o)

    if isinstance(o, float) and not math.isfinite(o):
        raise ParameterValidationError(f"Non-finite float value '{o}' found during serialization.")

    return o

_ENUM_LOOKUP_CACHE: Dict[Type[Enum], Dict[str, Enum]] = {}

def _build_enum_lookup(enum_cls: Type[EnumType]) -> Dict[str, EnumType]:
    """Build a case-insensitive lookup table for an enum class.

    Keys are: member.name.upper(), member.value (if str), member.value.upper()
    (if str), str(member.value) (if int/float and not bool).  Name match takes
    precedence on insertion order.
    """
    table: Dict[str, EnumType] = {}
    for m in enum_cls:
        table.setdefault(m.name.upper(), m)
        if isinstance(m.value, str):
            table.setdefault(m.value, m)
            table.setdefault(m.value.upper(), m)
        elif isinstance(m.value, (int, float)) and not isinstance(m.value, bool):
            table.setdefault(str(m.value), m)
    return table

def _coerce_enum(enum_cls: Type[EnumType], value: Any) -> EnumType:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        if enum_cls not in _ENUM_LOOKUP_CACHE:
            _ENUM_LOOKUP_CACHE[enum_cls] = _build_enum_lookup(enum_cls)
        table = _ENUM_LOOKUP_CACHE[enum_cls]
        s = value.strip()
        hit = table.get(s)
        if hit is None:
            hit = table.get(s.upper())
        if hit is not None:
            return hit
    if isinstance(value, bool):
        raise ParameterValidationError(
            f"Boolean value {value!r} is not valid for {enum_cls.__name__}. "
            f"Allowed names: {[m.name for m in enum_cls]}."
        )
    try:
        return enum_cls(value)
    except (ValueError, TypeError) as e:
        candidates = [m.name for m in enum_cls]
        candidates_upper = [m.name.upper() for m in enum_cls]
        suggestion_str = ""
        if isinstance(value, str):
            matches = difflib.get_close_matches(value.strip().upper(), candidates_upper, n=1)
            # Map back to the original (non-uppercased) name for display.
            if matches:
                matches = [candidates[candidates_upper.index(matches[0])]]
            if matches:
                suggestion_str = f" Did you mean '{matches[0]}'?"
        raise ParameterValidationError(
            f"Invalid value '{value}' for {enum_cls.__name__}. "
            f"Allowed names: {candidates}.{suggestion_str}"
        ) from e


def _coerce_type(d: Dict[str, Any], key: str, target_type: Type[Any], required: bool = True, default: Any = None) -> Any:
    """Extract and coerce *key* from *d*, removing it from the dict.

    .. warning:: This function **pops** the key from *d*, mutating the caller's
       dictionary.  This is intentional for the "consume-then-detect-unknowns"
       pattern in ``QKDParams.from_dict``.

    When *required* is False, a missing key **or** an explicit ``None``
    value both return *default*.  This matches the expectation that
    "null in config = use the default" for non-Optional fields.  For
    genuinely Optional fields, pass ``default=None`` so null round-trips
    as None.
    """
    if key not in d:
        if required:
            raise ParameterValidationError(f"Missing required parameter: '{key}'")
        return default

    val = d.pop(key)

    if val is None:
        if required:
            raise ParameterValidationError(f"Required parameter '{key}' cannot be null.")
        return default

    try:
        if target_type is bool:
            return parse_bool(val)

        if target_type is int:
            if isinstance(val, bool):
                raise ParameterValidationError(
                    f"Parameter '{key}' must be an int, not a bool."
                )
            if isinstance(val, float) and not val.is_integer():
                raise ParameterValidationError(
                    f"Parameter '{key}' must be an integer, got non-integer float {val}."
                )
            if isinstance(val, Decimal) and val != val.to_integral_value():
                raise ParameterValidationError(
                    f"Parameter '{key}' must be an integer, got non-integer Decimal {val}."
                )
            return int(val)


        if target_type is float:
            if isinstance(val, bool):
                raise ParameterValidationError(
                    f"Parameter '{key}' must be a float, not a bool."
                )
            return float(val)

        if target_type is list:
            if isinstance(val, str):
                raise ParameterValidationError(
                    f"Parameter '{key}' must be a list, but got a string. "
                    f"If you intended a JSON list, parse it before passing."
                )
            if isinstance(val, dict):
                raise ParameterValidationError(
                    f"Parameter '{key}' must be a list, got a dict."
                )
            if not isinstance(val, (list, tuple, set, frozenset)):
                raise ParameterValidationError(
                    f"Parameter '{key}' must be a list, got {type(val).__name__}."
                )
            return list(val)

        if target_type is dict:
            if not isinstance(val, dict):
                raise ParameterValidationError(
                    f"Parameter '{key}' must be a dict, got {type(val).__name__}."
                )
            return copy.deepcopy(val)

        if target_type is str:
            if not isinstance(val, str):
                raise ParameterValidationError(
                    f"Parameter '{key}' must be a string, got {type(val).__name__}."
                )
            return val

        return target_type(val)
    except ParameterValidationError:
        raise
    except (ValueError, TypeError) as e:
        raise ParameterValidationError(
            f"Parameter '{key}' must be of type {target_type.__name__}, but got value '{val}' of type {type(val).__name__}."
        ) from e


def _migrate_config_if_needed(d: Dict[str, Any]) -> Dict[str, Any]:
    # Only deep-copy when a migration is actually about to run, to avoid
    # the O(N) deepcopy cost in hot loops (parameter sweeps).  When the
    # first migration is added, gate the deepcopy on the schema version.
    # No migrations currently implemented.
    return d

def _compute_total_loss_db_from_scalars(
    *,
    distance_km: float,
    fiber_loss_db_km: float,
    filter_model: str,
    filter_awg_loss_db: float,
    N_channels: int,
    filter_fbg_loss_per_channel_db: float,
    modulator_insertion_loss_db: float,
    circulator_insertion_loss_db: float,
) -> float:
    """Single source of truth for the total-loss formula.

    Used by QKDParams._compute_total_loss_db(), from_dict(), and
    load_lim2014_*_params() so that all three paths stay consistent
    without any code duplication.
    """
    if filter_model == "awg":
        filter_loss = filter_awg_loss_db
    elif filter_model == "fbg_serial":
        filter_loss = N_channels * filter_fbg_loss_per_channel_db
    elif filter_model == "ideal":
        filter_loss = 0.0
    else:
        # _validate() also catches this, but raising here prevents the
        # channel from being built with 0 filter loss before _validate runs.
        raise ParameterValidationError(
            f"Unknown filter_model {filter_model!r} in loss computation. "
            f"Allowed: 'ideal', 'awg', 'fbg_serial'."
        )
    return (
        distance_km * fiber_loss_db_km
        + filter_loss
        + modulator_insertion_loss_db
        + circulator_insertion_loss_db
    )

@dataclass(frozen=True, slots=True)
class QKDParams:
    """
    High-level, immutable container for all QKD simulation parameters.
    """
    # 1. Fields without defaults (must come first)
    protocol: ProtocolFromModule
    source: OpticalSource
    channel: FiberChannel
    detector: ThresholdDetector
    num_bits: int
    photon_number_cap: int
    batch_size: int
    num_workers: int
    eps_sec: float
    eps_cor: float
    eps_pe: float
    eps_smooth: float
    security_proof: SecurityProof
    ci_method: ConfidenceBoundMethod
    # The following four fields are defaulted here so direct construction
    # matches from_dict()'s defaulting behavior.  They must come after the
    # no-default fields above (dataclass ordering rule).
    force_sequential: bool = False
    f_error_correction: float = 1.16
    enforce_monotonicity: bool = True
    assume_phase_equals_bit_error: bool = False

    # 2. Fields with defaults (must follow no-default fields)
    auto_optimize_lim2014: bool = False
    lp_solver_method: str = "highs"
    allow_unsafe_mdi_approx: bool = False
    require_tail_below: Optional[float] = DEFAULT_POISSON_TAIL_THRESHOLD
    master_seed: Optional[int] = None
    # New parameters
    eps_pa: float = 1e-10
    eps_sif: float = 1e-10
    
    # Ma 2005 Specific Enhancements
    assume_zero_background: bool = False 
    auto_optimize_ma2005: bool = False 
    use_weak_gllp: bool = False 
    system_duty_cycle: float = 1.0 
    f_ec_dynamic_config: Optional[Dict[str, Any]] = None

    # Lim 2014 Specific Enhancements
    security_constant_kappa: Optional[float] = None
    error_correction_model: str = "standard"

    # --- Paper Features ---
    decoder_architecture: DecoderArchitecture = DecoderArchitecture.LOCAL
    redundancy_M: int = 1
    ppbs_transmissivity: float = 0.66
    use_entangling_encoder: bool = False 
    use_adaptive_measurements: bool = False
    
    # [Chapman et al. 2018] Enhancements
    damping_parameter: Optional[float] = None # gamma for Amplitude Damping Channel
    encoding_rotation_angle: Optional[float] = None # theta_gamma for Coherent Scheme

    # SCM Parameters
    N_channels: int = 1
    modulation_index: float = 0.0
    target_carrier_frequency_hz: Optional[float] = None

    # Expanded Config for SCM Physics
    visibility_mismatch_dm: float = 0.0
    visibility_bias_drift_psi1: float = 0.0
    visibility_bias_drift_psi2: float = 0.0
    drift_step_size: float = 0.0

    # SCM Interference & Filtering Models
    cso_model: str = "approximate" # 'approximate', 'discrete_exact', or 'combinatorial_uniform'
    frequency_plan: Optional[List[float]] = None
    filter_model: str = "ideal" # 'ideal', 'awg', 'fbg_serial'
    filter_awg_loss_db: float = 3.0
    filter_fbg_loss_per_channel_db: float = 0.5 
    filter_fbg_bandwidth_hz: float = 1.3e9 
    filter_crosstalk_db: float = -30.0
    
    # [Improvement 1] Physical Circulator Loss for Serial Architectures
    # Default 0.4 dB per paper "Analysis of Subcarrier Multiplexed..."
    circulator_insertion_loss_db: float = 0.4 

    # Modulator Physics [Advancement 5]
    modulator_bandwidth_hz: Optional[float] = None
    modulator_insertion_loss_db: float = 6.0
    modulator_material: str = "linbo3"
    
    # [Advancement 1] WDM Hierarchy Support
    wdm_channel_count: int = 1
    wdm_channel_spacing_hz: float = 100e9 # Standard 100 GHz grid
    
    # [Improvement 3] Physical WDM Impairments (FWM/XPM/Raman)
    # Nonlinear coefficient gamma (1/W/km). Default ~1.3 for SMF-28.
    fiber_nonlinear_coeff: float = 1.3 
    # Per-channel launch power for coexisting classical WDM channels (dBm).
    # Default -20.0 dBm = 0.01 mW, representing a low-power classical channel
    # at the edge of coexistence.  Set to 0 for pure-QKD (no classical channels).
    wdm_channel_power_dbm: float = -20.0
    # [New] Raman Scattering Coefficient (~1e-9 per km per nm bandwidth
    # for standard SMF in the C-band with co-propagating classical channels).
    wdm_raman_coefficient: float = 0.0

    # [Advancement 4] Active Dispersion Control
    use_linear_modulation_approximation: bool = True
    dispersion_parameter_ps_nm_km: float = 0.0
    carrier_wavelength_nm: float = 1550.0
    active_dispersion_compensation: bool = False 

    # Option J: pure fiber attenuation coefficient (dB/km).  Distinct from
    # channel.fiber_loss_db_km, which in DWDM mode is derived from
    # total_loss_db / distance_km (via FiberChannel.from_total_loss) and
    # therefore includes AWG filter loss spread across distance.  Used by
    # _compute_raman_dark_rate_hz for the nonlinear effective length l_eff.
    fiber_loss_db_km: float = 0.2

    def __post_init__(self) -> None:
        # Defensive deep-copy of every mutable container field so callers
        # cannot mutate QKDParams state via the dict/list they passed in,
        # and so dataclasses.replace() does not share one container across
        # multiple QKDParams instances.  Handles both bare containers
        # (List, Dict) and Optional[Container[...]] / Container[...] | None
        # by unwrapping the Optional before checking the origin.
        import types as _types
        _hints = get_type_hints(type(self))
        _mutable_origins = (dict, list, set, frozenset)
        _union_origins = (Union, getattr(_types, "UnionType", ()))
        for _f in dataclasses.fields(self):
            _ann = _hints.get(_f.name)
            # Unwrap Optional[X] / X | None so Optional[List[...]] is deep-copied.
            if get_origin(_ann) in _union_origins:
                _args = [a for a in get_args(_ann) if a is not type(None)]
                # Unwrap any mutable-origin arg in a multi-way union
                # (e.g. Union[List[int], Dict[str, int], None]).
                for _a in _args:
                    if get_origin(_a) in _mutable_origins:
                        _ann = _a
                        break
            if get_origin(_ann) in _mutable_origins:
                _v = getattr(self, _f.name)
                if _v is not None:
                    object.__setattr__(self, _f.name, copy.deepcopy(_v))

        # Defensive deep-copy of the four component objects so callers cannot
        # mutate QKDParams state via the protocol/source/channel/detector they
        # passed in, and so dataclasses.replace() does not share component
        # state across multiple QKDParams instances.  _attach_proof_module()
        # below shallow-copies the protocol shell, so the deep copy here is
        # the authoritative isolation boundary.
        for _comp_name in ("protocol", "source", "channel", "detector"):
            object.__setattr__(self, _comp_name, copy.deepcopy(getattr(self, _comp_name)))

        for _str_field in ("lp_solver_method", "filter_model", "cso_model",
                           "error_correction_model", "modulator_material"):
            _val = getattr(self, _str_field)
            if not isinstance(_val, str):
                raise ParameterValidationError(
                    f"{_str_field} must be a string, got {type(_val).__name__}."
                )
            object.__setattr__(self, _str_field, _val.lower())
        # Coerce enum fields so direct construction matches from_dict behavior.
        object.__setattr__(self, "security_proof",
            _coerce_enum(SecurityProof, self.security_proof))
        object.__setattr__(self, "ci_method",
            _coerce_enum(ConfidenceBoundMethod, self.ci_method))
        object.__setattr__(self, "decoder_architecture",
            _coerce_enum(DecoderArchitecture, self.decoder_architecture))
        self._assert_component_interfaces()
        self._validate()
        self._attach_proof_module()

    def _assert_component_interfaces(self) -> None:
        components: Dict[str, Any] = {
            "protocol": self.protocol, "source": self.source,
            "channel": self.channel, "detector": self.detector,
        }
        for name, comp in components.items():
            if not (hasattr(comp, "to_config_dict") and callable(comp.to_config_dict)):
                raise ConfigurationError(f"Component '{name}' of type {type(comp).__name__} must implement a to_config_dict() method.")
        if not hasattr(self.protocol, "protocol_name"):
            raise ConfigurationError(f"Protocol {type(self.protocol).__name__} must have a 'protocol_name' property.")

    def _attach_proof_module(self) -> None:
        """Attach the appropriate proof module based on security_proof + protocol.

        Runs in __post_init__ so direct construction and from_dict() produce
        identical protocol state.  Always shallow-copies the protocol shell
        first (and clears any pre-existing ``proof_module``) so that
        ``dataclasses.replace()`` on a QKDParams whose protocol already
        carries a proof module does not leak the stale proof into the copy.
        """
        _detached = copy.copy(self.protocol)
        object.__setattr__(self, "protocol", _detached)
        object.__setattr__(_detached, "source", self.source)
        # Always clear first: switching security_proof away from PAPER_2009
        # via replace() must not leave the old proof attached.
        if hasattr(_detached, "proof_module"):
            object.__setattr__(_detached, "proof_module", None)
        if self.security_proof == SecurityProof.PAPER_2009:
            if (hasattr(_detached, "protocol_name")
                    and _detached.protocol_name in ("bb84-decoy", "b92")):
                try:
                    from .proofs.paper_2009_individual import Paper2009IndividualAttackProof
                except ImportError as e:
                    raise ConfigurationError(
                        f"security_proof=PAPER_2009 requires the "
                        f"paper_2009_individual proof module, which could not "
                        f"be imported: {e}"
                    ) from e
                object.__setattr__(
                    _detached, "proof_module",
                    Paper2009IndividualAttackProof(_detached),
                )
            else:
                raise ConfigurationError(
                    f"security_proof=PAPER_2009 is only compatible with "
                    f"protocols 'bb84-decoy' and 'b92', got "
                    f"{getattr(_detached, 'protocol_name', type(_detached).__name__)!r}."
                )
        # For other security_proofs (LIM_2014, MDI_QKD, ...), proof_module
        # remains None.  Downstream code MUST check before invoking it.
        # If a future proof gains a dedicated module, add the branch here.

    def _compute_total_loss_db(self) -> float:
        """Compute total channel loss from QKDParams scalar fields.

        Delegates to the standalone _compute_total_loss_db_from_scalars()
        so that from_dict() and the factory functions can reuse the same
        formula without constructing an intermediate QKDParams.
        """
        return _compute_total_loss_db_from_scalars(
            distance_km=self.channel.distance_km,
            fiber_loss_db_km=self.fiber_loss_db_km,
            filter_model=self.filter_model,
            filter_awg_loss_db=self.filter_awg_loss_db,
            N_channels=self.N_channels,
            filter_fbg_loss_per_channel_db=self.filter_fbg_loss_per_channel_db,
            modulator_insertion_loss_db=self.modulator_insertion_loss_db,
            circulator_insertion_loss_db=self.circulator_insertion_loss_db,
        )

    def with_channel_rebuilt(self, **changes: Any) -> "QKDParams":
        """Return a new QKDParams with `changes` applied AND the channel
        rebuilt to match the new loss-affecting fields.

        This is the only supported way to change loss-affecting fields
        (filter_model, filter_*_loss_db, modulator_insertion_loss_db,
        circulator_insertion_loss_db, N_channels, fiber_loss_db_km,
        channel.distance_km) on an existing QKDParams.  Using
        dataclasses.replace() directly will trigger the loss-consistency
        guard in _validate().
        """
        # Special-case distance_km first — it lives on channel, not on QKDParams,
        # so it must be popped before the unknown-key check.
        if "distance_km" in changes:
            new_distance_km = float(changes.pop("distance_km"))
            if not math.isfinite(new_distance_km) or new_distance_km < 0:
                raise ParameterValidationError(
                    f"distance_km must be a non-negative finite number, got {new_distance_km}."
                )
        else:
            new_distance_km = self.channel.distance_km

        # Validate changes: reject unknown keys and 'channel' (rebuilt
        # automatically by this method).
        _field_names = {f.name for f in dataclasses.fields(self)}
        _unknown = set(changes) - _field_names
        if _unknown:
            raise ParameterValidationError(
                f"Unknown fields in with_channel_rebuilt changes: {sorted(_unknown)}"
            )
        if "channel" in changes:
            raise ParameterValidationError(
                "with_channel_rebuilt() rebuilds the channel automatically; "
                "do not pass 'channel' in changes."
            )
        _new = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        _new.update(changes)
        new_total_loss = _compute_total_loss_db_from_scalars(
            distance_km=new_distance_km,
            fiber_loss_db_km=_new["fiber_loss_db_km"],
            filter_model=_new["filter_model"],
            filter_awg_loss_db=_new["filter_awg_loss_db"],
            N_channels=_new["N_channels"],
            filter_fbg_loss_per_channel_db=_new["filter_fbg_loss_per_channel_db"],
            modulator_insertion_loss_db=_new["modulator_insertion_loss_db"],
            circulator_insertion_loss_db=_new["circulator_insertion_loss_db"],
        )
        new_channel = FiberChannel.from_total_loss(new_distance_km, new_total_loss)
        return dataclasses.replace(self, channel=new_channel, **changes)

    def _validate(self) -> None:
        if not isinstance(self.photon_number_cap, int) or isinstance(self.photon_number_cap, bool) or self.photon_number_cap < 0:
            raise ParameterValidationError("photon_number_cap must be a non-negative integer.")

        # distance_km lives on the channel; from_dict validates it but
        # direct construction with a FiberChannel built from negative
        # distance would otherwise slip through (the loss-consistency
        # guard at the end of _validate passes because both sides match).
        if not isinstance(self.channel.distance_km, (int, float)) or isinstance(self.channel.distance_km, bool):
            raise ParameterValidationError(
                f"channel.distance_km must be a number, got {type(self.channel.distance_km).__name__}."
            )
        if not math.isfinite(self.channel.distance_km) or self.channel.distance_km < 0:
            raise ParameterValidationError(
                f"channel.distance_km must be a non-negative finite number, got {self.channel.distance_km}."
            )


        # Validate every declared bool field is actually a bool (direct
        # construction bypasses _coerce_type's parse_bool path).
        # Resolve annotations through get_type_hints because PEP 563
        # (__future__ annotations) stores them as strings.
        _type_hints = get_type_hints(type(self))
        for _bf in dataclasses.fields(self):
            if _type_hints.get(_bf.name) is bool and not isinstance(getattr(self, _bf.name), bool):
                raise ParameterValidationError(
                    f"{_bf.name} must be a bool, got {type(getattr(self, _bf.name)).__name__}."
                )

        if not (1.0 <= self.f_error_correction <= 2.0):
            raise ParameterValidationError(f"f_error_correction must be in [1.0, 2.0], got {self.f_error_correction}.")
        if self.f_error_correction > 1.2:
            logger.debug(
                f"f_error_correction is typically between 1.0 and 1.2, but got "
                f"{self.f_error_correction}. Enable DEBUG logging to suppress."
            )
        # Ensure all declared float fields (including Optional[float] and
        # float | None) are finite — NaN/Inf silently pass range checks
        # because NaN comparisons always return False.
        import types as _types
        _union_origins = (Union, getattr(_types, "UnionType", ()))
        for _ff in dataclasses.fields(self):
            _ann = _type_hints.get(_ff.name)
            _is_float = (
                _ann is float
                or (get_origin(_ann) in _union_origins and float in get_args(_ann))
            )
            if _is_float:
                _fv = getattr(self, _ff.name)
                if isinstance(_fv, bool):
                    raise ParameterValidationError(
                        f"{_ff.name} must be a float, got bool {_fv}."
                    )
                if isinstance(_fv, float) and not math.isfinite(_fv):
                    raise ParameterValidationError(
                        f"{_ff.name} must be finite, got {_fv}."
                    )
        if self.f_ec_dynamic_config is not None:
            if not isinstance(self.f_ec_dynamic_config, dict):
                raise ParameterValidationError(
                    f"f_ec_dynamic_config must be a dict, got {type(self.f_ec_dynamic_config).__name__}."
                )
            if not self.f_ec_dynamic_config:
                raise ParameterValidationError("f_ec_dynamic_config must not be empty when provided.")
            if "model" not in self.f_ec_dynamic_config:
                raise ParameterValidationError("f_ec_dynamic_config must contain a 'model' key (e.g., 'linear').")
            _ec_model = self.f_ec_dynamic_config["model"]
            if not isinstance(_ec_model, str) or not _ec_model.strip():
                raise ParameterValidationError(
                    f"f_ec_dynamic_config['model'] must be a non-empty string, got {_ec_model!r}."
                )
            _allowed_ec_models = {"linear", "constant"}
            if _ec_model.strip().lower() not in _allowed_ec_models:
                raise ParameterValidationError(
                    f"f_ec_dynamic_config['model']={_ec_model!r} not recognized. "
                    f"Allowed: {sorted(_allowed_ec_models)}."
                )
            # Linear model requires slope and intercept (finite numbers).
            if _ec_model.strip().lower() == "linear":
                for _req_key in ("slope", "intercept"):
                    if _req_key not in self.f_ec_dynamic_config:
                        raise ParameterValidationError(
                            f"f_ec_dynamic_config['{_req_key}'] is required when model='linear'."
                        )
                    _val = self.f_ec_dynamic_config[_req_key]
                    if isinstance(_val, bool) or not isinstance(_val, (int, float)):
                        raise ParameterValidationError(
                            f"f_ec_dynamic_config['{_req_key}'] must be a number, "
                            f"got {type(_val).__name__}."
                        )
                    if not math.isfinite(_val):
                        raise ParameterValidationError(
                            f"f_ec_dynamic_config['{_req_key}'] must be finite, got {_val}."
                        )

        # security_constant_kappa must be positive when set (Lim 2014 proof).
        if self.security_constant_kappa is not None:
            if not isinstance(self.security_constant_kappa, (int, float)) or isinstance(self.security_constant_kappa, bool):
                raise ParameterValidationError(
                    f"security_constant_kappa must be a number, got {type(self.security_constant_kappa).__name__}."
                )
            if not math.isfinite(self.security_constant_kappa) or self.security_constant_kappa <= 0.0:
                raise ParameterValidationError(
                    f"security_constant_kappa must be a positive finite number, got {self.security_constant_kappa}."
                )

        # modulator_material: lowercased in __post_init__; restrict to known.
        _allowed_modulator_materials = {"linbo3", "silicon", "gaas"}
        if self.modulator_material not in _allowed_modulator_materials:
            raise ParameterValidationError(
                f"Invalid modulator_material {self.modulator_material!r}. "
                f"Allowed: {sorted(_allowed_modulator_materials)}."
            )


        for eps_name, eps_val in (("eps_sec", self.eps_sec), ("eps_cor", self.eps_cor),
                                   ("eps_pe", self.eps_pe), ("eps_smooth", self.eps_smooth),
                                   ("eps_pa", self.eps_pa), ("eps_sif", self.eps_sif)):
            if not isinstance(eps_val, (int, float)) or isinstance(eps_val, bool):
                raise ParameterValidationError(
                    f"{eps_name} must be a number, got {type(eps_val).__name__}."
                )
            if not math.isfinite(eps_val):
                raise ParameterValidationError(f"{eps_name} must be finite, got {eps_val}.")
            if eps_val <= 0.0:
                raise ParameterValidationError(f"{eps_name} must be positive, got {eps_val}.")
            if eps_val >= 1.0:
                raise ParameterValidationError(
                    f"{eps_name} must be < 1.0 (absurd security parameter), got {eps_val}."
                )

        total_epsilon = self.eps_sec + self.eps_cor + self.eps_pe + self.eps_smooth + self.eps_pa + self.eps_sif
        if not math.isfinite(total_epsilon) or total_epsilon >= 1.0:
            raise ParameterValidationError(f"The sum of all epsilon values must be finite and < 1.0, but got {total_epsilon}.")
        
        # Validate security_proof ↔ protocol pairing.  Only MDI was checked
        # before, allowing physically meaningless combinations like
        # LIM_2014 + B92Protocol to slip through.
        from .protocols import (
            BB84DecoyProtocol as _BB84,
            MDIQKDProtocol as _MDI,
            B92Protocol as _B92,
            RedundantTransmissionProtocol as _RT,
        )
        _proof_protocol_pairs = {
            SecurityProof.MDI_QKD: (_MDI,),
            SecurityProof.LIM_2014: (_BB84,),
            SecurityProof.PAPER_2009: (_BB84, _B92),
        }
        if self.security_proof not in _proof_protocol_pairs:
            raise ParameterValidationError(
                f"security_proof={self.security_proof.value!r} has no protocol-pairing "
                f"rule defined. Add an entry to _proof_protocol_pairs or use a "
                f"supported proof."
            )
        _expected_protos = _proof_protocol_pairs[self.security_proof]
        if not isinstance(self.protocol, _expected_protos):
            raise ParameterValidationError(
                f"security_proof={self.security_proof.value!r} requires one of "
                f"{[p.__name__ for p in _expected_protos]}, got {type(self.protocol).__name__}."
            )

        _lp_methods_lower = {m.lower() for m in LP_SOLVER_METHODS}
        if self.lp_solver_method not in _lp_methods_lower:
            raise ParameterValidationError(
                f"Invalid lp_solver_method '{self.lp_solver_method}'. "
                f"Allowed methods: {sorted(LP_SOLVER_METHODS)}"
            )

        if not isinstance(self.N_channels, int) or isinstance(self.N_channels, bool) or self.N_channels < 1:
            raise ParameterValidationError("N_channels must be an integer >= 1.")
        
        if not (0.0 <= self.modulation_index <= 1.0):
            raise ParameterValidationError("modulation_index must be a float between 0.0 and 1.0.")

        if self.N_channels > 1 and self.modulation_index == 0.0:
            logger.warning("N_channels > 1 but modulation_index is 0.0. No intermodulation noise will be calculated.")
            
        if self.modulator_bandwidth_hz is not None and self.frequency_plan:
            max_freq = max(self.frequency_plan)
            if max_freq > self.modulator_bandwidth_hz:
                logger.warning(
                    f"Frequency plan (max: {max_freq/1e9:.1f} GHz) exceeds modulator bandwidth "
                    f"({self.modulator_bandwidth_hz/1e9:.1f} GHz). Physics model applies V_pi degradation."
                )
        # --- Additional physical-parameter validations ---
        if self.visibility_bias_drift_psi1 < 0.0:
            raise ParameterValidationError(
                f"visibility_bias_drift_psi1 must be non-negative, got {self.visibility_bias_drift_psi1}."
            )
        if self.visibility_bias_drift_psi2 < 0.0:
            raise ParameterValidationError(
                f"visibility_bias_drift_psi2 must be non-negative, got {self.visibility_bias_drift_psi2}."
            )
        if self.fiber_nonlinear_coeff < 0.0:
            raise ParameterValidationError(
                f"fiber_nonlinear_coeff must be non-negative, got {self.fiber_nonlinear_coeff}."
            )
        if self.wdm_raman_coefficient < 0.0:
            raise ParameterValidationError(
                f"wdm_raman_coefficient must be non-negative, got {self.wdm_raman_coefficient}."
            )
        if self.master_seed is not None:
            if not isinstance(self.master_seed, int) or isinstance(self.master_seed, bool):
                raise ParameterValidationError(
                    f"master_seed must be an int, got {type(self.master_seed).__name__}."
                )
            if self.master_seed < 0:
                raise ParameterValidationError(
                    f"master_seed must be non-negative when set, got {self.master_seed}."
                )
        if self.frequency_plan is not None:
            if not isinstance(self.frequency_plan, list):
                raise ParameterValidationError(
                    f"frequency_plan must be a list, got {type(self.frequency_plan).__name__}."
                )
            if len(self.frequency_plan) != self.N_channels:
                raise ParameterValidationError(
                    f"frequency_plan length ({len(self.frequency_plan)}) must match "
                    f"N_channels ({self.N_channels})."
                )
            for i, freq in enumerate(self.frequency_plan):
                if not isinstance(freq, (int, float)) or isinstance(freq, bool):
                    raise ParameterValidationError(
                        f"frequency_plan[{i}] must be a number, got {type(freq).__name__}."
                    )
                if not math.isfinite(freq) or freq <= 0.0:
                    raise ParameterValidationError(
                        f"frequency_plan[{i}] must be a positive finite number, got {freq}."
                    )
            if len(set(self.frequency_plan)) != len(self.frequency_plan):
                raise ParameterValidationError(
                    f"frequency_plan contains duplicate frequencies: {self.frequency_plan}"
                )
        
        if not isinstance(self.redundancy_M, int) or isinstance(self.redundancy_M, bool) or self.redundancy_M < 1:
            raise ParameterValidationError("redundancy_M must be a positive integer.")

        if self.decoder_architecture == DecoderArchitecture.ENTANGLING and self.redundancy_M < 2:
            logger.warning("Using ENTANGLING decoder with redundancy_M=1 may not provide quantum advantage.")
            
        if not (-200.0 <= self.dispersion_parameter_ps_nm_km <= 200.0):
            raise ParameterValidationError(
                f"dispersion_parameter_ps_nm_km out of plausible range [-200, 200]: {self.dispersion_parameter_ps_nm_km}."
            )
            
        if not (0.0 < self.system_duty_cycle <= 1.0):
            raise ParameterValidationError("system_duty_cycle must be in (0, 1].")
            
        if not isinstance(self.wdm_channel_count, int) or isinstance(self.wdm_channel_count, bool) or self.wdm_channel_count < 1:
            raise ParameterValidationError("wdm_channel_count must be >= 1.")
            
        if self.circulator_insertion_loss_db < 0:
            raise ParameterValidationError("circulator_insertion_loss_db cannot be negative.")
             
        if self.error_correction_model not in ("standard", "lim", "tomamichel"):
            raise ParameterValidationError(f"Invalid error_correction_model: {self.error_correction_model}")
            
        # Chapman 2018 Validations
        if self.damping_parameter is not None:
            if not (0.0 <= self.damping_parameter <= 1.0):
                raise ParameterValidationError("damping_parameter must be in [0, 1].")

        for _int_field, _iv in (("num_bits", self.num_bits),
                                 ("batch_size", self.batch_size),
                                 ("num_workers", self.num_workers)):
            if not isinstance(_iv, int) or isinstance(_iv, bool):
                raise ParameterValidationError(
                    f"{_int_field} must be an int, got {type(_iv).__name__}."
                )
            if _iv < 1:
                raise ParameterValidationError(f"{_int_field} must be >= 1.")
        if self.batch_size > self.num_bits:
            logger.warning(
                f"batch_size ({self.batch_size}) > num_bits ({self.num_bits}); "
                f"downstream batching may produce empty batches."
            )

        if not (0.0 < self.ppbs_transmissivity <= 1.0):
            raise ParameterValidationError(f"ppbs_transmissivity must be in (0, 1], got {self.ppbs_transmissivity}.")
        if not math.isfinite(self.carrier_wavelength_nm) or self.carrier_wavelength_nm <= 0.0:
            raise ParameterValidationError(f"carrier_wavelength_nm must be a positive finite number, got {self.carrier_wavelength_nm}.")
        if self.fiber_loss_db_km < 0.0:
            raise ParameterValidationError(f"fiber_loss_db_km must be non-negative, got {self.fiber_loss_db_km}.")
        if self.filter_awg_loss_db < 0.0:
            raise ParameterValidationError(f"filter_awg_loss_db must be non-negative, got {self.filter_awg_loss_db}.")
        if self.modulator_insertion_loss_db < 0.0:
            raise ParameterValidationError(f"modulator_insertion_loss_db must be non-negative, got {self.modulator_insertion_loss_db}.")
        if self.filter_model not in ("ideal", "awg", "fbg_serial"):
            raise ParameterValidationError(f"Invalid filter_model: {self.filter_model!r}. Allowed: 'ideal', 'awg', 'fbg_serial'.")
        if self.cso_model not in ("approximate", "discrete_exact", "combinatorial_uniform"):
            raise ParameterValidationError(f"Invalid cso_model: {self.cso_model!r}. Allowed: 'approximate', 'discrete_exact', 'combinatorial_uniform'.")
        if self.filter_fbg_loss_per_channel_db < 0.0:
            raise ParameterValidationError(f"filter_fbg_loss_per_channel_db must be non-negative, got {self.filter_fbg_loss_per_channel_db}.")
        if not math.isfinite(self.filter_fbg_bandwidth_hz) or self.filter_fbg_bandwidth_hz <= 0.0:
            raise ParameterValidationError(f"filter_fbg_bandwidth_hz must be a positive finite number, got {self.filter_fbg_bandwidth_hz}.")
        if self.require_tail_below is not None and not (0.0 < self.require_tail_below < 1.0):
            raise ParameterValidationError(f"require_tail_below must be in (0, 1) or None, got {self.require_tail_below}.")
        if self.target_carrier_frequency_hz is not None and self.target_carrier_frequency_hz <= 0.0:
            raise ParameterValidationError(f"target_carrier_frequency_hz must be positive when set, got {self.target_carrier_frequency_hz}.")
        if self.modulator_bandwidth_hz is not None and self.modulator_bandwidth_hz <= 0.0:
            raise ParameterValidationError(f"modulator_bandwidth_hz must be positive when set, got {self.modulator_bandwidth_hz}.")
        if self.visibility_mismatch_dm < 0.0:
            raise ParameterValidationError(f"visibility_mismatch_dm must be non-negative, got {self.visibility_mismatch_dm}.")
        if self.drift_step_size < 0.0:
            raise ParameterValidationError(f"drift_step_size must be non-negative, got {self.drift_step_size}.")
        if not math.isfinite(self.wdm_channel_spacing_hz) or self.wdm_channel_spacing_hz <= 0.0:
            raise ParameterValidationError(f"wdm_channel_spacing_hz must be a positive finite number, got {self.wdm_channel_spacing_hz}.")

        if self.filter_crosstalk_db > 0.0:
            raise ParameterValidationError(
                f"filter_crosstalk_db must be non-positive (a loss, not a gain), "
                f"got {self.filter_crosstalk_db}."
            )
        if not (-50.0 <= self.wdm_channel_power_dbm <= 30.0):
            raise ParameterValidationError(
                f"wdm_channel_power_dbm out of plausible range [-50, 30] dBm, "
                f"got {self.wdm_channel_power_dbm}."
            )

        if self.encoding_rotation_angle is not None:
            if not (0.0 <= self.encoding_rotation_angle <= math.pi):
                raise ParameterValidationError(
                    f"encoding_rotation_angle must be in [0, π], got {self.encoding_rotation_angle}."
                )

        # --- Channel loss consistency guard ---
        # Catches dataclasses.replace() calls that change loss-affecting
        # fields (filter_model, filter losses, modulator/circulator loss,
        # fiber_loss_db_km) without rebuilding the channel.  Without this
        # check, the channel's total_loss_db silently desynchronizes from
        # the QKDParams fields, producing incorrect physics.  Use
        # QKDParams.with_channel_rebuilt() to change loss fields safely.
        _expected_total_loss = self._compute_total_loss_db()
        if not (math.isfinite(_expected_total_loss) and math.isfinite(self.channel.total_loss_db)):
            raise ParameterValidationError(
                f"Total loss must be finite; got channel.total_loss_db="
                f"{self.channel.total_loss_db}, recomputed={_expected_total_loss}."
            )
        if abs(self.channel.total_loss_db - _expected_total_loss) > 1e-9 * max(1.0, abs(_expected_total_loss)):
            raise ParameterValidationError(
                f"channel.total_loss_db ({self.channel.total_loss_db}) does not match "
                f"recomputed loss ({_expected_total_loss}). Use "
                f"QKDParams.with_channel_rebuilt() or from_dict() when changing "
                f"loss-affecting fields."
            )

    def to_serializable_dict(self, *, redact_seed: bool = True) -> Dict[str, Any]:
        params_dict = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "protocol_name": self.protocol.protocol_name,
            "protocol_config": self.protocol.to_config_dict(),
            "source_config": self.source.to_config_dict(),
            "channel_config": self.channel.to_config_dict(),
            "detector_config": self.detector.to_config_dict(),
            **self.to_summary_dict(redact=redact_seed, enums_as_values=True)
        }
        result = _to_serializable(params_dict)
        if redact_seed:
            _redact_seed_recursive(result)
        _validate_serializable_object(result)
        return result

    def to_summary_dict(self, redact: bool = True, enums_as_values: bool = True) -> Dict[str, Any]:
        summary = {}
        for f in dataclasses.fields(self):
            if f.name not in {"protocol", "source", "channel", "detector"}:
                value = getattr(self, f.name)
                if isinstance(value, (dict, list, set, frozenset, tuple)):
                    value = copy.deepcopy(value)
                if enums_as_values and isinstance(value, Enum):
                    value = value.value
                summary[f.name] = value

        # Redact master_seed by *dropping* the key, not by writing None.
        # Writing None destroys the seed on every serialize→deserialize
        # cycle (from_dict treats a missing key the same as null), so
        # callers who want to preserve the seed must pass redact=False.
        if redact and "master_seed" in summary:
            summary.pop("master_seed")
        return summary

    def __repr__(self) -> str:
        try:
            summary = self.to_summary_dict(redact=True, enums_as_values=False)
            summary_str = ", ".join(f"{k}={v!r}" for k, v in summary.items())
            return f"QKDParams(protocol={self.protocol.protocol_name}, {summary_str})"
        except Exception:
            return f"<QKDParams repr failed; protocol={type(self.protocol).__name__}>"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> QKDParams:
        """Factory method to deserialize parameters."""
        if not isinstance(d, dict):
            raise ConfigurationError(
                f"from_dict() expects a dict, got {type(d).__name__}."
            )

        d_copy = dict(d)  # shallow copy so _coerce_type.pop doesn't mutate caller
        d_copy = _migrate_config_if_needed(d_copy)

        try:
            source_config = d_copy.pop("source_config")

            if not isinstance(source_config, dict):
                raise ConfigurationError(
                    f"source_config must be a dict, got {type(source_config).__name__}."
                )
            
            error_model_str = source_config.get("error_model", "random_gaussian")
            error_model = _coerce_enum(SourceErrorModel, error_model_str)
            
            # These three values are read from the top-level config and
            # passed to BOTH QKDParams and OpticalSource.from_pulse_ensemble().
            # If OpticalSource.to_config_dict() ever emits them, from_dict
            # must compare the two sources and reject conflicts.  See
            # test_round_trip_identity in tests/test_params.py.
            n_channels = _coerce_type(d_copy, "N_channels", int, required=False, default=1)
            modulation_index = _coerce_type(d_copy, "modulation_index", float, required=False, default=0.0)
            use_linear_approx = _coerce_type(d_copy, "use_linear_modulation_approximation", bool, required=False, default=True)

            pulse_list = source_config.get("pulse_configs", [])
            pulse_period = source_config.get("pulse_period_ns")
            stats_type_str = source_config.get("statistics_type", "poisson")
            stats_type = _coerce_enum(SourceStatisticsType, stats_type_str)
            intensity_jitter = _coerce_type(source_config, "intensity_jitter", float, required=False, default=0.0)
            is_bidirectional = _coerce_type(source_config, "is_bidirectional", bool, required=False, default=False)

            use_small_angle_approx = _coerce_type(source_config, "use_small_angle_approximation", bool, required=False, default=True)
            ideal_emission_prob = _coerce_type(source_config, "ideal_emission_probability", float, required=False, default=1.0)
            adversarial_block_size = _coerce_type(source_config, "adversarial_block_size", int, required=False, default=1000)

            # Validate each pulse_config entry before constructing so the
            # error names the offending index/key instead of a generic
            # TypeError.
            from .datatypes import PulseTypeConfig as _PTC
            _allowed_pulse_keys = set(_PTC.__dataclass_fields__.keys())
            pulse_configs = []
            for _i, _pc in enumerate(pulse_list):
                if not isinstance(_pc, dict):
                    raise ConfigurationError(
                        f"source_config.pulse_configs[{_i}] must be a dict, "
                        f"got {type(_pc).__name__}."
                    )
                _unknown = set(_pc) - _allowed_pulse_keys
                if _unknown:
                    raise ConfigurationError(
                        f"source_config.pulse_configs[{_i}] has unknown keys: {sorted(_unknown)}"
                    )
                pulse_configs.append(PulseTypeConfig(**_pc))

            if pulse_period is not None:
                pulse_period = float(pulse_period)
                if not math.isfinite(pulse_period) or pulse_period <= 0:
                    raise ParameterValidationError(
                        f"pulse_period_ns must be a positive finite number, got {pulse_period}."
                    )
                source_rate = 1e9 / pulse_period
            else:
                source_rate = 1e9

            from qkd.datatypes import OpticalSourceConfig  # add to imports at top of file if not present
            optical_config = OpticalSourceConfig(
                source_rate=1e9,
                pulse_configs=pulse_configs,
                statistics_type=SourceStatisticsType.POISSON,
                error_model=SourceErrorModel.RANDOM_GAUSSIAN,
                intensity_jitter=0.0,
                modulation_index=0.0,
                N_channels=1,
                use_small_angle_approximation=True,
                use_linear_modulation_approximation=True,
                ideal_emission_probability=1.0,
                adversarial_block_size=1000,
                is_bidirectional=False,
            )
            source = OpticalSource.create(optical_config)

            # Detect unknown keys in source_config.  The three SCM fields
            # below are also accepted (and conflict-checked) so that
            # OpticalSource.to_config_dict() can safely emit them without
            # breaking round-trip identity.
            _known_source_keys = {
                "pulse_configs", "pulse_period_ns", "statistics_type",
                "intensity_jitter", "error_model", "is_bidirectional",
                "use_small_angle_approximation", "ideal_emission_probability",
                "adversarial_block_size",
                "N_channels", "modulation_index", "use_linear_modulation_approximation",
            }
            _unknown_source = set(source_config) - _known_source_keys
            if _unknown_source:
                raise ConfigurationError(
                    f"Unknown keys in source_config: {sorted(_unknown_source)}"
                )
            # Conflict check: if source_config carries SCM fields, they must
            # agree with the top-level values already extracted.
            _scm_conflicts = []
            if "N_channels" in source_config and source_config["N_channels"] != n_channels:
                _scm_conflicts.append(("N_channels", source_config["N_channels"], n_channels))
            if "modulation_index" in source_config and not math.isclose(
                float(source_config["modulation_index"]), modulation_index, rel_tol=1e-9, abs_tol=0.0
            ):
                _scm_conflicts.append(("modulation_index", source_config["modulation_index"], modulation_index))
            if "use_linear_modulation_approximation" in source_config and bool(source_config["use_linear_modulation_approximation"]) != use_linear_approx:
                _scm_conflicts.append(("use_linear_modulation_approximation", source_config["use_linear_modulation_approximation"], use_linear_approx))
            if _scm_conflicts:
                raise ConfigurationError(
                    "source_config SCM fields conflict with top-level values: "
                    + "; ".join(f"{k}: source={sv!r} vs top-level={tv!r}" for k, sv, tv in _scm_conflicts)
                )

            channel_config = d_copy.pop("channel_config")
            if not isinstance(channel_config, dict):
                raise ConfigurationError(
                    f"channel_config must be a dict, got {type(channel_config).__name__}."
                )
            distance_km = float(channel_config["distance_km"])
            # Read top-level fiber_loss_db_km (emitted by to_summary_dict) as
            # the authoritative pure-fiber-loss value; fall back to the value
            # inside channel_config for older configs.  Using _coerce_type
            # also pops the top-level key so it doesn't trigger the
            # "unknown keys" error at the end of from_dict.
            fiber_loss_db_km = _coerce_type(
                d_copy, "fiber_loss_db_km", float, required=False,
                default=float(channel_config.get("fiber_loss_db_km", 0.2)),
            )

            if not math.isfinite(distance_km) or distance_km < 0:
                raise ParameterValidationError(
                    f"distance_km must be a non-negative finite number, got {distance_km}."
                )
            if not math.isfinite(fiber_loss_db_km) or fiber_loss_db_km < 0:
                raise ParameterValidationError(
                    f"fiber_loss_db_km must be a non-negative finite number, got {fiber_loss_db_km}."
                )

            # Coerce channel-affecting params consistently (same values used
            # for both channel reconstruction and QKDParams fields below).
            filter_model = _coerce_type(d_copy, "filter_model", str, required=False, default="ideal").lower()
            filter_awg_loss_db = _coerce_type(d_copy, "filter_awg_loss_db", float, required=False, default=3.0)
            filter_fbg_loss_per_channel_db = _coerce_type(d_copy, "filter_fbg_loss_per_channel_db", float, required=False, default=0.5)
            modulator_insertion_loss_db = _coerce_type(d_copy, "modulator_insertion_loss_db", float, required=False, default=6.0)
            circulator_insertion_loss_db = _coerce_type(d_copy, "circulator_insertion_loss_db", float, required=False, default=0.4)

            # Compute total loss using the single source-of-truth formula.
            total_loss_db = _compute_total_loss_db_from_scalars(
                distance_km=distance_km,
                fiber_loss_db_km=fiber_loss_db_km,
                filter_model=filter_model,
                filter_awg_loss_db=filter_awg_loss_db,
                N_channels=n_channels,
                filter_fbg_loss_per_channel_db=filter_fbg_loss_per_channel_db,
                modulator_insertion_loss_db=modulator_insertion_loss_db,
                circulator_insertion_loss_db=circulator_insertion_loss_db,
            )
            channel = FiberChannel.from_total_loss(distance_km, total_loss_db)

            # channel_config is fully consumed by the explicit reads above;
            # extra keys (e.g. total_loss_db emitted by FiberChannel.to_config_dict)
            # are informational only and intentionally ignored to preserve
            # round-trip identity.

            detector_config = d_copy.pop("detector_config")
            if not isinstance(detector_config, dict):
                raise ConfigurationError(
                    f"detector_config must be a dict, got {type(detector_config).__name__}."
                )
            if "double_click_policy" in detector_config:
                detector_config["double_click_policy"] = _coerce_enum(DoubleClickPolicy, detector_config["double_click_policy"])
            detector_config = {k: v for k, v in detector_config.items() if not k.startswith('_')}
            from .datatypes import DetectorType as _DT
            from .detectors import DeadTimeModel as _DTM, AfterpulseModel as _AM
            _enum_map = {'detector_type': _DT, 'dead_time_model': _DTM, 'afterpulse_model': _AM}
            for _ek, _ec in _enum_map.items():
                if _ek in detector_config:
                    detector_config[_ek] = _coerce_enum(_ec, detector_config[_ek])
            # Detect unknown keys in detector_config before constructing,
            # so the error names the bad key instead of a generic TypeError.
            from .detectors import ThresholdDetector as _TD
            _allowed_detector_keys = set(_TD.__dataclass_fields__.keys()) if hasattr(_TD, "__dataclass_fields__") else set()
            if _allowed_detector_keys:
                _unknown_detector = set(detector_config) - _allowed_detector_keys
                if _unknown_detector:
                    raise ConfigurationError(
                        f"Unknown keys in detector_config: {sorted(_unknown_detector)}"
                    )
            detector = ThresholdDetector(**detector_config)
            protocol_name = d_copy.pop("protocol_name")
            if isinstance(protocol_name, Enum):
                protocol_name = protocol_name.name
            if isinstance(protocol_name, str):
                protocol_name = protocol_name.strip().lower().replace("_", "-")
            else:
                raise ConfigurationError(
                    f"protocol_name must be a string, got {type(protocol_name).__name__}."
                )
            protocol_config = d_copy.pop("protocol_config")
            if not isinstance(protocol_config, dict):
                raise ConfigurationError(
                    f"protocol_config must be a dict, got {type(protocol_config).__name__}."
                )
            
            # [Chapman et al. 2018]
            damping_param = _coerce_type(d_copy, "damping_parameter", float, required=False, default=None)
            encoding_angle = _coerce_type(d_copy, "encoding_rotation_angle", float, required=False, default=None)
            
            # protocol: ProtocolFromModule  — type hint only; resolved below
            # The protocol's `source` is always the top-level source_config;
            # if a `source` key is also present in protocol_config, warn
            # the user (don't silently drop) and pop it to avoid
            # "got multiple values for argument 'source'" TypeError.
            if "source" in protocol_config:
                logger.warning(
                    "protocol_config.source is ignored — QKDParams uses the "
                    "top-level source_config. Remove 'source' from "
                    "protocol_config to silence this warning."
                )
                protocol_config.pop("source")
            # RedundantTransmissionProtocol takes damping_parameter and
            # rotation_angle from the top level; pop them from
            # protocol_config to avoid "got multiple values for argument".
            for _conflict_key in ("damping_parameter", "rotation_angle"):
                if _conflict_key in protocol_config:
                    logger.warning(
                        f"protocol_config.{_conflict_key} is ignored — QKDParams "
                        f"uses the top-level value. Remove '{_conflict_key}' from "
                        f"protocol_config to silence this warning."
                    )
                    protocol_config.pop(_conflict_key)

            # Detect unknown keys in protocol_config before constructing,
            # so the error names the bad key instead of a generic TypeError.
            from .protocols import BB84DecoyProtocol, MDIQKDProtocol, B92Protocol, RedundantTransmissionProtocol
            _proto_cls_map = {
                "bb84-decoy": BB84DecoyProtocol,
                "mdi-qkd": MDIQKDProtocol,
                "b92": B92Protocol,
                "redundant": RedundantTransmissionProtocol,
            }
            _proto_cls = _proto_cls_map.get(protocol_name)
            if _proto_cls is not None and hasattr(_proto_cls, "__dataclass_fields__"):
                _allowed_proto_keys = set(_proto_cls.__dataclass_fields__.keys())
                _unknown_proto = set(protocol_config) - _allowed_proto_keys
                if _unknown_proto:
                    raise ConfigurationError(
                        f"Unknown keys in protocol_config: {sorted(_unknown_proto)}"
                    )
            if protocol_name == "bb84-decoy":
                protocol = BB84DecoyProtocol(source=source, **protocol_config)  # type: ignore[misc]
            elif protocol_name == "mdi-qkd":
                protocol = MDIQKDProtocol(source=source, **protocol_config)  # type: ignore[misc]
            elif protocol_name == "b92":
                protocol = B92Protocol(source=source, **protocol_config)  # type: ignore[misc]
            elif protocol_name == "redundant":
                protocol = RedundantTransmissionProtocol(
                    source=source,
                    damping_parameter=damping_param,
                    rotation_angle=encoding_angle,
                    **protocol_config
                )  # type: ignore[misc]
            else:
                raise ConfigurationError(
                    f"Protocol '{protocol_name}' not supported. "
                    f"Supported: ['bb84-decoy', 'mdi-qkd', 'b92', 'redundant']"
                )
        except KeyError as e:
            raise ConfigurationError(
                f"Missing required key {e}. Check source_config, channel_config, "
                f"detector_config, protocol_config, and protocol_name."
            ) from e
        except TypeError as e:
            raise ConfigurationError(f"Mismatched parameters in config section: {e}") from e
        except ValueError as e:
            raise ConfigurationError(f"Invalid value in config section: {e}") from e
        except ParameterValidationError:
            raise

        try:
            schema_version = d_copy.pop("schema_version", None)
            if schema_version is None:
                logger.warning(
                    "Config lacks schema_version — assuming current version "
                    "%r. This may produce incorrect physics if the config was "
                    "generated by an older framework version.",
                    CURRENT_SCHEMA_VERSION,
                )
            elif _parse_schema_version(schema_version) != _parse_schema_version(CURRENT_SCHEMA_VERSION):
                # No migrations are implemented, so silently interpreting an
                # old config under the new schema would produce incorrect
                # physics.  Fail loudly instead.
                raise ConfigurationError(
                    f"Config schema_version {schema_version!r} is unsupported "
                    f"(expected {CURRENT_SCHEMA_VERSION!r}). No migrations are "
                    f"currently implemented; regenerate the config with the "
                    f"current version of the framework."
                )

            try:
                security_proof_val = d_copy.pop("security_proof")
                ci_method_val = d_copy.pop("ci_method")
            except KeyError as e:
                raise ConfigurationError(f"Missing required parameter: {e}") from e
            
            # proof_module attachment now happens in QKDParams._attach_proof_module()
            # (called from __post_init__), so both direct construction and
            # from_dict() produce identical protocol state.

            _decoder_arch_raw = _coerce_type(d_copy, "decoder_architecture", str, required=False, default="local")
            decoder_arch_val = _decoder_arch_raw.strip().lower() if _decoder_arch_raw is not None else "local"

            _f_error_correction = _coerce_type(d_copy, "f_error_correction", float, required=False, default=None)
            # Always pop the legacy alias so it never reaches the unknown-key check.
            _legacy_fec = _coerce_type(d_copy, "error_correction_efficiency", float, required=False, default=None)
            if _legacy_fec is not None:
                logger.warning(
                    "Config uses deprecated alias 'error_correction_efficiency'; "
                    "rename to 'f_error_correction'. Support may be removed in a future version."
                )
            if _f_error_correction is None:
                _f_error_correction = _legacy_fec if _legacy_fec is not None else 1.16

            _frequency_plan_raw = _coerce_type(d_copy, "frequency_plan", list, required=False, default=None)
            if _frequency_plan_raw is not None:
                _frequency_plan = [float(x) for x in _frequency_plan_raw]
            else:
                _frequency_plan = None                
            params = {
                "security_proof": _coerce_enum(SecurityProof, security_proof_val),
                "ci_method": _coerce_enum(ConfidenceBoundMethod, ci_method_val),
                "decoder_architecture": _coerce_enum(DecoderArchitecture, decoder_arch_val),
                
                "redundancy_M": _coerce_type(d_copy, "redundancy_M", int, required=False, default=1),
                "use_entangling_encoder": _coerce_type(d_copy, "use_entangling_encoder", bool, required=False, default=False),
                "use_adaptive_measurements": _coerce_type(d_copy, "use_adaptive_measurements", bool, required=False, default=False),
                
                # Chapman et al. 2018
                "damping_parameter": damping_param,
                "encoding_rotation_angle": encoding_angle,

                "dispersion_parameter_ps_nm_km": _coerce_type(d_copy, "dispersion_parameter_ps_nm_km", float, required=False, default=0.0),
                "carrier_wavelength_nm": _coerce_type(d_copy, "carrier_wavelength_nm", float, required=False, default=1550.0),
                
                "num_bits": _coerce_type(d_copy, "num_bits", int),
                "photon_number_cap": _coerce_type(d_copy, "photon_number_cap", int),
                "batch_size": _coerce_type(d_copy, "batch_size", int),
                "num_workers": _coerce_type(d_copy, "num_workers", int),
                "f_error_correction": _f_error_correction,
                "eps_sec": _coerce_type(d_copy, "eps_sec", float),
                "eps_cor": _coerce_type(d_copy, "eps_cor", float),
                "eps_pe": _coerce_type(d_copy, "eps_pe", float),
                "eps_smooth": _coerce_type(d_copy, "eps_smooth", float),
                "eps_pa": _coerce_type(d_copy, "eps_pa", float, required=False, default=1e-10),
                "eps_sif": _coerce_type(d_copy, "eps_sif", float, required=False, default=1e-10),
                "force_sequential": _coerce_type(d_copy, "force_sequential", bool, required=False, default=False),
                "enforce_monotonicity": _coerce_type(d_copy, "enforce_monotonicity", bool, required=False, default=True),
                "assume_phase_equals_bit_error": _coerce_type(d_copy, "assume_phase_equals_bit_error", bool, required=False, default=False),
                "allow_unsafe_mdi_approx": _coerce_type(d_copy, "allow_unsafe_mdi_approx", bool, required=False, default=False),
                "lp_solver_method": _coerce_type(d_copy, "lp_solver_method", str, required=False, default="highs"),
                "require_tail_below": _coerce_type(d_copy, "require_tail_below", float, required=False, default=DEFAULT_POISSON_TAIL_THRESHOLD),
                "master_seed": _coerce_type(d_copy, "master_seed", int, required=False, default=None),
                
                "assume_zero_background": _coerce_type(d_copy, "assume_zero_background", bool, required=False, default=False),
                "auto_optimize_ma2005": _coerce_type(d_copy, "auto_optimize_ma2005", bool, required=False, default=False),
                "use_weak_gllp": _coerce_type(d_copy, "use_weak_gllp", bool, required=False, default=False),
                "system_duty_cycle": _coerce_type(d_copy, "system_duty_cycle", float, required=False, default=1.0),
                "f_ec_dynamic_config": _coerce_type(d_copy, "f_ec_dynamic_config", dict, required=False, default=None),
                
                # Lim 2014 specific
                "security_constant_kappa": _coerce_type(d_copy, "security_constant_kappa", float, required=False, default=None),
                "error_correction_model": _coerce_type(d_copy, "error_correction_model", str, required=False, default="standard"),
                "auto_optimize_lim2014": _coerce_type(d_copy, "auto_optimize_lim2014", bool, required=False, default=False),

                "N_channels": n_channels,
                "modulation_index": modulation_index,
                "target_carrier_frequency_hz": _coerce_type(d_copy, "target_carrier_frequency_hz", float, required=False, default=None),

                "visibility_mismatch_dm": _coerce_type(d_copy, "visibility_mismatch_dm", float, required=False, default=0.0),
                "visibility_bias_drift_psi1": _coerce_type(d_copy, "visibility_bias_drift_psi1", float, required=False, default=0.0),
                "visibility_bias_drift_psi2": _coerce_type(d_copy, "visibility_bias_drift_psi2", float, required=False, default=0.0),
                "drift_step_size": _coerce_type(d_copy, "drift_step_size", float, required=False, default=0.0),

                "cso_model": _coerce_type(d_copy, "cso_model", str, required=False, default="approximate").lower(),
                "frequency_plan": _frequency_plan,
                "filter_model": filter_model,
                "filter_awg_loss_db": filter_awg_loss_db,
                "filter_fbg_loss_per_channel_db": filter_fbg_loss_per_channel_db,
                "filter_fbg_bandwidth_hz": _coerce_type(d_copy, "filter_fbg_bandwidth_hz", float, required=False, default=1.3e9),
                "filter_crosstalk_db": _coerce_type(d_copy, "filter_crosstalk_db", float, required=False, default=-30.0),

                # [Improvement 1] Circulator loss
                "circulator_insertion_loss_db": circulator_insertion_loss_db,

                "modulator_bandwidth_hz": _coerce_type(d_copy, "modulator_bandwidth_hz", float, required=False, default=None),
                "modulator_insertion_loss_db": modulator_insertion_loss_db,
                "modulator_material": _coerce_type(d_copy, "modulator_material", str, required=False, default="LiNbO3"),
                
                "use_linear_modulation_approximation": use_linear_approx,
                "ppbs_transmissivity": _coerce_type(d_copy, "ppbs_transmissivity", float, required=False, default=0.66),
                
                # New Params for WDM
                "wdm_channel_count": _coerce_type(d_copy, "wdm_channel_count", int, required=False, default=1),
                "wdm_channel_spacing_hz": _coerce_type(d_copy, "wdm_channel_spacing_hz", float, required=False, default=100e9),
                # [Improvement 3] WDM Physics
                "fiber_nonlinear_coeff": _coerce_type(d_copy, "fiber_nonlinear_coeff", float, required=False, default=1.3),
                "wdm_channel_power_dbm": _coerce_type(d_copy, "wdm_channel_power_dbm", float, required=False, default=-20.0),
                # [New] Raman
                "wdm_raman_coefficient": _coerce_type(d_copy, "wdm_raman_coefficient", float, required=False, default=0.0),
                
                "active_dispersion_compensation": _coerce_type(d_copy, "active_dispersion_compensation", bool, required=False, default=False),

                # Option J: round-trip pure fiber loss for Raman l_eff computation.
                # Parsed from channel_config above; thread through to QKDParams so
                # _compute_raman_dark_rate_hz reads it instead of the
                # AWG-contaminated channel.fiber_loss_db_km.
                "fiber_loss_db_km": fiber_loss_db_km,
            }
            d_copy = {k: v for k, v in d_copy.items() if not k.startswith("_")}
            if d_copy:
                raise ConfigurationError(f"Unknown parameter keys provided: {sorted(d_copy.keys())}")
            return cls(protocol=protocol, source=source, channel=channel, detector=detector, **params)
        
        except ParameterValidationError:
            raise  # The error message is already specific; don't wrap.
        except TypeError as e:
            raise ConfigurationError(f"Mismatched parameters during final object construction: {e}") from e
        except ValueError as e:
            # Downstream validators (ThresholdDetector, OpticalSource, etc.)
            # may raise plain ValueError; wrap it to honor the from_dict
            # contract that only ConfigurationError / ParameterValidationError
            # escape this method.
            raise ConfigurationError(f"Invalid value during final object construction: {e}") from e

# --- New Factory Methods for Lim 2014 Presets ---

def load_lim2014_dedicated_params(distance_km: float = 0.0,
                                  num_bits: int = 10000000,
                                  fiber_loss_db_km: float = 0.2) -> QKDParams:
    """
    Creates a QKDParams object with the 'Dedicated Fiber' configuration 
    specified in Lim et al. (2014) Section IV.
    """
    from .protocols import BB84DecoyProtocol
    
    pulse_configs = [
        PulseTypeConfig(name="signal", mean_photon_number=0.5, probability=0.6),
        PulseTypeConfig(name="decoy", mean_photon_number=0.1, probability=0.2),
        PulseTypeConfig(name="vacuum", mean_photon_number=0.0, probability=0.2),
    ]

    from qkd.datatypes import OpticalSourceConfig
    optical_config = OpticalSourceConfig(
        source_rate=1e9,
        pulse_configs=pulse_configs,
        statistics_type=SourceStatisticsType.POISSON,
        error_model=SourceErrorModel.RANDOM_GAUSSIAN,
        intensity_jitter=0.0,
        modulation_index=0.0,
        N_channels=1,
        use_small_angle_approximation=True,
        use_linear_modulation_approximation=True,
        ideal_emission_probability=1.0,
        adversarial_block_size=1000,
        is_bidirectional=False,
    )
    source = OpticalSource.create(optical_config)

    
    detector = ThresholdDetector(
        det_eff_d0=0.15, det_eff_d1=0.15,
        dark_rate=600,
        qber_intrinsic=5e-3,
        misalignment=0.0,
        double_click_policy=DoubleClickPolicy.RANDOM,
        afterpulse_prob=0.0,
        afterpulse_model=AfterpulseModel.GEOMETRIC
    )
    
    # Rebuild channel using the single source-of-truth loss formula so that
    # direct-construction and from_dict() produce identical channel loss.
    # Read defaults from QKDParams field definitions so this factory stays
    # in sync automatically.
    _fields = {}
    for _f in dataclasses.fields(QKDParams):
        if _f.default is not dataclasses.MISSING:
            _fields[_f.name] = _f.default
        elif _f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            _fields[_f.name] = _f.default_factory()
    # Dedicated (single-channel, no AWG/MZM/circulator) path: zero out all
    # baseline insertion losses. Only fiber loss * distance should remain.
    _total_loss_db = _compute_total_loss_db_from_scalars(
        distance_km=distance_km,
        fiber_loss_db_km=fiber_loss_db_km,
        filter_model="ideal",
        filter_awg_loss_db=0.0,
        N_channels=1,
        filter_fbg_loss_per_channel_db=0.0,
        modulator_insertion_loss_db=0.0,
        circulator_insertion_loss_db=0.0,
    )
    channel = FiberChannel.from_total_loss(distance_km, _total_loss_db)


    
    # Asymmetric basis probabilities will be optimized, start balanced
    protocol = BB84DecoyProtocol(
        alice_z_basis_prob=0.5,
        bob_z_basis_prob=0.5,
        source=source,
        double_click_policy="DISCARD",
    )
    
    return QKDParams(
        protocol=protocol, source=source, channel=channel, detector=detector,
        num_bits=num_bits, photon_number_cap=10, batch_size=num_bits, num_workers=1,
        f_error_correction=1.16,
        eps_sec=1e-7, eps_cor=1e-7, eps_pe=1e-7, eps_smooth=1e-7,
        security_proof=SecurityProof.LIM_2014,
        ci_method=ConfidenceBoundMethod.CLOPPER_PEARSON,
        force_sequential=True, enforce_monotonicity=True, assume_phase_equals_bit_error=False,
        security_constant_kappa=None,
        auto_optimize_lim2014=True,
        fiber_loss_db_km=fiber_loss_db_km,
        filter_awg_loss_db=0.0,
        filter_fbg_loss_per_channel_db=0.0,
        modulator_insertion_loss_db=0.0,
        circulator_insertion_loss_db=0.0,
    )

def load_lim2014_dwdm_params(distance_km: float = 0.0, num_bits: int = 10000000,
                              wdm_channel_count: int = 4,
                              wdm_channel_power_dbm: Optional[float] = None,
                              fiber_loss_db_km: float = 0.2,
                              wdm_raman_coefficient: float = 1e-9) -> QKDParams:
    """
    Creates a QKDParams object with the 'DWDM' configuration
    specified in Lim et al. (2014) Section IV and Figure 2.

    Parameters
    ----------
    distance_km : float
        Fiber length in kilometers.
    num_bits : int
        Total number of pulses (postprocessing block size).
    wdm_channel_count : int, default 4
        Number of classical WDM channels (paper uses 4, "4+1 architecture").
        The quantum channel is separate and not counted here.
    wdm_channel_power_dbm : float or None, default None
        Per-channel LAUNCH power in dBm.

        If None (default): distance-aware launch power is computed so that
        the RECEIVED power is -34 dBm at the detector, matching Lim2014
        Section IV and Ref. [38] (commercial DWDM SFP receiver sensitivity).

        p_launch_dbm = -34 + fiber_loss_db + awg_filter_loss_db

    This is the paper-faithful behavior.  At 50 km (10 dB channel loss +
    3 dB AWG): p_launch = -21 dBm.  At 100 km (20 dB + 3 dB): -11 dBm.
    At 150 km (30 dB + 3 dB): -1 dBm.

        If a float is provided: it is used as a constant launch power
        regardless of distance (legacy / sweep mode).

    Raman noise model:
        p_raman = p_launch * (N-1) * wdm_raman_coeff * l_eff
        where l_eff = (1 - exp(-alpha * L)) / alpha, alpha = 0.2 * 0.23026 Np/km.
        With p_launch = -34 dBm (received), Raman is ~4 orders of magnitude
        smaller than with p_launch = -3 dBm at 50 km, recovering the paper's
        130 km QKD reach instead of the 100 km cliff-edge.
    """
    params = load_lim2014_dedicated_params(distance_km, num_bits,
                                           fiber_loss_db_km=fiber_loss_db_km)

    # AWG filter loss (single source of truth — used for both launch-power
    # computation and channel rebuild below).
    _awg_filter_loss_db = next(
        f.default for f in dataclasses.fields(QKDParams)
        if f.name == "filter_awg_loss_db"
    )

    if (not isinstance(wdm_channel_count, int)
            or isinstance(wdm_channel_count, bool)
            or wdm_channel_count < 1):
        raise ParameterValidationError(
            f"wdm_channel_count must be a positive int, got {wdm_channel_count!r}"
        )

    if wdm_channel_power_dbm is None:
        # Classical WDM channels share ONLY the fiber and the AWG mux with the
        # quantum channel — they do not traverse the QKD modulator or circulator.
        # Compute launch power from fiber loss + AWG loss only, so that the
        # received power at the detector is -34 dBm (Lim2014 Section IV).
        _fiber_loss_db = distance_km * params.fiber_loss_db_km
        wdm_channel_power_dbm = -34.0 + _fiber_loss_db + _awg_filter_loss_db
        if wdm_channel_power_dbm > 30.0:
            raise ParameterValidationError(
                f"Auto-computed wdm_channel_power_dbm={wdm_channel_power_dbm:.1f} dBm "
                f"exceeds the 30 dBm plausible cap. distance_km={distance_km} is too "
                f"large for the -34 dBm received-power target. Pass an explicit "
                f"wdm_channel_power_dbm to override."
            )

    # Enable WDM effects
    wdm_update = {
        "wdm_channel_count": wdm_channel_count,
        "wdm_channel_power_dbm": float(wdm_channel_power_dbm),
        "wdm_raman_coefficient": wdm_raman_coefficient,
        "filter_model": "awg",
        "filter_awg_loss_db": _awg_filter_loss_db
    }   

    # Recompute total_loss_db using the single source-of-truth formula
    # with the WDM fields applied.  We read the post-change scalar values
    # directly (no intermediate QKDParams needed, so no __post_init__ /
    # loss-consistency guard fires).

    _new_filter_model = wdm_update.get("filter_model", params.filter_model)
    _new_filter_awg = wdm_update.get("filter_awg_loss_db", params.filter_awg_loss_db)
    _new_N_channels = wdm_update.get("N_channels", params.N_channels)
    _new_fbg_per_ch = wdm_update.get("filter_fbg_loss_per_channel_db", params.filter_fbg_loss_per_channel_db)
    _new_mod_loss = wdm_update.get("modulator_insertion_loss_db", params.modulator_insertion_loss_db)
    _new_circ_loss = wdm_update.get("circulator_insertion_loss_db", params.circulator_insertion_loss_db)
    _new_fiber_loss = wdm_update.get("fiber_loss_db_km", params.fiber_loss_db_km)
    new_total_loss_db = _compute_total_loss_db_from_scalars(
        distance_km=distance_km,
        fiber_loss_db_km=_new_fiber_loss,
        filter_model=_new_filter_model,
        filter_awg_loss_db=_new_filter_awg,
        N_channels=_new_N_channels,
        filter_fbg_loss_per_channel_db=_new_fbg_per_ch,
        modulator_insertion_loss_db=_new_mod_loss,
        circulator_insertion_loss_db=_new_circ_loss,
    )
    wdm_update["channel"] = FiberChannel.from_total_loss(distance_km, new_total_loss_db)
    # Single replace with channel already rebuilt — passes the loss-consistency guard.
    return dataclasses.replace(params, **wdm_update)
