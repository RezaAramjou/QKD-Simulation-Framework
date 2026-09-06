# qkd/datatypes.py
# -*- coding: utf-8 -*-
"""
Data structures and enumerations for the QKD simulation framework.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    TYPE_CHECKING,
    Union,
    Protocol,
    runtime_checkable,
)

from .modulators import MZMConfig
from .noise_models import ElectricalNoiseConfig
from .exceptions import ParameterValidationError
from .constants import is_close, is_valid_probability, is_finite_non_negative
from .utils.utils import sanitize_for_serialization

PROB_SUM_TOL = 1e-9


if TYPE_CHECKING:
    from .params import QKDParams

__version__ = "4.2.1"

__all__ = [
    "ConfidenceBoundMethod",
    "DoubleClickPolicy",
    "DecoderArchitecture",
    "SourceErrorModel",
    "EpsilonAllocation",
    "PulseTypeConfig",
    "PulseEnsembleConfig",
    "SecurityCertificate",
    "SecurityProof",
    "SimulationResults",
    "SimulationStatus",
    "SourceStatisticsType",
    "TallyCounts",
    "ProtocolType",
    "DetectorType",
    "IntensityNode",
    "IntensityConfig",
    "AttenuationConfig",
    "OpticalSourceConfig",
    "DetectionConfig",
    "ErrorCorrectionConfig",
    "ProtocolParameters",
]

# --- Type aliases ---

JSONDict = Dict[str, Any]
JSONMapping = Mapping[str, Any]
TallyStatsMap = Dict[str, "TallyCounts"]


# --- Serializable Protocol / Mixin ---

@runtime_checkable
class Serializable(Protocol):
    """Protocol for objects that support dict-based serialization."""

    def to_dict(self) -> JSONDict: ...
    @classmethod
    def from_dict(cls, data: JSONMapping) -> Any: ...


class SerializableDataclassMixin:
    """
    Mixin providing default to_dict/from_dict for dataclasses.

    - to_dict: uses dataclasses.asdict (shallow for enums/nested dataclasses)
    - from_dict: ignores unknown keys by default unless strict=True
    """

    __slots__ = ()

    def to_dict(self) -> JSONDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: JSONMapping, *, strict: bool = False) -> Any:
        valid_keys = {f.name for f in fields(cls)}
        extra_keys = set(data.keys()) - valid_keys
        if strict and extra_keys:
            raise ParameterValidationError(
                f"Unknown keys for {cls.__name__}.",
                context={"unknown_keys": sorted(extra_keys)},
            )
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


# --- Common validation helpers ---

def _validate_non_negative_float(name: str, value: Any) -> None:
    if not is_finite_non_negative(value):
        raise ParameterValidationError(
            f"{name} must be a non-negative finite float.",
            param_name=name,
            param_value=value,
        )


def _validate_probability(name: str, value: Any) -> None:
    if not is_valid_probability(value):
        raise ParameterValidationError(
            f"{name} must be a finite float between $0.0$ and $1.0$.",
            param_name=name,
            param_value=value,
        )


def _validate_non_negative_int(name: str, value: Any) -> None:
    # Avoid bool, since bool is a subclass of int
    if type(value) is not int or not is_finite_non_negative(value):
        raise ParameterValidationError(
            f"{name} must be a non-negative integer.",
            param_name=name,
            param_value=value,
        )


def _validate_enum(value: Any, enum_cls: type[Enum], field_name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except Exception as exc:
        raise ParameterValidationError(
            f"{field_name} must be a {enum_cls.__name__} enum.",
            param_name=field_name,
            param_value=value,
        ) from exc


def _validate_probability_range_01(name: str, value: Any) -> None:
    """
    Probability-like parameters that are expected in $[0, 1]$ (or $(0, 1]$ depending on usage).
    Currently used as a generic checker (non-negative and <= 1.0 if finite).
    """
    _validate_non_negative_float(name, value)
    if value > 1.0 and not is_close(value, 1.0):
        raise ParameterValidationError(
            f"{name} must be within $[0, 1]$.",
            param_name=name,
            param_value=value,
        )


# --- Enums ---

class ProtocolType(str, Enum):
    """Supported QKD Protocols."""
    BB84_DECOY = "bb84-decoy"
    MDI_QKD = "mdi-qkd"
    B92 = "b92"
    # Legacy / internal placeholder; kept for backward compatibility.
    # TODO: consider removal or renaming when legacy support is no longer needed.
    REDUNDANT = "redundant"


class DetectorType(str, Enum):
    """Types of single-photon detectors."""
    SPD = "spd"      # Standard Single-Photon Detector (e.g., InGaAs, APD)
    SNSPD = "snspd"  # Superconducting Nanowire Single-Photon Detector
    PNRD = "pnrd"    # Photon-Number Resolving Detector


class DoubleClickPolicy(str, Enum):
    """Policy for handling double-click events in detectors."""
    DISCARD = "discard"
    RANDOM = "random"


class DecoderArchitecture(str, Enum):
    """
    Specifies the physical architecture of the receiver's decoder.
    - LOCAL: Measures each qubit individually (standard QKD).
    - ENTANGLING: Interferes multiple qubits before measurement to distinguish
      non-orthogonal states.
    """
    LOCAL = "local"
    ENTANGLING = "entangling"


class SourceErrorModel(str, Enum):
    """
    Defines the physical model for source intensity fluctuations.
    - RANDOM_GAUSSIAN: Standard random jitter (Gaussian distribution).
    - ADVERSARIAL_BLOCK: Deterministic "strengthened" vs "weakened" blocks
      exploitable by Eve (Wang et al. 2008).
    """
    RANDOM_GAUSSIAN = "random_gaussian"
    ADVERSARIAL_BLOCK = "adversarial_block"


class SecurityProof(str, Enum):
    """Supported security proofs for key rate calculation."""
    LIM_2014 = "lim-2014"
    TIGHT_PROOF = "tight-proof"
    MDI_QKD = "mdi-qkd"
    PAPER_2009 = "paper-2009"
    MA_2005 = "ma-2005"
    MA_2005_ASYMPTOTIC = "ma-2005-asymptotic"  # [Paper Limit] Infinite decoy states


class ConfidenceBoundMethod(str, Enum):
    """Statistical methods for calculating confidence bounds."""
    CHERNOFF = "chernoff"
    CLOPPER_PEARSON = "clopper-pearson"
    HOEFFDING = "hoeffding"
    GAUSSIAN = "gaussian"  # Standard error analysis used in Ma et al. 2005


class SimulationStatus(str, Enum):
    """Represents the final status of a simulation run."""
    OK = "ok"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class SourceStatisticsType(str, Enum):
    """Defines the photon number statistics of the optical source."""
    POISSON = "poisson"  # Standard attenuated laser
    THERMAL = "thermal"  # LEDs, ASE sources (Bose-Einstein)


# --- Dataclasses ---

@dataclass(frozen=True, slots=True)
class IntensityNode(SerializableDataclassMixin):
    """Represents a single intensity setting (signal or decoy)."""
    mu: float
    probability: float

    def __post_init__(self) -> None:
        _validate_non_negative_float("mu", self.mu)
        _validate_probability("probability", self.probability)

    def with_probability(self, p: float) -> IntensityNode:
        """Return a copy of this node with a different probability."""
        _validate_probability("probability", p)
        return replace(self, probability=p)

    def with_mu(self, mu: float) -> IntensityNode:
        """Return a copy of this node with a different mean photon number."""
        _validate_non_negative_float("mu", mu)
        return replace(self, mu=mu)

    @classmethod
    def from_dict(cls, data: JSONMapping, *, strict: bool = False) -> IntensityNode:
        return super().from_dict(data, strict=strict)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class IntensityConfig(SerializableDataclassMixin):
    """
    Configuration for signal and decoy state intensities.

    Invariants:
    - Each node has $0 \le \mu < \infty$ and probability in $[0,1]$.
    - The sum of probabilities over signal and all decoys is $1$ (up to numerical tolerance).
    - Decoy nodes must have unique $(\mu, probability)$ combinations; duplicate $\mu$ can be
      disallowed if domain requires (currently uniqueness enforced on $(\mu, probability)$).
    """
    signal: IntensityNode
    decoys: List[IntensityNode] = field(default_factory=list)

    def __post_init__(self) -> None:
        total_prob = self.signal.probability + sum(d.probability for d in self.decoys)
        if not is_close(total_prob, 1.0):
            raise ParameterValidationError(
                "Total intensity probabilities (signal + decoys) must sum to $1.0$.",
                context={"total_probability": total_prob},
            )

        # Enforce uniqueness of decoy (mu, probability) tuples
        seen = set()
        for d in self.decoys:
            key = (d.mu, d.probability)
            if key in seen:
                raise ParameterValidationError(
                    "Duplicate decoy IntensityNode detected.",
                    context={"mu": d.mu, "probability": d.probability},
                )
            seen.add(key)

    def all_nodes(self) -> List[IntensityNode]:
        """Return a list containing the signal followed by all decoy nodes."""
        return [self.signal, *self.decoys]

    def all_probabilities(self) -> List[float]:
        """Return the list of probabilities for signal and all decoys."""
        return [n.probability for n in self.all_nodes()]

    def total_probability(self) -> float:
        """Return the total probability over signal and decoys (should be $1.0$)."""
        return self.signal.probability + sum(d.probability for d in self.decoys)

    @classmethod
    def from_dict(cls, data: JSONMapping, *, strict: bool = False) -> IntensityConfig:
        strict_flag = strict
        signal_data = data.get("signal")
        if signal_data is None:
            raise ParameterValidationError("IntensityConfig requires 'signal' field.")

        decoys_data = data.get("decoys", [])
        if not isinstance(decoys_data, list):
            raise ParameterValidationError(
                "IntensityConfig.decoys must be a list.",
                param_name="decoys",
                param_value=decoys_data,
            )

        signal = IntensityNode.from_dict(signal_data, strict=strict_flag)
        decoys = [IntensityNode.from_dict(d, strict=strict_flag) for d in decoys_data]
        return cls(signal=signal, decoys=decoys)


@dataclass(frozen=True, slots=True)
class AttenuationConfig(SerializableDataclassMixin):
    """
    Configuration for quantum channel attenuation.

    Attributes:
        fiber_length: Fiber length in km.
        attenuation_coefficient: Attenuation coefficient in dB/km (e.g., $0.2$ dB/km).

    Derived:
        total_loss_db = fiber_length * attenuation_coefficient
    """
    fiber_length: float  # km
    attenuation_coefficient: float  # dB/km

    def __post_init__(self) -> None:
        _validate_non_negative_float("fiber_length", self.fiber_length)
        _validate_non_negative_float("attenuation_coefficient", self.attenuation_coefficient)

    @property
    def total_loss_db(self) -> float:
        """Total channel loss in dB: $fiber\_length \times attenuation\_coefficient$."""
        return self.fiber_length * self.attenuation_coefficient

    @classmethod
    def from_dict(cls, data: JSONMapping, *, strict: bool = False) -> AttenuationConfig:
        return super().from_dict(data, strict=strict)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class OpticalSourceConfig(SerializableDataclassMixin):
    """
    Full configuration for an optical source in a QKD simulation.
    """
    source_rate: float
    pulse_configs: Tuple[PulseTypeConfig, ...]
    statistics_type: SourceStatisticsType = SourceStatisticsType.POISSON
    error_model: SourceErrorModel = SourceErrorModel.RANDOM_GAUSSIAN
    intensity_jitter: float = 0.0
    modulation_index: float = 0.0
    N_channels: int = 1
    use_small_angle_approximation: bool = True
    use_linear_modulation_approximation: bool = False
    ideal_emission_probability: float = 1.0
    adversarial_block_size: int = 1000
    is_bidirectional: bool = False
    mzm: MZMConfig = field(default_factory=MZMConfig)
    electrical_noise: ElectricalNoiseConfig = field(default_factory=ElectricalNoiseConfig)

    def __post_init__(self) -> None:
        _validate_non_negative_float("source_rate", self.source_rate)

        if not self.pulse_configs:
            raise ParameterValidationError(
                "pulse_configs must contain at least one pulse type.",
                param_name="pulse_configs",
                param_value=self.pulse_configs,
            )

        total_probability = sum(p.probability for p in self.pulse_configs)
        if abs(total_probability - 1.0) > PROB_SUM_TOL:
            raise ParameterValidationError(
                "Pulse probabilities must sum to 1.",
                param_name="pulse_configs",
                param_value=total_probability,
            )

        if self.N_channels < 1:
            raise ParameterValidationError(
                "N_channels must be at least 1.",
                param_name="N_channels",
                param_value=self.N_channels,
            )

        _validate_non_negative_float("intensity_jitter", self.intensity_jitter)
        _validate_non_negative_float("modulation_index", self.modulation_index)
        _validate_probability("ideal_emission_probability", self.ideal_emission_probability)

        if self.adversarial_block_size <= 0:
            raise ParameterValidationError(
                "adversarial_block_size must be positive.",
                param_name="adversarial_block_size",
                param_value=self.adversarial_block_size,
            )

        if not isinstance(self.use_small_angle_approximation, bool):
            raise ParameterValidationError(
                "use_small_angle_approximation must be a boolean.",
                param_name="use_small_angle_approximation",
                param_value=self.use_small_angle_approximation,
            )

        if not isinstance(self.use_linear_modulation_approximation, bool):
            raise ParameterValidationError(
                "use_linear_modulation_approximation must be a boolean.",
                param_name="use_linear_modulation_approximation",
                param_value=self.use_linear_modulation_approximation,
            )

        if not isinstance(self.is_bidirectional, bool):
            raise ParameterValidationError(
                "is_bidirectional must be a boolean.",
                param_name="is_bidirectional",
                param_value=self.is_bidirectional,
            )

        self.mzm.validate()
        self.electrical_noise.validate()

    @property
    def pulse_period_ns(self) -> float:
        return 1e9 / self.source_rate

    @classmethod
    def from_dict(cls, data: JSONMapping, *, strict: bool = False) -> OpticalSourceConfig:
        return super().from_dict(data, strict=strict)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class DetectionConfig(SerializableDataclassMixin):
    """
    Detector parameters.

    Attributes:
        efficiency: Detection efficiency, probability in $[0,1]$.
        dark_count_rate: Dark count rate per gate (or per pulse) – unit must be
            documented at the call-site (e.g., counts per gate).
        detector_type: Type of detector (SPD, SNSPD, PNRD).
    """
    efficiency: float
    dark_count_rate: float
    detector_type: DetectorType = DetectorType.SPD

    def __post_init__(self) -> None:
        _validate_probability("efficiency", self.efficiency)
        _validate_non_negative_float("dark_count_rate", self.dark_count_rate)
        object.__setattr__(
            self,
            "detector_type",
            _validate_enum(self.detector_type, DetectorType, "detector_type"),
        )

    def to_dict(self) -> JSONDict:
        d = asdict(self)
        d["detector_type"] = self.detector_type.value
        return d

    @classmethod
    def from_dict(cls, data: JSONMapping, *, strict: bool = False) -> DetectionConfig:
        valid_keys = {f.name for f in fields(cls)}
        extra_keys = set(data.keys()) - valid_keys
        if strict and extra_keys:
            raise ParameterValidationError(
                "Unknown keys for DetectionConfig.",
                context={"unknown_keys": sorted(extra_keys)},
            )
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        if "detector_type" in filtered:
            filtered["detector_type"] = DetectorType(filtered["detector_type"])
        return cls(**filtered)


@dataclass(frozen=True, slots=True)
class ErrorCorrectionConfig(SerializableDataclassMixin):
    """
    Error correction reconciliation parameters.

    Attributes:
        efficiency: Reconciliation efficiency $f(E)$, usually $\ge 1.0$.
            In many implementations $1.0 \le f(E) \le 3.0$; this bound is enforced
            here, based on typical literature assumptions. If a different range is
            required, this constraint should be updated accordingly and documented.
    """
    efficiency: float  # f(E) usually >= 1.0 (1.0 is Shannon limit)

    def __post_init__(self) -> None:
        _validate_non_negative_float("efficiency", self.efficiency)
        if self.efficiency < 1.0 or self.efficiency > 3.0:
            raise ParameterValidationError(
                "Error correction efficiency must be between $1.0$ and $3.0$.",
                param_name="efficiency",
                param_value=self.efficiency,
            )

    @classmethod
    def from_dict(cls, data: JSONMapping, *, strict: bool = False) -> ErrorCorrectionConfig:
        return super().from_dict(data, strict=strict)  # type: ignore[return-value]
        
@dataclass(frozen=True)
class OpticalComponent:
    name: str
    component_type: str = "generic"
    loss_db: float = 0.0
    efficiency: float = 1.0
    metadata: dict[str, Any] | None = None



@dataclass(frozen=True, slots=True, kw_only=True)
class ProtocolParameters(SerializableDataclassMixin):
    """
    Groups all fundamental protocol-level configurations.

    Attributes:
        protocol: QKD protocol type (e.g., BB84_DECOY).
        intensities: Intensity configuration (signal + decoys).
        attenuation: Channel attenuation configuration.
        optical: Optical source configuration.
        detection: Detection configuration.
        error_correction: Error correction configuration.
    """
    protocol: ProtocolType
    intensities: IntensityConfig
    attenuation: AttenuationConfig
    optical: OpticalSourceConfig
    detection: DetectionConfig
    error_correction: ErrorCorrectionConfig

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "protocol",
            _validate_enum(self.protocol, ProtocolType, "protocol"),
        )

    def to_dict(self) -> JSONDict:
        return {
            "protocol": self.protocol.value,
            "intensities": self.intensities.to_dict(),
            "attenuation": self.attenuation.to_dict(),
            "optical": self.optical.to_dict(),
            "detection": self.detection.to_dict(),
            "error_correction": self.error_correction.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: JSONMapping, *, strict: bool = False) -> ProtocolParameters:
        extra_keys = set(data.keys()) - {
            "protocol",
            "intensities",
            "attenuation",
            "optical",
            "detection",
            "error_correction",
        }
        if strict and extra_keys:
            raise ParameterValidationError(
                "Unknown keys for ProtocolParameters.",
                context={"unknown_keys": sorted(extra_keys)},
            )

        protocol = ProtocolType(data["protocol"])
        intensities = IntensityConfig.from_dict(data["intensities"], strict=strict)
        attenuation = AttenuationConfig.from_dict(data["attenuation"], strict=strict)
        optical = OpticalSourceConfig.from_dict(data["optical"], strict=strict)
        detection = DetectionConfig.from_dict(data["detection"], strict=strict)
        error_correction = ErrorCorrectionConfig.from_dict(
            data["error_correction"], strict=strict
        )
        return cls(
            protocol=protocol,
            intensities=intensities,
            attenuation=attenuation,
            optical=optical,
            detection=detection,
            error_correction=error_correction,
        )


@dataclass(frozen=True, slots=True)
class PulseTypeConfig(SerializableDataclassMixin):
    """
    Configuration for a single pulse type in a decoy-state protocol.

    Attributes:
        name: A unique identifier for the pulse type (e.g., "signal", "decoy").
        mean_photon_number: The unitless mean photon number ($\mu$).
        probability: The probability of sending this pulse type.
    """
    name: str
    mean_photon_number: float
    probability: float

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ParameterValidationError(
                "PulseTypeConfig.name cannot be empty or whitespace-only.",
                param_name="name",
                param_value=self.name,
            )

        nm = self.name.strip()
        object.__setattr__(self, "name", nm)

        _validate_non_negative_float("mean_photon_number", self.mean_photon_number)
        _validate_probability("probability", self.probability)

    @classmethod
    def from_dict(cls, data: JSONMapping, *, strict: bool = False) -> PulseTypeConfig:
        return super().from_dict(data, strict=strict)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class PulseEnsembleConfig(SerializableDataclassMixin):
    """
    Container for a set of PulseTypeConfig objects.

    Invariants:
        - Sum of probabilities over all pulses is 1 (within numerical tolerance).
        - Pulse names are unique.
    """
    pulses: List[PulseTypeConfig]

    def __post_init__(self) -> None:
        if not self.pulses:
            raise ParameterValidationError("PulseEnsembleConfig.pulses cannot be empty.")

        total_prob = sum(p.probability for p in self.pulses)
        if not is_close(total_prob, 1.0):
            raise ParameterValidationError(
                "Sum of pulse probabilities must be 1.0.",
                context={"total_probability": total_prob},
            )

        names = [p.name for p in self.pulses]
        if len(names) != len(set(names)):
            raise ParameterValidationError(
                "Pulse names in PulseEnsembleConfig must be unique.",
                context={"names": names},
            )

    @classmethod
    def from_dict(cls, data: JSONMapping, *, strict: bool = False) -> PulseEnsembleConfig:
        pulses_data = data.get("pulses")
        if not isinstance(pulses_data, list):
            raise ParameterValidationError(
                "PulseEnsembleConfig.pulses must be a list.",
                param_name="pulses",
                param_value=pulses_data,
            )
        pulses = [PulseTypeConfig.from_dict(d, strict=strict) for d in pulses_data]
        return cls(pulses=pulses)


@dataclass(slots=True)
class TallyCounts(SerializableDataclassMixin):
    """
    A mutable container for event counts during a QKD simulation.

    Invariants:
        - All fields are non-negative integers.
        - sifted <= sent
        - errors_sifted <= sifted
        - sifted_z <= sent_z, sifted_x <= sent_x
        - errors_sifted_z <= sifted_z, errors_sifted_x <= sifted_x
        - (sifted_z + sifted_x) <= sifted (if tighter invariants hold, they can be enforced)
        - (sent_z + sent_x) <= sent
        - (errors_sifted_z + errors_sifted_x) <= errors_sifted
    """
    sent: int = 0
    sifted: int = 0
    errors_sifted: int = 0
    double_clicks_discarded: int = 0
    sent_z: int = 0
    sent_x: int = 0
    sifted_z: int = 0
    sifted_x: int = 0
    errors_sifted_z: int = 0
    errors_sifted_x: int = 0

    def __post_init__(self) -> None:
        for fdef in fields(self):
            name = fdef.name
            value = getattr(self, name)
            _validate_non_negative_int(name, value)

        if self.sifted > self.sent:
            raise ParameterValidationError(
                "Sifted count cannot exceed sent count.",
                context={"sifted": self.sifted, "sent": self.sent},
            )
        if self.errors_sifted > self.sifted:
            raise ParameterValidationError(
                "Error count cannot exceed sifted count.",
                context={
                    "errors_sifted": self.errors_sifted,
                    "sifted": self.sifted,
                },
            )
        if self.sifted_z > self.sent_z or self.sifted_x > self.sent_x:
            raise ParameterValidationError(
                "Per-basis sifted counts cannot exceed per-basis sent counts.",
                context={
                    "sifted_z": self.sifted_z,
                    "sent_z": self.sent_z,
                    "sifted_x": self.sifted_x,
                    "sent_x": self.sent_x,
                },
            )
        if self.errors_sifted_z > self.sifted_z or self.errors_sifted_x > self.sifted_x:
            raise ParameterValidationError(
                "Per-basis error counts cannot exceed per-basis sifted counts.",
                context={
                    "errors_z": self.errors_sifted_z,
                    "sifted_z": self.sifted_z,
                    "errors_x": self.errors_sifted_x,
                    "sifted_x": self.sifted_x,
                },
            )

        # Aggregate basis consistency checks
        if self.sent_z + self.sent_x > self.sent:
            raise ParameterValidationError(
                "Sum of per-basis sent counts cannot exceed total sent.",
                context={
                    "sent_z": self.sent_z,
                    "sent_x": self.sent_x,
                    "sent": self.sent,
                },
            )
        if self.sifted_z + self.sifted_x > self.sifted:
            raise ParameterValidationError(
                "Sum of per-basis sifted counts cannot exceed total sifted.",
                context={
                    "sifted_z": self.sifted_z,
                    "sifted_x": self.sifted_x,
                    "sifted": self.sifted,
                },
            )
        if self.errors_sifted_z + self.errors_sifted_x > self.errors_sifted:
            raise ParameterValidationError(
                "Sum of per-basis error counts cannot exceed total errors_sifted.",
                context={
                    "errors_sifted_z": self.errors_sifted_z,
                    "errors_sifted_x": self.errors_sifted_x,
                    "errors_sifted": self.errors_sifted,
                },
            )

    # Derived properties

    @property
    def qber(self) -> Optional[float]:
        """Overall Quantum Bit Error Rate (QBER)."""
        if self.sifted == 0:
            return None
        return self.errors_sifted / self.sifted

    @property
    def qber_z(self) -> Optional[float]:
        """Z-basis QBER."""
        if self.sifted_z == 0:
            return None
        return self.errors_sifted_z / self.sifted_z

    @property
    def qber_x(self) -> Optional[float]:
        """X-basis QBER."""
        if self.sifted_x == 0:
            return None
        return self.errors_sifted_x / self.sifted_x

    @property
    def sifting_ratio(self) -> Optional[float]:
        """Ratio of sifted to sent counts."""
        if self.sent == 0:
            return None
        return self.sifted / self.sent

    # Operator helpers

    def merged(self, other: TallyCounts) -> TallyCounts:
        """Return a new TallyCounts representing the sum of this and another."""
        return self + other

    def __add__(self, other: TallyCounts) -> TallyCounts:
        if not isinstance(other, TallyCounts):
            raise TypeError(
                f"Unsupported operand type for +: 'TallyCounts' and '{type(other).__name__}'"
            )
        # Avoid asdict overhead; sum field-by-field
        data: Dict[str, int] = {}
        for fdef in fields(self):
            name = fdef.name
            data[name] = getattr(self, name) + getattr(other, name)
        return TallyCounts(**data)

    def __iadd__(self, other: TallyCounts) -> TallyCounts:
        # Since dataclass is mutable (no frozen=True), support in-place addition.
        merged = self + other
        for fdef in fields(self):
            setattr(self, fdef.name, getattr(merged, fdef.name))
        return self

    def to_dict(self) -> JSONDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: JSONMapping, *, strict: bool = False) -> TallyCounts:
        valid_keys = {f.name for f in fields(cls)}
        extra_keys = set(data.keys()) - valid_keys
        if strict and extra_keys:
            raise ParameterValidationError(
                "Unknown keys for TallyCounts.",
                context={"unknown_keys": sorted(extra_keys)},
            )
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


@dataclass(frozen=True, slots=True, kw_only=True)
class EpsilonAllocation(SerializableDataclassMixin):
    """
    Defines the allocation of the total security parameter (epsilon).

    Invariants:
        - Each epsilon parameter is finite and non-negative.
        - The sum $eps\_pe + 2 eps\_smooth + eps\_pa + eps\_cor + eps\_phase\_est$
          does not exceed $eps\_sec$ (within numerical tolerance).
        - Typically, each epsilon should be in $[0,1]$; this is enforced generically.
    """
    eps_sec: float
    eps_cor: float
    eps_pe: float
    eps_smooth: float
    eps_pa: float
    eps_phase_est: float

    def __post_init__(self) -> None:
        self._validate()
    def validate(self) -> None:
        """Validate epsilon allocation.

        Public compatibility wrapper used by proof classes.
        """
        self._validate()
        
    def _validate(self) -> None:
        # Basic non-negative float & [0,1] probability checks
        for fdef in fields(self):
            name = fdef.name
            val = getattr(self, name)
            _validate_probability_range_01(name, val)

        total_sum = (
            self.eps_pe
            + 2.0 * self.eps_smooth
            + self.eps_pa
            + self.eps_cor
            + self.eps_phase_est
        )
        if total_sum > self.eps_sec and not is_close(total_sum, self.eps_sec):
            raise ParameterValidationError(
                "Epsilon allocation insecure: sum of components exceeds eps_sec.",
                context={
                    "eps_sec": self.eps_sec,
                    "component_sum": total_sum,
                    "required": r"$eps\_pe + 2 \times eps\_smooth + eps\_pa + eps\_cor + eps\_phase\_est \le eps\_sec$",
                },
            )

    @property
    def allocated_sum(self) -> float:
        """Return $eps\_pe + 2 eps\_smooth + eps\_pa + eps\_cor + eps\_phase\_est$."""
        return (
            self.eps_pe
            + 2.0 * self.eps_smooth
            + self.eps_pa
            + self.eps_cor
            + self.eps_phase_est
        )

    @property
    def slack(self) -> float:
        """Return $eps\_sec - allocated\_sum$."""
        return self.eps_sec - self.allocated_sum

    @classmethod
    def from_dict(cls, data: JSONMapping, *, strict: bool = False) -> EpsilonAllocation:
        return super().from_dict(data, strict=strict)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, kw_only=True)
class SecurityCertificate(SerializableDataclassMixin):
    """
    An immutable record of the parameters and assumptions used to generate a secure key.

    Attributes:
        proof_name: SecurityProof enumeration.
        confidence_bound_method: ConfidenceBoundMethod enumeration.
        assumed_phase_equals_bit_error: Boolean assumption flag.
        epsilon_allocation: EpsilonAllocation details.
        lp_solver_diagnostics: Optional diagnostics for LP solver (mapping, JSON-serializable).
    """
    proof_name: SecurityProof
    confidence_bound_method: ConfidenceBoundMethod
    assumed_phase_equals_bit_error: bool
    epsilon_allocation: EpsilonAllocation
    lp_solver_diagnostics: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "proof_name",
            _validate_enum(self.proof_name, SecurityProof, "proof_name"),
        )
        object.__setattr__(
            self,
            "confidence_bound_method",
            _validate_enum(
                self.confidence_bound_method,
                ConfidenceBoundMethod,
                "confidence_bound_method",
            ),
        )

        if type(self.assumed_phase_equals_bit_error) is not bool:
            raise ParameterValidationError(
                "assumed_phase_equals_bit_error must be a bool.",
                param_name="assumed_phase_equals_bit_error",
                param_value=self.assumed_phase_equals_bit_error,
            )

        if not isinstance(self.epsilon_allocation, EpsilonAllocation):
            raise ParameterValidationError(
                "epsilon_allocation must be an EpsilonAllocation.",
                param_name="epsilon_allocation",
                param_value=self.epsilon_allocation,
            )

    def to_dict(self) -> JSONDict:
        raw_dict = {
            "proof_name": self.proof_name.value,
            "confidence_bound_method": self.confidence_bound_method.value,
            "assumed_phase_equals_bit_error": self.assumed_phase_equals_bit_error,
            "epsilon_allocation": self.epsilon_allocation.to_dict(),
            "lp_solver_diagnostics": self.lp_solver_diagnostics,
        }
        return sanitize_for_serialization(raw_dict)

    @classmethod
    def from_dict(cls, data: JSONMapping, *, strict: bool = False) -> SecurityCertificate:
        extra_keys = set(data.keys()) - {
            "proof_name",
            "confidence_bound_method",
            "assumed_phase_equals_bit_error",
            "epsilon_allocation",
            "lp_solver_diagnostics",
        }
        if strict and extra_keys:
            raise ParameterValidationError(
                "Unknown keys for SecurityCertificate.",
                context={"unknown_keys": sorted(extra_keys)},
            )

        return cls(
            proof_name=SecurityProof(data["proof_name"]),
            confidence_bound_method=ConfidenceBoundMethod(
                data["confidence_bound_method"]
            ),
            assumed_phase_equals_bit_error=data["assumed_phase_equals_bit_error"],
            epsilon_allocation=EpsilonAllocation.from_dict(
                data["epsilon_allocation"], strict=strict
            ),
            lp_solver_diagnostics=data.get("lp_solver_diagnostics"),
        )


@dataclass(slots=True, kw_only=True)
class SimulationResults(SerializableDataclassMixin):
    """
    A container for the complete results of a QKD simulation run.

    Attributes:
        params: Either a QKDParams object or a ProtocolParameters instance.
        metadata: Arbitrary metadata (JSON-serializable mapping).
        security_certificate: Optional SecurityCertificate.
        decoy_estimates: Optional mapping for decoy-state estimates (structure domain-specific).
        secure_key_length: Optional secure key length (non-negative integer).
        raw_sifted_key_length: Raw sifted key length (non-negative integer).
        simulation_time_seconds: Total simulation time in seconds (non-negative float).
        status: SimulationStatus enum.
        tally_stats: Optional mapping from labels to TallyCounts.
        weak_gllp_rate: Optional float, domain-specific.
        tagged_fraction_bound: Optional float, expected in $[0,1]$ if given.
        beta_y1_deviation: Optional non-negative deviation.
        beta_e1_deviation: Optional non-negative deviation.
        success_probability: Optional probability in $[0,1]$.

    Notes:
        - This class is mutable by design to allow incremental updates. If a frozen
          result object is preferred, consider building it separately.
    """
    params: Union["QKDParams", ProtocolParameters]
    metadata: JSONDict = field(default_factory=dict)
    security_certificate: Optional[SecurityCertificate] = None
    decoy_estimates: Optional[JSONDict] = None
    secure_key_length: Optional[int] = None
    raw_sifted_key_length: int = 0
    simulation_time_seconds: float = 0.0
    status: SimulationStatus = SimulationStatus.OK
    tally_stats: Optional[TallyStatsMap] = None

    weak_gllp_rate: Optional[float] = None
    tagged_fraction_bound: Optional[float] = None

    beta_y1_deviation: Optional[float] = None
    beta_e1_deviation: Optional[float] = None

    success_probability: Optional[float] = None
    schema_version: str = field(default=__version__)

    def __post_init__(self) -> None:
        # Defensive copies to avoid external mutation side-effects
        if not isinstance(self.metadata, dict):
            raise ParameterValidationError("metadata must be a dict.")
        self.metadata = dict(self.metadata)

        if self.decoy_estimates is not None:
            if not isinstance(self.decoy_estimates, dict):
                raise ParameterValidationError(
                    "decoy_estimates must be a dict if provided.",
                    param_name="decoy_estimates",
                    param_value=self.decoy_estimates,
                )
            self.decoy_estimates = dict(self.decoy_estimates)

        if self.tally_stats is not None:
            if not isinstance(self.tally_stats, dict):
                raise ParameterValidationError(
                    "tally_stats must be a mapping from str to TallyCounts.",
                    param_name="tally_stats",
                    param_value=self.tally_stats,
                )
            # Ensure mapping is str -> TallyCounts
            new_map: TallyStatsMap = {}
            for k, v in self.tally_stats.items():
                if not isinstance(k, str):
                    raise ParameterValidationError(
                        "tally_stats keys must be strings.",
                        param_name="tally_stats",
                        param_value=k,
                    )
                if not isinstance(v, TallyCounts):
                    raise ParameterValidationError(
                        "tally_stats values must be TallyCounts instances.",
                        param_name="tally_stats",
                        param_value=v,
                    )
                new_map[k] = v
            self.tally_stats = new_map

        if self.secure_key_length is not None:
            _validate_non_negative_int("secure_key_length", self.secure_key_length)
        _validate_non_negative_int("raw_sifted_key_length", self.raw_sifted_key_length)
        _validate_non_negative_float(
            "simulation_time_seconds", self.simulation_time_seconds
        )

        # Status must be a SimulationStatus; no silent coercion to FAILED
        self.status = _validate_enum(self.status, SimulationStatus, "status")  # type: ignore[assignment]

        # Validate probability-like fields when present
        if self.tagged_fraction_bound is not None:
            _validate_probability_range_01(
                "tagged_fraction_bound", self.tagged_fraction_bound
            )

        if self.weak_gllp_rate is not None:
            _validate_non_negative_float("weak_gllp_rate", self.weak_gllp_rate)

        if self.beta_y1_deviation is not None:
            _validate_non_negative_float("beta_y1_deviation", self.beta_y1_deviation)

        if self.beta_e1_deviation is not None:
            _validate_non_negative_float("beta_e1_deviation", self.beta_e1_deviation)

        if self.success_probability is not None:
            _validate_probability_range_01(
                "success_probability", self.success_probability
            )

        if self.security_certificate is not None and not isinstance(
            self.security_certificate, SecurityCertificate
        ):
            raise ParameterValidationError(
                "security_certificate must be a SecurityCertificate instance.",
                param_name="security_certificate",
                param_value=self.security_certificate,
            )

    # Convenience methods

    @property
    def is_successful(self) -> bool:
        """Return True if the simulation completed successfully."""
        return self.status == SimulationStatus.OK

    def with_status(self, status: SimulationStatus) -> SimulationResults:
        """Return a shallow copy with updated status."""
        status = _validate_enum(status, SimulationStatus, "status")  # type: ignore[assignment]
        return replace(self, status=status)

    def to_dict(self) -> JSONDict:
        # Strict serialization: params must implement to_dict or to_serializable_dict
        params_obj = self.params
        if hasattr(params_obj, "to_dict"):
            params_dict = params_obj.to_dict()  # type: ignore[assignment]
        elif hasattr(params_obj, "to_serializable_dict"):
            params_dict = params_obj.to_serializable_dict()  # type: ignore[assignment]
        else:
            raise TypeError(
                "params must implement 'to_dict' or 'to_serializable_dict' for serialization."
            )

        raw_dict: JSONDict = {
            "params": params_dict,
            "metadata": self.metadata,
            "security_certificate": self.security_certificate.to_dict()
            if self.security_certificate
            else None,
            "decoy_estimates": self.decoy_estimates,
            "secure_key_length": self.secure_key_length,
            "raw_sifted_key_length": self.raw_sifted_key_length,
            "simulation_time_seconds": self.simulation_time_seconds,
            "status": self.status.value,
            "tally_stats": {
                k: v.to_dict() for k, v in self.tally_stats.items()
            }
            if self.tally_stats is not None
            else None,
            "weak_gllp_rate": self.weak_gllp_rate,
            "tagged_fraction_bound": self.tagged_fraction_bound,
            "beta_y1_deviation": self.beta_y1_deviation,
            "beta_e1_deviation": self.beta_e1_deviation,
            "success_probability": self.success_probability,
            "schema_version": self.schema_version,
        }
        return sanitize_for_serialization(raw_dict)

    @classmethod
    def from_dict(
        cls,
        data: JSONMapping,
        *,
        params_factory: Optional[
            callable  # (JSONMapping) -> Union[QKDParams, ProtocolParameters]
        ] = None,
        strict: bool = False,
    ) -> SimulationResults:
        """
        Deserialize SimulationResults from a mapping.

        Args:
            data: JSON-like mapping.
            params_factory: Callable to reconstruct `params` from the serialized
                mapping. If None, the raw mapping is left as-is and must be
                converted by the caller.
            strict: If True, unknown top-level keys cause an error.
        """
        valid_keys = {
            "params",
            "metadata",
            "security_certificate",
            "decoy_estimates",
            "secure_key_length",
            "raw_sifted_key_length",
            "simulation_time_seconds",
            "status",
            "tally_stats",
            "weak_gllp_rate",
            "tagged_fraction_bound",
            "beta_y1_deviation",
            "beta_e1_deviation",
            "success_probability",
            "schema_version",
        }
        extra_keys = set(data.keys()) - valid_keys
        if strict and extra_keys:
            raise ParameterValidationError(
                "Unknown keys for SimulationResults.",
                context={"unknown_keys": sorted(extra_keys)},
            )

        params_data = data.get("params")
        if params_factory is not None:
            params = params_factory(params_data)
        else:
            # Fallback: store raw mapping; caller should wrap into appropriate object later.
            params = params_data

        status_val = data.get("status", SimulationStatus.OK.value)
        status_enum = SimulationStatus(status_val)

        sec_cert_data = data.get("security_certificate")
        security_certificate = (
            SecurityCertificate.from_dict(sec_cert_data, strict=strict)
            if sec_cert_data is not None
            else None
        )

        tally_stats_data = data.get("tally_stats")
        tally_stats: Optional[TallyStatsMap] = None
        if tally_stats_data is not None:
            if not isinstance(tally_stats_data, Mapping):
                raise ParameterValidationError(
                    "tally_stats must be a mapping from str to dict.",
                    param_name="tally_stats",
                    param_value=tally_stats_data,
                )
            tally_stats = {
                k: TallyCounts.from_dict(v, strict=strict)
                for k, v in tally_stats_data.items()
            }

        return cls(
            params=params,
            metadata=dict(data.get("metadata", {})),
            security_certificate=security_certificate,
            decoy_estimates=data.get("decoy_estimates"),
            secure_key_length=data.get("secure_key_length"),
            raw_sifted_key_length=data.get("raw_sifted_key_length", 0),
            simulation_time_seconds=data.get("simulation_time_seconds", 0.0),
            status=status_enum,
            tally_stats=tally_stats,
            weak_gllp_rate=data.get("weak_gllp_rate"),
            tagged_fraction_bound=data.get("tagged_fraction_bound"),
            beta_y1_deviation=data.get("beta_y1_deviation"),
            beta_e1_deviation=data.get("beta_e1_deviation"),
            success_probability=data.get("success_probability"),
            schema_version=data.get("schema_version", __version__),
        )
        


