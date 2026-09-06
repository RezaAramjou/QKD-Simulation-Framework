# qkd/proofs/base.py
# -*- coding: utf-8 -*-
"""
Abstract base class for finite-key security proofs in quantum key distribution.

This module defines the core interface for all security proof implementations,
ensuring a standardized structure for epsilon allocation, parameter estimation,
and secure key length calculation.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import time
import traceback
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from functools import lru_cache
from typing import (Any, Callable, Dict, Generator, Iterable, Literal,
                    Protocol, Tuple, Type, TypeVar, Optional, Union)

import numpy as np

# --- QKD Parameter Imports ---
try:
    from ..params import QKDParams
except ImportError as exc:
    raise RuntimeError("Could not import from 'params.py'.") from exc

# --- QKD Data Type Imports ---
try:
    from ..datatypes import ConfidenceBoundMethod
except ImportError as exc:
    raise RuntimeError("Could not import from 'datatypes.py'.") from exc

# --- QKD Constants Imports ---
try:
    from ..constants import (
        ENTROPY_PROB_CLAMP,
        NUMERIC_ABS_TOL,
        MAX_SEED_INT_PCG64
    )
except ImportError as exc:
    raise RuntimeError("Could not import from 'constants.py'.") from exc

# --- QKD Exception Imports ---
try:
    from ..exceptions import (
        ConfigurationError,
        ParameterValidationError as QKDParamError,
        QKDSimulationError,
        ErrorCode as QKDErrorCode
    )
except ImportError as exc:
    raise RuntimeError("Could not import custom exceptions from 'exceptions.py'.") from exc

# --- Safe SciPy Import ---
try:
    import scipy
    from scipy import special
    from scipy.stats import beta
    if tuple(map(int, scipy.__version__.split('.')[:2])) < (1, 8):
        raise ConfigurationError(
            "SciPy >= 1.8 is required for reliable beta.ppf behavior.",
            code=QKDErrorCode.CONFIG,
            context={"installed_version": scipy.__version__, "required_version": ">=1.8"}
        )
except ImportError as exc:
    raise ConfigurationError(
        "SciPy is required for Clopper-Pearson confidence intervals.",
        code=QKDErrorCode.CONFIG,
        cause=exc
    ) from exc

# --- High-Precision Math ---
try:
    import mpmath  # type: ignore
except ImportError:
    mpmath = None

from ..utils import math as qkd_math  # For mathematical helper functions

__version__ = "4.2.0"
__all__ = [
    'FiniteKeyProof', 'DecoyEstimates', 'KeyCalculationResult', 'TallyCounts',
    'EpsilonAllocation', 'SolverDiagnostics', 'ProofMode', 'ErrorCode',
    'ConfidenceBoundMethod', 'TALLY_KEY_Z_SIGNAL', 'TALLY_KEY_X_SIGNAL', 'derive_child_seed'
]

T = TypeVar("T")
TALLY_KEY_Z_SIGNAL = "Z_signal"
TALLY_KEY_X_SIGNAL = "X_signal"

class ProofMode(Enum):
    PRODUCTION = auto()
    DEBUG = auto()
    AUDIT = auto()

class ErrorCode(Enum):
    INSUFFICIENT_STATISTICS = auto()
    LP_INFEASIBLE = auto()
    NUMERICAL_CLAMP_APPLIED = auto()
    LEAKAGE_EXCEEDS_ENTROPY = auto()
    HIGH_SOLVER_RESIDUAL = auto()
    SAFE_DIVIDE_FALLBACK = auto()

def _json_safe_dict(data: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    """Recursively converts numpy/enum types to native Python types for JSON."""
    d: Dict[str, Any] = {}
    for key, value in data:
        if isinstance(value, (np.integer, np.int64)):
            d[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            d[key] = float(value)
        elif isinstance(value, np.ndarray):
            if value.size > 1000:
                d[key] = {'data': value[:1000].tolist(), 'truncated': True}
            else:
                d[key] = value.tolist()
        elif isinstance(value, np.bool_):
            d[key] = bool(value)
        elif isinstance(value, Enum):
            d[key] = value.name
        elif isinstance(value, dict):
            d[key] = _json_safe_dict(value.items())
        else:
            d[key] = value
    return d

@dataclass(frozen=True)
class TallyCounts:
    detections: int = 0
    errors: int = 0

    def __post_init__(self):
        if self.detections < 0 or self.errors < 0:
            raise QKDParamError(
                "Tally counts cannot be negative.",
                code=QKDErrorCode.PARAM_VALIDATION,
                context={"detections": self.detections, "errors": self.errors}
            )
        if self.errors > self.detections:
            raise QKDParamError(
                f"Error count ({self.errors}) cannot exceed detection count ({self.detections}).",
                code=QKDErrorCode.PARAM_VALIDATION,
                context={"detections": self.detections, "errors": self.errors}
            )

@dataclass
class EpsilonAllocation:
    eps_sec: float
    eps_pe: float
    eps_smooth: float
    eps_pa: float

    def validate(self, policy: Callable[["EpsilonAllocation"], float]) -> None:
        consumed = policy(self)
        if consumed > self.eps_sec:
            raise QKDParamError(
                f"Epsilon allocation exceeds budget: {consumed:.2e} > {self.eps_sec:.2e}",
                code=QKDErrorCode.PARAM_VALIDATION,
                context={"consumed": consumed, "budget": self.eps_sec, "policy": policy.__name__}
            )

@dataclass(frozen=True)
class SolverDiagnostics:
    solver_name: str
    is_success: bool
    status_message: str
    residual_norm: Optional[float] = None
    timings: Dict[str, float] = field(default_factory=dict)
    numeric_diagnostics: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class DecoyEstimates:
    yield_1_lower_bound: float
    error_rate_1_upper_bound: float
    is_feasible: bool
    failure_prob_used: float
    diagnostics: SolverDiagnostics

    def __repr__(self) -> str:
        return (f"DecoyEstimates(Y1_L={self.yield_1_lower_bound:.2e}, "
                f"e1_U={self.error_rate_1_upper_bound:.3f}, feasible={self.is_feasible})")

    def as_serializable(self) -> Dict[str, Any]:
        data = asdict(self, dict_factory=_json_safe_dict)
        data['metadata'] = {
            'schema_version': '1.2', 'lib_version': __version__,
            'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
        }
        return data

@dataclass(frozen=True)
class KeyCalculationResult:
    """
    Standard result object for finite-key secure key rate calculations.
    
    Updated to include optional fields used by Ma et al. 2005 proofs and to
    allow flexible diagnostics (list or dict).
    """
    secure_key_length: int
    privacy_amplification_term: float
    error_correction_leakage: float
    phase_error_rate_upper_bound: float
    
    # Diagnostics can be a list of strings or a structured dict
    diagnostics: Any = field(default_factory=dict, repr=False)
    error_codes: list[ErrorCode] = field(default_factory=list)
    
    # Optional fields for detailed analysis (e.g. Ma 2005 comparison logic)
    weak_gllp_rate: Optional[float] = None
    tagged_fraction_bound: Optional[float] = None

    def __repr__(self) -> str:
        return f"KeyCalculationResult(len={self.secure_key_length}, e_ph={self.phase_error_rate_upper_bound:.3f})"

    def as_serializable(self) -> Dict[str, Any]:
        data = asdict(self, dict_factory=_json_safe_dict)
        data['error_codes'] = [e.name for e in self.error_codes]
        data['metadata'] = {'schema_version': '1.3', 'lib_version': __version__}
        return data


# --- Abstract Base Class for Security Proofs ---
class FiniteKeyProof(ABC):
    __implementation_version__: str = "0.0.0"
    allowed_ci_methods: frozenset[ConfidenceBoundMethod] = frozenset(ConfidenceBoundMethod)
    min_y1_threshold: float = 1e-9

    def __init__(self, params: QKDParams, mode: ProofMode = ProofMode.PRODUCTION):
        self.p = params
        self.mode = mode
        self.logger = logging.getLogger(self.__class__.__name__)
        self.run_id = getattr(params, 'run_id', 'no-run-id')
        self.min_prob_clamp = getattr(params, 'entropy_clamp', ENTROPY_PROB_CLAMP)
        self.use_mpmath = getattr(params, 'use_mpmath', False) and mpmath is not None

        self.logger.info("event=init run_id=%s proof=%s version=%s numpy=%s scipy=%s",
                         self.run_id, self.__class__.__name__, self.__implementation_version__,
                         np.__version__, scipy.__version__)

        self.validate_physical_params()
        self._check_notation_map()

        if self.p.ci_method not in self.allowed_ci_methods:
            msg = f"CI method '{self.p.ci_method.name}' is not recommended for this proof."
            if self.is_audit_mode():
                raise QKDParamError(
                    msg,
                    code=QKDErrorCode.PARAM_VALIDATION,
                    context={"ci_method": self.p.ci_method.name,
                             "allowed": [m.name for m in self.allowed_ci_methods]}
                )
            self.logger.warning("[%s] %s", self.run_id, msg)

        self.eps_alloc = self.allocate_epsilons()
        self.eps_alloc.validate()

    def is_audit_mode(self) -> bool: return self.mode == ProofMode.AUDIT
    def is_debug_mode(self) -> bool: return self.mode in (ProofMode.DEBUG, ProofMode.AUDIT)

    @abstractmethod
    def notation_map(self) -> Dict[str, str]: ...
    @abstractmethod
    def allocate_epsilons(self) -> EpsilonAllocation: ...
    @abstractmethod
    def get_epsilon_policy(self) -> Callable[[EpsilonAllocation], float]: ...
    @abstractmethod
    def estimate_yields_and_errors(self, stats_map: Dict[str, TallyCounts]) -> DecoyEstimates: ...
    @abstractmethod
    def calculate_key_length(self, decoy_estimates: DecoyEstimates, stats_map: Dict[str, TallyCounts]) -> KeyCalculationResult: ...

    def validate_physical_params(self) -> None:
        num_pulses = getattr(self.p, 'num_pulses', None)
        if num_pulses is not None and num_pulses < 100:
            msg = f"Number of pulses ({num_pulses}) is very low for a meaningful run."
            if self.is_audit_mode():
                raise QKDParamError(
                    msg,
                    param_name="num_pulses",
                    param_value=num_pulses,
                    code=QKDErrorCode.PARAM_VALIDATION
                )
            self.logger.warning(msg)

    def validate_stats_map(self, stats_map: Dict[str, TallyCounts], required_keys: list[str]):
        for key in required_keys:
            if key not in stats_map:
                raise QKDParamError(
                    f"Missing required key in stats_map: '{key}'",
                    param_name="stats_map",
                    code=QKDErrorCode.PARAM_VALIDATION,
                    context={"missing_key": key, "required_keys": required_keys}
                )
            if not isinstance(stats_map[key], TallyCounts):
                raise QKDParamError(
                    f"Value for key '{key}' must be a TallyCounts instance.",
                    param_name=key,
                    param_value=type(stats_map[key]).__name__,
                    code=QKDErrorCode.PARAM_VALIDATION
                )

    def get_bounds(self, k: int, n: int, failure_prob: float, sided: Literal["two", "one_upper", "one_lower"] = "two", diagnostics: Dict | None = None) -> Tuple[float, float]:
        k, n = int(k), int(n)
        if not (0 <= k <= n):
            raise QKDParamError(
                f"k must be in [0, n], got k={k}, n={n}.",
                code=QKDErrorCode.PARAM_VALIDATION,
                context={"k": k, "n": n}
            )
        if not (1e-300 < failure_prob < 1):
            raise QKDParamError(
                f"Failure probability must be in (1e-300, 1), got {failure_prob}.",
                param_name="failure_prob",
                param_value=failure_prob,
                code=QKDErrorCode.PARAM_VALIDATION
            )
        if n == 0: return (0.0, 1.0)
        alpha = float(failure_prob if sided != "two" else failure_prob / 2.0)

        if diagnostics is not None:
            diagnostics['alpha_used'] = alpha

        sided_arg = "two-sided" if sided == "two" else ("upper" if sided == "one_upper" else "lower")

        if self.p.ci_method == ConfidenceBoundMethod.CLOPPER_PEARSON:
            interval = qkd_math.clopper_pearson_bounds(k, n, failure_prob, side=sided_arg)
            lower, upper = interval.lower, interval.upper
        elif self.p.ci_method == ConfidenceBoundMethod.HOEFFDING:
            interval = qkd_math.hoeffding_bounds(k, n, failure_prob, side=sided_arg)
            lower, upper = interval.lower, interval.upper
        elif self.p.ci_method == ConfidenceBoundMethod.GAUSSIAN:
            interval = qkd_math.gaussian_bounds(k, n, failure_prob, side=sided_arg)
            lower, upper = interval.lower, interval.upper
        elif self.p.ci_method == ConfidenceBoundMethod.CHERNOFF:
            interval = qkd_math.chernoff_bounds(k, n, failure_prob, side=sided_arg)
            lower, upper = interval.lower, interval.upper
        else:
            raise QKDSimulationError(
                f"CI method '{self.p.ci_method.name}' not implemented.",
                code=QKDErrorCode.CONFIG,
                context={"ci_method": self.p.ci_method.name}
            )

        self._assert_finite("lower bound", lower); self._assert_finite("upper bound", upper)
        return (lower, upper)

    def binary_entropy(self, p_err: float | np.ndarray | list) -> float | np.ndarray:
        p = np.asarray(p_err, dtype=np.float64)
        p = np.clip(p, self.min_prob_clamp, 1.0 - self.min_prob_clamp)
        with np.errstate(divide='ignore', invalid='ignore', over='raise' if self.is_audit_mode() else 'warn'):
            h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        h = np.nan_to_num(h, nan=0.0)
        return h.item() if isinstance(p_err, (float, int)) else h

    def _check_notation_map(self):
        required = {"Y_1^L", "e_ph", "s_z_1^L"}
        if not required.issubset(self.notation_map().keys()):
            raise ConfigurationError(
                f"Subclass '{self.__class__.__name__}' must implement notation_map with keys: {required}",
                code=QKDErrorCode.CONFIG,
                context={"required_keys": required, "class": self.__class__.__name__}
            )

    def _assert_finite(self, name: str, value: Any):
        if not math.isfinite(value):
            msg = f"Computation of '{name}' resulted in non-finite value: {value}"
            if self.is_audit_mode():
                raise QKDSimulationError(
                    msg,
                    code=QKDErrorCode.SIMULATION,
                    context={"name": name, "value": value}
                )
            self.logger.error(msg)

    def _safe_log(self, x: float) -> float:
        if x < 1e-300:
            if self.is_audit_mode():
                raise QKDSimulationError(
                    "Log argument too small in AUDIT mode.",
                    code=QKDErrorCode.SIMULATION,
                    context={"x": x}
                )
            self.logger.warning("Log argument %e is smaller than safe limit, clamping.", x)
            x = 1e-300
        return math.log(x)

    def _safe_divide(self, num: float, den: float, default: float = 0.0, name: str = '', error_codes: list | None = None) -> float:
        if abs(den) < NUMERIC_ABS_TOL:
            if error_codes is not None: error_codes.append(ErrorCode.SAFE_DIVIDE_FALLBACK)
            self.logger.warning("Denominator for '%s' is near zero (%e). Returning default value %f.", name, den, default)
            return default
        return num / den

    def _clamp_nonneg(self, x: float, name: str, error_codes: list | None = None) -> float:
        if x < 0:
            if error_codes is not None: error_codes.append(ErrorCode.NUMERICAL_CLAMP_APPLIED)
            if x < -NUMERIC_ABS_TOL and self.is_audit_mode():
                raise QKDSimulationError(
                    f"Negative value for {name} ({x}) exceeded tolerance in AUDIT mode.",
                    code=QKDErrorCode.SIMULATION,
                    context={"name": name, "value": x, "tolerance": -NUMERIC_ABS_TOL}
                )
            self.logger.debug("Clamped negative value for %s: %e -> 0.0", name, x)
            return 0.0
        return x

    def _clamp_key_length(self, l_float: float) -> int:
        return int(max(0.0, math.floor(l_float)))

    @contextmanager
    def _timed(self, timings: Dict, key: str) -> Generator:
        t0 = time.perf_counter(); yield; timings[key] = time.perf_counter() - t0

    def _conservative_decoy_estimate(self, failure_prob: float, diagnostics: SolverDiagnostics) -> DecoyEstimates:
        return DecoyEstimates(0.0, 0.5, False, failure_prob, diagnostics)

    def audit_dump(self, result: KeyCalculationResult, decoy: DecoyEstimates, stats: Dict, start_time: float) -> Dict:
        dump = {
            "params": self.p.to_summary_dict(redact=True), "mode": self.mode.name,
            "proof_implementation": f"{self.__class__.__name__} v{self.__implementation_version__}",
            "stats_map": {k: asdict(v) for k, v in stats.items()},
            "decoy_estimates": decoy.as_serializable(),
            "key_calculation_result": result.as_serializable(),
            "timestamps": {'start': start_time, 'end': time.time()},
        }
        if self.is_audit_mode():
             y1_margin = decoy.yield_1_lower_bound - self.min_y1_threshold
             e1_margin = 0.5 - decoy.error_rate_1_upper_bound 
             dump['sensitivity_summary'] = {'y1_margin': y1_margin, 'e1_margin': e1_margin}
        return dump

    def explain_key_decision(self, result: KeyCalculationResult, decoy: DecoyEstimates) -> str:
        if result.secure_key_length > 0:
            return f"Secure key generated. Final length: {result.secure_key_length} bits."
        reasons = []
        if ErrorCode.INSUFFICIENT_STATISTICS in result.error_codes:
            reasons.append(f"insufficient single-photon yield (Y1_L={decoy.yield_1_lower_bound:.2e} < threshold={self.min_y1_threshold:.2e})")
        if ErrorCode.LP_INFEASIBLE in result.error_codes:
            reasons.append("decoy-state analysis was infeasible")
        if ErrorCode.LEAKAGE_EXCEEDS_ENTROPY in result.error_codes:
            reasons.append(f"information leakage ({result.error_correction_leakage + result.privacy_amplification_term:.2f}) exceeded available entropy")
        if not reasons: reasons.append("an unspecified condition led to a non-positive key length")
        return f"No secure key generated because {', and '.join(reasons)}."


@lru_cache(maxsize=128)
def derive_child_seed(master_seed: int, index: int) -> int:
    """Deterministically derives a child seed. Not for cryptographic use."""
    key = str(master_seed).encode()
    msg = str(index).encode()
    digest = hmac.new(key, msg, digestmod=hashlib.sha256).digest()
    return int.from_bytes(digest[:8], 'big') % MAX_SEED_INT_PCG64 or 1
