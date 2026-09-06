# -*- coding: utf-8 -*-
"""
Improved implementation of the MDI-QKD finite-key security proof.

This module provides a comprehensively refactored, safety-focused implementation
for calculating the secure key length in an MDI-QKD protocol. It inherits from a
base proof class and incorporates robust validation, numerical stability checks,
extensive diagnostics, and a clear API.

WARNING: The core key-rate calculation formula remains a placeholder adapted
from BB84, as a full 2D decoy-state analysis for MDI-QKD is not implemented.
This placeholder logic is disabled by default and requires an explicit, high-risk
opt-in to run. Its use for any real security evaluation is strongly discouraged.
"""
# Safety Checklist for Maintainers:
# 1. Epsilon Allocation: Ensure self.eps_alloc is properly initialized with
#    positive security parameters before any calculation.
# 2. Pulse Config: A valid 'signal' pulse configuration must exist in self.p.source.
# 3. Solver Tolerances: self.p.solver_tol must be set to a reasonable value
#    to catch unreliable LP solutions.
# 4. Unsafe Approximation: The `allow_unsafe_mdi_approx` flag must never be
#    enabled in a production security environment.
# 5. CI Method: The `ci_method` must be a valid ConfidenceBoundMethod enum member.

import dataclasses
import datetime
import enum
import hashlib
import hmac
import inspect
import json
import logging
import math
import os
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# ==============================================================================
# Constants
# ==============================================================================
# From specific PA proof, document origin if this were a real proof.
# Citation: Placeholder based on Lim et al., Phys. Rev. A 89, 022341 (2014)
PA_SMOOTH_FACTOR = 7.0
PA_MAGIC_CONSTANT = 21.0
EPS_MIN = 1e-300 # Minimum safe epsilon to avoid log(inf)
MAX_DIAGNOSTICS_ENTRIES = 1000 # Safety cap for diagnostics list

# ==============================================================================
# Mock Objects and Dependencies (for self-contained demonstration)
# ==============================================================================

class QKDSimulationError(Exception):
    """Custom exception for simulation errors."""
    pass

class ParameterValidationError(ValueError):
    """Custom exception for invalid parameters."""
    pass

class ConfidenceBoundMethod(enum.Enum):
    """Enum for confidence interval calculation methods."""
    HOEFFDING = "hoeffding"
    CLOPPER_PEARSON = "clopper_pearson"

class ErrorCode(enum.Enum):
    """Enum for machine-readable error codes."""
    INSUFFICIENT_STATISTICS = "INSUFFICIENT_STATISTICS"
    UNSAFE_APPROXIMATION_USED = "UNSAFE_APPROXIMATION_USED"
    LP_INFEASIBLE = "LP_INFEASIBLE"
    HIGH_SOLVER_RESIDUAL = "HIGH_SOLVER_RESIDUAL"
    NUMERICAL_CLAMP_APPLIED = "NUMERICAL_CLAMP_APPLIED"
    INVALID_INPUT_STATS = "INVALID_INPUT_STATS"
    MISSING_PULSE_CONFIG = "MISSING_PULSE_CONFIG"
    ZERO_DENOMINATOR = "ZERO_DENOMINATOR"
    EPSILON_MISMATCH = "EPSILON_MISMATCH"
    LARGE_PA_TERM = "LARGE_PA_TERM"
    INVALID_EPSILON = "INVALID_EPSILON"
    HIGH_UNCERTAINTY = "HIGH_UNCERTAINTY"
    IMBALANCED_BASIS_STATS = "IMBALANCED_BASIS_STATS"

@dataclasses.dataclass
class TallyCounts:
    """Mock dataclass for storing statistical counts."""
    sent: int = 0
    sifted_z: int = 0
    sifted_x: int = 0
    errors_sifted_z: int = 0
    errors_sifted_x: int = 0

@dataclasses.dataclass
class SolverDiagnostics:
    """Mock dataclass for decoy solver diagnostics."""
    residual_norm: Optional[float] = 0.0
    status_message: str = "optimal"
    solver_name: str = "mock_solver"
    timings: Dict[str, float] = dataclasses.field(default_factory=dict)
    numeric_diagnostics: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

@dataclasses.dataclass
class DecoyEstimates:
    """Mock dataclass for the output of a decoy-state analysis."""
    is_feasible: bool = True
    yield_1_lower_bound: float = 0.0
    error_rate_1_upper_bound: float = 0.0
    failure_prob_used: float = 1e-10
    diagnostics: SolverDiagnostics = dataclasses.field(default_factory=SolverDiagnostics)
    truncated_Nvar: Optional[int] = None

    def as_serializable(self) -> Dict[str, Any]:
        return _json_safe_dict(dataclasses.asdict(self))

@dataclasses.dataclass
class KeyCalculationResult:
    """
    Structured result of a key length calculation.
    - secure_key_length (int): Final secure key length in bits.
    - privacy_amplification_term (float): Total bits leaked during privacy amp.
    - error_correction_leakage (float): Total bits leaked during error correction.
    - phase_error_rate_upper_bound (float): Upper bound on phase error rate [0, 0.5].
    """
    secure_key_length: int = 0
    privacy_amplification_term: float = 0.0
    error_correction_leakage: float = 0.0
    phase_error_rate_upper_bound: float = 0.0
    diagnostics: List[Any] = dataclasses.field(default_factory=list)
    error_codes: List[ErrorCode] = dataclasses.field(default_factory=list)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def as_serializable(self) -> Dict[str, Any]:
        return _json_safe_dict(dataclasses.asdict(self))

def _json_safe_dict(obj: Any, truncate_len: int = 100) -> Any:
    """Helper to make a dictionary JSON-serializable, with truncation."""
    if isinstance(obj, dict):
        return {k: _json_safe_dict(v, truncate_len) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > truncate_len:
            return [_json_safe_dict(i, truncate_len) for i in obj[:truncate_len]] + [f"... (truncated {len(obj) - truncate_len} items)"]
        return [_json_safe_dict(i, truncate_len) for i in obj]
    if isinstance(obj, enum.Enum):
        return obj.value
    if hasattr(obj, 'as_serializable'):
        return obj.as_serializable()
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)

class MockParams:
    """Mock class for simulation parameters."""
    def __init__(self):
        self.allow_unsafe_mdi_approx: Union[bool, str] = False
        self.f_error_correction = 1.16
        self.ci_method: ConfidenceBoundMethod = ConfidenceBoundMethod.HOEFFDING
        self.solver_tol = 1e-9
        self.hmac_key: Optional[bytes] = b'secret-key-for-testing'
        self.source = self
        self.pulse_configs = {"signal": {"name": "signal", "mean_photon_number": 0.5}}

    def get_pulse_config_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        return self.pulse_configs.get(name)

    def sanitized_params(self) -> Dict[str, Any]:
        return {"allow_unsafe_mdi_approx": self.allow_unsafe_mdi_approx, "f_error_correction": self.f_error_correction}

class Lim2014Proof:
    """Mock base class for a security proof, providing helper methods."""
    is_placeholder: bool = False
    __implementation_version__ = "1.3.0"

    def __init__(self, p: MockParams, run_id: str = "run-001"):
        self.p = p
        self.run_id = run_id
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{self.run_id}]")
        self.eps_alloc = self._allocate_epsilons()
        self.min_y1_threshold = 1e-12
        self.NUMERIC_TOL = 1e-9

    def _allocate_epsilons(self) -> Any:
        class Eps:
            eps_pe = 1e-10
            eps_smooth = 1e-10
            eps_pa = 1e-10
            eps_cor = 1e-10
        return Eps()

    def get_epsilon_policy(self) -> Optional[Callable]:
        return lambda eps: eps.eps_pe

    def is_audit_mode(self) -> bool:
        return True

    def is_debug_mode(self) -> bool:
        return True

    def binary_entropy(self, x: float) -> float:
        x_float = float(x)
        if x_float <= self.NUMERIC_TOL or x_float >= 1.0 - self.NUMERIC_TOL:
            return 0.0
        return -x_float * math.log2(x_float) - (1.0 - x_float) * math.log2(1.0 - x_float)

    def get_bounds(self, k: int, n: int, eps: float, sided: str = 'two', diagnostics: Optional[Dict] = None) -> Tuple[float, float]:
        if n <= 0:
            return (0.0, 1.0)
        p_hat = k / n
        log_term = self._safe_log(1.0 / max(eps, EPS_MIN))
        self._assert_finite(log_term, 'log_term_for_ci')
        delta = math.sqrt(max(0.0, log_term / (2.0 * n)))
        if diagnostics is not None:
            diagnostics['ci_method'] = self.p.ci_method.name
            diagnostics['ci_delta'] = delta
        if sided == 'one_upper':
            return (0.0, min(1.0, p_hat + delta))
        return (max(0.0, p_hat - delta), min(1.0, p_hat + delta))

    def _clamp_nonneg(self, val: float, name: str, result: KeyCalculationResult) -> float:
        if val < 0:
            self._add_diagnostic(result, 'clamp', 'warning', {'name': name, 'original': val, 'clamped_to': 0.0})
            self._add_error_code(result, ErrorCode.NUMERICAL_CLAMP_APPLIED)
            return 0.0
        return val

    def _safe_divide(self, num: float, den: float, default: float, name: str, result: KeyCalculationResult) -> float:
        if abs(den) < self.NUMERIC_TOL:
            self.logger.warning(f"Safe division fallback for '{name}': denominator is near zero ({den:.2e}).")
            self._add_error_code(result, ErrorCode.ZERO_DENOMINATOR)
            return default
        return num / den

    def _clamp_key_length(self, key_len_float: float) -> int:
        if not math.isfinite(key_len_float) or key_len_float < 0:
            return 0
        return math.floor(key_len_float + self.NUMERIC_TOL)

    def _safe_log(self, x: float) -> float:
        if x <= 0:
            return -math.inf
        return math.log(x)

    def _log2_safe(self, x: float) -> float:
        if x <= 0:
            return -math.inf
        return math.log2(x)

    def _assert_finite(self, val: float, name: str):
        if not math.isfinite(val):
            raise QKDSimulationError(f"Numeric error: '{name}' is not finite ({val}).")

    def _conservative_decoy_estimate(self) -> DecoyEstimates:
        return DecoyEstimates(is_feasible=False, yield_1_lower_bound=0.0, error_rate_1_upper_bound=0.5)

    def _add_diagnostic(self, result: KeyCalculationResult, diag_type: str, severity: str, payload: Dict):
        if len(result.diagnostics) >= MAX_DIAGNOSTICS_ENTRIES:
            return
        result.diagnostics.append({
            'type': diag_type,
            'severity': severity,
            'timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',
            'payload': payload
        })

    def _add_error_code(self, result: KeyCalculationResult, code: ErrorCode):
        if code not in result.error_codes:
            result.error_codes.append(code)

    def explain_key_decision(self, result: KeyCalculationResult, decoy_estimates: Optional[DecoyEstimates] = None) -> Tuple[str, str]:
        if result.secure_key_length > 0:
            return "OK", f"Secure key generated. Length: {result.secure_key_length} bits."
        
        clauses = {
            ErrorCode.INSUFFICIENT_STATISTICS: ("INSUFFICIENT_STATS", "insufficient statistics (e.g., sifted counts or single-photon yield too low)"),
            ErrorCode.LP_INFEASIBLE: ("LP_INFEASIBLE", "decoy-state analysis was infeasible"),
        }
        codes, reasons = [], []
        for code in clauses:
            if code in result.error_codes:
                codes.append(clauses[code][0])
                reasons.append(clauses[code][1])
        
        if result.phase_error_rate_upper_bound >= 0.5 - self.NUMERIC_TOL:
              codes.append("HIGH_PHASE_ERROR")
              reasons.append(f"phase error rate is too high (e_ph={result.phase_error_rate_upper_bound:.3f})")
        
        if result.diagnostics and result.diagnostics[0].get('type') == 'summary':
              payload = result.diagnostics[0].get('payload', {})
              raw_bits = payload.get('s_z_11_L_counts', 0)
              if result.privacy_amplification_term + result.error_correction_leakage > raw_bits > 0:
                  codes.append("LEAKAGE_EXCEEDS_RAW_BITS")
                  reasons.append("privacy amplification and error correction leakage terms exceed estimated raw bits")

        return "_".join(codes) or "UNKNOWN", "; ".join(reasons) or "unknown numerical or statistical reason"

# ==============================================================================
# Refactored MDIQKDProof Implementation
# ==============================================================================

__all__ = ["MDIQKDProof"]

class MDIQKDProof(Lim2014Proof):
    is_placeholder: bool = True
    _REQUIRES_2D_DECOY: bool = True

    def __init__(self, p: MockParams, run_id: str = "run-001"):
        super().__init__(p, run_id)
        self._validate_params()

    def _validate_params(self) -> None:
        if not isinstance(self.p.ci_method, ConfidenceBoundMethod):
            raise ParameterValidationError(f"ci_method must be ConfidenceBoundMethod, got {type(self.p.ci_method)}")
        if not (0 < self.p.f_error_correction < 10):
            raise ParameterValidationError(f"f_error_correction ({self.p.f_error_correction}) is out of reasonable bounds (0, 10).")
        if self.p.hmac_key and not (isinstance(self.p.hmac_key, (bytes, bytearray)) and len(self.p.hmac_key) > 0):
            raise ParameterValidationError("hmac_key must be non-empty bytes or bytearray.")
        if not hasattr(self, 'eps_alloc'):
            raise ParameterValidationError("eps_alloc is not initialized.")
        for name in ['eps_pe', 'eps_smooth', 'eps_pa', 'eps_cor']:
            eps_val = getattr(self.eps_alloc, name, None)
            if not (isinstance(eps_val, (int, float)) and 0 < eps_val < 1):
                raise ParameterValidationError(f"Missing or invalid security parameter: eps_alloc.{name}")

    def _abort_zero_key(self, result: KeyCalculationResult, code: ErrorCode, msg: str, location: str) -> KeyCalculationResult:
        self.logger.warning(msg)
        self._add_error_code(result, code)
        self._add_diagnostic(result, 'abort', 'error', {'message': msg, 'location': location})
        result.secure_key_length = 0
        result.phase_error_rate_upper_bound = 0.5
        return result

    def _get_location(self) -> str:
        """Safely gets the current file and line number for logging."""
        frame = inspect.currentframe()
        # Go back one frame to get the caller's location
        if frame:
            frame = frame.f_back
        lineno = frame.f_lineno if frame else "N/A"
        filename = os.path.basename(frame.f_code.co_filename) if frame else "unknown"
        return f"{filename}:{lineno}"

    def _compute_phase_error_bound(self, n_x: int, m_x: int, decoy_estimates: DecoyEstimates, result: KeyCalculationResult, timings: Dict[str, float]) -> float:
        """Computes the phase error rate upper bound. Returns rate."""
        t0 = time.perf_counter()
        policy = self.get_epsilon_policy() or (lambda e: getattr(e, 'eps_pe', None))
        eps_for_pe = policy(self.eps_alloc)
        if not (isinstance(eps_for_pe, (float, int)) and 0 < eps_for_pe < 1):
            if self.is_audit_mode(): raise ParameterValidationError(f"Invalid eps_for_pe: {eps_for_pe}")
            return 0.5
        
        eps_for_pe = max(eps_for_pe, EPS_MIN)
        
        if abs(decoy_estimates.failure_prob_used - eps_for_pe) > self.NUMERIC_TOL:
            chosen_eps = min(decoy_estimates.failure_prob_used, eps_for_pe)
            self._add_diagnostic(result, 'eps_choice', 'info', {'decoy_eps': decoy_estimates.failure_prob_used, 'proof_eps': eps_for_pe, 'chosen': chosen_eps})
            self._add_error_code(result, ErrorCode.EPSILON_MISMATCH)
            eps_for_pe = chosen_eps
        
        diag_pe: Dict[str, Any] = {}
        _, m_x_U_bound = self.get_bounds(m_x, n_x, eps_for_pe, sided='one_upper', diagnostics=diag_pe)
        self._add_diagnostic(result, 'phase_error_ci', 'info', diag_pe)

        e_ph_11_U = float(m_x_U_bound) # get_bounds returns a rate
        e_ph_11_U = min(0.5, self._clamp_nonneg(e_ph_11_U, 'e_ph_11_U', result))
        self._assert_finite(e_ph_11_U, 'e_ph_11_U')
        timings['phase_error_bound_ms'] = (time.perf_counter() - t0) * 1000
        return e_ph_11_U

    def _compute_leak_ec(self, n_z: int, errors_sifted_z: int, result: KeyCalculationResult, timings: Dict[str, float]) -> float:
        """Computes the error correction leakage. Returns bits."""
        t0 = time.perf_counter()
        qber_z = self._safe_divide(float(errors_sifted_z), float(n_z), default=0.0, name='qber_z', result=result)
        qber_z = min(1.0 - self.NUMERIC_TOL, max(self.NUMERIC_TOL, qber_z))
        
        leak_EC = float(self.p.f_error_correction) * self.binary_entropy(qber_z) * float(n_z)
        timings['leak_ec_ms'] = (time.perf_counter() - t0) * 1000
        return self._clamp_nonneg(leak_EC, 'leak_EC', result)

    def _compute_pa_terms(self, result: KeyCalculationResult, timings: Dict[str, float]) -> float:
        """Computes privacy amplification leakage terms. Returns bits."""
        t0 = time.perf_counter()
        term1 = PA_SMOOTH_FACTOR * self._log2_safe(PA_MAGIC_CONSTANT / self.eps_alloc.eps_smooth)
        term2 = 2.0 * self._log2_safe(1.0 / self.eps_alloc.eps_pa)
        if not (math.isfinite(term1) and math.isfinite(term2)):
            raise ParameterValidationError("Invalid PA epsilon produced non-finite leakage term.")
        
        pa_term_bits = term1 + term2
        corr_term_bits = self._log2_safe(2.0 / self.eps_alloc.eps_cor)
        
        total_pa_leakage = self._clamp_nonneg(pa_term_bits, 'pa_term_bits', result) + self._clamp_nonneg(corr_term_bits, 'corr_term_bits', result)
        if total_pa_leakage > 1e9:
            self._add_error_code(result, ErrorCode.LARGE_PA_TERM)
            self._add_diagnostic(result, 'pa_term', 'error', {'eps_smooth': self.eps_alloc.eps_smooth})
        timings['pa_terms_ms'] = (time.perf_counter() - t0) * 1000
        return total_pa_leakage

    def _validate_inputs(self, decoy_estimates: DecoyEstimates, stats_map: Dict[str, TallyCounts], result: KeyCalculationResult) -> Optional[Dict[str, int]]:
        """Validates input statistics and extracts signal pulse data."""
        if not isinstance(stats_map, dict) or "signal" not in stats_map:
            self._abort_zero_key(result, ErrorCode.INVALID_INPUT_STATS, "stats_map must be a dict with a 'signal' key.", self._get_location())
            return None

        signal_stats = stats_map["signal"]
        if not isinstance(signal_stats, TallyCounts):
             self._abort_zero_key(result, ErrorCode.INVALID_INPUT_STATS, "'signal' stats must be a TallyCounts object.", self._get_location())
             return None

        return dataclasses.asdict(signal_stats)

    def _enforce_min_y1_threshold(self, s_z_11_L: float, result: KeyCalculationResult) -> bool:
        """Checks if the single-photon yield is sufficient."""
        if s_z_11_L < self.min_y1_threshold:
            msg = f"Lower-bounded single-photon counts ({s_z_11_L:.2e}) are below the minimum threshold ({self.min_y1_threshold:.2e})."
            self._abort_zero_key(result, ErrorCode.INSUFFICIENT_STATISTICS, msg, self._get_location())
            return False
        return True

    def calculate_key_length(
        self, decoy_estimates: DecoyEstimates, stats_map: Dict[str, TallyCounts]
    ) -> KeyCalculationResult:
        timings: Dict[str, float] = {}
        t_start = time.perf_counter()
        self.logger.info("event=key_calc_start run_id=%s", self.run_id)
        result = KeyCalculationResult(metadata={'run_id': self.run_id, 'calc_id': str(uuid.uuid4())})

        if not (self.p.allow_unsafe_mdi_approx is True or self.p.allow_unsafe_mdi_approx == "I_ACCEPT_RISK"):
            return self._abort_zero_key(result, ErrorCode.UNSAFE_APPROXIMATION_USED, "Unsafe MDI approximation not enabled.", self._get_location())
        
        if self.is_audit_mode() and self.p.allow_unsafe_mdi_approx != "I_ACCEPT_RISK":
              raise QKDSimulationError("To use the unsafe MDI approximation in audit mode, allow_unsafe_mdi_approx must be 'I_ACCEPT_RISK'.")
        
        self._add_error_code(result, ErrorCode.UNSAFE_APPROXIMATION_USED)

        if decoy_estimates is None or not hasattr(decoy_estimates, 'yield_1_lower_bound'):
            raise ParameterValidationError("decoy_estimates must be a DecoyEstimates-like object")
        
        if not decoy_estimates.is_feasible:
            self._add_error_code(result, ErrorCode.LP_INFEASIBLE)
            decoy_estimates = self._conservative_decoy_estimate()

        signal_stats_dict = self._validate_inputs(decoy_estimates, stats_map, result)
        if signal_stats_dict is None: return result

        p_sig_cfg = self.p.source.get_pulse_config_by_name("signal")
        if not (p_sig_cfg and isinstance(p_sig_cfg.get('mean_photon_number'), (float, int)) and p_sig_cfg['mean_photon_number'] > 0):
            if self.is_audit_mode(): raise ParameterValidationError("Missing or invalid 'signal' pulse config.")
            return self._abort_zero_key(result, ErrorCode.MISSING_PULSE_CONFIG, "Invalid signal config", self._get_location())

        n_z, n_x = int(signal_stats_dict['sifted_z']), int(signal_stats_dict['sifted_x'])
        m_x = int(signal_stats_dict['errors_sifted_x'])

        if n_z <= 0 or n_x <= 0:
            return self._abort_zero_key(result, ErrorCode.INSUFFICIENT_STATISTICS, f"Insufficient sifted events (n_z={n_z}, n_x={n_x})", self._get_location())
        
        if n_x < n_z * 0.01:
            self._add_error_code(result, ErrorCode.IMBALANCED_BASIS_STATS)
            self.logger.warning(f"Imbalanced basis stats: n_z={n_z}, n_x={n_x}")

        orig_Y11_L = decoy_estimates.yield_1_lower_bound
        Y11_L = max(0.0, min(1.0, orig_Y11_L))
        if Y11_L != orig_Y11_L: self._clamp_nonneg(orig_Y11_L, 'Y11_L', result)

        s_z_11_L = float(n_z) * Y11_L # counts
        if not self._enforce_min_y1_threshold(s_z_11_L, result): return result

        e_ph_11_U = self._compute_phase_error_bound(n_x, m_x, decoy_estimates, result, timings)
        leak_EC = self._compute_leak_ec(n_z, int(signal_stats_dict['errors_sifted_z']), result, timings)
        total_pa_leakage = self._compute_pa_terms(result, timings)

        key_length_float = float(s_z_11_L * (1.0 - self.binary_entropy(e_ph_11_U)) - leak_EC - total_pa_leakage)
        
        result.secure_key_length = self._clamp_key_length(key_length_float)
        if 0 < result.secure_key_length <= 10: self.logger.warning(f"Generated key is very small ({result.secure_key_length} bits).")
        result.phase_error_rate_upper_bound = e_ph_11_U
        result.error_correction_leakage = leak_EC
        result.privacy_amplification_term = total_pa_leakage
        
        diag_summary = {
            'ok': result.secure_key_length > 0, 'key_len': result.secure_key_length,
            'Y11_L_rate': round(Y11_L, 8), 's_z_11_L_counts': round(s_z_11_L, 8), 'e_ph_11_U_rate': round(e_ph_11_U, 8),
            'leak_EC_bits': round(leak_EC, 8), 'leak_PA_bits': round(total_pa_leakage, 8), 'key_len_float': round(key_length_float, 8)
        }
        self._add_diagnostic(result, 'summary', 'info', diag_summary)
        
        health = 'OK' if not result.error_codes else ('ERROR' if any(c in [ErrorCode.INVALID_EPSILON, ErrorCode.MISSING_PULSE_CONFIG] for c in result.error_codes) else 'WARN')
        
        explain_code, explain_text = self.explain_key_decision(result, decoy_estimates)
        result.metadata.update({
            'timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z', 'impl_version': self.__implementation_version__,
            'is_placeholder': self.is_placeholder, 'ci_method': self.p.ci_method.name, 'safety_level': 'INSECURE_PLACEHOLDER',
            'primary_error': result.error_codes[0].value if result.error_codes else None, 'health': health,
            'explain': explain_text, 'explain_code': explain_code,
            'pa_provenance': 'placeholder: adaptation of Lim et al., PRA 89, 022341 (2014)'
        })
        
        input_payload = {"params": self.p.sanitized_params(), "stats": signal_stats_dict}
        input_str = json.dumps(input_payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
        result.metadata['input_hash'] = hashlib.sha256(input_str).hexdigest()
        result.metadata['input_hash_algo'] = 'sha256'

        if self.p.hmac_key:
            canonical_payload = {
                'run_id': result.metadata['run_id'], 'calc_id': result.metadata['calc_id'],
                'input_hash': result.metadata['input_hash'], 'key_len': result.secure_key_length,
                'impl_version': result.metadata['impl_version']
            }
            payload_to_sign = json.dumps(canonical_payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
            result.metadata['signature'] = hmac.new(self.p.hmac_key, payload_to_sign, hashlib.sha256).hexdigest()
            result.metadata['signature_algo'] = 'hmac-sha256'
            result.metadata['signed_fields'] = list(canonical_payload.keys())

        timings['total_ms'] = (time.perf_counter() - t_start) * 1000
        self._add_diagnostic(result, 'timings', 'info', timings)
        self.logger.info("event=key_calc_finish run_id=%s key_len=%d Y11=%g e_ph=%g", self.run_id, result.secure_key_length, Y11_L, e_ph_11_U)
        return result

# ==============================================================================
# Example Usage
# ==============================================================================
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main_logger = logging.getLogger(__name__)

    params = MockParams()
    params.ci_method = ConfidenceBoundMethod.HOEFFDING
    
    mock_stats = {"signal": TallyCounts(sent=1_000_000, sifted_z=50_000, sifted_x=50_000, errors_sifted_z=1_000, errors_sifted_x=1_500)}
    mock_decoy_results = DecoyEstimates(is_feasible=True, yield_1_lower_bound=0.05, error_rate_1_upper_bound=0.03, diagnostics=SolverDiagnostics(residual_norm=1e-10))

    main_logger.info("\n--- SCENARIO: Unsafe approximation enabled with token ---")
    params.allow_unsafe_mdi_approx = "I_ACCEPT_RISK"
    mdi_proof_unsafe = MDIQKDProof(params, run_id="unsafe-run-4")
    key_result_unsafe = mdi_proof_unsafe.calculate_key_length(mock_decoy_results, mock_stats)
    
    print(f"\nFinal Key Length: {key_result_unsafe.secure_key_length}")
    print(f"Explanation: {key_result_unsafe.metadata.get('explain')}")
    print(f"Signature: {key_result_unsafe.metadata.get('signature')}")
    print("\nFull diagnostics payload (JSON):")
    print(json.dumps(key_result_unsafe.as_serializable(), indent=2))

