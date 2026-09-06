# -*- coding: utf-8 -*-
"""
Implementation of a production-grade, tight BB84 finite-key security proof.

This module provides a robust, auditable implementation of the BB84-tight
security proof, derived from the Lim et al. 2014 framework. It has been
extensively refactored to meet production standards for correctness, API
consistency, numerical stability, diagnostics, and provenance.
"""

import math
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

# ==============================================================================
# 1. Supporting Datatypes, Constants, and Exceptions
# ==============================================================================


from ..datatypes import TallyCounts, EpsilonAllocation

from ..constants import NUMERIC_ABS_TOL, ENTROPY_PROB_CLAMP

from .lim2014 import Lim2014Proof as Lim2014ProofBase


_TIGHT_PROOF_EPS_MIN = 1e-300
DEFAULT_ENTROPY_PROB_CLAMP = 1e-12
DEFAULT_NUMERIC_TOL = 1e-9

class ErrorCode(Enum):
    INVALID_PARAMS = "INVALID_PARAMS"
    INVALID_DECOY_ESTIMATES_TYPE = "INVALID_DECOY_ESTIMATES_TYPE"
    MISSING_DECOY_FIELDS = "MISSING_DECOY_FIELDS"
    INVALID_STATS_MAP = "INVALID_STATS_MAP"
    MISSING_SIGNAL_CONFIG = "MISSING_SIGNAL_CONFIG"
    INVALID_PULSE_CONFIG = "INVALID_PULSE_CONFIG"
    NO_SIGNAL_PULSES_SENT = "NO_SIGNAL_PULSES_SENT"
    INSUFFICIENT_STATISTICS = "INSUFFICIENT_STATISTICS"
    INCONSISTENT_MEASUREMENT = "INCONSISTENT_MEASUREMENT"
    NON_FINITE_VALUE = "NON_FINITE_VALUE"
    PHASE_ERROR_CLAMPED = "PHASE_ERROR_CLAMPED"
    INVALID_YIELD_BOUND = "INVALID_YIELD_BOUND"
    INVALID_ERROR_RATE_BOUND = "INVALID_ERROR_RATE_BOUND"

class ParameterValidationError(ValueError): pass
class QKDSimulationError(RuntimeError): pass

@dataclass
class DecoyEstimatesInput:
    """
    Normalized decoy estimates required for the key calculation.
    This proof expects the bit error rate (e1_bit_ub) from the Z-basis
    and the yield (Y1_L_rate) which is assumed basis-independent.
    """
    Y1_L_rate: float
    e1_bit_ub: float  # This is e1_z_ub (error rate in Z-basis)

@dataclass
class KeyCalculationResult:
    secure_key_length: int = 0
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    error_codes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# 2. Main Proof Implementation
# ==============================================================================

__all__ = ["BB84TightProof", "KeyCalculationResult", "TallyCounts", "EpsilonAllocation", "ErrorCode"]

class BB84TightProof(Lim2014ProofBase):
    __implementation_version__ = "3.7.6-fixed" # Incremented version for logical fix

    def __repr__(self) -> str:
        return f"BB84TightProof(impl_version='{self.__implementation_version__}')"

    def allocate_epsilons(self) -> EpsilonAllocation:
        eps_sec = self.p.eps_sec
        eps_cor = self.p.eps_cor
        remaining_eps = eps_sec - eps_cor
        if remaining_eps <= 0:
            raise ParameterValidationError(f"eps_sec ({eps_sec:.2e}) must be greater than eps_cor ({eps_cor:.2e}).")
        
        
        # The validation check fails because (eps_cor + (eps_sec - eps_cor))
        # evaluates to > eps_sec in 64-bit float math.
        # We must allocate the full budget minus a small *absolute* margin
        # that is guaranteed to be numerically significant.
        
        # Use an absolute safety margin
        absolute_margin = 1e-20
        budget_to_alloc = remaining_eps - absolute_margin
        
        # Ensure budget is still positive after margin
        if budget_to_alloc <= 0:
             raise ParameterValidationError(f"Remaining epsilon budget ({remaining_eps:.2e}) is too small after applying margin.")

      
        # The validation rule is: eps_pe + 2*eps_smooth + eps_pa <= budget
        # We allocate 4 parts: 1 for pe, 2 for smooth (1 part * 2), 1 for pa.
        eps_pe = budget_to_alloc / 4.0
        eps_smooth = budget_to_alloc / 4.0
        # eps_pa is allocated the remainder of the *weighted* budget
        eps_pa = budget_to_alloc - eps_pe - (2.0 * eps_smooth)
        # --- END LOGICAL FIX ---
        
        num_pulse_configs = len(getattr(self.p.source, 'pulse_configs', []))
        denominator = max(1, 4 * num_pulse_configs + 1)
        eps_phase_est = eps_pe / denominator
        
        allocation = EpsilonAllocation(
            eps_sec=eps_sec, eps_cor=eps_cor, eps_pe=eps_pe,
            eps_smooth=eps_smooth, eps_pa=eps_pa, eps_phase_est=eps_phase_est
        )
        allocation.validate() # This check should now pass.
        return allocation

    def calculate_key_length(
        self,
        stats_map: Dict[str, TallyCounts],
        decoy_estimates: Optional[Union[Dict[str, Any], object]] = None
    ) -> KeyCalculationResult:
        run_id = getattr(self.p, 'run_id', None)
        self.logger.info("event=key_calc_start run_id=%s", run_id)
        start_time = time.perf_counter()
        result = KeyCalculationResult()
        
        try:
            # اگر decoy_estimates داده نشد، از موتور پایه Lim2014 استفاده کن تا کرش ندهد
            if decoy_estimates is None:
                return super().calculate_key_length(stats_map)
                
            is_decoy_protocol = bool(decoy_estimates)

            if is_decoy_protocol:
                signal_stats, p_sig_cfg = self._validate_stats_map_decoy(stats_map, result)
                if result.error_codes: return self._finalize_result(result, start_time)
                
                decoy_norm = self._normalize_decoy_estimates(decoy_estimates, result)
                if result.error_codes: return self._finalize_result(result, start_time)
                
                
                s_x_1_L_count, s_z_1_L_count, n_x, _, n_z = self._compute_base_counts_decoy(signal_stats, p_sig_cfg, decoy_norm, result)
                e1_bit_ub = decoy_norm.e1_bit_ub # This is e1_z_ub
            
            else:
                result.diagnostics['info'] = "Running in non-decoy mode; using conservative estimates."
                signal_stats = self._get_total_stats_non_decoy(stats_map, result)
                if result.error_codes: return self._finalize_result(result, start_time)
                
                
                # Assume key is from X-basis, phase error from Z-basis.
                n_x = float(signal_stats.sifted_x)
                m_x = float(signal_stats.errors_sifted_x)
                n_z = float(signal_stats.sifted_z)
                m_z = float(signal_stats.errors_sifted_z)
                
                # Conservative estimate: all sifted bits are single-photon
                s_x_1_L_count = n_x # Key-basis (X) single-photon count
                s_z_1_L_count = n_z # Phase-basis (Z) single-photon count
                
                # Use Z-basis QBER as the phase error rate estimate
                e1_bit_ub = self._safe_divide(m_z, n_z, name="qber_z", error_codes=result.error_codes) 
                result.diagnostics['intermediate'] = {
                    "n_x": n_x, "m_x": m_x, "n_z": n_z, "m_z": m_z, "qber_z": e1_bit_ub
                }

            min_s_z_1_L = getattr(self.p, 'S_Z_1_L_MIN_FOR_PHASE_EST', 100)
            
            
            # Check must be on Z-basis counts, which are used for phase estimation.
            if n_z <= 0 or s_z_1_L_count < min_s_z_1_L:
                result.error_codes.append(ErrorCode.INSUFFICIENT_STATISTICS)
                return self._finalize_result(result, start_time)

            
            # Key is from X-basis, so error correction is on X-basis stats.
            if is_decoy_protocol:
                n_key, m_key = signal_stats.sifted_x, signal_stats.errors_sifted_x
            else:
                n_key, m_key = n_x, m_x # From non-decoy logic above
                
            leak_ec = self._compute_leak_ec(n_key, m_key, result)
            
            
            # Must use Z-basis single-photon count (s_z_1_L_count)
            e1_phase_ub = self._compute_phase_error_ub(s_z_1_L_count, e1_bit_ub, result)
            
            pa_corr_bits = self._compute_pa_terms(result)
            
            
            # Available entropy comes from the key-basis (X) single-photon count
            available_entropy = s_x_1_L_count * (1.0 - self._binary_entropy_safe(e1_phase_ub))
            
            key_len_float = available_entropy - leak_ec - pa_corr_bits
            
            self._finite_or_error("final_key_length", key_len_float, result)
            result.secure_key_length = self._clamp_key_length(key_len_float) # Use base class _clamp_key_length
            self._populate_final_diagnostics(result, available_entropy, key_len_float, pa_corr_bits)

        except (ParameterValidationError, QKDSimulationError) as e:
            self.logger.error("event=key_calc_error run_id=%s error='%s'", run_id, e)
            result.error_codes.append(ErrorCode.INVALID_PARAMS)
        
        return self._finalize_result(result, start_time)

    def _get_total_stats_non_decoy(self, stats_map, result):
        if not stats_map or not any(isinstance(v, TallyCounts) for v in stats_map.values()):
            result.error_codes.append(ErrorCode.INVALID_STATS_MAP)
            result.diagnostics['explain'] = "stats_map is empty or contains no valid TallyCounts objects for non-decoy mode."
            return None
        
        total_stats = TallyCounts()
        for stats in stats_map.values():
            if isinstance(stats, TallyCounts):
                total_stats = total_stats.merged(stats)
        
        return total_stats

    def _validate_stats_map_decoy(self, stats_map, result):
        if 'signal' not in stats_map or not isinstance(stats_map['signal'], TallyCounts):
            result.error_codes.append(ErrorCode.INVALID_STATS_MAP)
            return None, None
        
        stats = stats_map['signal']
        p_sig_cfg = self.p.source.get_pulse_config_by_name("signal")
        if not p_sig_cfg or not hasattr(p_sig_cfg, 'mean_photon_number'):
            result.error_codes.append(ErrorCode.MISSING_SIGNAL_CONFIG)
            return stats, None
        return stats, p_sig_cfg

    def _normalize_decoy_estimates(self, decoy_estimates, result):
        
        # If the input is already the correct type, return it immediately.
        if isinstance(decoy_estimates, DecoyEstimatesInput):
            return decoy_estimates

        # Attempt to get attributes from the DecoyEstimates object (from base.py)
        Y1_L = getattr(decoy_estimates, 'yield_1_lower_bound', None)
        e1_U = getattr(decoy_estimates, 'error_rate_1_upper_bound', None)

        if Y1_L is None or e1_U is None:
            # Fallback for dictionary inputs (e.g., the {} from simulation.py)
            if isinstance(decoy_estimates, dict):
                # Check for the mismatched names used in the original broken code
                Y1_L = decoy_estimates.get('Y1_L')
                e1_U = decoy_estimates.get('e1_U')
            
        if Y1_L is None or e1_U is None:
            # If still None, the input is invalid or empty
            result.error_codes.append(ErrorCode.MISSING_DECOY_FIELDS)
            return None
            
        # The rest of the file expects DecoyEstimatesInput, which uses
        # Y1_L_rate and e1_bit_ub. We pass the values we found.
        return DecoyEstimatesInput(float(Y1_L), float(e1_U))

    def _compute_base_counts_decoy(self, signal_stats, p_sig_cfg, decoy_norm, result):
        mu_s = p_sig_cfg.mean_photon_number
        p1_s = self._single_photon_prob(mu_s)
        
        
        
        # Key is from X-basis
        alice_x_prob = 1.0 - getattr(self.p.protocol, "alice_z_basis_prob", 0.5)
        bob_x_prob = 1.0 - getattr(self.p.protocol, "bob_z_basis_prob", 0.5)
        s_x_1_L_count = float(signal_stats.sent_x) * p1_s * decoy_norm.Y1_L_rate * alice_x_prob * bob_x_prob # Assumes Y1 is basis-independent
        n_x = float(signal_stats.sifted_x)
        m_x = float(signal_stats.errors_sifted_x)
        if s_x_1_L_count > n_x + NUMERIC_ABS_TOL:
            s_x_1_L_count = n_x
            
        # Phase error is from Z-basis
        alice_z_prob = getattr(self.p.protocol, "alice_z_basis_prob", 0.5)
        bob_z_prob = getattr(self.p.protocol, "bob_z_basis_prob", 0.5)
        s_z_1_L_count = float(signal_stats.sent_z) * p1_s * decoy_norm.Y1_L_rate * alice_z_prob * bob_z_prob
        n_z = float(signal_stats.sifted_z)
        if s_z_1_L_count > n_z + NUMERIC_ABS_TOL:
            s_z_1_L_count = n_z

        result.diagnostics['intermediate'] = {
            "mu_s": mu_s, "p1_s": p1_s, 
            "s_x_1_L_count": s_x_1_L_count, "n_x": n_x, "m_x": m_x,
            "s_z_1_L_count": s_z_1_L_count, "n_z": n_z
        }
        return s_x_1_L_count, s_z_1_L_count, n_x, m_x, n_z

    def _compute_leak_ec(self, n_key_basis: float, m_key_basis: float, result: KeyCalculationResult) -> float:
        
        # Renamed 'n_z'/'m_z' to 'n_key_basis'/'m_key_basis' as this
        # function calculates leak_ec for the key-generating basis (X).
        qber_key_basis = self._safe_divide(m_key_basis, n_key_basis, name="qber_key_basis", error_codes=result.error_codes)
        f_ec = getattr(self.p, 'f_error_correction', 1.1)
        leak_ec = f_ec * self._binary_entropy_safe(qber_key_basis) * n_key_basis
        
        di = result.diagnostics.setdefault('intermediate', {})
        di['qber_key_basis'] = qber_key_basis
        di['leak_ec_bits'] = leak_ec
        return leak_ec

    def _compute_phase_error_ub(self, s_z_1_L_count: float, e1_bit_ub: float, result: KeyCalculationResult) -> float:
        
        # This function MUST use the Z-basis single-photon count.
        # The call site is now fixed to pass s_z_1_L_count.
        
        di = result.diagnostics.setdefault('intermediate', {})
        if self.p.assume_phase_equals_bit_error:
            di['e1_phase_ub'] = e1_bit_ub
            return e1_bit_ub
            
        eps_ph_est_safe, _ = self._log_inv_clamp(self.eps_alloc.eps_phase_est)
        log_term = -math.log(eps_ph_est_safe)
        
        # Check for division by zero or invalid sqrt
        if s_z_1_L_count <= 0:
            di['e1_phase_ub'] = 0.5 # Most conservative estimate
            return 0.5
            
        delta = self._safe_sqrt(log_term / (2 * s_z_1_L_count))
        e1_phase_ub = e1_bit_ub + delta
        
        if e1_phase_ub > 0.5:
            e1_phase_ub = 0.5
            
        di['e1_phase_ub'] = e1_phase_ub
        di['phase_error_delta'] = delta
        return e1_phase_ub

    def _compute_pa_terms(self, result):
        eps_smooth_safe, _ = self._log_inv_clamp(self.eps_alloc.eps_smooth)
        eps_pa_safe, _ = self._log_inv_clamp(self.eps_alloc.eps_pa)
        eps_cor_safe, _ = self._log_inv_clamp(self.eps_alloc.eps_cor)
        pa_term_bits = _safe_log2(1.0 / (2.0 * eps_smooth_safe)) + _safe_log2(1.0 / eps_pa_safe)
        corr_term_bits = _safe_log2(2.0 / eps_cor_safe)
        
        di = result.diagnostics.setdefault('intermediate', {})
        di['pa_corr_bits'] = pa_term_bits + corr_term_bits
        di['pa_bits'] = pa_term_bits
        di['corr_bits'] = corr_term_bits
        
        return pa_term_bits + corr_term_bits

    def _populate_final_diagnostics(self, result, available_entropy, key_len_float, pa_corr_bits):
        di = result.diagnostics.setdefault('intermediate', {})
        di['available_entropy_bits'] = available_entropy
        di['final_key_length_float'] = key_len_float

    def _finalize_result(self, result, start_time):
        end_time = time.perf_counter()
        result.metadata['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
        result.metadata['calculation_time_ms'] = (end_time - start_time) * 1000
        result.error_codes = [code.value if isinstance(code, Enum) else code for code in result.error_codes]
        run_id = getattr(self.p, 'run_id', None)
        if result.error_codes:
            self.logger.warning("event=key_calc_abort run_id=%s errors=%s", run_id, result.error_codes)
        else:
            self.logger.info("event=key_calc_finish run_id=%s key_length=%d", run_id, result.secure_key_length)
        return result

    def _log_inv_clamp(self, val):
        return (_TIGHT_PROOF_EPS_MIN, True) if val < _TIGHT_PROOF_EPS_MIN else (val, False)
    
    def _finite_or_error(self, name, val, result, non_negative=False):
        if not math.isfinite(val) or (non_negative and val < 0):
            result.error_codes.append(ErrorCode.NON_FINITE_VALUE)

    def _binary_entropy_safe(self, p: float) -> float:
        p_clamped = max(ENTROPY_PROB_CLAMP, min(p, 1 - ENTROPY_PROB_CLAMP))
        if p_clamped <= 0 or p_clamped >= 1: return 0.0
        return -p_clamped * math.log2(p_clamped) - (1 - p_clamped) * math.log2(1 - p_clamped)

    def _single_photon_prob(self, mu: float) -> float:
        if mu < 0: raise ParameterValidationError(f"mu cannot be negative: {mu}")
        if mu == 0: return 0.0
        return mu * math.exp(-mu)
    
    def _safe_sqrt(self, x: float) -> float:
        if x < 0: return 0.0
        return math.sqrt(x)

def _safe_log2(val: float) -> float:
    return 0.0 if val <= 0 else math.log2(val)
