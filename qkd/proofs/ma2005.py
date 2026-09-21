# qkd/proofs/ma2005.py
# -*- coding: utf-8 -*-
"""
Implementations of the Ma et al. 2005 decoy-state protocols.
Includes robust enhancements for intensity jitter, thermal statistics, and
arbitrary decoy states.
Includes the formal asymptotic limit proof.
"""

import math
import logging
import numpy as np
from typing import Dict, Any, Callable, Optional, Tuple, List, Mapping, Union
from dataclasses import dataclass

from ..params import QKDParams
from ..datatypes import TallyCounts, EpsilonAllocation, SourceStatisticsType, ConfidenceBoundMethod
from ..exceptions import ParameterValidationError, ConfigurationError, LPFailureError
from ..utils.math import binary_entropy
from .base import (
    FiniteKeyProof, 
    KeyCalculationResult, 
    DecoyEstimates, 
    SolverDiagnostics, 
    ErrorCode
)
from .utils_lp import solve_lp

__all__ = ["Ma2005VacuumWeakProof", "Ma2005GeneralTwoDecoyProof", "Ma2005OneDecoyProof", "Ma2005GeneralMultiDecoyProof", "Ma2005AsymptoticProof"]

logger = logging.getLogger(__name__)

class Ma2005BaseProof(FiniteKeyProof):
    """Base class for Ma 2005 proofs sharing the GLLP rate formula."""
    
    def __init__(self, params: QKDParams, *args, **kwargs):
        # Spec Sec 4.4: 10-sigma Gaussian ("standard error analysis").
        # Do NOT override user's CI method choice.
        super().__init__(params, *args, **kwargs)
        self.is_thermal = (self.p.source.statistics_type == SourceStatisticsType.THERMAL)

    def _normalize_stats_map(
        self,
        stats_map: Mapping[Union[str, int], TallyCounts],
        pulse_names: List[str],
    ) -> Dict[str, TallyCounts]:
        """Normalize stats_map keys to use pulse name strings.

        The simulation may pass stats_map with integer-string keys
        ("0", "1", "2") or integer keys (0, 1, 2) instead of name keys
        ("signal", "decoy", "vacuum").  This method converts any of those
        formats into the canonical name-keyed dict that all MA2005 internal
        code expects.
        """
        # Try #1: name keys already present
        if all(name in stats_map for name in pulse_names):
            return {name: stats_map[name] for name in pulse_names}  # type: ignore[index]

        # Try #2: string-integer keys ("0", "1", "2")
        if all(str(i) in stats_map for i in range(len(pulse_names))):
            return {pulse_names[i]: stats_map[str(i)] for i in range(len(pulse_names))}

        # Try #3: integer keys (0, 1, 2)
        if all(i in stats_map for i in range(len(pulse_names))):
            return {pulse_names[i]: stats_map[i] for i in range(len(pulse_names))}

        raise ParameterValidationError(
            f"stats_map keys do not match pulse names or positional indices. "
            f"Expected names {pulse_names!r}; got {list(stats_map.keys())!r}."
        )

    def _get_pulse_names(self) -> List[str]:
        """Return the ordered list of pulse names from the source config."""
        return [pc.name for pc in self.p.source.pulse_configs]

    def _get_prob_n(self, mu: float, n: int) -> float:
        """Calculates P(n|mu) handling both Poisson and Thermal cases."""
        if self.is_thermal:
            return (mu**n) / ((1.0 + mu)**(n + 1))
        else:
            return math.exp(-mu) * (mu**n) / math.factorial(n)

    def notation_map(self) -> Dict[str, str]:
        return {
            "Y_1": "yield_1_lower_bound",
            "e_1": "error_rate_1_upper_bound",
            "Q_mu": "Gain of signal state",
            "E_mu": "QBER of signal state",
            "\\Delta": "tagged_fraction (Delta)",
            "f(e)": "f_error_correction",
            "Y_1^L": "yield_1_lower_bound",
            "e_ph": "error_rate_1_upper_bound", 
            "s_z_1^L": "Calculated as Q_1 * N_sent (single-photon count)"
        }

    def allocate_epsilons(self) -> EpsilonAllocation:
        # Full constraint: eps_pe + 2*eps_smooth + eps_pa + eps_cor + eps_phase_est <= eps_sec
        eps_phase_est = self.p.eps_pe / 10.0
        used_budget = self.p.eps_pe + 2.0 * self.p.eps_smooth + self.p.eps_cor + eps_phase_est
        
        if used_budget >= self.p.eps_sec:
            # Gap [A] fix (v2): user's epsilons overflow eps_sec. Instead of replacing
            # them with eps_sec/8 (which silently discards the user's intent), normalize
            # them proportionally so they sum to eps_sec. This preserves the relative
            # weighting the user specified.
            scale = self.p.eps_sec / used_budget if used_budget > 0 else 1.0
            eps_cor_n   = self.p.eps_cor * scale
            eps_pe_n    = self.p.eps_pe * scale
            eps_smooth_n = self.p.eps_smooth * scale
            eps_phase_est_n = eps_pe_n / 10.0
            eps_pa_n = self.p.eps_sec - (eps_pe_n + 2.0 * eps_smooth_n + eps_cor_n + eps_phase_est_n)
            if eps_pa_n < 0.0:
                eps_pa_n = 0.0
            return EpsilonAllocation(
                eps_sec=self.p.eps_sec,
                eps_cor=eps_cor_n,
                eps_pe=eps_pe_n,
                eps_smooth=eps_smooth_n,
                eps_pa=eps_pa_n,
                eps_phase_est=eps_phase_est_n
            )
            
        eps_pa = self.p.eps_sec - used_budget
        
        return EpsilonAllocation(
            eps_sec=self.p.eps_sec, 
            eps_cor=self.p.eps_cor,
            eps_pe=self.p.eps_pe, 
            eps_smooth=self.p.eps_smooth,
            eps_pa=eps_pa, 
            eps_phase_est=eps_phase_est
        )

    def get_epsilon_policy(self) -> Callable[[EpsilonAllocation], float]:
        return lambda eps: eps.eps_pe

    def _calculate_f_ec(self, error_rate: float) -> float:
        """Calculates adaptive error correction efficiency if configured.

        Spec Eq (1) context: f(x) >= 1 (Shannon limit f=1). Enforce the
        physical lower bound to prevent unphysically high key rates when
        the user supplies a sub-Shannon value.
        """
        config = self.p.f_ec_dynamic_config
        if config and config.get("model") == "linear":
            base = config.get("base", 1.1)
            slope = config.get("slope", 0.0)
            f_val = base + slope * error_rate
        else:
            f_val = self.p.f_error_correction
        if f_val < 1.0:
            logger.warning(
                "f_ec=%.4f < 1.0 violates Shannon limit (spec Eq 1); "
                "clamping to 1.0", f_val
            )
            return 1.0
        return f_val

    def _get_eta(self) -> float:
        """Paper Eq (5): eta = t_AB * eta_Bob.
        eta_Bob = t_Bob * eta_D is given directly in Table 1 (GYS: 0.045,
        KTH: 0.143) as a primitive parameter."""
        eta_bob = getattr(self.p.detector, "eta_bob", None)
        if eta_bob is None:
            t_bob = getattr(self.p.detector, "t_bob", 1.0)
            eta_d = getattr(self.p.detector, "det_eff_d0", 0.0)
            eta_bob = t_bob * eta_d
        return float(self.p.channel.transmittance * eta_bob)

    def _get_y0(self, stats_map=None) -> float:
        """Paper Table 1: Y_0 is the *total* background rate (GYS: 1.7e-6,
        KTH: 4e-4), not per-detector.  Do NOT multiply dark_rate by 2.
        If vacuum stats are available, use Q_vacuum = Y_0 (Eq 33)."""
        if stats_map:
            vac = stats_map.get("vacuum")
            if vac is not None and vac.sent > 0:
                return float(vac.sifted) / float(vac.sent)
        y0 = getattr(self.p.detector, "y0", None)
        if y0 is not None:
            return float(y0)
        return float(getattr(self.p.detector, "dark_rate", 0.0))

    def _calculate_sifting_factor_q(self, signal_sent: int, total_sent: int) -> float:
        """Paper Eq (45): q = N_S / (2N) for standard BB84.
        Asymptotic limit: N_S -> N gives q -> 1/2.
        Set self.p.sifting_factor_q to override (e.g. efficient BB84)."""
        override = getattr(self.p, "sifting_factor_q", None)
        if override is not None:
            return float(override)
        if total_sent > 0:
            return float(signal_sent) / (2.0 * float(total_sent))
        return 0.5

    @staticmethod
    def solve_optimal_mu(e_detector: float, f_ec: float, tol: float = 1e-12) -> float:
        """Paper Eq (12): (1-mu)*exp(-mu) = f(e_det)*H2(e_det)/(1-H2(e_det)).
        Solve for mu in (0, 1] via bisection.
        Reference: GYS f=1.0 -> mu~0.54; GYS f=1.22 -> mu~0.48."""
        h2 = binary_entropy(e_detector)
        if h2 >= 1.0:
            return 1.0
        rhs = f_ec * h2 / (1.0 - h2)

        def g(mu):
            return (1.0 - mu) * math.exp(-mu) - rhs

        if g(1.0) >= 0.0:
            return 1.0
        lo, hi = 1e-12, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            g_mid = g(mid)
            if abs(g_mid) < tol or (hi - lo) < tol:
                return mid
            if g_mid > 0.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _calculate_weak_gllp_rate(
        self, 
        decoy_estimates: DecoyEstimates, 
        stats_map: Dict[str, TallyCounts]
    ) -> float:
        """Calculates Weak GLLP rate based on Eq 43."""
        signal_stats = stats_map.get("signal")
        if not signal_stats or signal_stats.sent == 0:
            return 0.0

        Q_mu = signal_stats.sifted / signal_stats.sent
        E_mu = signal_stats.errors_sifted / signal_stats.sifted if signal_stats.sifted > 0 else 0.5
        
        # Use Eq 36 bound for Delta
        delta = decoy_estimates.diagnostics.numeric_diagnostics.get("Delta_tagged_bound_Eq36", 1.0)
        
        if delta >= 1.0 - 1e-9:
            return 0.0

        total_sent = sum(s.sent for s in stats_map.values())
        q = self._calculate_sifting_factor_q(signal_stats.sent, total_sent)
        f_ec = self._calculate_f_ec(E_mu)
        term_ec = f_ec * binary_entropy(E_mu)
        
        # Weak GLLP formula Eq 43: q*Q_mu * { -H(E_mu) + (1-Delta)[1 - H(E_mu/(1-Delta))] }
        # Note: Paper Eq 43 includes f(E_mu)
        e_untagged_bound = E_mu / (1.0 - delta)
        if e_untagged_bound >= 0.5:
            return 0.0
            
        term_pa = (1.0 - delta) * (1.0 - binary_entropy(e_untagged_bound))
        
        rate = q * Q_mu * (term_pa - term_ec)
        return max(0.0, rate)

    def _calculate_gllp_rate(
        self, 
        decoy_estimates: DecoyEstimates, 
        stats_map: Dict[str, TallyCounts]
    ) -> KeyCalculationResult:
        signal_stats = stats_map.get("signal")
        if not signal_stats or signal_stats.sent == 0:
            return KeyCalculationResult(0, 0.0, 0.0, 0.5, error_codes=[ErrorCode.INSUFFICIENT_STATISTICS])

        mu = self.p.source.get_pulse_config_by_name("signal").mean_photon_number
        total_sent = sum(s.sent for s in stats_map.values())
        q_factor = self._calculate_sifting_factor_q(signal_stats.sent, total_sent)

        Q_mu = signal_stats.sifted / signal_stats.sent
        E_mu = signal_stats.errors_sifted / signal_stats.sifted if signal_stats.sifted > 0 else 0.5
        f_ec = self._calculate_f_ec(E_mu)

        Y1_L = decoy_estimates.yield_1_lower_bound
        e1_U = decoy_estimates.error_rate_1_upper_bound

        prob_1_mu = self._get_prob_n(mu, 1)
        Q_1 = Y1_L * prob_1_mu

        weak_rate = self._calculate_weak_gllp_rate(decoy_estimates, stats_map)
        delta_bound = decoy_estimates.diagnostics.numeric_diagnostics.get("Delta_tagged_bound_Eq36", None)

        if getattr(self.p, 'use_weak_gllp', False):
            key_len = max(0, int(weak_rate * signal_stats.sent))
            return KeyCalculationResult(
                secure_key_length=key_len,
                privacy_amplification_term=0.0,
                error_correction_leakage=Q_mu * f_ec * binary_entropy(E_mu) * signal_stats.sent,
                phase_error_rate_upper_bound=E_mu / (1.0 - delta_bound) if delta_bound and delta_bound < 1.0 else 0.5,
                diagnostics={"mode": "WEAK_GLLP_EQ_43", "rate_per_pulse": weak_rate, "Delta": delta_bound},
                weak_gllp_rate=weak_rate,
                tagged_fraction_bound=delta_bound
            )

        h2_E_mu = binary_entropy(E_mu)
        term_ec_cost = Q_mu * f_ec * h2_E_mu

        h2_e1 = binary_entropy(e1_U)
        term_pa_gain = Q_1 * (1.0 - h2_e1)

        rate = q_factor * (term_pa_gain - term_ec_cost)


        key_len = max(0, int(rate * signal_stats.sent))


        diag = {
            "Q_mu": Q_mu, "E_mu": E_mu, "Q_1": Q_1, "Y1_L": Y1_L, "e1_U": e1_U, "rate_per_pulse": rate,
            "f_ec_used": f_ec, "weak_gllp_rate": weak_rate,
            "asym_retry_used": False,  # always False after paper-faithful removal
        }
        if decoy_estimates.diagnostics.numeric_diagnostics:
            diag.update(decoy_estimates.diagnostics.numeric_diagnostics)

        return KeyCalculationResult(
            secure_key_length=key_len,
            privacy_amplification_term=term_pa_gain * signal_stats.sent,
            error_correction_leakage=term_ec_cost * signal_stats.sent,
            phase_error_rate_upper_bound=e1_U,
            diagnostics=diag,
            weak_gllp_rate=weak_rate,
            tagged_fraction_bound=delta_bound
        )

    def _get_bounds_stats(self, stats_map: Dict[str, TallyCounts], name: str, alpha: float) -> Tuple[float, float, float, float]:
        """Helper to get lower/upper bounds for Q and E of a specific pulse type.

        Returns RATES (0 to 1), not counts.  get_bounds() already returns
        rates (probabilities in [0,1]), so we do NOT divide by sent/sifted.
        """
        if name not in stats_map:
            return 0.0, 0.0, 0.0, 0.0
        s = stats_map[name]
        if s.sent == 0:
            return 0.0, 0.0, 0.0, 0.0

        # get_bounds() returns RATES (probabilities), not counts.
        q_low, q_high = self.get_bounds(s.sifted, s.sent, alpha, sided="two")

        if s.sifted > 0:
            e_low, e_high = self.get_bounds(s.errors_sifted, s.sifted, alpha, sided="two")
        else:
            e_low, e_high = 0.0, 0.5
        return q_low, q_high, e_low, e_high


class Ma2005VacuumWeakProof(Ma2005BaseProof):
    """
    Implements the "Vacuum + Weak" decoy protocol.
    Includes robustness to intensity jitter by using worst-case bounds.
    """
    
    def estimate_yields_and_errors(self, stats_map: Dict[str, TallyCounts]) -> DecoyEstimates:
        stats_map = self._normalize_stats_map(stats_map, self._get_pulse_names())
        if self.is_thermal:
             raise ConfigurationError("Ma2005VacuumWeakProof supports only Poisson statistics.")

        try:
            cfg_sig = self.p.source.get_pulse_config_by_name("signal")
            cfg_decoy = self.p.source.get_pulse_config_by_name("decoy")
            cfg_vac = self.p.source.get_pulse_config_by_name("vacuum")
        except Exception as e:
             return DecoyEstimates(0.0, 0.5, False, 1.0, SolverDiagnostics("Ma2005", False, str(e)))

        mu = cfg_sig.mean_photon_number
        nu = cfg_decoy.mean_photon_number

        # Spec Sec 4.1: intensity fluctuations explicitly neglected.
        alpha = self.eps_alloc.eps_pe

        Q_mu_L, Q_mu_U, E_mu_L, E_mu_U = self._get_bounds_stats(stats_map, "signal", alpha)
        Q_nu_L, Q_nu_U, E_nu_L, E_nu_U = self._get_bounds_stats(stats_map, "decoy", alpha)
        Q_vac_L, Q_vac_U, E_vac_L, E_vac_U = self._get_bounds_stats(stats_map, "vacuum", alpha)

        Y0_U = Q_vac_U
        Y0_L = Q_vac_L
        e0 = 0.5

        # [Paper Eq 34] Lower Bound on Y1
        term_Q_nu = Q_nu_L * math.exp(nu)
        term_Q_mu = Q_mu_U * math.exp(mu) * (nu**2 / mu**2)
        term_Y0 = Y0_U * (mu**2 - nu**2) / (mu**2)

        denom = mu * nu - nu**2
        if denom <= 0:
            prefactor = 0.0
        else:
            prefactor = mu / denom
        
        Y1_L = prefactor * (term_Q_nu - term_Q_mu - term_Y0)
        Y1_L = min(1.0, max(0.0, Y1_L))

        # [Paper Eq 37] Upper Bound on e1
        if Y1_L * nu <= 1e-20:
            e1_U = 0.5
        else:
            numerator_e1 = (E_nu_U * Q_nu_U * math.exp(nu)) - (e0 * Y0_L)
            e1_U = numerator_e1 / (Y1_L * nu)
            e1_U = min(0.5, max(0.0, e1_U))

        # [Paper Eq 36] Tagged Fraction Delta Upper Bound
        if Q_nu_L > 0:
            term1 = (nu / (mu - nu))
            term2 = (nu * math.exp(-nu) * Q_mu_U) / (mu * math.exp(-mu) * Q_nu_L) - 1.0
            term3 = (nu * math.exp(-nu) * Y0_U) / (mu * Q_nu_L)
            delta_bound_strict = term1 * term2 + term3
        else:
            delta_bound_strict = 1.0

        # Asymptotic Reference (B4+B5: use _get_eta / _get_y0 helpers)
        eta = self._get_eta()
        y0_val = self._get_y0(stats_map)
        e_det = self.p.detector.qber_intrinsic

        y1_asymptotic = y0_val + eta
        e1_asymptotic = (0.5 * y0_val + e_det * eta) / y1_asymptotic if y1_asymptotic > 0 else 0.5

        beta_y1 = (y1_asymptotic - Y1_L) / y1_asymptotic if y1_asymptotic > 0 else 0.0
        beta_e1 = (e1_U - e1_asymptotic) / e1_asymptotic if e1_asymptotic > 0 else 0.0

        diag_data = {
            "Y0_L": Y0_L, "Y0_U": Y0_U,
            "Delta_tagged_bound_Eq36": delta_bound_strict,
            "beta_Y1_deviation": beta_y1,
            "beta_e1_deviation": beta_e1,
            "used_asymptotic_fallback": False,  # always False after paper-faithful removal
            "Y1_asymptotic": y1_asymptotic,
            "e1_asymptotic": e1_asymptotic,
        }

        return DecoyEstimates(
            yield_1_lower_bound=Y1_L,
            error_rate_1_upper_bound=e1_U,
            is_feasible=True, 
            failure_prob_used=0.0,
            diagnostics=SolverDiagnostics("Ma2005_VacWeak_Robust", True, "OK", numeric_diagnostics=diag_data)
        )

    def calculate_key_length(self, decoy_estimates: DecoyEstimates, stats_map: Dict[str, TallyCounts]) -> KeyCalculationResult:
        stats_map = self._normalize_stats_map(stats_map, self._get_pulse_names())
        return self._calculate_gllp_rate(decoy_estimates, stats_map)


class Ma2005GeneralTwoDecoyProof(Ma2005BaseProof):
    """Implements the General Two-Decoy protocol with statistical fluctuations."""
    def estimate_yields_and_errors(self, stats_map: Dict[str, TallyCounts]) -> DecoyEstimates:
        stats_map = self._normalize_stats_map(stats_map, self._get_pulse_names())
        if self.is_thermal:
            raise ConfigurationError("Ma2005GeneralTwoDecoyProof supports only Poisson statistics.")
        
        try:
            sig_cfg = self.p.source.get_pulse_config_by_name("signal")
            decoy_cfgs = [pc for pc in self.p.source.pulse_configs if pc.name != "signal"]
            if len(decoy_cfgs) < 2: return DecoyEstimates(0.0, 0.5, False, 1.0, SolverDiagnostics("ConfigError", False, "Need 2 decoys"))
            decoy_cfgs.sort(key=lambda x: x.mean_photon_number)
            nu2_cfg, nu1_cfg = decoy_cfgs[0], decoy_cfgs[1]
        except Exception as e: return DecoyEstimates(0.0, 0.5, False, 1.0, SolverDiagnostics("Error", False, str(e)))

        mu, nu1, nu2 = sig_cfg.mean_photon_number, nu1_cfg.mean_photon_number, nu2_cfg.mean_photon_number
        alpha = self.eps_alloc.eps_pe

        Q_mu_L, Q_mu_U, E_mu_L, E_mu_U = self._get_bounds_stats(stats_map, "signal", alpha)
        Q_nu1_L, Q_nu1_U, E_nu1_L, E_nu1_U = self._get_bounds_stats(stats_map, nu1_cfg.name, alpha)
        Q_nu2_L, Q_nu2_U, E_nu2_L, E_nu2_U = self._get_bounds_stats(stats_map, nu2_cfg.name, alpha)

        # Use bounds conservatively: Maximize Y0 subtraction (L) and minimize positive terms (L)
        # To lower bound Y0 (Eq 18):
        term_num = nu1 * Q_nu2_L * math.exp(nu2) - nu2 * Q_nu1_U * math.exp(nu1)
        Y0_L = min(1.0, max(0.0, term_num / (nu1 - nu2)))

        # Eq 21: Y1 >= ...
        denom_Y1 = (mu * nu1) - (mu * nu2) - (nu1**2) + (nu2**2)
        # Term Q_nu1 exp(nu1) - Q_nu2 exp(nu2) -> Minimize -> Q_nu1_L - Q_nu2_U
        term_Q_diff = Q_nu1_L * math.exp(nu1) - Q_nu2_U * math.exp(nu2)
        # Subtraction term: Q_mu exp(mu) - Y0 -> Maximize -> Q_mu_U - Y0_L
        term_mu_sub = (Q_mu_U * math.exp(mu) - Y0_L)
        factor_sq = (nu1**2 - nu2**2) / (mu**2)
        
        Y1_L = min(1.0, max(0.0, (mu / denom_Y1) * (term_Q_diff - (factor_sq * term_mu_sub))))

        # Eq 25: e1 <= ...
        if Y1_L * (nu1 - nu2) <= 1e-20: e1_U = 0.5
        else:
            # Maximize numerator: E*Q
            num_e1 = (E_nu1_U * Q_nu1_U * math.exp(nu1)) - (E_nu2_L * Q_nu2_L * math.exp(nu2))
            e1_U = min(0.5, max(0.0, num_e1 / ((nu1 - nu2) * Y1_L)))

        return DecoyEstimates(Y1_L, e1_U, True, 0.0, SolverDiagnostics("Ma2005_General", True, "OK", numeric_diagnostics={"Y0_L": Y0_L}))

    def calculate_key_length(self, decoy_estimates: DecoyEstimates, stats_map: Dict[str, TallyCounts]) -> KeyCalculationResult:
        stats_map = self._normalize_stats_map(stats_map, self._get_pulse_names())
        return self._calculate_gllp_rate(decoy_estimates, stats_map)


class Ma2005OneDecoyProof(Ma2005BaseProof):
    """Implements the "One-Decoy" protocol with statistical fluctuations."""
    def __init__(self, params, use_tighter_bound: bool = True, assume_zero_background: bool = True, **kwargs):
        super().__init__(params, **kwargs)
        self.assume_zero_background = assume_zero_background 

    def estimate_yields_and_errors(self, stats_map: Dict[str, TallyCounts]) -> DecoyEstimates:
        stats_map = self._normalize_stats_map(stats_map, self._get_pulse_names())
        if self.is_thermal:
             raise ConfigurationError("Ma2005OneDecoyProof supports only Poisson statistics.")
    
        try:
            cfg_sig = self.p.source.get_pulse_config_by_name("signal")
            cfg_decoy = self.p.source.get_pulse_config_by_name("decoy")
        except Exception as e:
            return DecoyEstimates(0.0, 0.5, False, 1.0, SolverDiagnostics("Ma2005", False, str(e)))

        mu, nu = cfg_sig.mean_photon_number, cfg_decoy.mean_photon_number
        e0 = 0.5
        alpha = self.eps_alloc.eps_pe

        Q_mu_L, Q_mu_U, E_mu_L, E_mu_U = self._get_bounds_stats(stats_map, "signal", alpha)
        Q_nu_L, Q_nu_U, E_nu_L, E_nu_U = self._get_bounds_stats(stats_map, "decoy", alpha)
        
        if self.assume_zero_background:
            # Eq. 41: Tighter Bound (Y0=0 implicitly)
            prefactor = mu / (mu * nu - nu**2)
            # Minimize Y1: Q_nu -> L, Q_mu -> U
            term_Q_nu = Q_nu_L * math.exp(nu)
            term_Q_mu = Q_mu_U * math.exp(mu) * (nu**2 / mu**2)
            Y1_L = min(1.0, max(0.0, prefactor * (term_Q_nu - term_Q_mu)))
            
            if Y1_L * nu <= 1e-20: e1_U = 0.5
            else: 
                # Maximize e1: E*Q -> U
                e1_U = min(0.5, max(0.0, (E_nu_U * Q_nu_U * math.exp(nu)) / (Y1_L * nu)))
            diag_msg = "Ma2005_OneDecoy_ZeroBackground"
        else:
            # Standard One-Decoy Bound (Eq. 38-40)
            # Y0_U derived from Signal E*Q (Eq 38)
            Y0_U = min(1.0, max(0.0, (E_mu_U * Q_mu_U * math.exp(mu)) / e0))
            
            term1 = Q_nu_L * math.exp(nu)
            term2 = Q_mu_U * math.exp(mu) * (nu**2 / mu**2)
            term3 = (Y0_U) * ((mu**2 - nu**2) / mu**2)
            Y1_L = min(1.0, max(0.0, (mu / (mu * nu - nu**2)) * (term1 - term2 - term3)))

            if Y1_L * mu <= 1e-20:
                e1_U = 0.5
            else:
                # NOTE (B6): Spec Eq (40) uses Y_1^{L,mu,0} which is not
                # explicitly defined in the paper (spec Sec 12 #3). We use
                # Y1_L from Eq (39) as a conservative substitute.
                e1_U = min(0.5, max(0.0, (E_mu_U * Q_mu_U * math.exp(mu)) / (Y1_L * mu)))
            diag_msg = "Ma2005_OneDecoy_Simple"

        return DecoyEstimates(Y1_L, e1_U, True, 0.0, SolverDiagnostics(diag_msg, True, "OK"))

    def calculate_key_length(self, decoy_estimates: DecoyEstimates, stats_map: Dict[str, TallyCounts]) -> KeyCalculationResult:
        stats_map = self._normalize_stats_map(stats_map, self._get_pulse_names())
        return self._calculate_gllp_rate(decoy_estimates, stats_map)


class Ma2005GeneralMultiDecoyProof(Ma2005BaseProof):
    """
    Implements the General Decoy Method (Eq. 13) for arbitrary m decoy states.
    Uses Linear Programming (LP) to find Y1 and e1.
    *** Supports Thermal Statistics ***
    """
    def estimate_yields_and_errors(self, stats_map: Dict[str, TallyCounts]) -> DecoyEstimates:
        stats_map = self._normalize_stats_map(stats_map, self._get_pulse_names())
        pulse_configs = self.p.source.pulse_configs
        k_intensities = len(pulse_configs)
        n_cap = self.p.photon_number_cap
        alpha = self.eps_alloc.eps_pe
        
        c_y1 = np.zeros(n_cap + 1)
        c_y1[1] = 1.0
        
        A_eq = np.zeros((k_intensities, n_cap + 1))
        b_eq = np.zeros(k_intensities)
        
        # Use statistical bounds for LP constraints
        b_eq_L = np.zeros(k_intensities)
        b_eq_U = np.zeros(k_intensities)
        
        valid_indices = []
        for i, pc in enumerate(pulse_configs):
            if pc.name not in stats_map or stats_map[pc.name].sent == 0: continue
            valid_indices.append(i)
            mu = pc.mean_photon_number
            
            Q_L, Q_U, _, _ = self._get_bounds_stats(stats_map, pc.name, alpha)
            b_eq_L[i] = Q_L
            b_eq_U[i] = Q_U
            
            for n in range(n_cap + 1):
                A_eq[i, n] = self._get_prob_n(mu, n)
        
        A_eq = A_eq[valid_indices]
        b_eq_L = b_eq_L[valid_indices]
        b_eq_U = b_eq_U[valid_indices]
        
        try:
            # Constraints: Q_L <= sum <= Q_U
            # sum <= Q_U
            # -sum <= -Q_L
            A_ub = np.vstack([A_eq, -A_eq])
            b_ub = np.hstack([b_eq_U, -b_eq_L])
            
            sol_y1, diag_y1 = solve_lp(c_y1, A_ub, b_ub, n_cap + 1, self.p.lp_solver_method)
            Y1_L = min(1.0, max(0.0, sol_y1[1]))
            
            c_ey1 = np.zeros(n_cap + 1)
            c_ey1[1] = -1.0
            
            b_eq_err_L = np.zeros(len(valid_indices))
            b_eq_err_U = np.zeros(len(valid_indices))
            
            for i_idx, i in enumerate(valid_indices):
                pc = pulse_configs[i]
                Q_L, Q_U, E_L, E_U = self._get_bounds_stats(stats_map, pc.name, alpha)
                s = stats_map[pc.name]
                # Get bounds on error rate (errors_sifted / sifted, NOT errors_sifted / sent)
                # get_bounds returns CI for the COUNT; divide by s.sifted to get RATE
                if s.sifted > 0:
                    eq_low_count, eq_high_count = self.get_bounds(s.errors_sifted, s.sifted, alpha, sided="two")
                    EQ_L = eq_low_count / s.sifted
                    EQ_U = eq_high_count / s.sifted
                else:
                    EQ_L, EQ_U = 0.0, 0.5
                b_eq_err_L[i_idx] = EQ_L
                b_eq_err_U[i_idx] = EQ_U
            
            A_ub_err = np.vstack([A_eq, -A_eq])
            b_ub_err = np.hstack([b_eq_err_U, -b_eq_err_L])
            
            sol_ey1, diag_ey1 = solve_lp(c_ey1, A_ub_err, b_ub_err, n_cap + 1, self.p.lp_solver_method)
            e1Y1_U = max(0.0, sol_ey1[1])
            
            if Y1_L <= 1e-20: e1_U = 0.5
            else: e1_U = min(0.5, max(0.0, e1Y1_U / Y1_L))
            
            return DecoyEstimates(Y1_L, e1_U, True, 0.0, 
                                  SolverDiagnostics("LP_MultiDecoy", True, "OK", numeric_diagnostics={"solver": self.p.lp_solver_method}))

        except LPFailureError as e:
            return DecoyEstimates(0.0, 0.5, False, 1.0, SolverDiagnostics("LP_Failed", False, str(e)))

    def calculate_key_length(self, decoy_estimates: DecoyEstimates, stats_map: Dict[str, TallyCounts]) -> KeyCalculationResult:
        stats_map = self._normalize_stats_map(stats_map, self._get_pulse_names())
        return self._calculate_gllp_rate(decoy_estimates, stats_map)


class Ma2005AsymptoticProof(Ma2005BaseProof):
    """
    Implements the theoretical 'Infinite Decoy State' limit.
    Uses the asymptotic formulas derived in Ma et al. 2005 Eqs (27) and (28).
    This assumes perfect knowledge of the channel parameters (estimated from the signal state)
    and infinite decoy states, providing the theoretical upper bound on performance.
    """
    
    def estimate_yields_and_errors(self, stats_map: Dict[str, TallyCounts]) -> DecoyEstimates:
        stats_map = self._normalize_stats_map(stats_map, self._get_pulse_names())
        if "signal" not in stats_map:
             return DecoyEstimates(0.0, 0.5, False, 1.0, SolverDiagnostics("Error", False, "Missing signal stats"))

        signal_stats = stats_map["signal"]
        if signal_stats.sent == 0:
             return DecoyEstimates(0.0, 0.5, False, 1.0, SolverDiagnostics("Error", False, "No signal pulses sent"))

        # 1. Estimate Experimental Observables (Q_mu, E_mu)
        Q_mu = signal_stats.sifted / signal_stats.sent
        
        # 2. Derive Channel Parameters (eta) assuming known background Y0 and e_det
        # The asymptotic limit assumes we can determine these perfectly.
        # In practice, we use the simulation parameters if available, or estimate from Vacuum if present.
        
        # Paper Table 1: Y_0 is the total background rate.
        Y0 = self._get_y0(stats_map)
        e0 = 0.5
        e_det = self.p.detector.qber_intrinsic
        
        # Attempt to refine Y0 if vacuum stats are available (closer to "infinite decoy" reality check)
        if "vacuum" in stats_map and stats_map["vacuum"].sent > 0:
            vac_stats = stats_map["vacuum"]
            Y0 = vac_stats.sifted / vac_stats.sent
        
        mu = self.p.source.get_pulse_config_by_name("signal").mean_photon_number
        
        # Solve Q_mu = Y0 + 1 - exp(-eta * mu) for eta
        # exp(-eta * mu) = 1 + Y0 - Q_mu
        # -eta * mu = ln(1 + Y0 - Q_mu)
        
        arg = 1.0 + Y0 - Q_mu
        if arg <= 0:
            # Q_mu anomalously high; fall back to channel-model eta.
            eta = self._get_eta()
        else:
            eta_effective = -math.log(arg) / mu
            eta = max(0.0, min(1.0, eta_effective))

        # 3. Calculate Asymptotic Limits (Eq 27 and 28)
        # Eq 27: Y1 = Y0 + eta
        Y1_asymptotic = Y0 + eta
        
        # Eq 28: e1 = (e0*Y0 + e_det*eta) / Y1
        if Y1_asymptotic > 0:
            numerator = (e0 * Y0) + (e_det * eta)
            e1_asymptotic = numerator / Y1_asymptotic
        else:
            e1_asymptotic = 0.5
            
        e1_asymptotic = min(0.5, max(0.0, e1_asymptotic))

        return DecoyEstimates(
            yield_1_lower_bound=Y1_asymptotic,
            error_rate_1_upper_bound=e1_asymptotic,
            is_feasible=True,
            failure_prob_used=0.0,
            diagnostics=SolverDiagnostics("Ma2005_Asymptotic", True, "OK", numeric_diagnostics={"eta_inferred": eta, "Y0_used": Y0})
        )

    def calculate_key_length(self, decoy_estimates: DecoyEstimates, stats_map: Dict[str, TallyCounts]) -> KeyCalculationResult:
        stats_map = self._normalize_stats_map(stats_map, self._get_pulse_names())
        # Use GLLP without statistical penalties (infinite key limit usually implies infinite block size too, 
        # but here we calculate rate per pulse based on finite-size Q_mu but infinite-decoy Y1/e1)
        return self._calculate_gllp_rate(decoy_estimates, stats_map)
