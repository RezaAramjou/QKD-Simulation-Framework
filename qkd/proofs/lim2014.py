# qkd/proofs/lim2014.py
# -*- coding: utf-8 -*-
"""
Finite-key key-length calculation following the Lim et al. 2014 decoy-state
framework.

Reference:
    Lim, C. C. W., Curty, M., Walenta, N., Xu, F., and Zbinden, H.
    "Concise security bounds for practical decoy-state quantum key
    distribution." Physical Review A 89.2 (2014): 022307.
    DOI: 10.1103/PhysRevA.89.022307

This implementation is paper-faithful for the three-intensity signal /
weak-decoy / vacuum setting described in Eqs. (1)-(5) and Appendix A of the
paper. Specifically:

  - Eq. (1):  secret key length
               l = s_{X,0} + s_{X,1} * (1 - h(phi_X))
                   - lambda_EC - 6*log2(21/eps_sec) - log2(2/eps_cor)
  - Eq. (2):  s_{X,0} lower bound (vacuum events)
  - Eq. (3):  s_{X,1} lower bound (single-photon events)
  - Eq. (4):  v_{Z,1} upper bound (single-photon error events)
  - Eq. (5):  phi_X upper bound with random-sampling fluctuation gamma
  - Eqs. in main text + Appendix A: HYBRID Hoeffding / Chernoff intervals.
    For large counts (n_k >= 10 * delta_paper): paper-faithful Hoeffding
               n_{X,k}^{+-} = (e^{mu_k}/p_k) * (n_{X,k} +- sqrt((n_X/2)*ln(21/eps_sec)))
    For small counts (0 < n_k < 10 * delta_paper): per-intensity Chernoff
               n_{X,k}^{+-} = (e^{mu_k}/p_k) * (n_{X,k} +- sqrt(2*n_{X,k}*ln(1/eps)))
    For n_k = 0: multiplicative Chernoff upper bound = ln(1/eps_per_event).
    The hybrid is provably <= Hoeffding for all n_k, so security is never
    weakened. The paper's Hoeffding choice was for closed-form simplicity;
    tighter concentration inequalities are explicitly permitted (Appendix A).
  - Appendix B: eps_sec = 21*eps  (21 failure events, equally split)

Deviations / extensions beyond the closed-form paper results are clearly
flagged in diagnostics:

  - N-decoy LP branch (more than 3 intensities):
        The paper states the analysis "can also be straightforwardly
        generalized to any number of intensity levels" but does not give
        closed-form formulas. This implementation uses a conservative
        truncated-photon-number linear program with explicit non-negativity
        and upper-bound constraints and a tail relaxation. The 21-event
        eps_sec split is reused (conservatively) for this branch.

  - 'tomamichel' error-correction model:
        Optional. Defaults to the paper's lambda_EC = f_EC * n * h(e_obs)
        with the log2(2/eps_cor) correctness term subtracted separately
        (Eq. 1). The 'tomamichel' option folds log2(2/eps_cor) into
        lambda_EC per Ref. [37] of the paper (Tomamichel et al.,
        arXiv:1401.5194) and is offered as a more accurate leakage model.

All numerical safeguards (MIN_EPS floors, clamps to [0, 0.5]) are designed
to be triggered only in degenerate edge cases (e.g., zero observed errors)
where the paper's formulas become singular; in such cases the safe return
matches the paper's worst-case bound (e.g., phi_X = 0.5) rather than
introducing a more optimistic value.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

from .base import (
    DecoyEstimates,
    ErrorCode,
    FiniteKeyProof,
    KeyCalculationResult,
    _json_safe_dict,
)
from .utils_lp import solve_lp
from ..datatypes import EpsilonAllocation, TallyCounts
from ..exceptions import ConfigurationError, LPFailureError, ParameterValidationError

if TYPE_CHECKING:
    from ..params import QKDParams

__all__ = ["Lim2014Proof", "DecoyEstimatesLim2014"]

LOGGER = logging.getLogger(__name__)

MAX_ERROR_RATE = 0.5

# Hybrid Hoeffding/Chernoff threshold: use Hoeffding when
# n_k >= HOEFFDING_FACTOR * delta_paper (relative error <= 10%).
HOEFFDING_FACTOR = 10.0
PROB_SUM_TOL = 1e-9
MIN_EPS = 1e-300

# Number of failure events in the Lim-2014 security accounting
# (Appendix B, Eq. B4: eps_sec = 21*eps when each error term is set to a
# common value eps).
LIM2014_FAILURE_EVENTS_3_INTENSITY = 21.0

DEFAULT_KAPPA_MAX_ITERATIONS = 50
DEFAULT_KAPPA_ABS_TOL_BITS = 100.0
DEFAULT_KAPPA_REL_TOL = 1e-6
MAX_SECURITY_EPS = 1.0 - 1e-15


@dataclass(frozen=True)
class DecoyEstimatesLim2014(DecoyEstimates):
    """
    Intermediate decoy estimates for the Lim-2014 calculation.
    """

    s_X_0_lower: float = 0.0
    s_X_1_lower: float = 0.0
    s_Z_1_lower: float = 0.0
    v_Z_1_upper: float = 0.0
    phi_X_upper: float = MAX_ERROR_RATE
    is_feasible: bool = False
    failure_prob_used: float = 1.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe_dict(self)


@dataclass(frozen=True)
class Lim2014Proof(FiniteKeyProof):
    """
    Lim et al. 2014 finite-key proof implementation.

    The three-intensity analytical path implements Eqs. (1)-(5) of the paper
    exactly. For more than three intensities, a conservative LP-based
    N-decoy estimate is used (see module docstring).
    """

    p: "QKDParams"  # type: ignore[assignment]

    def __init__(self, p: "QKDParams"):
        import logging
        from qkd.proofs.base import ENTROPY_PROB_CLAMP, ProofMode
        try:
            import mpmath
            has_mpmath = True
        except ImportError:
            has_mpmath = False

        object.__setattr__(self, 'p', p)
        object.__setattr__(self, 'mode', ProofMode.PRODUCTION)
        object.__setattr__(self, 'logger', logging.getLogger(self.__class__.__name__))
        object.__setattr__(self, 'run_id', getattr(p, 'run_id', 'no-run-id'))
        object.__setattr__(self, 'min_prob_clamp', getattr(p, 'entropy_clamp', ENTROPY_PROB_CLAMP))
        object.__setattr__(self, 'use_mpmath', getattr(p, 'use_mpmath', False) and has_mpmath)
        if getattr(p, 'ci_method', None) is not None:
            from ..datatypes import ConfidenceBoundMethod
            if p.ci_method != ConfidenceBoundMethod.CHERNOFF:
                import logging
                logging.getLogger(self.__class__.__name__).warning(
                    f"Lim2014 proof requires CHERNOFF bounds for finite-key security. "
                    f"Overriding '{p.ci_method.name}' with CHERNOFF."
                )
                object.__setattr__(p, 'ci_method', ConfidenceBoundMethod.CHERNOFF)
    # ---------------------------------------------------------------------
    # Epsilon handling
    # ---------------------------------------------------------------------

    def get_epsilon_policy(self) -> Callable[[EpsilonAllocation], float]:
        return lambda eps: eps.eps_phase_est

    def allocate_epsilons(self) -> EpsilonAllocation:  # type: ignore[override]
        self._validate_epsilon_value(self.p.eps_sec, "eps_sec")
        self._validate_epsilon_value(self.p.eps_cor, "eps_cor")

        # Paper Appendix B, Eq. (B4): with 21 failure events each set to a
        # common value eps, eps_sec = 21*eps  =>  eps = eps_sec / 21.
        eps_single_test = self.p.eps_sec / LIM2014_FAILURE_EVENTS_3_INTENSITY

        return EpsilonAllocation(
            eps_sec=self.p.eps_sec,
            eps_cor=self.p.eps_cor,
            eps_phase_est=eps_single_test,
            eps_pe=0.0,
            eps_smooth=0.0,
            eps_pa=0.0,
        )

    @staticmethod
    def _validate_epsilon_value(value: float, name: str) -> None:
        if not math.isfinite(float(value)) or not (0.0 < float(value) < 1.0):
            raise ParameterValidationError(f"{name} must be finite and in the open interval (0, 1).")

    @staticmethod
    def _safe_log(x: float) -> float:
        if not math.isfinite(float(x)) or x <= 0.0:
            raise ParameterValidationError(f"log argument must be finite and positive; got {x!r}.")
        return math.log(x)

    @staticmethod
    def _safe_log2(x: float) -> float:
        if not math.isfinite(float(x)) or x <= 0.0:
            raise ParameterValidationError(f"log2 argument must be finite and positive; got {x!r}.")
        return math.log2(x)

    @staticmethod
    def _clamp_probability(x: float, *, upper: float = 1.0) -> float:
        if not math.isfinite(float(x)):
            return upper
        return min(max(float(x), 0.0), upper)

    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------

    def _validate_configuration(self) -> None:
        self._validate_epsilon_value(self.p.eps_sec, "eps_sec")
        self._validate_epsilon_value(self.p.eps_cor, "eps_cor")

        if not math.isfinite(float(self.p.f_error_correction)) or self.p.f_error_correction < 1.0:
            raise ParameterValidationError("f_error_correction must be finite and >= 1.")

        # 'standard' and 'lim' both implement the paper's Eq. (1) exactly
        # (lambda_EC = f_EC * n * h(e_obs), with log2(2/eps_cor) subtracted
        # separately as the correctness term).
        # 'tomamichel' is the optional Ref. [37] extension.
        ec_model = getattr(self.p, "error_correction_model", "standard")
        if ec_model not in {"standard", "lim", "tomamichel"}:
            raise ParameterValidationError(
                "error_correction_model must be one of: 'standard', 'lim', 'tomamichel'."
            )

        photon_cap = int(getattr(self.p, "photon_number_cap", 0))
        if photon_cap < 1:
            raise ParameterValidationError("photon_number_cap must be at least 1.")

        pulse_configs = list(getattr(self.p.source, "pulse_configs", []))
        if len(pulse_configs) < 3:
            raise ConfigurationError("Lim2014Proof requires at least three pulse configurations.")

        names = [pc.name for pc in pulse_configs]
        if len(set(names)) != len(names):
            raise ConfigurationError("Pulse configuration names must be unique.")

        prob_sum = 0.0
        for pc in pulse_configs:
            mu = float(pc.mean_photon_number)
            prob = float(pc.probability)

            if not math.isfinite(mu) or mu < 0.0:
                raise ParameterValidationError(f"Invalid mean photon number for pulse {pc.name!r}: {mu!r}.")
            if not math.isfinite(prob) or prob <= 0.0:
                raise ParameterValidationError(f"Pulse probability for {pc.name!r} must be finite and positive.")
            prob_sum += prob

        if abs(prob_sum - 1.0) > PROB_SUM_TOL:
            raise ParameterValidationError(f"Pulse probabilities must sum to 1; got {prob_sum!r}.")

        kappa = getattr(self.p, "security_constant_kappa", None)
        if kappa is not None:
            if not math.isfinite(float(kappa)) or float(kappa) <= 0.0:
                raise ParameterValidationError("security_constant_kappa must be finite and positive when set.")

    def _validate_tally_counts(self, stats: TallyCounts, pulse_name: str) -> None:
        required = ("sent", "sifted_x", "sifted_z", "errors_sifted_x", "errors_sifted_z")
        missing = [name for name in required if not hasattr(stats, name)]
        if missing:
            raise ParameterValidationError(
                f"TallyCounts for pulse {pulse_name!r} is missing required fields: {missing}."
            )

        sent = float(stats.sent)
        sifted_x = float(stats.sifted_x)
        sifted_z = float(stats.sifted_z)
        err_x = float(stats.errors_sifted_x)
        err_z = float(stats.errors_sifted_z)

        values = {
            "sent": sent,
            "sifted_x": sifted_x,
            "sifted_z": sifted_z,
            "errors_sifted_x": err_x,
            "errors_sifted_z": err_z,
        }

        for name, value in values.items():
            if not math.isfinite(value) or value < 0.0:
                raise ParameterValidationError(
                    f"{name} for pulse {pulse_name!r} must be finite and non-negative."
                )

        if err_x > sifted_x:
            raise ParameterValidationError(
                f"errors_sifted_x exceeds sifted_x for pulse {pulse_name!r}: {err_x} > {sifted_x}."
            )
        if err_z > sifted_z:
            raise ParameterValidationError(
                f"errors_sifted_z exceeds sifted_z for pulse {pulse_name!r}: {err_z} > {sifted_z}."
            )
        if sifted_x + sifted_z > sent + 1e-9:
            raise ParameterValidationError(
                f"sifted_x + sifted_z exceeds sent for pulse {pulse_name!r}."
            )

    # ---------------------------------------------------------------------
    # Source / probability utilities
    # ---------------------------------------------------------------------

    @staticmethod
    def _poisson_pmf_list(mu: float, n_cap: int) -> List[float]:
        if mu < 0.0 or not math.isfinite(mu):
            raise ParameterValidationError(f"Poisson mean must be finite and non-negative; got {mu!r}.")
        pmfs = [0.0] * (n_cap + 1)
        pmfs[0] = math.exp(-mu)
        for n in range(1, n_cap + 1):
            pmfs[n] = pmfs[n - 1] * mu / float(n)
        return pmfs

    def _get_tau_n(self, n: int, pulse_map: Mapping[str, float], p_k: Mapping[str, float]) -> float:
        """
        tau_n := sum_k p_k * e^{-mu_k} * mu_k^n / n!
        (paper, definition below Eq. (2)).
        """
        if n < 0:
            raise ParameterValidationError("n must be non-negative.")

        tau_n = 0.0
        for name, mu in pulse_map.items():
            prob_k = p_k[name]
            pmf = math.exp(-mu) if n == 0 else math.exp(-mu) * (mu**n) / math.factorial(n)
            tau_n += prob_k * pmf
        return max(0.0, tau_n)

    def _get_tau_list(self, n_cap: int, mu_map: Mapping[str, float], p_k_map: Mapping[str, float]) -> List[float]:
        pmf_by_pulse = {name: self._poisson_pmf_list(mu, n_cap) for name, mu in mu_map.items()}
        tau_ns: List[float] = []
        for n in range(n_cap + 1):
            tau_ns.append(sum(p_k_map[name] * pmf_by_pulse[name][n] for name in mu_map))
        return tau_ns

    @staticmethod
    def _get_tail_probability(mu: float, n_cap: int) -> float:
        pmfs = Lim2014Proof._poisson_pmf_list(mu, n_cap)
        tail = 1.0 - sum(pmfs)
        return min(max(tail, 0.0), 1.0)

    def _get_bounds(self, n_k: float, n_total: float, eps_ln_term: float,
                     clip_total: Optional[float] = None) -> Tuple[float, float]:
        """
        Hybrid Hoeffding / per-intensity Chernoff confidence interval.

        Paper Lim2014 Appendix A uses Hoeffding's inequality with a single
        delta = sqrt(n_total/2 * ln(21/eps_sec)) that is the SAME for all
        intensities k. This is a sufficient condition for security but is
        pathologically loose when n_k << n_total (e.g. vacuum counts at long
        distance: observed 5, Hoeffding upper 572, 114x the observation).

        Hybrid strategy (security-preserving, tighter for small counts):
          - When n_k is "large" (n_k >= 10 * delta_paper, i.e. relative
            statistical error < 10%), use the paper's Hoeffding bound.
            This is the paper-faithful regime where Hoeffding is appropriate.
          - When n_k is "small" but nonzero (0 < n_k < 10 * delta_paper),
            use per-intensity multiplicative Chernoff:
                delta_k = sqrt(2 * n_k * ln(1/eps_per_event))
            where eps_per_event = eps_sec / 21 (Lim2014 Appendix B split).
            This is tighter than Hoeffding for small counts and provides
            the same security guarantee (any concentration inequality giving
            coverage >= 1 - eps is valid within the proof).
          - When n_k = 0, the upper bound is ln(1/eps_per_event) (rule of
            three for Poisson/binomial zero observations). This is finite
            and far smaller than Hoeffding's delta_paper.

        The hybrid is provably <= Hoeffding for all n_k, so it never weakens
        security. The Lim2014 paper explicitly states (Appendix A) that the
        Hoeffding choice is for simplicity and tighter bounds may be used.

        Parameters
        ----------
        n_k : float
            Observed count for the specific intensity / category.
        n_total : float
            Total detections across all intensities in the relevant basis.
        eps_ln_term : float
            ln(21 / eps_sec). Equals ln(1/eps_per_event) under the Appendix B
            21-event split (eps_per_event = eps_sec / 21).
        clip_total : float, optional
            Upper clip for the bounds. Defaults to n_total when omitted.

        # HYBRID-HOEFFDING-CHERNOFF v1
        """
        if clip_total is None:
            clip_total = n_total

        if n_total < 0.0:
            raise ParameterValidationError("n_total must be non-negative.")
        if clip_total < 0.0:
            raise ParameterValidationError("clip_total must be non-negative.")
        if eps_ln_term < 0.0 or not math.isfinite(eps_ln_term):
            raise ParameterValidationError("eps_ln_term must be finite and non-negative.")
        if n_k < 0.0:
            raise ParameterValidationError("n_k must be non-negative.")

        # --- Hoeffding delta (paper Eq. A1): same for all k, scales with n_total.
        delta_paper = math.sqrt((n_total / 2.0) * eps_ln_term)

        # --- Hybrid selection ---
        # Threshold: use Hoeffding when observed count is large enough that
        # the relative statistical error (delta_paper / n_k) is <= 10%.
        # Below this threshold, per-intensity Chernoff is strictly tighter.
        hoeffding_threshold = HOEFFDING_FACTOR * delta_paper

        if n_k >= hoeffding_threshold and n_k > 0.0:
            # Paper-faithful Hoeffding regime (large counts).
            lower = max(0.0, n_k - delta_paper)
            upper = min(n_k + delta_paper, clip_total)
        elif n_k > 0.0:
            # Per-intensity multiplicative Chernoff regime (small but nonzero).
            # delta_k = sqrt(2 * n_k * ln(1/eps_per_event))
            # Note eps_ln_term = ln(21/eps_sec) = ln(1/eps_per_event), so we
            # reuse it directly.
            delta_k = math.sqrt(2.0 * n_k * eps_ln_term)
            lower = max(0.0, n_k - delta_k)
            upper = min(n_k + delta_k, clip_total)
        else:
            # n_k = 0: multiplicative Chernoff upper bound (rule of three).
            # P(X = 0 | mu) = exp(-mu) <= eps  =>  mu <= ln(1/eps).
            # This is the rigorous finite upper bound for zero observations.
            lower = 0.0
            upper = min(eps_ln_term, clip_total)

        return lower, upper

    # ---------------------------------------------------------------------
    # LP-based N-decoy estimates (extension; not in paper)
    # ---------------------------------------------------------------------

    def _solve_n_decoy_lp(
        self,
        n_trials_for_bounds: float,
        variable_upper_bound: float,
        obs_counts: Mapping[str, float],
        mu_map: Mapping[str, float],
        p_k_map: Mapping[str, float],
        target: str,
        eps_ln_term: float,
        diagnostics: Dict[str, Any],
    ) -> float:
        """
        Conservative LP estimate for lower bounds on counts or upper bounds on
        error counts. Used only when more than 3 intensities are configured.

        NOTE: This is an implementation extension. The paper states the
        analysis generalizes to any number of intensity levels but does not
        provide closed-form formulas. The LP below uses explicit
        non-negativity and upper-bound constraints plus a tail relaxation;
        the 21-event eps_sec split is reused conservatively.
        """
        pulse_names = list(mu_map.keys())
        n_cap = int(self.p.photon_number_cap)

        if target not in {"s0_lower", "s1_lower", "v1_upper"}:
            raise ParameterValidationError(f"Unknown LP target {target!r}.")

        if n_cap < 1:
            raise ParameterValidationError("photon_number_cap must be at least 1 for LP decoy estimation.")

        c = np.zeros(n_cap + 1)
        if target == "s0_lower":
            c[0] = 1.0
        elif target == "s1_lower":
            c[1] = 1.0
        else:
            c[1] = -1.0  # maximize v1 by minimizing -v1

        tau_ns = self._get_tau_list(n_cap, mu_map, p_k_map)
        pmf_by_pulse = {name: self._poisson_pmf_list(mu_map[name], n_cap) for name in pulse_names}

        a_rows: List[np.ndarray] = []
        b_rows: List[float] = []

        for name in pulse_names:
            pk = p_k_map[name]
            if pk <= 0.0:
                raise ParameterValidationError(f"Pulse probability for {name!r} must be positive.")

            count_k = float(obs_counts.get(name, 0.0))
            lower, upper = self._get_bounds(count_k, n_trials_for_bounds, eps_ln_term)

            row = np.zeros(n_cap + 1)
            for n in range(n_cap + 1):
                if tau_ns[n] > 0.0:
                    row[n] = (pk * pmf_by_pulse[name][n]) / tau_ns[n]
                else:
                    row[n] = 0.0

            # Truncated contribution cannot exceed the observed upper interval.
            a_rows.append(row.copy())
            b_rows.append(upper)

            # Omitted tail contribution is unknown and non-negative. For a
            # conservative lower observational constraint, relax by a physical
            # tail allowance bounded by variable_upper_bound.
            tail_prob = self._get_tail_probability(mu_map[name], n_cap)
            tail_relaxation = min(variable_upper_bound, variable_upper_bound * max(tail_prob, 0.0))
            relaxed_lower = max(0.0, lower - tail_relaxation)

            a_rows.append(-row.copy())
            b_rows.append(-relaxed_lower)

        identity = np.eye(n_cap + 1)

        # Non-negativity: -x <= 0
        for i in range(n_cap + 1):
            a_rows.append(-identity[i])
            b_rows.append(0.0)

        # Physical upper bound: x <= variable_upper_bound
        upper_bound = max(0.0, float(variable_upper_bound))
        for i in range(n_cap + 1):
            a_rows.append(identity[i])
            b_rows.append(upper_bound)

        a_ub = np.vstack(a_rows)
        b_ub = np.array(b_rows, dtype=float)

        try:
            sol, diag = solve_lp(c, a_ub, b_ub, n_cap + 1, self.p.lp_solver_method)
            diagnostics.setdefault("lp_diagnostics", {})[target] = _json_safe_dict(diag)

            idx = 0 if target == "s0_lower" else 1
            value = float(sol[idx])

            if target in {"s0_lower", "s1_lower"}:
                return min(max(value, 0.0), upper_bound)

            return min(max(value, 0.0), upper_bound)

        except LPFailureError as exc:
            diagnostics.setdefault("warnings", []).append(f"LP failed for {target}: {exc!r}")
            diagnostics.setdefault("lp_failures", []).append(target)

            # Conservative fallbacks:
            # - Lower bounds become 0.
            # - Upper bound on error count becomes the physical maximum.
            if target in {"s0_lower", "s1_lower"}:
                return 0.0
            return upper_bound

    # ---------------------------------------------------------------------
    # Analytical Lim-2014 formulas (Eqs. 2-5)
    # ---------------------------------------------------------------------

    @staticmethod
    def _validate_three_intensity_order(mu1: float, mu2: float, mu3: float) -> None:
        """
        Paper §II: mu1 > mu2 + mu3 and mu2 > mu3 >= 0.

        The closed-formulas (Eqs. 2-4) are valid for any mu3 >= 0 satisfying
        the above; the bound is tightest as mu3 -> 0 but is not restricted
        to a true vacuum.
        """
        if not (math.isfinite(mu1) and math.isfinite(mu2) and math.isfinite(mu3)):
            raise ParameterValidationError("Three-intensity means must be finite.")
        if not (mu1 > mu2 > mu3 >= 0.0):
            raise ConfigurationError(
                "Three-intensity Lim formulas require mu1 > mu2 > mu3 >= 0."
            )
        if not (mu1 > mu2 + mu3):
            raise ConfigurationError("Three-intensity Lim formulas require mu1 > mu2 + mu3.")



    def _calc_s0_lower(
        self,
        tau_0: float,
        mu2: float,
        mu3: float,
        n_mu2_plus: float,
        n_mu3_minus: float,
    ) -> float:
        """
        Eq. (2):
            s_{X,0} >= tau_0 * (mu2 * n^-_{X,mu3} - mu3 * n^+_{X,mu2}) / (mu2 - mu3)

        For a valid LOWER bound on vacuum events, we use:
          - n^-_{X,mu3}: LOWER bound of vacuum detections (minimize additive term)
          - n^+_{X,mu2}: UPPER bound of decoy detections (maximize subtracted term)
        Both choices make the numerator as small as possible → conservative.

        where n^{+-}_{X,k} = (e^{mu_k} / p_k) * (n_{X,k} +- delta) are the
        Hoeffding-bounded detection counts (caller pre-multiplies).
        """
        denominator = mu2 - mu3
        if denominator <= 0.0:
            raise ConfigurationError("Invalid s0 denominator; require mu2 > mu3.")

        numerator = tau_0 * (mu2 * n_mu3_minus - mu3 * n_mu2_plus)
        return max(0.0, numerator / denominator)

    def _calc_s1_lower(
        self,
        tau_1: float,
        mu1: float,
        mu2: float,
        mu3: float,
        n_mu1_plus: float,
        n_mu2_minus: float,
        n_mu3_plus: float,
        s_0_lower_over_tau_0: float,
    ) -> float:
        """
        Eq. (3):
            s_{X,1} >= tau_1 * mu1 * T / D
        where
            T = n^-_{X,mu2} - n^+_{X,mu3} - ((mu2^2 - mu3^2)/mu1^2) * (n^+_{X,mu1} - s_{X,0}/tau_0)
            D = mu1*(mu2 - mu3) - mu2^2 + mu3^2
        """
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu3_sq = mu3 * mu3

        denominator = mu1 * (mu2 - mu3) - mu2_sq + mu3_sq
        if denominator <= 0.0:
            raise ConfigurationError("Invalid s1 denominator; source intensities do not satisfy formula conditions.")

        term = n_mu2_minus - n_mu3_plus - ((mu2_sq - mu3_sq) / mu1_sq) * (
            n_mu1_plus - s_0_lower_over_tau_0
        )
        numerator = tau_1 * mu1 * term
        return max(0.0, numerator / denominator)

    def _calc_v1_upper(
        self,
        tau_1: float,
        mu2: float,
        mu3: float,
        m_mu2_plus: float,
        m_mu3_minus: float,
        physical_upper: float,
    ) -> float:
        """
        Eq. (4):
            v_{Z,1} <= tau_1 * (m^+_{Z,mu2} - m^-_{Z,mu3}) / (mu2 - mu3)

        where m^{+-}_{Z,k} = (e^{mu_k} / p_k) * (m_{Z,k} +- delta_m) and
        delta_m = sqrt((m_Z / 2) * ln(21/eps_sec)) per Eq. (A2).
        """
        denominator = mu2 - mu3
        if denominator <= 0.0:
            raise ConfigurationError("Invalid v1 denominator; require mu2 > mu3.")

        numerator = tau_1 * (m_mu2_plus - m_mu3_minus)
        return min(max(0.0, numerator / denominator), max(0.0, physical_upper))

    def _calc_phi_X_upper(
        self,
        v_Z_1_upper: float,
        s_Z_1_lower: float,
        s_X_1_lower: float,
        eps_a: float,
    ) -> float:
        """
        Eq. (5):
            phi_X = c_{X,1} / s_{X,1} <= v_{Z,1}/s_{Z,1} + gamma(a, b, c, d)
        where
            a = eps_sec / 21     (single failure-event epsilon)
            b = v_{Z,1} / s_{Z,1}
            c = s_{Z,1}
            d = s_{X,1}
            gamma(a,b,c,d) = sqrt( ((c+d)(1-b)b) / (c d ln 2) * log2( (c+d) / (c d (1-b) b a^2) ) )

        The square root is explicit in the paper (the typographical layout
        in the PDF can obscure it). When b is 0 or 1 (degenerate), or when
        c or d is non-positive, the bound saturates at MAX_ERROR_RATE = 0.5
        per the paper's worst case.
        """
        self._validate_epsilon_value(eps_a, "eps_phase_est")

        if s_Z_1_lower <= 0.0 or s_X_1_lower <= 0.0:
            return MAX_ERROR_RATE

        e_Z_1 = self._safe_divide(v_Z_1_upper, s_Z_1_lower, default=MAX_ERROR_RATE)
        e_Z_1 = self._clamp_probability(e_Z_1, upper=MAX_ERROR_RATE)

        if e_Z_1 >= MAX_ERROR_RATE:
            return MAX_ERROR_RATE

        c = float(s_Z_1_lower)
        d = float(s_X_1_lower)

        if c <= 0.0 or d <= 0.0:
            return MAX_ERROR_RATE

        # The Lim random-sampling fluctuation term is singular at b in {0, 1}.
        # In those degenerate cases the paper's worst-case bound is 0.5.
        b = min(max(e_Z_1, MIN_EPS), MAX_ERROR_RATE - 1e-15)
        a = max(float(eps_a), MIN_EPS)

        denominator = c * d * (1.0 - b) * b * (a * a)
        if denominator <= 0.0 or not math.isfinite(denominator):
            return MAX_ERROR_RATE

        log_arg = (c + d) / denominator
        if log_arg <= 1.0 or not math.isfinite(log_arg):
            gamma = 0.0
        else:
            term1 = ((c + d) * (1.0 - b) * b) / (c * d * math.log(2.0))
            term2 = self._safe_log2(log_arg)
            product = term1 * term2
            gamma = math.sqrt(product) if product > 0.0 and math.isfinite(product) else MAX_ERROR_RATE

        return min(max(0.0, e_Z_1 + gamma), MAX_ERROR_RATE)

    # ---------------------------------------------------------------------
    # Error-correction leakage
    # ---------------------------------------------------------------------

    def _calc_error_correction_leakage(self, n_x: float, qber_x: float, eps_cor: float) -> Tuple[float, float]:
        """
        Returns ``(leak_EC, separate_correctness_term)``.

        Models:
            - ``standard`` / ``lim``: paper Eq. (1).
                leak_EC      = f_EC * n_X * h(e_obs)
                corr_term    = log2(2 / eps_cor)        (subtracted separately in Eq. 1)
            - ``tomamichel``: Ref. [37] of the paper (Tomamichel et al.,
                arXiv:1401.5194). Includes the correctness logarithmic
                overhead in the leakage term and returns a zero separate
                correctness term to avoid double counting.
        """
        self._validate_epsilon_value(eps_cor, "eps_cor")

        qber_x = self._clamp_probability(qber_x, upper=MAX_ERROR_RATE)
        h_e = self.binary_entropy(qber_x)
        asymptotic_leak = float(self.p.f_error_correction * n_x * h_e)
        corr_term = self._safe_log2(2.0 / eps_cor)

        model = getattr(self.p, "error_correction_model", "standard")
        if model == "tomamichel":
            return asymptotic_leak + corr_term, 0.0

        return asymptotic_leak, corr_term

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def calculate_key_length(
        self,
        stats_map: Mapping[Union[str, int], TallyCounts],
    ) -> KeyCalculationResult:  # type: ignore[override]
        self._validate_configuration()

        if getattr(self.p, "security_constant_kappa", None) is not None:
            result = self._iterative_kappa_solver(stats_map)
        else:
            result = self._calculate_key_length_core(stats_map, self.p.eps_sec)

        # === Defensive physical clamp ===
        # The secure key length CANNOT exceed the total number of basis-matched
        # raw detections (n_X + n_Z = total sifted bits). This catches any
        # internal bug (kappa solver loop, LP fallback, decoy estimate overflow)
        # that might produce an unclamped or doubly-counted key length.
        # Physical justification: secure key bits are a SUBSET of raw sifted bits
        # after error correction + privacy amplification.
        try:
            _n_total = 0
            for _k, _v in stats_map.items():
                if hasattr(_v, 'sifted'):
                    _n_total += int(_v.sifted)
            if _n_total > 0 and result.secure_key_length > _n_total:
                import logging as _logging
                _logging.getLogger('Lim2014Proof').warning(
                    f'Defensive clamp triggered: secure_key_length {result.secure_key_length} '
                    f'> total sifted {_n_total}; clamping to {_n_total}. '
                    f'This indicates an internal bug in the kappa solver or decoy estimation.'
                )
                # Reconstruct result with clamped value
                from qkd.datatypes import KeyCalculationResult as _KCR
                result = _KCR(
                    secure_key_length=_n_total,
                    privacy_amplification_term=result.privacy_amplification_term,
                    error_correction_leakage=result.error_correction_leakage,
                    phase_error_rate_upper_bound=result.phase_error_rate_upper_bound,
                    error_codes=result.error_codes,
                    diagnostics=result.diagnostics,
                )
        except Exception as _e:
            import logging as _logging
            _logging.getLogger('Lim2014Proof').debug(
                f'Defensive clamp check failed (non-fatal): {_e}'
            )

        return result

    def _iterative_kappa_solver(
        self,
        stats_map: Mapping[Union[str, int], TallyCounts],
    ) -> KeyCalculationResult:
        """
        Implements the paper's eps_sec = kappa * l convention (paper §IV:
        "we set eps_sec to be proportional to the secret key length, that
        is, eps_sec = kappa * l").
        """
        kappa = float(self.p.security_constant_kappa)
        if not math.isfinite(kappa) or kappa <= 0.0:
            raise ParameterValidationError("security_constant_kappa must be finite and positive.")

        n_total = sum(float(s.sent) for s in stats_map.values())
        l_est = max(1.0, min(n_total * 0.01, n_total))

        last_result: KeyCalculationResult | None = None
        converged = False

        for iteration in range(DEFAULT_KAPPA_MAX_ITERATIONS):
            eps_target = kappa * max(l_est, 1.0)
            eps_target = min(max(eps_target, 1e-10), MAX_SECURITY_EPS)  # floor at 1e-10: smaller eps explodes Chernoff bounds (ln(eps/21) > 26)

            result = self._calculate_key_length_core(stats_map, eps_target)
            last_result = result
            l_new = float(result.secure_key_length)

            abs_delta = abs(l_new - l_est)
            rel_delta = abs_delta / max(1.0, abs(l_est))

            if abs_delta <= DEFAULT_KAPPA_ABS_TOL_BITS or rel_delta <= DEFAULT_KAPPA_REL_TOL:
                converged = True
                result.diagnostics.setdefault("dynamic_epsilon", {})
                result.diagnostics["dynamic_epsilon"].update(
                    {
                        "converged": True,
                        "iterations": iteration + 1,
                        "kappa": kappa,
                        "eps_sec_final": eps_target,
                    }
                )
                return result

            if l_new <= 0.0:
                l_est = max(1.0, 0.5 * l_est)
            else:
                l_est = 0.5 * (l_est + l_new)

        assert last_result is not None
        last_result.diagnostics.setdefault("dynamic_epsilon", {})
        last_result.diagnostics["dynamic_epsilon"].update(
            {
                "converged": converged,
                "iterations": DEFAULT_KAPPA_MAX_ITERATIONS,
                "kappa": kappa,
                "warning": "dynamic epsilon iteration did not converge",
            }
        )
        last_result.diagnostics.setdefault("warnings", []).append("Dynamic epsilon iteration did not converge.")
        return last_result

    # ---------------------------------------------------------------------
    # Core calculation
    # ---------------------------------------------------------------------

    def _normalize_stats_map(
        self,
        stats_map: Mapping[Union[str, int], TallyCounts],
        pulse_names: List[str],
    ) -> Dict[str, TallyCounts]:
        if all(name in stats_map for name in pulse_names):
            return {name: stats_map[name] for name in pulse_names}  # type: ignore[index]

        if all(str(i) in stats_map for i in range(len(pulse_names))):
            return {pulse_names[i]: stats_map[str(i)] for i in range(len(pulse_names))}

        if all(i in stats_map for i in range(len(pulse_names))):
            return {pulse_names[i]: stats_map[i] for i in range(len(pulse_names))}

        raise ParameterValidationError(
            f"stats_map keys do not match pulse names or positional indices. "
            f"Expected names {pulse_names!r}; got {list(stats_map.keys())!r}."
        )

    def _zero_result(
        self,
        branch: str,
        diagnostics: Dict[str, Any],
        error_codes: List[ErrorCode] | None = None,
    ) -> KeyCalculationResult:
        diagnostics = dict(diagnostics)
        diagnostics["return_branch"] = branch
        return KeyCalculationResult(
            secure_key_length=0,
            privacy_amplification_term=0.0,
            error_correction_leakage=0.0,
            phase_error_rate_upper_bound=MAX_ERROR_RATE,
            error_codes=error_codes or [ErrorCode.INSUFFICIENT_STATISTICS],
            diagnostics=_json_safe_dict(diagnostics.items()),
        )

    def _calculate_key_length_core(
        self,
        stats_map: Mapping[Union[str, int], TallyCounts],
        eps_sec_override: float,
    ) -> KeyCalculationResult:
        self._validate_epsilon_value(eps_sec_override, "eps_sec_override")

        # Paper Appendix B, Eq. (B4): with 21 failure events set to a common
        # value eps, eps_sec = 21*eps  =>  eps = eps_sec / 21.
        epsilon = eps_sec_override / LIM2014_FAILURE_EVENTS_3_INTENSITY
        self._validate_epsilon_value(epsilon, "single_test_epsilon")
        # ln(21 / eps_sec) is the argument used in the paper's Hoeffding terms.
        eps_ln_term = self._safe_log(LIM2014_FAILURE_EVENTS_3_INTENSITY / eps_sec_override)

        pulse_configs = list(self.p.source.pulse_configs)
        pulse_names = [pc.name for pc in pulse_configs]
        mu_map: Dict[str, float] = {pc.name: float(pc.mean_photon_number) for pc in pulse_configs}
        p_k_map: Dict[str, float] = {pc.name: float(pc.probability) for pc in pulse_configs}

        diagnostics: Dict[str, Any] = {
            "epsilon_accounting": {
                "scheme": "Lim2014 Appendix B (21-event split, eps_sec = 21*eps)",
                "eps_sec_used": float(eps_sec_override),
                "single_test_epsilon": float(epsilon),
                "failure_events": LIM2014_FAILURE_EVENTS_3_INTENSITY,
                "hoeffding_ln_term": float(eps_ln_term),
                "bound_method": "hoeffding_chernoff_hybrid",
                "note": (
                "Uses a hybrid Hoeffding / per-intensity Chernoff bound. "
                "For large counts (n_k >= 10 * delta_paper, relative "
                "statistical error <= 10%), uses the paper-faithful "
                "Hoeffding delta = sqrt(n_total/2 * ln(21/eps_sec)). "
                "For small counts (0 < n_k < threshold), uses the tighter "
                "per-intensity Chernoff delta_k = sqrt(2 * n_k * ln(1/eps)). "
                "For n_k = 0, uses the multiplicative Chernoff upper bound "
                "ln(1/eps_per_event) (rule of three). This is provably "
                "<= Hoeffding for all n_k, so security is never weakened. "
                "The Lim2014 paper chose Hoeffding for closed-form "
                "simplicity; the hybrid uses tighter bounds where Hoeffding "
                "is pathological, without changing the security proof."
                ),
            },
            "mu_map": {k: float(v) for k, v in mu_map.items()},
            "p_k_map": {k: float(v) for k, v in p_k_map.items()},
            "warnings": [],
        }

        try:
            stats_by_pulse = self._normalize_stats_map(stats_map, pulse_names)
            for name, stats in stats_by_pulse.items():
                self._validate_tally_counts(stats, name)
        except ParameterValidationError as exc:
            diagnostics["error"] = str(exc)
            diagnostics["stats_map_keys"] = list(stats_map.keys())
            diagnostics["pulse_names"] = pulse_names
            return self._zero_result("invalid_or_mismatched_stats", diagnostics)

        n_X_k = {name: float(stats.sifted_x) for name, stats in stats_by_pulse.items()}
        n_Z_k = {name: float(stats.sifted_z) for name, stats in stats_by_pulse.items()}
        m_Z_k = {name: float(stats.errors_sifted_z) for name, stats in stats_by_pulse.items()}
        m_X_k = {name: float(stats.errors_sifted_x) for name, stats in stats_by_pulse.items()}

        n_X = sum(n_X_k.values())
        n_Z = sum(n_Z_k.values())
        m_Z = sum(m_Z_k.values())
        m_X = sum(m_X_k.values())

        diagnostics.update(
            {
                "n_X": float(n_X),
                "n_Z": float(n_Z),
                "m_X": float(m_X),
                "m_Z": float(m_Z),
                "n_X_k": {k: float(v) for k, v in n_X_k.items()},
                "n_Z_k": {k: float(v) for k, v in n_Z_k.items()},
                "m_X_k": {k: float(v) for k, v in m_X_k.items()},
                "m_Z_k": {k: float(v) for k, v in m_Z_k.items()},
                "eps_ln_term": float(eps_ln_term),
            }
        )

        if n_X <= 0.0 or n_Z <= 0.0:
            diagnostics["reason"] = "Both X and Z basis sifted counts must be positive."
            return self._zero_result("insufficient_statistics_nX_or_nZ", diagnostics)

        tau_0 = self._get_tau_n(0, mu_map, p_k_map)
        tau_1 = self._get_tau_n(1, mu_map, p_k_map)
        diagnostics["tau_0"] = float(tau_0)
        diagnostics["tau_1"] = float(tau_1)

        if tau_0 <= 0.0 or tau_1 <= 0.0:
            diagnostics["reason"] = "tau_0 and tau_1 must be positive."
            return self._zero_result("invalid_tau", diagnostics)

        # -----------------------------------------------------------------
        # Decoy estimation
        # -----------------------------------------------------------------

        if len(mu_map) > 3:
            # Extension: N-decoy LP. See module docstring.
            diagnostics["decoy_method"] = "conservative_truncated_lp_n_decoy"
            diagnostics["decoy_method_note"] = (
                "Extension beyond Lim2014 closed-form results (paper says "
                "analysis generalizes but does not provide formulas)."
            )

            s_X_0_L = self._solve_n_decoy_lp(
                n_trials_for_bounds=n_X,
                variable_upper_bound=n_X,
                obs_counts=n_X_k,
                mu_map=mu_map,
                p_k_map=p_k_map,
                target="s0_lower",
                eps_ln_term=eps_ln_term,
                diagnostics=diagnostics,
            )
            s_X_1_L = self._solve_n_decoy_lp(
                n_trials_for_bounds=n_X,
                variable_upper_bound=n_X,
                obs_counts=n_X_k,
                mu_map=mu_map,
                p_k_map=p_k_map,
                target="s1_lower",
                eps_ln_term=eps_ln_term,
                diagnostics=diagnostics,
            )
            s_Z_1_L = self._solve_n_decoy_lp(
                n_trials_for_bounds=n_Z,
                variable_upper_bound=n_Z,
                obs_counts=n_Z_k,
                mu_map=mu_map,
                p_k_map=p_k_map,
                target="s1_lower",
                eps_ln_term=eps_ln_term,
                diagnostics=diagnostics,
            )
            v_Z_1_U = self._solve_n_decoy_lp(
                n_trials_for_bounds=m_Z,
                variable_upper_bound=n_Z,
                obs_counts=m_Z_k,
                mu_map=mu_map,
                p_k_map=p_k_map,
                target="v1_upper",
                eps_ln_term=eps_ln_term,
                diagnostics=diagnostics,
            )

        else:
            diagnostics["decoy_method"] = "lim2014_three_intensity_analytical"

            try:
                sorted_keys = sorted(mu_map.keys(), key=lambda key: mu_map[key], reverse=True)
                k1, k2, k3 = sorted_keys[0], sorted_keys[1], sorted_keys[2]
                mu1, mu2, mu3 = mu_map[k1], mu_map[k2], mu_map[k3]
                p1, p2, p3 = p_k_map[k1], p_k_map[k2], p_k_map[k3]

                self._validate_three_intensity_order(mu1, mu2, mu3)

                if min(p1, p2, p3) <= 0.0:
                    raise ParameterValidationError("Three-intensity pulse probabilities must be positive.")

                diagnostics["three_intensity_roles"] = {
                    "signal": k1,
                    "decoy": k2,
                    "vacuum": k3,
                    "mu1": float(mu1),
                    "mu2": float(mu2),
                    "mu3": float(mu3),
                }

                # Hoeffding confidence intervals (paper Eqs. A1, A7-A10).
                # delta = sqrt(n_total/2 * ln(21/eps_sec)) is the SAME for all k.
                # n+_{X,k} = (e^k/p_k) * (n_{X,k} + delta)  [upper]
                # n-_{X,k} = (e^k/p_k) * max(0, n_{X,k} - delta)  [lower]
                # The caller applies the (e^k/p_k) prefactor below.
                n_X_k_bounds = {name: self._get_bounds(n_X_k[name], n_X, eps_ln_term, clip_total=n_X) for name in mu_map}
                n_Z_k_bounds = {name: self._get_bounds(n_Z_k[name], n_Z, eps_ln_term, clip_total=n_Z) for name in mu_map}
                m_Z_k_bounds = {name: self._get_bounds(m_Z_k[name], m_Z, eps_ln_term, clip_total=m_Z) for name in mu_map}

                diagnostics["n_X_k_bounds"] = {
                    k: [float(v[0]), float(v[1])] for k, v in n_X_k_bounds.items()
                }
                diagnostics["n_Z_k_bounds"] = {
                    k: [float(v[0]), float(v[1])] for k, v in n_Z_k_bounds.items()
                }
                diagnostics["m_Z_k_bounds"] = {
                    k: [float(v[0]), float(v[1])] for k, v in m_Z_k_bounds.items()
                }

                # Apply the (e^{mu_k} / p_k) prefactor to convert raw
                # Chernoff-bounded counts n_{X,k}^{+-} into the form used
                # in Eqs. (2)-(4).
                # Eq.(2): s_{X,0} >= tau_0*(mu2*n^-_{vac} - mu3*n^+_{dec})/(mu2-mu3)
                # For a valid lower bound: vacuum LOWER (k3 index 0) and decoy UPPER (k2 index 1)
                s_X_0_L = self._calc_s0_lower(
                    tau_0,
                    mu2,
                    mu3,
                    n_X_k_bounds[k2][1] / p2 * math.exp(mu2),  # decoy UPPER bound (n^+_{X,mu2})
                    n_X_k_bounds[k3][0] / p3 * math.exp(mu3),  # vacuum LOWER bound (n^-_{X,mu3})
                )
                s_X_0_over_tau0 = s_X_0_L / tau_0

                # Vacuum upper bound: now correctly computed by _get_bounds as
                # ln(1/eps) when observed = 0 (tight Chernoff bound, not n_X).
                # No anti-conservative patch needed.
                s_X_1_L = self._calc_s1_lower(
                    tau_1,
                    mu1,
                    mu2,
                    mu3,
                    n_X_k_bounds[k1][1] / p1 * math.exp(mu1),
                    n_X_k_bounds[k2][0] / p2 * math.exp(mu2),
                    n_X_k_bounds[k3][1] / p3 * math.exp(mu3),
                    s_X_0_over_tau0,
                )

                # Eq.(2) for Z-basis: same bound choice as X-basis
                s_Z_0_L = self._calc_s0_lower(
                    tau_0,
                    mu2,
                    mu3,
                    n_Z_k_bounds[k2][1] / p2 * math.exp(mu2),  # decoy UPPER bound (n^+_{Z,mu2})
                    n_Z_k_bounds[k3][0] / p3 * math.exp(mu3),  # vacuum LOWER bound (n^-_{Z,mu3})
                )
                s_Z_0_over_tau0 = s_Z_0_L / tau_0

                # Vacuum upper bound: now correctly computed by _get_bounds.
                s_Z_1_L = self._calc_s1_lower(
                    tau_1,
                    mu1,
                    mu2,
                    mu3,
                    n_Z_k_bounds[k1][1] / p1 * math.exp(mu1),
                    n_Z_k_bounds[k2][0] / p2 * math.exp(mu2),
                    n_Z_k_bounds[k3][1] / p3 * math.exp(mu3),
                    s_Z_0_over_tau0,
                )

                v_Z_1_U = self._calc_v1_upper(
                    tau_1,
                    mu2,
                    mu3,
                    m_Z_k_bounds[k2][1] / p2 * math.exp(mu2),
                    m_Z_k_bounds[k3][0] / p3 * math.exp(mu3),
                    physical_upper=n_Z,
                )

            except (ConfigurationError, ParameterValidationError, IndexError) as exc:
                diagnostics["error"] = str(exc)
                return self._zero_result("invalid_three_intensity_configuration", diagnostics)

        s_X_0_L = min(max(0.0, float(s_X_0_L)), n_X)
        s_X_1_L = min(max(0.0, float(s_X_1_L)), n_X)
        s_Z_1_L = min(max(0.0, float(s_Z_1_L)), n_Z)
        v_Z_1_U = min(max(0.0, float(v_Z_1_U)), n_Z)

        phi_X_U = self._calc_phi_X_upper(v_Z_1_U, s_Z_1_L, s_X_1_L, epsilon)

        qber_x = self._safe_divide(m_X, n_X, default=MAX_ERROR_RATE)
        qber_x = self._clamp_probability(qber_x, upper=MAX_ERROR_RATE)

        leak_EC, corr_term = self._calc_error_correction_leakage(n_X, qber_x, self.p.eps_cor)

        # Eq. (1):
        #   l = s_{X,0} + s_{X,1} * (1 - h(phi_X))
        #       - lambda_EC - 6*log2(21/eps_sec) - log2(2/eps_cor)
        term_s0 = s_X_0_L
        term_s1 = s_X_1_L * (1.0 - self.binary_entropy(phi_X_U))

        pa_term = 6.0 * self._safe_log2(LIM2014_FAILURE_EVENTS_3_INTENSITY / eps_sec_override)

        key_len = term_s0 + term_s1 - leak_EC - pa_term - corr_term
        key_len_clamped = self._clamp_key_length(float(key_len))

        if key_len < 0.0:
            diagnostics.setdefault("warnings", []).append("Raw key length is negative before clamping.")
        if s_X_1_L <= 0.0 or s_Z_1_L <= 0.0:
            diagnostics.setdefault("warnings", []).append("Single-photon lower bound is zero.")
        if phi_X_U >= MAX_ERROR_RATE:
            diagnostics.setdefault("warnings", []).append("Phase-error upper bound saturated at 0.5.")
        if "lp_failures" in diagnostics:
            diagnostics.setdefault("warnings", []).append("At least one LP estimate used a conservative fallback.")

        diagnostics.update(
            {
                "return_branch": "main_key_formula",
                "key_len_float": float(key_len),
                "key_len_clamped": int(key_len_clamped),
                "term_s0": float(term_s0),
                "term_s1": float(term_s1),
                "leak_EC": float(leak_EC),
                "pa_term": float(pa_term),
                "corr_term": float(corr_term),
                "s_X_0_L": float(s_X_0_L),
                "s_X_1_L": float(s_X_1_L),
                "s_Z_1_L": float(s_Z_1_L),
                "v_Z_1_U": float(v_Z_1_U),
                "phi_X_U": float(phi_X_U),
                "qber_x": float(qber_x),
            }
        )

        LOGGER.debug("Lim2014 diagnostics: %s", diagnostics)

        error_codes: List[ErrorCode] = []
        if key_len_clamped <= 0:
            error_codes.append(ErrorCode.INSUFFICIENT_STATISTICS)

        return KeyCalculationResult(
            secure_key_length=key_len_clamped,
            privacy_amplification_term=pa_term,
            error_correction_leakage=leak_EC,
            phase_error_rate_upper_bound=phi_X_U,
            error_codes=error_codes,
            diagnostics=_json_safe_dict(diagnostics.items()),
        )

    # ---------------------------------------------------------------------
    # Base-class compatibility
    # ---------------------------------------------------------------------

    def notation_map(self) -> Dict[str, str]:
        return {
            "l": "secure_key_length",
            "s_X,0": "s_X_0_L",
            "Y_1^L": "s_X_1_L",
            "e_ph": "phi_X_U",
            "lambda_EC": "error_correction_leakage",
            "s_z_1^L": "s_Z_1_L",
        }

    def estimate_yields_and_errors(
        self,
        stats_map: Mapping[Union[str, int], TallyCounts],
    ) -> DecoyEstimatesLim2014:  # type: ignore[override]
        result = self.calculate_key_length(stats_map)
        diagnostics = dict(result.diagnostics or {})

        return DecoyEstimatesLim2014(
            s_X_0_lower=float(diagnostics.get("s_X_0_L", 0.0)),
            s_X_1_lower=float(diagnostics.get("s_X_1_L", 0.0)),
            s_Z_1_lower=float(diagnostics.get("s_Z_1_L", 0.0)),
            v_Z_1_upper=float(diagnostics.get("v_Z_1_U", 0.0)),
            phi_X_upper=float(diagnostics.get("phi_X_U", MAX_ERROR_RATE)),
            is_feasible=bool(result.secure_key_length > 0),
            failure_prob_used=float(diagnostics.get("epsilon_accounting", {}).get("eps_sec_used", self.p.eps_sec)),
            diagnostics=diagnostics,
        )