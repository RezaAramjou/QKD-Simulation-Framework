# qkd/utils/math.py

# -*- coding: utf-8 -*-
"""
Numerical and statistical helper functions for QKD simulations.
This module provides robust, validated, and well-documented implementations of
common mathematical functions used in quantum information and statistics.
"""
from __future__ import annotations

import logging
import math
import numbers
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Literal, Tuple, Any, Optional

# Set up a logger for this module to report numeric warnings.
logger = logging.getLogger(__name__)

# A very small probability, used to guard against floating point errors in logs.
_MIN_ALPHA = 1e-300

# Declare np with type Any to accommodate the case where it's not installed.
np: Any

# Attempt to import numpy for type hinting and robust type checking.
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    np = None
    _NUMPY_AVAILABLE = False

# Lazy import of scipy only where needed.
_SCIPY_AVAILABLE = False
_SCIPY_VERSION = "N/A"

# Declare scipy components with type Any to handle the optional import.
_scipy_beta: Any
_scipy_gammaln: Any
_scipy_logsumexp: Any
_scipy_norm: Any

try:
    from scipy import __version__ as _scipy_version_str
    from scipy.stats import beta as _scipy_beta, norm as _scipy_norm  # type: ignore
    from scipy.special import gammaln as _scipy_gammaln, logsumexp as _scipy_logsumexp
    _SCIPY_AVAILABLE = True
    _SCIPY_VERSION = _scipy_version_str
except ImportError:  # pragma: no cover
    _scipy_beta = None
    _scipy_gammaln = None
    _scipy_logsumexp = None
    _scipy_norm = None

from ..constants import (
    ENTROPY_PROB_CLAMP,
    DEFAULT_POISSON_TAIL_THRESHOLD,
    clamp_probability,
    is_close,
    is_finite_non_negative,
    is_prob_vector,
    is_valid_probability,
)

__all__ = [
    "binary_entropy",
    "hoeffding_bounds",
    "clopper_pearson_bounds",
    "gaussian_bounds",
    "ConfidenceInterval",
    "IntervalMethod",
    "calculate_total_click_probability",
    "p_n_mu_vector",
]


class IntervalMethod(Enum):
    """Enumeration for the statistical methods used to compute intervals."""
    HOEFFDING = "hoeffding"
    CLOPPER_PEARSON = "clopper-pearson"
    GAUSSIAN = "gaussian"


@dataclass(frozen=True)
class ConfidenceInterval:
    """Represents a confidence interval with metadata."""
    lower: float
    upper: float
    alpha: float
    method: IntervalMethod

    def as_tuple(self) -> Tuple[float, float]:
        """Returns the interval as a standard (lower, upper) tuple."""
        return self.lower, self.upper


def _validate_k_n(k: int | float, n: int | float) -> Tuple[int, int]:
    """Internal helper to validate and coerce k and n for binomial trials."""
    try:
        k_int, n_int = int(k), int(n)
        if k_int != k or n_int != n:
            raise ValueError("Non-integer values provided for k or n.")
    except (ValueError, TypeError):
        raise ValueError(f"Inputs 'k' and 'n' must be integers or integer-like, but got k={k}, n={n}.")

    if not isinstance(n_int, numbers.Integral) or n_int < 0:
        raise ValueError(f"Number of trials 'n' must be a non-negative integer, but got {n}.")
    if not isinstance(k_int, numbers.Integral) or not (0 <= k_int <= n_int):
        raise ValueError(f"Number of successes 'k' must be an integer satisfying 0 <= k <= n, but got k={k}, n={n}.")
    return k_int, n_int


def _validate_alpha(alpha: float) -> None:
    """Internal helper to validate the significance level."""
    if not isinstance(alpha, (int, float)) or not (_MIN_ALPHA < alpha < 1.0):
        raise ValueError(
            f"Significance level 'alpha' must be a number in ({_MIN_ALPHA}, 1), but got {alpha}."
        )


def binary_entropy(p_err: float, clamp: bool = True) -> float:
    """Calculates the binary Shannon entropy h(p) in bits."""
    if not is_valid_probability(p_err):
        raise ValueError(f"Input probability 'p_err' must be a valid probability, but got {p_err}.")
    
    p = float(p_err)

    if p == 0.0 or p == 1.0:
        return 0.0

    if clamp:
        p = clamp_probability(p)

    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def hoeffding_bounds(
    k: int, n: int, alpha: float, side: Literal["two-sided", "upper", "lower"] = "two-sided"
) -> ConfidenceInterval:
    """Calculates the Hoeffding (additive Chernoff) confidence interval."""
    _validate_alpha(alpha)
    if side not in ("two-sided", "upper", "lower"):
        raise ValueError("Parameter 'side' must be one of 'two-sided', 'upper', or 'lower'.")
    
    k, n = _validate_k_n(k, n)
    if n == 0:
        raise ValueError("Number of trials 'n' must be positive for Hoeffding bounds.")

    p_hat = float(k) / n
    log_arg = (2.0 if side == "two-sided" else 1.0) / alpha
    delta = math.sqrt(math.log(log_arg) / (2.0 * n))

    lower = max(0.0, p_hat - delta) if side != "upper" else 0.0
    upper = min(1.0, p_hat + delta) if side != "lower" else 1.0

    return ConfidenceInterval(lower=lower, upper=upper, alpha=alpha, method=IntervalMethod.HOEFFDING)


def clopper_pearson_bounds(
    k: int, n: int, alpha: float, side: Literal["two-sided", "upper", "lower"] = "two-sided"
) -> ConfidenceInterval:
    """Calculates the exact Clopper-Pearson confidence interval using the Beta distribution."""
    _validate_alpha(alpha)
    if side not in ("two-sided", "upper", "lower"):
        raise ValueError("Parameter 'side' must be one of 'two-sided', 'upper', or 'lower'.")
    if not _SCIPY_AVAILABLE:
        raise ImportError(
            "The 'scipy' library is required for clopper_pearson_bounds. "
            "Please install it (`pip install scipy`) or use hoeffding_bounds."
        )
    k, n = _validate_k_n(k, n)

    if n == 0:
        return ConfidenceInterval(0.0, 1.0, alpha=alpha, method=IntervalMethod.CLOPPER_PEARSON)

    ppf_alpha = alpha if side != "two-sided" else alpha / 2.0

    if k == 0 or side == "upper":
        lower = 0.0
    else:
        lower = _scipy_beta.ppf(ppf_alpha, k, n - k + 1)

    if k == n or side == "lower":
        upper = 1.0
    else:
        upper = _scipy_beta.ppf(1.0 - ppf_alpha, k + 1, n - k)

    try:
        lower_f = float(lower)
        if math.isnan(lower_f):
            logger.debug(f"scipy.beta.ppf returned NaN for lower bound (k={k}, n={n}). Defaulting to 0.0.")
            lower_f = 0.0
    except (TypeError, ValueError):
        logger.warning(f"scipy.beta.ppf returned a non-scalar or non-numeric value for lower bound: {lower}")
        lower_f = 0.0
        
    try:
        upper_f = float(upper)
        if math.isnan(upper_f):
            logger.debug(f"scipy.beta.ppf returned NaN for upper bound (k={k}, n={n}). Defaulting to 1.0.")
            upper_f = 1.0
    except (TypeError, ValueError):
        logger.warning(f"scipy.beta.ppf returned a non-scalar or non-numeric value for upper bound: {upper}")
        upper_f = 1.0

    lower_f = max(0.0, min(1.0, lower_f))
    upper_f = max(0.0, min(1.0, upper_f))

    if lower_f > upper_f:
        logger.warning(
            "Clopper-Pearson calculation resulted in lower > upper "
            f"(k={k}, n={n}, alpha={alpha}, scipy_version={_SCIPY_VERSION}). "
            "Falling back to the trivial [0, 1] interval."
        )
        lower_f, upper_f = 0.0, 1.0

    return ConfidenceInterval(lower=lower_f, upper=upper_f, alpha=alpha, method=IntervalMethod.CLOPPER_PEARSON)


def gaussian_bounds(
    k: int, n: int, alpha: float, side: Literal["two-sided", "upper", "lower"] = "two-sided"
) -> ConfidenceInterval:
    """
    Calculates confidence intervals using Standard Error Analysis (Gaussian approximation).
    Used for reproducing results that rely on large-N approximations.
    
    Args:
        k: Number of successes (integers).
        n: Number of trials.
        alpha: Significance level (e.g., 1e-9).
        side: One of 'two-sided', 'upper', 'lower'.
    """
    _validate_alpha(alpha)
    k, n = _validate_k_n(k, n)
    
    if n == 0:
        return ConfidenceInterval(0.0, 1.0, alpha=alpha, method=IntervalMethod.GAUSSIAN)

    if not _SCIPY_AVAILABLE:
        raise ImportError("Scipy is required for Gaussian bounds to calculate inverse CDF (ppf/isf).")

    # [CORRECTION] Use ISF (Inverse Survival Function) instead of PPF.
    # For very small alpha (e.g., 1e-23), (1.0 - alpha) rounds to 1.0 in float64,
    # causing ppf to return infinity. ISF handles small tail probabilities accurately.
    tail_prob = alpha / 2.0 if side == "two-sided" else alpha
    z_score = float(_scipy_norm.isf(tail_prob))
    
    p_hat = float(k) / n
    
    # Standard Error Analysis uses p_hat for variance estimate
    std_dev = math.sqrt(p_hat * (1.0 - p_hat) / n)
    
    delta = z_score * std_dev
    
    lower = max(0.0, p_hat - delta) if side != "upper" else 0.0
    upper = min(1.0, p_hat + delta) if side != "lower" else 1.0
    
    return ConfidenceInterval(lower=lower, upper=upper, alpha=alpha, method=IntervalMethod.GAUSSIAN)


@lru_cache(maxsize=128)

def chernoff_bounds(k: int, n: int, alpha: float, side: str = "two-sided"):
    """Multiplicative Chernoff confidence intervals for Binomial(k|n, p).
    
    Inverting the multiplicative Chernoff tail bounds:
      Lower tail: P(X <= (1-d)*mu) <= exp(-d^2*mu/2)  =>  factor 2
      Upper tail: P(X >= (1+d)*mu) <= exp(-d^2*mu/3)  =>  factor 3
    
    Confidence interval inversion:
      Lower CI inverts the UPPER tail => factor 3
      Upper CI inverts the LOWER tail => factor 2
    
    So:  delta_lower = sqrt(3 * k * ln(1/eps))
        delta_upper = sqrt(2 * k * ln(1/eps))
    """
    from dataclasses import dataclass
    @dataclass
    class _Interval:
        lower: float
        upper: float
    
    if n == 0:
        return _Interval(0.0, 1.0)
    
    eps = alpha / 2.0 if side == "two-sided" else alpha
    
    if k <= 0:
        if eps > 0 and eps < 1 and n > 0:
            upper = max(1.0, n * (1.0 - eps ** (1.0 / n)))
        else:
            upper = float(n)
        if side == "upper":
            return _Interval(0.0, upper)
        return _Interval(0.0, float(n))
    
    if eps <= 0 or eps >= 1:
        return _Interval(0.0, float(n))
    
    ln_term = math.log(1.0 / eps)
    
    # FIXED factors: lower CI = factor 3, upper CI = factor 2
    lower = max(0.0, k - math.sqrt(3.0 * k * ln_term))
    upper = min(float(n), k + math.sqrt(2.0 * k * ln_term))
    
    if side == "upper":
        return _Interval(0.0, upper)
    elif side == "lower":
        return _Interval(lower, float(n))
    return _Interval(lower, upper)

def p_n_mu_vector(
    mu: float, n_cap: int, *,
    tail_threshold: Optional[float] = DEFAULT_POISSON_TAIL_THRESHOLD,
    return_log: bool = False,
    hard_cap_limit: bool = False, max_n_cap: int = 100_000,
) -> np.ndarray:
    """Calculates the Poisson PMF vector robustly using log-space computation."""
    if not isinstance(mu, (int, float)) or not is_finite_non_negative(mu):
        raise ValueError("`mu` must be a non-negative finite number.")
    if not isinstance(n_cap, int) or n_cap < 1:
        raise ValueError("`n_cap` must be a positive integer.")
    if n_cap > max_n_cap and hard_cap_limit:
        raise ValueError(f"n_cap={n_cap} exceeds max_n_cap={max_n_cap}.")
    if not _NUMPY_AVAILABLE or not _SCIPY_AVAILABLE:
        raise ImportError("Numpy and Scipy are required for p_n_mu_vector.")

    if mu < np.finfo(float).tiny:
        vec = np.full(n_cap + 1, -np.inf if return_log else 0.0, dtype=np.float64)
        vec[0] = 0.0 if return_log else 1.0
        return vec

    ns = np.arange(n_cap, dtype=np.int64)
    with np.errstate(all='ignore'):
        log_pmf_body = ns * np.log(mu) - mu - _scipy_gammaln(ns + 1.0)

    if not np.all(np.isfinite(log_pmf_body)):
        raise RuntimeError(f"Numeric overflow in PMF calculation for mu={mu:.6g}.")

    log_sum_body = _scipy_logsumexp(log_pmf_body)
    with np.errstate(divide='ignore'):
        log_tail = -np.inf if log_sum_body >= 0.0 else np.log1p(-np.exp(log_sum_body))

    log_vec = np.append(log_pmf_body, log_tail)
    log_vec_normalized = log_vec - _scipy_logsumexp(log_vec)

    if return_log:
        return log_vec_normalized

    vec = np.exp(log_vec_normalized)
    if tail_threshold is not None and vec[-1] > tail_threshold:
        raise ValueError(f"Tail probability {vec[-1]:.3e} exceeds threshold {tail_threshold:.3e}.")

    if not is_prob_vector(vec):
        raise RuntimeError(f"Final PMF is not a valid probability vector. Sum: {np.sum(vec)}")

    return vec


def calculate_total_click_probability(
    mu: float,
    efficiency: float,
    dark_rate: float,
    n_cap: int = 100
) -> float:
    """Calculates the total probability of a detector click for a given pulse."""
    if not is_valid_probability(efficiency):
        raise ValueError("Parameter 'efficiency' must be a valid probability in [0, 1].")
    if not is_valid_probability(dark_rate):
        raise ValueError("Parameter 'dark_rate' must be a valid probability in [0, 1].")

    if not _NUMPY_AVAILABLE:
        raise ImportError("Numpy is required for calculate_total_click_probability.")

    pn_vector = p_n_mu_vector(mu=mu, n_cap=n_cap, tail_threshold=None, return_log=False)

    n_values = np.arange(n_cap + 1, dtype=np.int64)
    p_signal_click_given_n = 1.0 - (1.0 - efficiency)**n_values.astype(np.float64)

    p_signal_total = np.sum(pn_vector * p_signal_click_given_n)

    p_total = p_signal_total + dark_rate - (p_signal_total * dark_rate)

    return float(np.clip(p_total, 0.0, 1.0))
