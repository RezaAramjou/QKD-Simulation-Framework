"""Validation and numerical helper functions for the QKD simulation framework."""

from __future__ import annotations

import math
import warnings
from numbers import Real
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants_definitions import (
    ENTROPY_PROB_CLAMP,
    NUMERIC_ABS_TOL,
    NUMERIC_REL_TOL,
    PROB_SUM_TOL,
)

assert 0.0 < ENTROPY_PROB_CLAMP < 0.5, (
    f"ENTROPY_PROB_CLAMP must be in (0, 0.5), got {ENTROPY_PROB_CLAMP}"
)

PUBLIC_FUNCTION_NAMES: Final[tuple[str, ...]] = (
    "clamp_probabilities",
    "clamp_probability",
    "db_to_linear",
    "is_close",
    "is_finite_non_negative",
    "is_prob_vector",
    "is_valid_probability",
    "renormalize_probabilities",
)

__all__ = list(PUBLIC_FUNCTION_NAMES)

def _is_real_scalar(value: Any) -> bool:
    """Return True for real-typed scalars, excluding bool and np.bool_.

    Note: this checks type only, not value — NaN and Inf will return True.
    Finiteness validation is the caller's responsibility.
    """
    return isinstance(value, (Real, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def _to_real_scalar(value: Any, *, name: str) -> float:
    """Validate and coerce a scalar real input."""
    if not _is_real_scalar(value):
        raise TypeError(f"{name} must be a real scalar.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _to_real_1d_array(values: ArrayLike, *, name: str, allow_empty: bool) -> NDArray[np.float64]:
    """Validate and coerce an input into a 1D finite real NumPy array."""
    arr = np.asarray(values)

    if arr.dtype == np.bool_:
        raise TypeError(f"{name} must contain only real numeric values, not booleans.")
    if arr.ndim == 0:
        raise ValueError(f"{name} must be a 1D array, but a scalar was provided.")
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, but got {arr.ndim}D input.")
    if not allow_empty and arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.iscomplexobj(arr):
        raise TypeError(f"{name} must contain only real values.")

    try:
        arr = arr.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain only real numeric values.") from exc

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")

    return arr


def is_close(a: Real, b: Real, *, rel_tol: float = NUMERIC_REL_TOL, abs_tol: float = NUMERIC_ABS_TOL) -> bool:
    """Return True when two real scalar values are close under module tolerances."""
    a_val = _to_real_scalar(a, name="a")
    b_val = _to_real_scalar(b, name="b")
    return math.isclose(a_val, b_val, rel_tol=rel_tol, abs_tol=abs_tol)


def is_valid_probability(p: Any) -> bool:
    """
    Return True when `p` is a finite real scalar in approximately [0, 1],
    tolerating up to NUMERIC_ABS_TOL drift at either boundary.

    Complex numbers and non-scalar inputs are rejected.
    """
    if not _is_real_scalar(p):
        return False

    p_val = float(p)
    return math.isfinite(p_val) and -NUMERIC_ABS_TOL <= p_val <= 1.0 + NUMERIC_ABS_TOL


def is_finite_non_negative(value: Any) -> bool:
    """
    Return True when `value` is a finite real scalar that is non-negative up to
    `NUMERIC_ABS_TOL` numerical drift.
    """
    if not _is_real_scalar(value):
        return False

    value_f = float(value)
    return math.isfinite(value_f) and value_f >= -NUMERIC_ABS_TOL


def clamp_probability(p: Real) -> float:
    """
    Clamp a finite real scalar probability to `[ENTROPY_PROB_CLAMP, 1 - ENTROPY_PROB_CLAMP]`.

    Raises:
        TypeError: if `p` is not a real scalar.
        ValueError: if `p` is NaN or infinite.
    """
    p_val = _to_real_scalar(p, name="p")
    return min(max(p_val, ENTROPY_PROB_CLAMP), 1.0 - ENTROPY_PROB_CLAMP)


def clamp_probabilities(probs: ArrayLike) -> NDArray[np.float64]:
    """
    Clamp a 1D finite real vector elementwise to
    `[ENTROPY_PROB_CLAMP, 1 - ENTROPY_PROB_CLAMP]`.

    The input may be empty; in that case an empty array is returned.
    """
    arr = _to_real_1d_array(probs, name="probs", allow_empty=True)
    return np.clip(arr, ENTROPY_PROB_CLAMP, 1.0 - ENTROPY_PROB_CLAMP)


def renormalize_probabilities(probs: ArrayLike) -> NDArray[np.float64]:
    """
    Return a valid 1D probability vector derived from `probs`.

    Contract:
    - `probs` must be a non-empty 1D finite real vector.
    - Values are first clamped away from exact 0 and 1 to improve numerical
      stability for entropy-related downstream code.
    - The clamped vector is then L1-normalized.
    - If the total mass is numerically zero after preprocessing, a deterministic
      one-hot fallback with all mass at index 0 is returned.

    Note:
    The output is guaranteed to be a valid PDF, but it is not guaranteed that
    every component remains within `[ENTROPY_PROB_CLAMP, 1 - ENTROPY_PROB_CLAMP]`
    after the final normalization step.
    """
    arr = _to_real_1d_array(probs, name="probs", allow_empty=False)
    p_clamped = clamp_probabilities(arr)
    total_p = float(np.sum(p_clamped))

    if total_p < NUMERIC_ABS_TOL:
        warnings.warn(
            "renormalize_probabilities: total mass ≈ 0 after clamping; "
            "falling back to deterministic one-hot at index 0.",
            RuntimeWarning,
            stacklevel=2,
        )
        p_new = np.zeros_like(p_clamped)
        p_new[0] = 1.0
        return p_new

    return p_clamped / total_p


def db_to_linear(val_db: Real) -> float:
    """
    Convert a finite real scalar decibel value to linear power ratio.

    Formula:
        10^(x / 10)
    """
    val_db_f = _to_real_scalar(val_db, name="val_db")
    exponent = val_db_f / 10.0
    if exponent > 308:
        raise OverflowError(f"db_to_linear({val_db_f}) would overflow to infinity.")
    result = math.pow(10.0, exponent)
    if result == 0.0 and val_db_f != 0.0:
        raise OverflowError(f"db_to_linear({val_db_f}) underflows to zero.")
    return result


def is_prob_vector(p: ArrayLike) -> bool:
    """
    Return True when `p` is a non-empty 1D finite real probability vector.

    Validation rules:
    - one-dimensional only
    - finite real values only
    - each entry is non-negative up to `NUMERIC_ABS_TOL`
    - total sum is close to 1 within `PROB_SUM_TOL`
    """
    try:
        arr = _to_real_1d_array(p, name="p", allow_empty=False)
    except (TypeError, ValueError) as exc:
        msg = str(exc)
        if ("must be a 1D array" in msg
                or "must contain only" in msg
                or "must not be empty" in msg):
            return False
        raise

    if np.any(arr < -NUMERIC_ABS_TOL):
        return False

    return bool(np.isclose(np.sum(arr), 1.0, atol=PROB_SUM_TOL, rtol=0.0))

