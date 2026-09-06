# qkd/utils/validation.py

# -*- coding: utf-8 -*-
"""
Parameter validation helper functions.

This module provides robust boolean parsing and other validation utilities.
"""
from typing import Any
import numbers
import numpy as np

# --- FIX: Import from the project's centralized modules ---
from ..exceptions import ParameterValidationError
from ..constants import LP_SOLVER_METHODS

__all__ = ["parse_bool", "validate_lp_solver"] # Added the new function

# Canonical true/false string tokens (lowercased for case-insensitive matching)
_TRUE_TOKENS = {"true", "1", "yes", "y", "on"}
_FALSE_TOKENS = {"false", "0", "no", "n", "off"}


def parse_bool(x: Any) -> bool:
    """
    Strictly parses a value to a boolean, handling common representations.
    ... (docstring remains the same) ...
    """
    if isinstance(x, (bool, np.bool_)):
        return bool(x)

    if isinstance(x, numbers.Integral):
        if x == 1:
            return True
        if x == 0:
            return False
        raise ParameterValidationError(
            f"Invalid integer for boolean parsing: {x!r}. Only 0 or 1 are accepted."
        )

    if isinstance(x, numbers.Real) and not isinstance(x, numbers.Integral):
        if x == 1.0:
            return True
        if x == 0.0:
            return False
        raise ParameterValidationError(
            f"Invalid float for boolean parsing: {x!r}. Only exact 0.0 or 1.0 are accepted."
        )

    if isinstance(x, bytes):
        try:
            s = x.decode("utf-8")
        except UnicodeDecodeError:
            raise ParameterValidationError(
                f"Cannot parse boolean from bytes with invalid UTF-8 sequence: {x!r}"
            )
    elif isinstance(x, str):
        s = x
    else:
        raise ParameterValidationError(
            f"Cannot coerce type '{type(x).__name__}' to bool: {x!r}"
        )

    s_normalized = s.strip().lower()
    if s_normalized in _TRUE_TOKENS:
        return True
    if s_normalized in _FALSE_TOKENS:
        return False

    raise ParameterValidationError(
        f"Invalid string for boolean parsing: {s!r}. "
        f"Recognized values are (case-insensitive): {sorted(_TRUE_TOKENS | _FALSE_TOKENS)}"
    )

# --- NEW: Function moved from constants.py to its correct location ---
def validate_lp_solver(method: str) -> str:
    """
    Validates if a given LP solver method is supported by this framework.

    :param method: The name of the LP solver (e.g., 'highs').
    :return: The method name if it is valid.
    :raises ParameterValidationError: If the method is not in the supported list.
    """
    if method not in LP_SOLVER_METHODS:
        raise ParameterValidationError(
            message=f"LP solver '{method}' is not supported.",
            param_name="lp_solver_method",
            param_value=method,
            context={"available_methods": LP_SOLVER_METHODS}
        )
    return method
