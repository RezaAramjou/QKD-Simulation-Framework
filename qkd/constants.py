"""Shared constants, numerical tolerances, and utility functions for the QKD simulation framework."""

from __future__ import annotations

from typing import Any

from .constants_definitions import (
    CONST_BOLTZMANN,
    CONST_PLANCK,
    CONST_ELECTRON_CHARGE,
    DEFAULT_POISSON_TAIL_THRESHOLD,
    ENTROPY_PROB_CLAMP,
    EPS,
    LP_CONSTRAINT_VIOLATION_TOL,
    LP_SOLVER_METHODS,
    LPSolverMethod,
    MAX_SEED_INT_PCG64,
    MAX_SEED_INT_UINT32,
    MIN_SUCCESSFUL_Z1_BASIS_EVENTS_FOR_PHASE_EST,
    NUMERIC_ABS_TOL,
    NUMERIC_REL_TOL,
    PROB_SUM_TOL,
    PUBLIC_CONSTANT_NAMES,
    Y1_SAFE_THRESHOLD,
)
from .constants_helpers import (
    PUBLIC_FUNCTION_NAMES,
    clamp_probabilities,
    clamp_probability,
    db_to_linear,
    is_close,
    is_finite_non_negative,
    is_prob_vector,
    is_valid_probability,
    renormalize_probabilities,
)

__version__ = "4.3.0"

__all__ = [*PUBLIC_CONSTANT_NAMES, *PUBLIC_FUNCTION_NAMES, "as_dict"]


CONST_PLANCK = 6.62607015e-34          # J*s
CONST_SPEED_OF_LIGHT = 2.99792458e8    # m/s

def as_dict() -> dict[str, Any]:
    """
    Return public constants from this module as an ordered dictionary copy.

    The key order follows `PUBLIC_CONSTANT_NAMES`, which keeps serialization and
    logging output stable across runs.
    """
    module_globals = globals()
    return {name: module_globals[name] for name in PUBLIC_CONSTANT_NAMES}

