"""Type aliases and public constant definitions for the QKD simulation framework.

Tolerance Hierarchy
-------------------
ENTROPY_PROB_CLAMP ≤ NUMERIC_ABS_TOL << PROB_SUM_TOL

- ENTROPY_PROB_CLAMP: element-level clamp bound for entropy stability.
- NUMERIC_ABS_TOL:   element-level absolute tolerance for fuzzy comparison.
- PROB_SUM_TOL:      vector-level tolerance for probability-sum-to-1 checks.

For a probability vector of length N, element-level errors of up to
NUMERIC_ABS_TOL accumulate to at most N × NUMERIC_ABS_TOL in the sum.
This remains within PROB_SUM_TOL for N < ~10^4. For longer vectors,
PROB_SUM_TOL may need to be increased.
"""

from __future__ import annotations

from typing import Final, Literal, TypeAlias

import numpy as np

LPSolverMethod: TypeAlias = Literal["highs", "highs-ds", "highs-ipm"]

MAX_SEED_INT_PCG64: Final[int] = (1 << 63) - 1
MAX_SEED_INT_UINT32: Final[int] = (1 << 32) - 1
LP_SOLVER_METHODS: Final[tuple[LPSolverMethod, ...]] = ("highs", "highs-ds", "highs-ipm")

MIN_SUCCESSFUL_Z1_BASIS_EVENTS_FOR_PHASE_EST: Final[int] = 10
DEFAULT_POISSON_TAIL_THRESHOLD: Final[float] = 1e-9  # Unified with NUMERIC_REL_TOL per framework convention.

EPS: Final[float] = float(np.finfo(float).eps)
NUMERIC_ABS_TOL: Final[float] = 1e-12
NUMERIC_REL_TOL: Final[float] = 1e-9
_Y1_EPS_MULTIPLIER: Final[float] = 1e4
Y1_SAFE_THRESHOLD: Final[float] = max(1e-12, _Y1_EPS_MULTIPLIER * EPS)
ENTROPY_PROB_CLAMP: Final[float] = NUMERIC_ABS_TOL
PROB_SUM_TOL: Final[float] = 1e-8
LP_CONSTRAINT_VIOLATION_TOL: Final[float] = NUMERIC_REL_TOL

CONST_PLANCK: Final[float] = 6.62607015e-34  # J·s (exact, 2019 SI)
CONST_BOLTZMANN: Final[float] = 1.380649e-23  # J/K (exact, 2019 SI)
CONST_ELECTRON_CHARGE: Final[float] = 1.602176634e-19  # C (exact, 2019 SI)

PUBLIC_CONSTANT_NAMES: Final[tuple[str, ...]] = (
    "MAX_SEED_INT_PCG64",
    "MAX_SEED_INT_UINT32",
    "LP_SOLVER_METHODS",
    "MIN_SUCCESSFUL_Z1_BASIS_EVENTS_FOR_PHASE_EST",
    "EPS",
    "NUMERIC_ABS_TOL",
    "NUMERIC_REL_TOL",
    "Y1_SAFE_THRESHOLD",
    "ENTROPY_PROB_CLAMP",
    "PROB_SUM_TOL",
    "DEFAULT_POISSON_TAIL_THRESHOLD",
    "LP_CONSTRAINT_VIOLATION_TOL",
    "CONST_PLANCK",
    "CONST_BOLTZMANN",
    "CONST_ELECTRON_CHARGE",
)

__all__ = list(PUBLIC_CONSTANT_NAMES)