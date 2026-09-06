# qkd/sources.py
# -*- coding: utf-8 -*-
"""
Photon-number source models for the QKD simulation framework.

Scope
-----
This module models the **photon-number statistics** of optical sources used in
QKD simulations, together with a small set of classical intensity
imperfections (laser intensity jitter, MZM modulation, electrical driver
noise) and a small set of source flaws used in security analyses (emission
failure, block-wise adversarial intensity fluctuation).

It is **not** a full "quantum optical source model": it deliberately does not
model phase randomization, spectral/temporal/polarization side-channels,
phase coherence between pulses, or afterpulsing (which is a detector effect).
Those concerns live elsewhere in the framework.

**Critical note on phase randomization (F-04 fix):** The
``phase_randomization_assumed`` flag (renamed from ``phase_randomized``)
is **metadata-only**: it is a CLAIM by the user that their external setup
provides per-pulse phase randomization. This module does NOT perform actual
per-pulse phase randomization (sampling phi ~ Uniform[0, 2pi]). When the
flag is True, the user asserts that the Poisson/thermal diagonal sampling
is security-proof-compliant because external phase randomization makes each
pulse a diagonal mixture of Fock states. When False, the diagonal sampling
is not suitable for decoy-state BB84 security analysis.

Architecture
------------
Two layers are kept as separate as practical inside a single ``OpticalSource``
abstraction:

1. **State preparation / pulse-type selection** — which pulse (signal vs.
   decoys, with their probabilities and base mean photon numbers).
2. **Photon-number sampling** — Poisson, thermal/geometric, or
   density-matrix-diagonal sampling, optionally followed by classical
   imperfections (MZM attenuation, channel splitting, SCM sideband factor,
   intensity jitter, emission failure).

For :class:`DensityMatrixSource`, the density matrix **MUST** be expressed in
the Fock basis ``{|0>, |1>, |2>, ...}``. Only its diagonal is used for
photon-number sampling; off-diagonal coherences are validated (Hermiticity,
PSD, trace) but otherwise discarded by the sampler. The diagonal of a valid
density matrix in the Fock basis IS the photon-number distribution.

Selection policy for :meth:`OpticalSource.create` is documented in that
method's docstring and is deterministic given the inputs.
"""
from __future__ import annotations

import logging
import math
import threading
import warnings
import dataclasses
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.special import jv, comb as scipy_comb, erf as scipy_erf

from .constants import (
    EPS,
    MAX_SEED_INT_PCG64,
    MAX_SEED_INT_UINT32,
    NUMERIC_ABS_TOL,
    NUMERIC_REL_TOL,
    clamp_probability,
    is_close,
    is_finite_non_negative,
    is_valid_probability,
)
from .datatypes import (
    IntensityConfig,
    IntensityNode,
    OpticalSourceConfig,
    ProtocolParameters,
    PulseEnsembleConfig,
    PulseTypeConfig,
    SourceErrorModel,
    SourceStatisticsType,
)
from .exceptions import ParameterValidationError
from .security_metadata import SourceSecurityMetadata
from .type_defs import RNGType
from .utils.utils import sanitize_for_serialization

logger = logging.getLogger(__name__)

__all__ = [
    "OpticalSource", "DensityMatrixSource", "PoissonSource",
    "photon_number_distribution", "poisson_pn_array", "thermal_pn_array",
    "PN_TRUNCATION_DIM",
    "SourceSimParams", "build_source_sim_params",
]


# ---------------------------------------------------------------------------
# Module-local numerical constants
#
# These constants are *local* to the source layer. They MUST NOT be reused
# from unrelated framework layers (entropy estimation, LP constraints, etc.)
# because doing so couples unrelated numerical tolerances and silently
# corrupts the source's physical model.
# ---------------------------------------------------------------------------

# Below SCM_SMALL_ANGLE_LIMIT, the small-angle approximation J1(m) ~ m/2 is
# used. The threshold is set very low (1e-3) so that the first-derivative
# discontinuity at the boundary between the quadratic approximation and the
# exact Bessel function is negligible (issue 2.2 in 3rd review). At m=1e-3,
# the relative error between (m^2)/4 and J1(m)^2 is ~2.5e-7 (from the
# J1(m) ~ m/2 - m^3/16 expansion), which is < 0.1% — a 1000x improvement
# over the old threshold of 0.1 (where the discontinuity was ~0.5%). This
# is safe for gradient-based optimization of the modulation index.
SCM_SMALL_ANGLE_LIMIT: float = 1e-3

# Divisor for the (m^2)/4 small-angle / linear approximation of J1(m)^2.
SCM_POWER_DIVISOR: float = 4.0

# Number of first-order sidebands used for SCM power scaling (issue 2.1 in
# 3rd review). Phase modulation of an optical carrier generates two
# symmetric first-order sidebands (n=+1 and n=-1), each carrying J1(m)^2
# of the total optical power (since J_{-1}(m) = -J_1(m), the power is the
# same). The default SCM_NUM_SIDEBANDS=1 assumes optical single-sideband
# (SSB) filtering, where only one sideband is detected/used. Set this to 2
# for double-sideband (DSB) detection, where both sidebands contribute.
# This constant is module-level because SCM configuration is typically
# uniform across all sources in a simulation; per-source configuration
# would require extending OpticalSourceConfig in datatypes.py.
SCM_NUM_SIDEBANDS: int = 1

# Hard safety cap on the Gamma shape parameter ``k = 1/jitter^2``. With the
# analytic Gaussian branch below, this cap is essentially never reached in
# practice; it exists only to guard against floating-point overflow for
# pathological jitter values. Note: as of the 3rd-review fixes, the Gamma
# model has been replaced by a log-normal model for jitter >=
# GAUSSIAN_LIMIT_JITTER (issue 2.4, 3.1 in 3rd review), so this cap is now
# only a legacy safeguard.
MAX_GAMMA_SHAPE: float = 1e12

# 9th-review fix (F-27): moved from local scope inside
# _validate_base_invariants to module level for consistency with
# other source constants (SCM_SMALL_ANGLE_LIMIT, etc.) and for
# external testing/reference.
GAUSSIAN_LARGE_JITTER_THRESHOLD: float = 0.1

# 4th-review fix (issue 1.4): the GAUSSIAN_LIMIT_JITTER constant and the
# two-branch (Gaussian / log-normal) split inside RANDOM_GAUSSIAN have been
# REMOVED. RANDOM_GAUSSIAN now ALWAYS uses an additive Gaussian noise
# model with reflection at zero (mu_new = mu * |1 + jitter * z|, z ~ N(0,1)),
# which matches the enum name. The previous behavior silently substituted
# a log-normal distribution for jitter >= 1e-3, violating domain
# expectations and introducing systematic skewness when modeling additive
# Gaussian intensity noise. The reflection (|.|) introduces a small
# positive bias E[|X|] - E[X] = 2*|E[X|X<0]|*P(X<0) that grows with
# jitter; users who require strictly positive intensity fluctuations with
# exact mean preservation should request a dedicated RANDOM_LOGNORMAL
# error model (not yet implemented in the SourceErrorModel enum). The
# reflection bias is documented in _apply_intensity_jitter.

# Absolute tolerance for probability row-sum normalization. Diagonals of
# valid density matrices sum to 1 up to floating-point error. As of the
# 3rd-review fixes (issue 3.2), renormalization is ALWAYS applied after
# clipping tiny negative values, regardless of the drift magnitude, to
# guarantee the returned distribution sums to exactly 1.0. This constant
# is retained for the drift-logging threshold (debug messages only).
SOURCE_NORMALIZATION_TOL: float = 1e-10

# Tolerance for the (mu_config vs. mu_from_DM) consistency check. We use a
# relative tolerance because mean photon numbers can span many orders of
# magnitude across decoys (e.g., mu_signal=0.5, mu_vacuum=0).
SOURCE_DM_MEAN_RTOL: float = 1e-3
SOURCE_DM_MEAN_ATOL: float = 1e-9

# Minimum density-matrix dimension (must be at least 1, i.e., |0><0|).
SOURCE_DM_MIN_DIM: int = 1

# 4th-review fix (issue 2.3): split the known-keys set into base keys
# (valid for any OpticalSource subclass) and DM-specific keys (valid only
# for DensityMatrixSource). Non-DM sources constructed via from_dict with
# strict=True now reject DM-specific keys instead of silently accepting
# and ignoring them, catching typos like ``missing_dm_polcy`` and
# preventing users from accidentally setting DM policies on Poisson sources.
_KNOWN_BASE_CONFIG_KEYS: frozenset = frozenset({
    "source_rate",
    "pulse_configs",
    "statistics_type",
    "error_model",
    "intensity_jitter",
    "modulation_index",
    "N_channels",
    "use_small_angle_approximation",
    "use_linear_modulation_approximation",
    "ideal_emission_probability",
    "adversarial_block_size",
    "is_bidirectional",
    "mzm",
    "electrical_noise",
    "security_metadata",
})

# DM-specific keys, valid only for DensityMatrixSource.from_dict.
_KNOWN_DM_CONFIG_KEYS: frozenset = frozenset({
    "density_matrices",
    "missing_dm_policy",
    "heterogeneous_dim_policy",
})

# Union of both sets, for backward compatibility with code that checks
# against any known source key (e.g., the previous _KNOWN_SOURCE_CONFIG_KEYS).
_KNOWN_SOURCE_CONFIG_KEYS: frozenset = _KNOWN_BASE_CONFIG_KEYS | _KNOWN_DM_CONFIG_KEYS

# 4th-review fix (issue 4.1): chunk size for inverse-CDF sampling in
# generate_photons_dm. Limits peak memory to ~80 MB per chunk for a Fock
# truncation dimension of 100 (chunk_size * max_dim * 8 bytes = 80 MB).
# For larger dimensions, the chunk size is automatically reduced.
_DM_SAMPLING_CHUNK_BYTES: int = 80 * 1024 * 1024  # 80 MB budget per chunk

# 8th-review fix (F-07, F-25): hard safety cap on mean photon numbers.
# rng.poisson(np.inf) raises ValueError; rng.geometric(p=1e-15)
# returns photon counts ~10^15, which is nonsensical for QKD. Extreme
# jitter or MZM noise can produce infinite or near-infinite effective
# mus. This constant clamps mu to a physically reasonable maximum,
# preventing overflow and ensuring numerical stability.
MAX_MU: float = 1e6

# 8th-review fix (F-08): maximum size for the adversarial block factor
# cache. Without eviction, the cache grows as O(total_pulses / block_size).
ADVERSARIAL_CACHE_MAX_SIZE: int = 10000


# 9th-review fix (F-04): renamed from phase_randomized. This flag is
# a CLAIM about the user's external setup, NOT an implementation feature.
# No actual per-pulse phase randomization is performed by this module.
# Decoy-state BB84 security proofs assume per-pulse phase randomization.
PHASE_RANDOMIZATION_ASSUMED_DEFAULT: bool = True

# 9th-review fix (F-03): default for the scm_enabled flag.
# When True, modulation_index controls SCM sideband scaling (mu is
# interpreted as total emitted energy). When False, SCM is disabled
# and mu is interpreted as post-SCM (detector-relevant) photon number.
# This decouples the physical SCM switch from the continuous modulation
# index parameter, eliminating the semantic discontinuity at m=0.
SCM_ENABLED_DEFAULT: bool = False


# ---------------------------------------------------------------------------
# Module-level photon-number distribution helpers
#
# These pure-NumPy functions compute analytical photon-number probability
# vectors P(n) for Poisson and thermal/geometric distributions. They are
# designed to be called from tight analytical loops (e.g., finite-key
# parameter estimation, gain/QBER estimation) without creating Python
# objects that would bottleneck Numba-compiled code. They return raw
# NumPy arrays and accept only scalar/array parameters -- no dataclass
# or enum arguments.
#
# The default truncation dimension is 20, which covers the physically
# relevant range for QKD (mu typically < 1, so P(n>20) < 1e-18).
# Callers can override this for high-mu scenarios.
#
# Integration with OpticalSource:
#   - OpticalSource.number_probabilities_for_pulse() delegates to these
#     helpers for Poisson/thermal statistics, and uses cached DM
#     diagonals for DensityMatrixSource.
#   - main_optimized.py and proof modules should prefer
#     source.number_probabilities_for_pulse() for per-pulse queries,
#     and these helpers for standalone analytical computation (e.g.,
#     when constructing gain matrices over a grid of mu values).
# ---------------------------------------------------------------------------

# Default photon-number truncation dimension for analytical helpers.
# P(n >= PN_TRUNCATION_DIM) < 1e-18 for mu < 1, making truncation
# error negligible for QKD parameters. Increase for mu >> 1.
PN_TRUNCATION_DIM: int = 20


# --- Source simulation parameters bundle -------------------------------------
# Mirrors ChannelSimParams and DetectorSimParams: a frozen bundle of
# source-derived quantities for simulation use.

@dataclasses.dataclass(frozen=True)
class SourceSimParams:
    """Frozen bundle of source-derived quantities for simulation use.

    This dataclass captures the source parameters that the simulation
    driver needs at runtime for Rogers analytic SBR, CSV output, and
    finite-key gain-curve pre-computation.  By freezing these values
    from a validated :class:`OpticalSource` object, we ensure
    downstream code reads from the source's frozen state — not from a
    stale config dict.

    Construction
    ------------
    Use the factory :func:`build_source_sim_params` rather than
    constructing directly.
    """
    pulse_period_ns: float
    source_rate: float
    statistics_type: str  # .value of SourceStatisticsType enum
    error_model: str      # .value of SourceErrorModel enum
    modulation_index: float
    intensity_jitter: float
    N_channels: int
    ideal_emission_probability: float
    n_pulse_types: int
    signal_pulse_index: int


def build_source_sim_params(source: OpticalSource) -> SourceSimParams:
    """Build :class:`SourceSimParams` from a validated source object.

    This is the single approved path for main_optimized.py to extract
    all source-derived quantities for the simulation loop.  Every
    parameter is read from the source's validated, frozen state —
    not from the mutable config dict.
    """
    return SourceSimParams(
        pulse_period_ns=float(source.pulse_period_ns),
        source_rate=float(source.source_rate),
        statistics_type=source.statistics_type.value,
        error_model=source.error_model.value,
        modulation_index=float(source.modulation_index),
        intensity_jitter=float(source.intensity_jitter),
        N_channels=int(source.N_channels),
        ideal_emission_probability=float(source.ideal_emission_probability),
        n_pulse_types=len(source.pulse_names()),
        signal_pulse_index=source.get_pulse_index_by_name("signal"),
    )


def poisson_pn_array(
    mu: float,
    max_n: int = PN_TRUNCATION_DIM,
) -> np.ndarray:
    """Compute the Poisson photon-number probability vector P(n) for mu.

    Returns a 1-D float64 array of length ``max_n`` where element ``n``
    holds P(n) = mu^n * exp(-mu) / n!.

    Parameters
    ----------
    mu : float
        Mean photon number. Must be >= 0.
    max_n : int, default PN_TRUNCATION_DIM
        Truncation dimension (array length). Must be >= 1.

    Returns
    -------
    np.ndarray
        Probability vector of shape (max_n,). Sums to approximately 1
        (truncation error < P(n >= max_n), which is < 1e-18 for mu < 1).

    Notes
    -----
    This function is pure NumPy -- no Python objects, no dataclasses --
    so it can be called from Numba-compiled loops without type-unification
    bottlenecks. For per-pulse queries on an OpticalSource instance, use
    ``source.number_probabilities_for_pulse(name)`` instead, which handles
    emission-failure semantics and density-matrix sources.

    The computation uses the numerically stable recurrence:
        P(0) = exp(-mu)
        P(n+1) = P(n) * mu / (n+1)
    This avoids overflow in factorial / power terms for large mu or n.
    """
    if mu < 0.0:
        raise ParameterValidationError(
            "mu must be >= 0 for Poisson distribution.",
            param_name="mu",
            param_value=mu,
        )
    if max_n < 1:
        raise ParameterValidationError(
            "max_n must be >= 1 for photon-number truncation.",
            param_name="max_n",
            param_value=max_n,
        )
    if mu == 0.0:
        # Special case: P(0) = 1, P(n>0) = 0.
        result = np.zeros(max_n, dtype=np.float64)
        result[0] = 1.0
        return result

    result = np.empty(max_n, dtype=np.float64)
    # Numerically stable recurrence: P(0) = exp(-mu), P(n+1) = P(n)*mu/(n+1)
    result[0] = math.exp(-mu)
    for n in range(1, max_n):
        result[n] = result[n - 1] * mu / n
    return result


def thermal_pn_array(
    mu: float,
    max_n: int = PN_TRUNCATION_DIM,
) -> np.ndarray:
    """Compute the thermal (Bose-Einstein) photon-number probability vector.

    Returns a 1-D float64 array of length ``max_n`` where element ``n``
    holds P(n) = mu^n / (1 + mu)^(n+1).

    This is the geometric distribution with parameter p = 1/(1+mu),
    which is the photon-number distribution of single-mode thermal light
    (Bose-Einstein statistics).

    Parameters
    ----------
    mu : float
        Mean photon number. Must be >= 0.
    max_n : int, default PN_TRUNCATION_DIM
        Truncation dimension (array length). Must be >= 1.

    Returns
    -------
    np.ndarray
        Probability vector of shape (max_n,). Sums to approximately 1
        (truncation error for mu < 1 is < 1e-18).

    Notes
    -----
    Pure NumPy -- Numba-safe. For per-pulse queries on an OpticalSource,
    use ``source.number_probabilities_for_pulse(name)`` instead.

    Uses the recurrence:
        P(0) = 1 / (1 + mu)
        P(n+1) = P(n) * mu / (1 + mu)
    """
    if mu < 0.0:
        raise ParameterValidationError(
            "mu must be >= 0 for thermal distribution.",
            param_name="mu",
            param_value=mu,
        )
    if max_n < 1:
        raise ParameterValidationError(
            "max_n must be >= 1 for photon-number truncation.",
            param_name="max_n",
            param_value=max_n,
        )
    if mu == 0.0:
        # Special case: P(0) = 1, P(n>0) = 0 (same as Poisson).
        result = np.zeros(max_n, dtype=np.float64)
        result[0] = 1.0
        return result

    result = np.empty(max_n, dtype=np.float64)
    ratio = mu / (1.0 + mu)  # P(n+1) / P(n)
    result[0] = 1.0 / (1.0 + mu)
    for n in range(1, max_n):
        result[n] = result[n - 1] * ratio
    return result


def photon_number_distribution(
    mu: float,
    statistics_type: Union[SourceStatisticsType, str],
    max_n: int = PN_TRUNCATION_DIM,
) -> np.ndarray:
    """Compute P(n) for a given mu and statistics type (Poisson or thermal).

    This is a convenience wrapper that dispatches to :func:`poisson_pn_array`
    or :func:`thermal_pn_array` based on ``statistics_type``. It accepts
    either a :class:`SourceStatisticsType` enum or a string
    (``"POISSON"`` / ``"THERMAL"``).

    Parameters
    ----------
    mu : float
        Mean photon number.
    statistics_type : SourceStatisticsType or str
        Photon-number statistics type. ``"POISSON"`` or ``"THERMAL"``.
    max_n : int, default PN_TRUNCATION_DIM
        Truncation dimension.

    Returns
    -------
    np.ndarray
        Probability vector of shape (max_n,).

    Notes
    -----
    This function is the primary API for analytical P(n) computation
    in proof modules and simulation drivers. It returns a raw NumPy
    array (no Python objects) for Numba / tight-loop compatibility.
    """
    if isinstance(statistics_type, str):
        st_upper = statistics_type.strip().upper()
        if st_upper == "POISSON":
            return poisson_pn_array(mu, max_n)
        elif st_upper == "THERMAL":
            return thermal_pn_array(mu, max_n)
        else:
            raise ParameterValidationError(
                f"Unknown statistics_type string: {statistics_type!r}. "
                f"Use 'POISSON' or 'THERMAL'.",
                param_name="statistics_type",
                param_value=statistics_type,
            )
    elif isinstance(statistics_type, SourceStatisticsType):
        if statistics_type == SourceStatisticsType.POISSON:
            return poisson_pn_array(mu, max_n)
        elif statistics_type == SourceStatisticsType.THERMAL:
            return thermal_pn_array(mu, max_n)
        else:
            raise ParameterValidationError(
                f"Unsupported SourceStatisticsType: {statistics_type!r}. "
                f"Only POISSON and THERMAL are supported.",
                param_name="statistics_type",
                param_value=statistics_type,
            )
    else:
        raise ParameterValidationError(
            f"statistics_type must be SourceStatisticsType or str, "
            f"got {type(statistics_type).__name__}.",
            param_name="statistics_type",
            param_value=type(statistics_type).__name__,
        )


# 8th-review fix (F-08): bounded LRU cache for adversarial block factors.
class _BoundedLRUCache(OrderedDict):
    """LRU cache with a maximum size for adversarial block factors.

    8th-review fix (F-08): the default dict-based
    _adversarial_block_factor_cache grows without bound. This OrderedDict
    subclass evicts the least-recently-used entry when the cache exceeds
    ADVERSARIAL_CACHE_MAX_SIZE, bounding memory usage.

    9th-review fix (F-16): added threading.Lock for thread-safety.
    Multi-threaded simulations with shared source objects can corrupt
    the cache via concurrent access. All mutating operations now
    acquire the lock before proceeding.
    """
    def __init__(self, max_size: int = ADVERSARIAL_CACHE_MAX_SIZE):
        super().__init__()
        self.max_size = max_size
        self._lock = threading.Lock()

    def __setitem__(self, key, value):
        with self._lock:
            if key in self:
                self.move_to_end(key)
                super().__setitem__(key, value)
            else:
                if len(self) >= self.max_size:
                    self.popitem(last=False)  # evict oldest
                super().__setitem__(key, value)

    def __getitem__(self, key):
        with self._lock:
            return super().__getitem__(key)

    def __contains__(self, key):
        with self._lock:
            return super().__contains__(key)



# ---------------------------------------------------------------------------
# 5th-review fix (issue 4.1): bool-as-int rejection helper
#
# In Python, ``bool`` is a subclass of ``int``, so ``isinstance(True, int)``
# returns ``True``. This means any validation that uses
# ``isinstance(val, (int, np.integer))`` will silently accept ``True`` and
# ``False`` as valid integers, masking likely user errors (e.g., passing a
# boolean flag where an integer count was expected). The ``_is_integer``
# helper explicitly rejects ``bool`` values, so callers get a clear error
# message instead of silent acceptance.
# ---------------------------------------------------------------------------

def _is_integer(val: Any) -> bool:
    """Return True iff ``val`` is an integer (but NOT a bool).

    5th-review fix (issue 4.1): rejects ``bool`` values explicitly.
    In Python, ``isinstance(True, int)`` is ``True`` because ``bool``
    subclasses ``int``. This helper ensures that boolean values are
    rejected wherever a genuine integer is expected (pulse indices,
    sample counts, block sizes, tally counts, etc.), preventing silent
    acceptance of likely user errors.
    """
    if isinstance(val, bool):
        return False
    return isinstance(val, (int, np.integer))


# ---------------------------------------------------------------------------
# Density-matrix validation helpers
# ---------------------------------------------------------------------------


def _validate_density_matrix(rho: np.ndarray, atol: float = 1e-10) -> None:
    """Validate that ``rho`` is a proper density matrix.

    Checks performed:
      * Square 2D array
      * Hermitian (``rho == rho.conj().T``)
      * Trace = 1 (real)
      * Positive semidefinite (all eigenvalues >= -atol)
      * Dimension >= 1

    Notes
    -----
    This function does **not** validate the *basis* of the matrix. For
    photon-number sampling, the caller MUST ensure the matrix is expressed
    in the Fock basis ``{|0>, |1>, |2>, ...}``. The sampler only uses the
    diagonal; off-diagonal coherences are ignored.
    """
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ParameterValidationError(
            "Density matrix must be a square 2D array.",
            param_name="density_matrix",
            param_value=f"shape={rho.shape}",
        )
    if rho.shape[0] < SOURCE_DM_MIN_DIM:
        raise ParameterValidationError(
            f"Density matrix dimension must be >= {SOURCE_DM_MIN_DIM}.",
            param_name="density_matrix",
            param_value=f"dim={rho.shape[0]}",
        )
    if not np.allclose(rho, rho.conj().T, atol=atol, rtol=0.0):
        raise ParameterValidationError(
            "Density matrix must be Hermitian (rho == rho.conj().T).",
            param_name="density_matrix",
        )
    tr = np.trace(rho)
    if not np.isclose(np.real(tr), 1.0, atol=atol, rtol=0.0) or abs(np.imag(tr)) > atol:
        raise ParameterValidationError(
            f"Density matrix trace must be 1 (got {tr}).",
            param_name="density_matrix",
        )
    try:
        evals = np.linalg.eigvalsh(rho)
    except np.linalg.LinAlgError as exc:
        raise ParameterValidationError(
            f"Failed to compute eigenvalues of density matrix: {exc}",
            param_name="density_matrix",
        ) from exc
    min_eval = float(np.min(evals))
    if min_eval < -atol:
        raise ParameterValidationError(
            f"Density matrix must be positive semidefinite "
            f"(min eigenvalue = {min_eval:.3e} < -{atol:.0e}).",
            param_name="density_matrix",
        )


def _coerce_density_matrices(
    density_matrices: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, np.ndarray]]:
    """Coerce a mapping of pulse-name -> matrix-like into complex128 arrays.

    Returns ``None`` if the input is ``None`` or an empty mapping (the empty
    case is normalized to ``None`` so downstream code can use a single
    ``is None`` check).

    Raises :class:`ParameterValidationError` on:
      * Non-mapping input
      * Non-string or empty/whitespace-only keys
      * Key collisions after ``.strip()`` (e.g., ``"signal"`` vs ``" signal "``)
      * Non-square or non-2D matrices
      * Array coercion failures
    """
    if density_matrices is None:
        return None

    if not isinstance(density_matrices, Mapping):
        raise ParameterValidationError(
            "density_matrices must be a mapping from pulse name to matrix-like data.",
            param_name="density_matrices",
            param_value=type(density_matrices).__name__,
        )

    coerced: Dict[str, np.ndarray] = {}
    seen_raw: Dict[str, str] = {}

    for name, rho in density_matrices.items():
        if not isinstance(name, str) or not name.strip():
            raise ParameterValidationError(
                "density_matrices keys must be non-empty strings.",
                param_name="density_matrices",
                param_value=name,
            )
        stripped = name.strip()
        if stripped in seen_raw:
            raise ParameterValidationError(
                f"density_matrices key collision after stripping: "
                f"{seen_raw[stripped]!r} and {name!r} both map to {stripped!r}.",
                param_name="density_matrices",
            )
        seen_raw[stripped] = name

        try:
            arr = np.asarray(rho, dtype=np.complex128)
        except (TypeError, ValueError) as exc:
            raise ParameterValidationError(
                f"Cannot coerce density matrix for pulse {stripped!r} to "
                f"complex128: {exc}",
                param_name="density_matrices",
            ) from exc

        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ParameterValidationError(
                f"Density matrix for pulse {stripped!r} must be square 2D, "
                f"got shape {arr.shape}.",
                param_name="density_matrices",
            )

        coerced[stripped] = arr

    # Normalize empty mapping to None.
    return coerced if coerced else None


def _decode_density_matrices_from_serialized(
    raw: Any,
) -> Optional[Dict[str, np.ndarray]]:
    """Decode density matrices from a serialized representation.

    Supports two payload formats per pulse name:

    1. **JSON-safe** (produced by :meth:`DensityMatrixSource.to_config_dict`):
       ``{"real": [[...]], "imag": [[...]]}`` where ``real`` and ``imag``
       are nested lists of floats.
    2. **Direct array-like**: list-of-lists of complex numbers (Python only,
       not JSON-safe).

    Returns ``None`` if the input is ``None`` or an empty mapping.
    """
    if raw is None:
        return None

    if not isinstance(raw, Mapping):
        raise ParameterValidationError(
            "density_matrices payload must be a mapping.",
            param_name="density_matrices",
            param_value=type(raw).__name__,
        )

    if not raw:
        return None

    decoded: Dict[str, np.ndarray] = {}
    for name, payload in raw.items():
        if (
            isinstance(payload, Mapping)
            and "real" in payload
            and "imag" in payload
        ):
            real_arr = np.asarray(payload["real"], dtype=np.float64)
            imag_arr = np.asarray(payload["imag"], dtype=np.float64)
            if real_arr.shape != imag_arr.shape:
                raise ParameterValidationError(
                    f"real and imag parts for pulse {name!r} have mismatched "
                    f"shapes: {real_arr.shape} vs {imag_arr.shape}.",
                    param_name="density_matrices",
                )
            decoded[name] = real_arr + 1j * imag_arr
        else:
            try:
                decoded[name] = np.asarray(payload, dtype=np.complex128)
            except (TypeError, ValueError) as exc:
                raise ParameterValidationError(
                    f"Cannot decode density matrix for pulse {name!r}: {exc}",
                    param_name="density_matrices",
                ) from exc

    # Re-coerce to apply strip + square-shape validation.
    return _coerce_density_matrices(decoded)


def _encode_density_matrices_for_serialization(
    dms: Optional[Mapping[str, np.ndarray]],
) -> Optional[Dict[str, Any]]:
    """Encode density matrices in a JSON-safe ``{real, imag}`` format.

    This guarantees that :meth:`to_dict` / :meth:`from_dict` round-trip
    preserves complex-valued density matrices, since JSON has no native
    complex-number type.
    """
    if not dms:
        return None
    encoded: Dict[str, Any] = {}
    for name, rho in dms.items():
        arr = np.asarray(rho, dtype=np.complex128)
        encoded[name] = {
            "real": arr.real.tolist(),
            "imag": arr.imag.tolist(),
        }
    return encoded


# ---------------------------------------------------------------------------
# Pulse-config helpers
# ---------------------------------------------------------------------------


def _pulse_name_for_node(node: IntensityNode, index: int, is_signal: bool) -> str:
    """Generate a stable, role-based pulse name.

    The name reflects the *role* of the pulse (signal / vacuum / decoy) and
    is **not** purely index-based. The first pulse (``index == 0``) is
    treated as the signal; subsequent zero-mu pulses are ``vacuum_i``; other
    pulses are ``decoy_i``.

    Note: callers using :meth:`OpticalSource.from_pulse_ensemble` bypass this
    helper and pass an explicit :class:`PulseEnsembleConfig`, in which case
    the user-supplied names are used as-is.

    Issue 1.4 in 3rd review: the previous code had a dead branch
    ``"vacuum" if index == 0 else f"vacuum_{index}"`` where ``index == 0``
    was unreachable (because ``is_signal = (index == 0)`` causes the first
    branch to return "signal"). The dead code has been removed; all vacuum
    decoys are now consistently named ``vacuum_{index}``.
    """
    if is_signal:
        return "signal"
    if is_close(node.mu, 0.0):
        return f"vacuum_{index}"
    return f"decoy_{index}"


def _pulse_types_from_intensity_config(
    intensities: IntensityConfig,
) -> Tuple[PulseTypeConfig, ...]:
    """Build a tuple of :class:`PulseTypeConfig` from an :class:`IntensityConfig`.

    The first node is treated as the signal (role-based naming). Subsequent
    nodes are vacuum or decoys based on their mean photon number.
    """
    nodes = intensities.all_nodes()
    pulses = [
        PulseTypeConfig(
            name=_pulse_name_for_node(node, idx, idx == 0),
            mean_photon_number=node.mu,
            probability=node.probability,
        )
        for idx, node in enumerate(nodes)
    ]
    ensemble = PulseEnsembleConfig(pulses=pulses)
    return tuple(ensemble.pulses)


def _coerce_pulse_configs(
    pulse_configs: Union[Sequence[PulseTypeConfig], Sequence[Mapping[str, Any]]],
) -> Tuple[PulseTypeConfig, ...]:
    """Coerce a sequence of pulse configs / mappings into typed configs.

    Issue 3.4 in 3rd review: this helper previously constructed a
    :class:`PulseEnsembleConfig` to delegate uniqueness/probability
    validation, but that same validation is also performed (more
    defensively) in :meth:`OpticalSource._validate_base_invariants`.
    The redundant construction has been removed; we now just coerce
    the types and return the tuple. Validation runs once, at source
    construction time, in ``_validate_base_invariants``.
    """
    coerced: List[PulseTypeConfig] = []
    for item in pulse_configs:
        if isinstance(item, PulseTypeConfig):
            coerced.append(item)
        elif isinstance(item, Mapping):
            coerced.append(PulseTypeConfig.from_dict(item))
        else:
            raise ParameterValidationError(
                "pulse_configs entries must be PulseTypeConfig or mapping.",
                param_name="pulse_configs",
                param_value=type(item).__name__,
            )
    return tuple(coerced)


def _coerce_optical_source_config(
    data: Union[OpticalSourceConfig, Mapping[str, Any]],
) -> OpticalSourceConfig:
    """Coerce a mapping or :class:`OpticalSourceConfig` into a typed config.

    4th-review fix (issue 2.2): the previous code accessed
    ``OpticalSourceConfig.__dataclass_fields__["mzm"].default_factory()``
    directly, which would raise ``TypeError: 'MISSING' object is not
    callable`` if the dataclass definition were ever changed to use a
    static ``default = ...`` instead of ``default_factory``. We now use
    a safe helper that handles both cases (factory or static default),
    future-proofing against such refactors.
    """
    if isinstance(data, OpticalSourceConfig):
        return data

    if not isinstance(data, Mapping):
        raise ParameterValidationError(
            "config must be an OpticalSourceConfig or mapping.",
            param_name="config",
            param_value=type(data).__name__,
        )

    pulse_configs_raw = data.get("pulse_configs")
    if pulse_configs_raw is None:
        raise ParameterValidationError(
            "OpticalSourceConfig requires 'pulse_configs'.",
            param_name="pulse_configs",
        )

    pulse_configs = _coerce_pulse_configs(pulse_configs_raw)

    statistics_type = data.get("statistics_type", SourceStatisticsType.POISSON)
    if not isinstance(statistics_type, SourceStatisticsType):
        statistics_type = SourceStatisticsType(statistics_type)

    error_model = data.get("error_model", SourceErrorModel.RANDOM_GAUSSIAN)
    if not isinstance(error_model, SourceErrorModel):
        error_model = SourceErrorModel(error_model)

    # 4th-review fix (issue 2.2): use the safe default-extraction helper
    # instead of directly calling .default_factory(). This handles both
    # default_factory and static default values without raising.
    mzm_default = _get_dataclass_field_default(OpticalSourceConfig, "mzm")
    electrical_noise_default = _get_dataclass_field_default(
        OpticalSourceConfig, "electrical_noise"
    )

    return OpticalSourceConfig(
        source_rate=_coerce_float(data["source_rate"], 0.0, "source_rate"),
        pulse_configs=pulse_configs,
        statistics_type=statistics_type,
        error_model=error_model,
        intensity_jitter=_coerce_float(data.get("intensity_jitter"), 0.0, "intensity_jitter"),
        modulation_index=_coerce_float(data.get("modulation_index"), 0.0, "modulation_index"),
        N_channels=_coerce_int(data.get("N_channels"), 1, "N_channels"),
        use_small_angle_approximation=_coerce_bool(data.get("use_small_angle_approximation"), True),
        use_linear_modulation_approximation=_coerce_bool(
            data.get("use_linear_modulation_approximation"), False
        ),
        ideal_emission_probability=_coerce_float(data.get("ideal_emission_probability"), 1.0, "ideal_emission_probability"),
        adversarial_block_size=_coerce_int(data.get("adversarial_block_size"), 1000, "adversarial_block_size"),
        is_bidirectional=_coerce_bool(data.get("is_bidirectional"), False),
        mzm=data.get("mzm", mzm_default),
        electrical_noise=data.get("electrical_noise", electrical_noise_default),
    )


def _get_dataclass_field_default(datacls: type, field_name: str) -> Any:
    """Safely extract the default value of a dataclass field.

    4th-review fix (issue 2.2): handles both ``default_factory`` and
    static ``default`` field definitions. Returns ``None`` if the field
    has no default (which would surface a clear error from the dataclass
    constructor rather than the opaque ``TypeError: 'MISSING' object is
    not callable`` from a raw ``.default_factory()`` call).

    9th-review fix (F-01): the previous implementation had the function
    body split across two locations. Lines 628-630 contained only the
    early-return guard, making the function ALWAYS return None. The
    actual default-extraction logic (lines 749-755) was unreachable
    dead code after the ``return result`` of _coerce_int. This fix
    consolidates the logic into one reachable location.
    """
    fields_obj = getattr(datacls, "__dataclass_fields__", None)
    if fields_obj is None or field_name not in fields_obj:
        return None
    f = fields_obj[field_name]
    # default_factory takes precedence over default per the dataclass spec.
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    if f.default is not dataclasses.MISSING:
        return f.default
    return None


# ---------------------------------------------------------------------------
# 8th-review fix (F-05, F-06): safe coercion helpers for config deserialization
# ---------------------------------------------------------------------------
# F-05: bool('False') returns True in Python because any non-empty
# string is truthy. Config deserialization from JSON/YAML often produces
# string-valued booleans (e.g., 'False' for use_small_angle_approximation),
# which bool() silently converts to True, inverting the user's intent.
# F-06: float(None) and int(None) raise opaque TypeError instead
# of the framework's ParameterValidationError, making deserialized configs
# with None values (common in JSON) crash unhelpfully.
# These helpers parse string booleans correctly and raise clear errors on None.


def _coerce_bool(v: Any, default: bool) -> bool:
    """Coerce a value to bool safely, handling string-to-bool pitfalls.

    8th-review fix (F-05): bool('False') returns True because
    bool checks truthiness, not semantic content. This helper explicitly
    parses string representations of booleans.

    9th-review fix (F-12): removed the redundant ``isinstance(v, (np.bool_,))``
    check at the int/float branch. In numpy >= 1.20, ``np.bool_`` subclasses
    ``bool``, so the ``isinstance(v, bool)`` check already catches it. The old
    np.bool_ branch was unreachable dead code. Now both Python bool and numpy
    bool are handled at the same level.

    Rules:
      * None -> returns default
      * bool (Python or numpy) -> returns bool(v) directly
      * str -> v.lower() in ('true', '1', 'yes', 'on')
      * int / float / np.integer / np.floating -> bool(v)
      * Other types -> ParameterValidationError
    """
    if v is None:
        return default
    # 9th-review fix (F-12): handle both Python bool and numpy bool together.
    # In numpy >= 1.20, np.bool_ subclasses Python bool, so isinstance(v, bool)
    # catches both. For older numpy, add explicit np.bool_ check.
    if isinstance(v, bool) or isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, str):
        return v.lower() in ('true', '1', 'yes', 'on')
    if isinstance(v, (int, float, np.integer, np.floating)):
        return bool(v)
    raise ParameterValidationError(
        f'Cannot coerce value to bool: got {v!r} of type {type(v).__name__}.',
        param_name='bool_coercion',
        param_value=v,
    )


def _coerce_float(v: Any, default: float, name: str) -> float:
    """Coerce a value to float, raising clear error on None.

    8th-review fix (F-06): float(None) raises an opaque TypeError.
    This helper checks for None first and raises
    ParameterValidationError with a clear message.
    """
    if v is None:
        raise ParameterValidationError(
            f'{name} must be a real number, got None. '
            f'If this value came from a serialized config, ensure the key '
            f'is present and not null.',
            param_name=name,
            param_value=None,
        )
    try:
        result = float(v)
    except (TypeError, ValueError) as exc:
        raise ParameterValidationError(
            f'{name} must be a real number, got {v!r} of type '
            f'{type(v).__name__}.',
            param_name=name,
            param_value=v,
        ) from exc
    if not np.isfinite(result):
        raise ParameterValidationError(
            f'{name} must be finite, got {result}.',
            param_name=name,
            param_value=result,
        )
    return result


def _coerce_int(v: Any, default: int, name: str) -> int:
    """Coerce a value to int, rejecting bool and None.

    8th-review fix (F-06): int(None) raises an opaque TypeError.
    F-05 companion: rejects bool values explicitly.
    """
    if v is None:
        raise ParameterValidationError(
            f'{name} must be an integer, got None. '
            f'If this value came from a serialized config, ensure the key '
            f'is present and not null.',
            param_name=name,
            param_value=None,
        )
    if isinstance(v, bool):
        raise ParameterValidationError(
            f'{name} must be an integer, got bool {v!r}. '
            f'Boolean values are not valid integers.',
            param_name=name,
            param_value=v,
        )
    if isinstance(v, np.bool_):
        raise ParameterValidationError(
            f'{name} must be an integer, got numpy bool {v!r}.',
            param_name=name,
            param_value=v,
        )
    try:
        result = int(v)
    except (TypeError, ValueError) as exc:
        raise ParameterValidationError(
            f'{name} must be an integer, got {v!r} of type '
            f'{type(v).__name__}.',
            param_name=name,
            param_value=v,
        ) from exc
    return result


# ---------------------------------------------------------------------------
# OpticalSource
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OpticalSource:
    """
    Generic optical source backed by :class:`OpticalSourceConfig`.

    The source produces photon numbers according to:

    1. Pulse-type selection (signal vs. decoys, with their probabilities).
    2. Base mean photon number ``mu_base`` for the selected pulse.
    3. MZM attenuation / electrical-driver noise: ``mzm.apply(...)``.
    4. Channel split: ``mu / N_channels`` (energy-conserving uniform split).
    5. SCM sideband factor: ``mu * J1(m)^2`` (or small-angle approximation).
    6. Intensity jitter (multiplicative; see :meth:`_apply_intensity_jitter`).
    7. Photon-number sampling: Poisson or thermal/geometric.
    8. Emission-failure thinning: with prob ``1 - p_emit``, photons -> 0.

    Steps 3-6 are classical intensity imperfections applied to the base mu.
    Step 7 samples the photon number from the (possibly imperfect) mean.
    Step 8 models source emission failure as a *thinning* (loss) operation.

    For :class:`DensityMatrixSource`, steps 3-7 are replaced by direct
    sampling from the diagonal of the density matrix. The classical
    imperfections (MZM, channel split, SCM, jitter, emission failure) are
    **not** applied in that subclass because the density matrix is assumed
    to already encode the emitted state (post-imperfection). This is
    documented in :class:`DensityMatrixSource`.
    """

    config: OpticalSourceConfig
    security_metadata: Optional[SourceSecurityMetadata] = None

    # 7th-review fix (issue I.4): per-instance SCM sideband count.
    # Previously, ``SCM_NUM_SIDEBANDS`` was a module-level constant, which
    # introduced thread-safety violations and race conditions in
    # heterogeneous optical-network simulations where different links
    # require single-sideband (SSB, n=1) and double-sideband (DSB, n=2)
    # detection simultaneously. The per-instance field reads the module
    # constant as its default (via ``default_factory``) so existing
    # behavior is preserved, but each source can now override the value
    # without affecting other sources in the same process.
    scm_num_sidebands: int = field(
        default_factory=lambda: SCM_NUM_SIDEBANDS,
        repr=False,
        compare=False,
    )

    _base_mus_cache: np.ndarray = field(init=False, repr=False, compare=False)

    # 9th-review fix (F-04): renamed from phase_randomized to
    # phase_randomization_assumed. This flag is a CLAIM about the
    # user's external experimental setup, NOT an implementation
    # feature of this module. The module does NOT perform actual
    # per-pulse phase randomization (sampling phi ~ Uniform[0, 2pi]
    # per pulse). Decoy-state BB84 security proofs REQUIRE phase
    # randomization so each pulse is a diagonal mixture of Fock
    # states. When this flag is True, the user asserts that their
    # external setup provides this. When False, the Poisson/thermal
    # diagonal sampling is NOT security-proof-compliant.
    phase_randomization_assumed: bool = field(
        default_factory=lambda: PHASE_RANDOMIZATION_ASSUMED_DEFAULT,
        repr=False,
        compare=False,
    )
    # 9th-review fix (F-03): scm_enabled flag decouples the physical SCM
    # switch from the continuous modulation_index parameter. When False,
    # SCM is disabled, the modulation_index is ignored, and mu is
    # interpreted as the post-SCM (detector-relevant) photon number
    # (scaling = 1.0). When True, SCM is enabled, mu is interpreted as
    # the total emitted pulse energy, and the scaling factor
    # (n_sb * J1(m)^2) determines the fraction reaching the detector.
    # This eliminates the semantic discontinuity at m=0 where the
    # *meaning* of mu silently switches between "post-SCM" and "total
    # energy" interpretations.
    scm_enabled: bool = field(
        default_factory=lambda: SCM_ENABLED_DEFAULT,
        repr=False,
        compare=False,
    )
    _probabilities_cache: np.ndarray = field(init=False, repr=False, compare=False)
    _driver_voltage_noise_std: float = field(init=False, repr=False, compare=False, default=0.0)

    # Stateful block tracking for ADVERSARIAL_BLOCK error model (issue 2.6 in
    # 3rd review). The previous stateless design sampled a new lognormal
    # factor per call, which broke block correlation when the simulation
    # was chunked into small batches (e.g., n=10 per call with block_size=1000
    # would sample a new factor every 10 pulses, neutralizing the configured
    # correlation length). These fields track the global pulse counter and
    # the current block's factor across calls, so block boundaries are
    # respected regardless of call granularity.
    _adversarial_block_counter: int = field(init=False, repr=False, compare=False, default=0)
    _adversarial_current_factor: float = field(init=False, repr=False, compare=False, default=1.0)

    # 7th-review fix (issue I.1): persistent cache mapping block_id ->
    # lognormal factor, used when ``pulse_positions`` is provided to
    # ``_apply_intensity_jitter``. The previous implementation sampled
    # fresh factors for ALL blocks on every call, breaking block correlation
    # across non-sequential access patterns (e.g., querying signal pulses
    # in one call and decoy pulses in the next, where both calls reference
    # the same physical block). The cache ensures that the same physical
    # block always receives the same fluctuation factor, regardless of
    # how the caller chunks the queries.
    _adversarial_block_factor_cache: Dict[int, float] = field(
        init=False, repr=False, compare=False, default_factory=_BoundedLRUCache
    )

    # 7th-review fix (issue III.2): flag indicating that ``__post_init__``
    # has completed. Used by the ``__setattr__`` override to detect
    # post-init reassignment of ``config`` (which would invalidate the
    # internal caches without triggering a re-computation). Set to True
    # at the end of ``__post_init__`` via ``object.__setattr__``.
    _post_init_done: bool = field(init=False, repr=False, compare=False, default=False)

    # 4th-review fix (issue 4.3): pre-computed name-to-index mapping for
    # O(1) pulse lookups. The previous implementation performed an O(N)
    # linear scan on every call to get_pulse_config_by_name /
    # get_pulse_index_by_name. Since pulse names are validated for
    # uniqueness upon initialization, a dict-based lookup is safe and
    # eliminates the linear-scan overhead in repetitive query patterns
    # (e.g., DM sampling, security-metadata bookkeeping, per-pulse-type
    # analytics over long simulations).
    _pulse_name_to_index: Dict[str, int] = field(init=False, repr=False, compare=False, default_factory=dict)

    # 9th-review fix (F-07): explicit user acknowledgment for compound
    # fluctuations. Thermal (Bose-Einstein) statistics have intrinsic
    # super-Poissonian variance (Var(n) = mu^2 + mu). Adding classical
    # multiplicative jitter compounds fluctuations beyond what any known
    # single-mode light source produces: Var(n_jittered) = Var(n)*(1+sigma^2)
    # + mu^2*sigma^2. This field requires the user to explicitly acknowledge
    # this unphysical combination rather than just receiving a warning.
    allow_compound_fluctuations: bool = field(default=False, repr=False, compare=False)

    # ------------------------------------------------------------------
    # 7th-review fix (issue III.2): post-init immutability for ``config``.
    # ------------------------------------------------------------------
    # The internal NumPy caches (``_base_mus_cache``, ``_probabilities_cache``,
    # ``_pulse_name_to_index``, and in subclasses also ``_rho_tensor`` /
    # ``_number_probs_table``) are computed from ``self.config`` at
    # construction time and are NOT recomputed if ``self.config`` is
    # mutated post-init. Previously, reassigning ``source.config =
    # new_config`` would silently invalidate every cached array without
    # raising, leading to subtle desynchronization bugs.
    #
    # ``__setattr__`` is overridden to raise ``ParameterValidationError``
    # when ``config`` is reassigned after ``__post_init__`` completes
    # (signalled by ``_post_init_done``). The override is bypassed during
    # ``__init__`` (when ``_post_init_done`` is still False) so the
    # initial assignment from the dataclass-generated constructor still
    # works.
    #
    # In-place mutation of ``self.config.pulse_configs`` (a tuple) is
    # structurally prevented because tuples are immutable; in-place
    # mutation of individual ``PulseTypeConfig`` instances is NOT
    # prevented (this would require modifying ``OpticalSourceConfig`` in
    # datatypes.py), but is documented as unsupported.
    def __setattr__(self, name: str, value: Any) -> None:
        if (
            name == "config"
            and getattr(self, "_post_init_done", False)
        ):
            raise ParameterValidationError(
                "OpticalSource.config is immutable after construction. "
                "Reassigning it would invalidate the internal caches "
                "(_base_mus_cache, _probabilities_cache, "
                "_pulse_name_to_index, _rho_tensor, _number_probs_table) "
                "without triggering a re-computation, leading to subtle "
                "desynchronization bugs. Construct a new OpticalSource "
                "instance with the new config instead.",
                param_name="config",
            )
        object.__setattr__(self, name, value)

    def __deepcopy__(self, memo):
        """Deep-copy support that bypasses the __setattr__ guard.

        8th-review fix (F-12): deepcopy calls __setattr__('config', ...)
        after the copy is created, which triggers the immutability guard.
        This implementation bypasses the guard by using object.__setattr__.
        """
        import copy as _copy
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for slot in getattr(cls, '__slots__', ()):
            val = getattr(self, slot)
            object.__setattr__(result, slot, _copy.deepcopy(val, memo))
        object.__setattr__(result, '_post_init_done', True)
        return result

    def __getstate__(self):
        """Pickle support: return state dict from slots."""
        return {slot: getattr(self, slot) for slot in self.__slots__}

    def __setstate__(self, state):
        """Pickle support: restore state dict to slots."""
        for slot, value in state.items():
            object.__setattr__(self, slot, value)
        object.__setattr__(self, '_post_init_done', True)

    # ------------------------------------------------------------------
    # Construction & validation
    # ------------------------------------------------------------------
    #
    # Chaining strategy (issue 4 in 2nd review; 4th-review issue 2.1):
    # ``@dataclass(slots=True)`` subclasses in Python 3.12 cannot reliably
    # use the zero-argument ``super()`` form because the dataclass
    # decorator rebuilds the class with ``__slots__``, which breaks the
    # ``__class__`` cell used by the implicit ``super()``. Rather than
    # asking every subclass to remember to call
    # ``OpticalSource.__post_init__(self)`` (fragile under future
    # renames or hierarchy reshaping), we use a hook pattern:
    #
    #   * ``__post_init__`` is the single entry point. It calls the
    #     subclass-overridable ``_post_init_hook`` *after* the base
    #     validation runs. Subclasses override ``_post_init_hook`` (not
    #     ``__post_init__``), so the base validation is always invoked
    #     automatically regardless of subclass structure.
    #
    #   * Subclasses that need to run code *before* base validation
    #     override ``_pre_init_hook`` instead.
    #
    # **Multi-level inheritance (4th-review issue 2.1):** Subclasses that
    # override ``_pre_init_hook`` or ``_post_init_hook`` MUST explicitly
    # chain to the parent implementation via
    # ``super()._pre_init_hook()`` / ``super()._post_init_hook()`` at the
    # appropriate point in their override. Failure to do so will silently
    # bypass the parent's initialization logic (such as DM tensor
    # construction in ``DensityMatrixSource._post_init_hook`` or
    # statistics-type validation in ``PoissonSource._post_init_hook``).
    # This is the standard Python cooperative-multilevel-dispatch contract
    # for hook methods; it is documented here rather than enforced at
    # runtime because runtime enforcement would require metaclass magic
    # that is incompatible with ``@dataclass(slots=True)``.
    #
    # This eliminates the fragile explicit ``Parent.__post_init__(self)``
    # pattern while remaining forward-compatible with deeper hierarchies.

    def __post_init__(self) -> None:
        # Pre-init hook (subclass-customizable).
        # 8th-review fix (F-28): wrap in try/finally so _post_init_done
        # is set even if _pre_init_hook, _validate_base_invariants, or
        # _post_init_hook raises. Previously, an exception left
        # _post_init_done=False, bypassing the immutability guard.
        try:
            self._pre_init_hook()
            self._validate_base_invariants()
            self._post_init_hook()
        finally:
            # Mark construction as complete UNCONDITIONALLY.
            object.__setattr__(self, '_post_init_done', True)

    def _pre_init_hook(self) -> None:
        """Override in subclasses to run code BEFORE base validation.

        The default implementation is a no-op.

        Multi-level inheritance (4th-review issue 2.1)
        ----------------------------------------------
        If you override this in a subclass of a subclass (e.g., a class
        derived from :class:`DensityMatrixSource`), you MUST chain to the
        parent implementation via
        ``ParentClass._pre_init_hook(self)`` at the start of your override
        (using the **explicit class-reference form**, not zero-argument
        ``super()``, because ``@dataclass(slots=True)`` breaks the
        implicit ``super()`` in Python 3.12). Failing to chain will
        silently skip the parent's validation, leading to subtle
        construction-time bugs.
        """
        # Default: no-op.
        return None

    def _post_init_hook(self) -> None:
        """Override in subclasses to run code AFTER base validation.

        The default implementation is a no-op.

        Multi-level inheritance (4th-review issue 2.1)
        ----------------------------------------------
        If you override this in a subclass of a subclass (e.g., a class
        derived from :class:`DensityMatrixSource`), you MUST chain to the
        parent implementation via
        ``ParentClass._post_init_hook(self)`` at the start of your
        override (using the **explicit class-reference form**, not
        zero-argument ``super()``, because ``@dataclass(slots=True)``
        breaks the implicit ``super()`` in Python 3.12). Failing to chain
        will silently skip the parent's initialization, leaving internal
        caches (``_rho_tensor``, ``_number_probs_table``) unset.
        """
        # Default: no-op.
        return None

    def _validate_base_invariants(self) -> None:
        """Validate the invariants shared by all :class:`OpticalSource` subclasses.

        This method is called by :meth:`__post_init__` and is **not** meant
        to be overridden by subclasses. Subclasses should instead override
        :meth:`_pre_init_hook` and :meth:`_post_init_hook`.

        Note on DRY (issue 3.4 in 3rd review): the pulse-name uniqueness
        and probability-normalization checks performed here overlap with
        the validation in :class:`PulseEnsembleConfig.__post_init__`. We
        keep these checks in ``_validate_base_invariants`` as a defensive
        layer because:

          1. Users can construct an :class:`OpticalSource` directly with an
             :class:`OpticalSourceConfig` whose ``pulse_configs`` tuple did
             not go through :func:`_coerce_pulse_configs` (and thus not
             through :class:`PulseEnsembleConfig` validation).

          2. The non-pulse-field validation (source_rate, N_channels,
             adversarial_block_size, etc.) must be performed here regardless,
             so the pulse checks are co-located for clarity.

        The :func:`_coerce_pulse_configs` helper no longer constructs a
        :class:`PulseEnsembleConfig` separately (it just coerces the types),
        eliminating the redundant double-construction that the reviewer
        flagged.
        """
        if not isinstance(self.config, OpticalSourceConfig):
            raise ParameterValidationError(
                "config must be an OpticalSourceConfig instance.",
                param_name="config",
                param_value=type(self.config).__name__,
            )

        # Validate pulse configs.
        pulse_configs = self.config.pulse_configs
        if not pulse_configs:
            raise ParameterValidationError(
                "pulse_configs must be a non-empty sequence.",
                param_name="pulse_configs",
            )

        # Validate pulse-name uniqueness (also done by PulseEnsembleConfig,
        # but kept here as a defensive check — see method docstring).
        names: List[str] = [pc.name for pc in pulse_configs]
        if len(set(names)) != len(names):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ParameterValidationError(
                f"pulse_configs names must be unique. Duplicate names: {duplicates}.",
                param_name="pulse_configs",
            )

        # Validate mean photon numbers and probabilities.
        mus = np.array(
            [pc.mean_photon_number for pc in pulse_configs], dtype=np.float64
        )
        probs = np.array(
            [pc.probability for pc in pulse_configs], dtype=np.float64
        )

        if not np.all([is_finite_non_negative(mu) for mu in mus]):
            raise ParameterValidationError(
                "All mean photon numbers must be finite and non-negative.",
                param_name="pulse_configs",
            )

        if not np.all([is_valid_probability(p) for p in probs]):
            raise ParameterValidationError(
                "All pulse probabilities must be valid probabilities in [0, 1].",
                param_name="pulse_configs",
            )

        if not is_close(float(np.sum(probs)), 1.0):
            raise ParameterValidationError(
                f"pulse probabilities must sum to 1.0 (got {float(np.sum(probs))}).",
                param_name="pulse_configs",
            )

        # Validate source_rate (issue 2.11).
        # 5th-review fix (issue 4.1): reject bool explicitly, since
        # isinstance(True, int) is True in Python.
        source_rate = self.config.source_rate
        if isinstance(source_rate, bool) or not isinstance(
            source_rate, (int, float, np.integer, np.floating)
        ):
            raise ParameterValidationError(
                "source_rate must be a real number.",
                param_name="source_rate",
                param_value=type(source_rate).__name__,
            )
        if not np.isfinite(source_rate) or source_rate <= 0:
            raise ParameterValidationError(
                f"source_rate must be finite and positive (got {source_rate}).",
                param_name="source_rate",
            )

        # Validate N_channels.
        # 5th-review fix (issue 4.1): reject bool via _is_integer.
        if not _is_integer(self.config.N_channels):
            raise ParameterValidationError(
                "N_channels must be an integer.",
                param_name="N_channels",
                param_value=type(self.config.N_channels).__name__,
            )
        if self.config.N_channels < 1:
            raise ParameterValidationError(
                f"N_channels must be >= 1 (got {self.config.N_channels}).",
                param_name="N_channels",
            )

        # 8th-review fix (F-63): validate scm_num_sidebands >= 1.
        scm_nsb = getattr(self, 'scm_num_sidebands', SCM_NUM_SIDEBANDS)
        if not _is_integer(scm_nsb):
            raise ParameterValidationError(
                f'scm_num_sidebands must be an integer (got {type(scm_nsb).__name__}).',
                param_name='scm_num_sidebands',
                param_value=scm_nsb,
            )
        if scm_nsb < 1:
            raise ParameterValidationError(
                f'scm_num_sidebands must be >= 1 (got {scm_nsb}).',
                param_name='scm_num_sidebands',
                param_value=scm_nsb,
            )

        # 9th-review fix (F-03): validate scm_enabled flag and its
        # relationship with modulation_index. When scm_enabled=True,
        # modulation_index must be > 0 (no SCM with zero modulation).
        # When scm_enabled=False, modulation_index is ignored (a warning
        # is logged if it is > 0, to catch likely configuration errors).
        # NOTE: mod_idx is assigned here (before the SCM validation block)
        # because it is used in the scm_enabled checks below.  The later
        # modulation_index validation block (range check) also uses it.
        mod_idx = self.config.modulation_index
        scm_en = getattr(self, 'scm_enabled', SCM_ENABLED_DEFAULT)
        if not isinstance(scm_en, (bool, np.bool_)):
            raise ParameterValidationError(
                f'scm_enabled must be a boolean (got {type(scm_en).__name__}).',
                param_name='scm_enabled',
                param_value=scm_en,
            )
        if scm_en and mod_idx <= NUMERIC_ABS_TOL:
            raise ParameterValidationError(
                f"When scm_enabled=True, modulation_index must be > 0 "
                f"(got {mod_idx}). SCM with zero modulation depth means "
                f"no sideband signal -- set scm_enabled=False if you want "
                f"mu interpreted as the post-SCM photon number.",
                param_name='modulation_index',
                param_value=mod_idx,
            )
        if not scm_en and mod_idx > NUMERIC_ABS_TOL:
            logger.warning(
                "scm_enabled=False but modulation_index=%g > 0: "
                "modulation_index is IGNORED when SCM is disabled. "
                "Set scm_enabled=True to enable SCM sideband scaling, "
                "or set modulation_index=0 to suppress this warning.",
                mod_idx,
            )

        # Validate adversarial_block_size.
        # 5th-review fix (issue 4.1): reject bool via _is_integer.
        if not _is_integer(self.config.adversarial_block_size):
            raise ParameterValidationError(
                "adversarial_block_size must be an integer.",
                param_name="adversarial_block_size",
                param_value=type(self.config.adversarial_block_size).__name__,
            )
        if self.config.adversarial_block_size < 1:
            raise ParameterValidationError(
                f"adversarial_block_size must be >= 1 "
                f"(got {self.config.adversarial_block_size}).",
                param_name="adversarial_block_size",
            )

        # Validate ideal_emission_probability.
        p_emit = self.config.ideal_emission_probability
        if not is_valid_probability(p_emit):
            raise ParameterValidationError(
                f"ideal_emission_probability must be in [0, 1] (got {p_emit}).",
                param_name="ideal_emission_probability",
            )

        # Validate intensity_jitter.
        jitter = self.config.intensity_jitter
        if not np.isfinite(jitter) or jitter < 0:
            raise ParameterValidationError(
                f"intensity_jitter must be finite and >= 0 (got {jitter}).",
                param_name="intensity_jitter",
            )

        # Validate modulation_index (mod_idx was assigned earlier for
        # the SCM validation block above).
        if not np.isfinite(mod_idx) or mod_idx < 0:
            raise ParameterValidationError(
                f"modulation_index must be finite and >= 0 (got {mod_idx}).",
                param_name="modulation_index",
            )

        # 9th-review fix (F-07): require explicit user acknowledgment for
        # thermal + jitter compound fluctuations, rather than just a warning.
        # The combined variance Var(n_jittered) = Var(n)*(1+sigma^2) +
        # mu^2*sigma^2 exceeds any known single-mode light source.
        if (
            self.config.statistics_type == SourceStatisticsType.THERMAL
            and jitter > NUMERIC_ABS_TOL
        ):
            if not getattr(self, 'allow_compound_fluctuations', False):
                raise ParameterValidationError(
                    f"THERMAL statistics with non-zero intensity_jitter={jitter}: "
                    f"the model applies BOTH intrinsic Bose-Einstein fluctuations "
                    f"(Var(n) = mu^2 + mu) AND classical multiplicative laser "
                    f"jitter, which compounds fluctuations beyond what any known "
                    f"single-mode light source produces. This is unphysical for "
                    f"an ideal thermal source. Set allow_compound_fluctuations=True "
                    f"to explicitly acknowledge this, or set intensity_jitter=0 "
                    f"for pure thermal light.",
                    param_name="intensity_jitter",
                    param_value=jitter,
                )
            else:
                logger.warning(
                    "THERMAL statistics with non-zero intensity_jitter=%g "
                    "(allow_compound_fluctuations=True): the model applies "
                    "BOTH intrinsic Bose-Einstein fluctuations AND classical "
                    "multiplicative laser jitter, which compounds fluctuations "
                    "beyond what any known single-mode light source produces.",
                    jitter,
                )

        # 5th-review fix (issue 1.2): warn when RANDOM_GAUSSIAN is combined
        # with large intensity_jitter. The Gaussian reflection model
        # (mu_new = mu * |1 + jitter * z|) introduces a positive mean bias
        # E[|X|] - E[X] = 2*|E[X|X<0]|*P(X<0) that grows with jitter
        # (e.g., ~0.8% at jitter=0.5, ~16% at jitter=1.0). For small
        # jitter (< 0.1), the bias is negligible (P(X<0) is astronomically
        # small). For large jitter, the bias may affect security analyses
        # that assume mean-preserving intensity fluctuations. Users who
        # need strictly positive intensity with exact mean preservation
        # should request a dedicated RANDOM_LOGNORMAL error model (not
        # yet implemented in the SourceErrorModel enum).
        # 9th-review fix (F-27): moved to module level for consistency.

        # 9th-review fix (F-04): renamed flag from phase_randomized to
        # phase_randomization_assumed. The flag is a CLAIM about the user's
        # external setup, NOT an implementation feature.
        if not getattr(self, 'phase_randomization_assumed', True):
            logger.warning(
                'phase_randomization_assumed=False: this source is NOT '
                'suitable for decoy-state BB84 security analysis. The '
                'phase_randomization_assumed flag is a CLAIM about your '
                'external setup -- this module does NOT perform actual '
                'per-pulse phase randomization. Security proofs require '
                'uniformly random phase per pulse so each pulse is a '
                'diagonal mixture of Fock states. Without phase '
                'randomization, the gain/QBER/yield estimates are not '
                'security-proof-compliant.',
            )
        if (
            self.config.error_model == SourceErrorModel.RANDOM_GAUSSIAN
            and jitter > GAUSSIAN_LARGE_JITTER_THRESHOLD
        ):
            # 9th-review fix (F-02): the reflection model now includes
            # analytical bias correction. The remaining bias after correction
            # is negligible for all practical jitter values (< 1e-6 relative
            # error). This warning is retained for informational purposes.
            logger.warning(
                "RANDOM_GAUSSIAN with large intensity_jitter=%g (>= %.1f): "
                "the reflection model (mu_new = mu * |1 + jitter * z|) "
                "now includes analytical mean-bias correction "
                "(dividing by E[|1 + jitter * z|]). The corrected model "
                "preserves the mean photon number to within ~1e-6 relative "
                "error. For small jitter (< 0.1) the correction is a no-op. "
                "If you need strictly positive intensity with exact mean "
                "preservation, request a dedicated RANDOM_LOGNORMAL error "
                "model (not yet in the SourceErrorModel enum).",
                jitter,
                GAUSSIAN_LARGE_JITTER_THRESHOLD,
            )

        # Validate mzm / electrical_noise duck-typed contracts (issue 2.12).
        # 5th-review fix (issue 4.2): use ``callable(getattr(...))`` instead
        # of ``hasattr(...)``. The previous ``hasattr`` check would accept
        # any attribute named ``apply`` or ``total_voltage_std``, even if it
        # were a non-callable attribute (e.g., a string or int). The call
        # would then fail at runtime with a ``TypeError: 'str' object is not
        # callable`` instead of a clear ``ParameterValidationError`` at
        # construction time. Using ``callable(getattr(obj, name, None))``
        # ensures the attribute exists AND is callable before we accept it.
        if self.config.mzm is None or not callable(
            getattr(self.config.mzm, "apply", None)
        ):
            raise ParameterValidationError(
                "mzm must implement an 'apply(base_mus, voltage_noise_std, rng)' method.",
                param_name="mzm",
                param_value=type(self.config.mzm).__name__ if self.config.mzm is not None else None,
            )
        if (
            self.config.electrical_noise is None
            or not callable(getattr(self.config.electrical_noise, "total_voltage_std", None))
        ):
            raise ParameterValidationError(
                "electrical_noise must implement a 'total_voltage_std()' method.",
                param_name="electrical_noise",
                param_value=(
                    type(self.config.electrical_noise).__name__
                    if self.config.electrical_noise is not None
                    else None
                ),
            )

        # Cache driver voltage noise std (used by mzm.apply downstream).
        try:
            self._driver_voltage_noise_std = float(
                self.config.electrical_noise.total_voltage_std()
            )
        except (TypeError, ValueError) as exc:
            raise ParameterValidationError(
                f"electrical_noise.total_voltage_std() returned an invalid value: {exc}",
                param_name="electrical_noise",
            ) from exc

        # Freeze caches.
        mus.setflags(write=False)
        probs.setflags(write=False)
        self._base_mus_cache = mus
        self._probabilities_cache = probs

        # 4th-review fix (issue 4.3): build the O(1) name-to-index lookup
        # cache. We use a dict comprehension over the (already-validated
        # unique) pulse names. This eliminates the O(N) linear scan in
        # get_pulse_config_by_name / get_pulse_index_by_name, which is
        # called frequently during DM sampling, security-metadata
        # bookkeeping, and per-pulse-type analytics.
        self._pulse_name_to_index = {
            pc.name: i for i, pc in enumerate(self.pulse_configs)
        }

        # Initialize stateful block tracking for ADVERSARIAL_BLOCK (issue 2.6
        # in 3rd review). These are reset on every construction/validation
        # cycle to ensure reproducibility from a clean state.
        self._adversarial_block_counter = 0
        self._adversarial_current_factor = 1.0

        if self.security_metadata is not None:
            self.security_metadata.ensure_defaults()

    # ------------------------------------------------------------------
    # Pass-through properties
    # ------------------------------------------------------------------

    @property
    def pulse_configs(self) -> Tuple[PulseTypeConfig, ...]:
        """Tuple of pulse-type configurations (signal + decoys)."""
        return self.config.pulse_configs

    @property
    def pulse_period_ns(self) -> float:
        """Pulse period in nanoseconds.

        Notes
        -----
        For ``N_channels > 1`` or non-zero SCM, the *per-channel* pulse
        period may differ from this aggregate value. This property returns
        the **aggregate** source period; downstream code is responsible for
        any per-channel timing interpretation.
        """
        return self.config.pulse_period_ns

    @property
    def source_rate(self) -> float:
        """Aggregate source repetition rate in Hz (must be finite > 0)."""
        return self.config.source_rate

    @property
    def is_bidirectional(self) -> bool:
        """Metadata-only flag indicating bidirectional source use.

        Notes
        -----
        This flag is **metadata only**; it does not affect photon-number
        sampling in this class. Downstream protocol code may consult it to
        decide on bidirectional vs. unidirectional security analysis.

        7th-review fix (issue I.5): this property is *deprecated* because
        it represents **improper domain mixing** between the source layer
        (photon-number statistics) and the channel/protocol layer
        (transmission architecture). Whether an optical system operates
        bidirectionally (e.g., plug-and-play architectures) or
        unidirectionally is an architectural property of the quantum
        transmission channel and protocol execution layer, not an
        intrinsic physical property of the photon source. Including it
        at the source layer invites users to mistakenly believe it
        affects photon generation, which it does not.

        A ``FutureWarning`` is emitted on each access to alert users that
        the property will be removed in a future refactor (where it will
        live on a protocol-level config object). Callers who need the
        value should access ``self.config.is_bidirectional`` directly
        (suppressing the warning) or migrate to the future
        protocol-level config when it becomes available.

        The property is retained for backwards compatibility; existing
        callers that suppress the warning will continue to work.
        """
        warnings.warn(
            "OpticalSource.is_bidirectional is deprecated and will be "
            "removed in a future refactor. The flag represents a "
            "channel/protocol-layer property (bidirectional vs. "
            "unidirectional transmission architecture) that has no "
            "behavioral effect on photon-number generation; including "
            "it at the source layer is improper domain mixing. Access "
            "self.config.is_bidirectional directly if you need the "
            "value, or migrate to a future protocol-level config when "
            "available.",
            FutureWarning,
            stacklevel=2,
        )
        return self.config.is_bidirectional

    @property
    def statistics_type(self) -> SourceStatisticsType:
        """Photon-number statistics type (POISSON or THERMAL)."""
        return self.config.statistics_type

    @property
    def intensity_jitter(self) -> float:
        """Relative intensity jitter (std / mean), dimensionless, >= 0."""
        return self.config.intensity_jitter

    @property
    def error_model(self) -> SourceErrorModel:
        """Intensity-fluctuation error model."""
        return self.config.error_model

    @property
    def modulation_index(self) -> float:
        """SCM phase-modulation index ``m`` (>= 0)."""
        return self.config.modulation_index

    @property
    def N_channels(self) -> int:
        """Number of optical channels for energy-splitting (>= 1)."""
        return self.config.N_channels

    @property
    def use_small_angle_approximation(self) -> bool:
        """Whether to use ``J1(m) ~ m/2`` for small ``m``."""
        return self.config.use_small_angle_approximation

    @property
    def use_linear_modulation_approximation(self) -> bool:
        """Whether to force the linear ``(m^2)/4`` approximation for all ``m``."""
        return self.config.use_linear_modulation_approximation

    @property
    def ideal_emission_probability(self) -> float:
        """Probability ``p_emit`` that the source actually emits a pulse.

        With probability ``1 - p_emit``, the source fails to emit and the
        photon count is set to 0 (a thinning / loss model). With probability
        ``p_emit``, the photon count is sampled from the configured
        distribution. This is *not* a thermalization switch.
        """
        return self.config.ideal_emission_probability

    @property
    def adversarial_block_size(self) -> int:
        """Block size (in pulses) for :attr:`ADVERSARIAL_BLOCK` correlation."""
        return self.config.adversarial_block_size

    @property
    def mzm(self):
        """Modulator object (must implement ``apply``)."""
        return self.config.mzm

    @property
    def electrical_noise(self):
        """Electrical-driver noise object (must implement ``total_voltage_std``)."""
        return self.config.electrical_noise

    @property
    def pulse_ensemble(self) -> PulseEnsembleConfig:
        """Convenience view of pulse configs as a :class:`PulseEnsembleConfig`."""
        return PulseEnsembleConfig(pulses=list(self.pulse_configs))

    @property
    def intensity_config(self) -> IntensityConfig:
        """Build an :class:`IntensityConfig` from pulse configs.

        6th-review fix (issue 14): instead of unconditionally treating
        ``pulse_configs[0]`` as the signal, this property now inspects
        pulse names. If a pulse named ``"signal"`` exists, it is used as
        the signal; otherwise, the first pulse is used (preserving
        backwards compatibility) and a warning is logged. This handles
        deserialized configs where index 0 may not be the signal pulse.
        Vacuum pulses (``name.startswith("vacuum")`` or ``mu == 0``) and
        decoy pulses (other names) are mapped to the decoys list.

        Sources built via :meth:`from_intensity_config` use role-based
        naming (first node = "signal"), so this property round-trips
        correctly. Sources built via :meth:`from_pulse_ensemble` or
        :meth:`from_dict` may have arbitrary pulse names; the name-based
        lookup ensures the correct pulse is selected as the signal
        regardless of config ordering.
        """
        if not self.pulse_configs:
            raise ParameterValidationError(
                "Cannot build IntensityConfig from empty pulse_configs.",
                param_name="pulse_configs",
            )

        # 6th-review fix (issue 14): look for a pulse named "signal" first.
        signal_idx = None
        for i, pc in enumerate(self.pulse_configs):
            if pc.name == "signal":
                signal_idx = i
                break

        if signal_idx is None:
            # No pulse named "signal"; fall back to index 0 with a warning.
            # This preserves backwards compatibility for configs that rely
            # on the index-0 convention.
            logger.warning(
                "intensity_config: no pulse named 'signal' found in "
                "pulse_configs (names: %s). Falling back to index 0 as the "
                "signal pulse. This may misrepresent the QKD protocol "
                "roles if index 0 is not intended to be the signal. "
                "Rename the signal pulse to 'signal' to suppress this "
                "warning, or use get_pulse_config_by_name for explicit "
                "role-based access.",
                [pc.name for pc in self.pulse_configs],
            )
            signal_idx = 0

        signal_pc = self.pulse_configs[signal_idx]
        signal = IntensityNode(
            mu=signal_pc.mean_photon_number,
            probability=signal_pc.probability,
        )
        decoys = [
            IntensityNode(mu=pc.mean_photon_number, probability=pc.probability)
            for i, pc in enumerate(self.pulse_configs)
            if i != signal_idx
        ]
        return IntensityConfig(signal=signal, decoys=decoys)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        config: Union[OpticalSourceConfig, Mapping[str, Any]],
        *,
        density_matrices: Optional[Mapping[str, Any]] = None,
        security_metadata: Optional[SourceSecurityMetadata] = None,
    ) -> "OpticalSource":
        """Create an :class:`OpticalSource` (or subclass) from configuration.

        Selection policy (deterministic given the inputs):

          * If ``density_matrices`` is a **non-empty** mapping:
              -> returns :class:`DensityMatrixSource` (regardless of
                 ``statistics_type``). The DM diagonal is used for
                 photon-number sampling.
          * Elif ``statistics_type == POISSON``:
              -> returns :class:`PoissonSource`.
          * Else (e.g., ``THERMAL``):
              -> returns :class:`OpticalSource` (base class).

        Notes
        -----
        - An empty mapping ``{}`` for ``density_matrices`` is treated as
          "not provided" and does **not** trigger the DM branch.
        - **Subtype-dependent behavior (issue 3.3 in 3rd review):** While
          the returned subtype is deterministic given the inputs, downstream
          code MUST be aware that :class:`DensityMatrixSource.generate_photons`
          bypasses the classical-imperfection pipeline (MZM attenuation,
          channel splitting, SCM factor, intensity jitter, emission-failure
          thinning) because the DM is assumed to already encode the emitted
          state. :class:`OpticalSource` and :class:`PoissonSource` apply
          the full pipeline. If your downstream analysis depends on whether
          classical effects are applied, check ``isinstance(source,
          DensityMatrixSource)`` before calling ``generate_photons``.
        """
        optical_config = _coerce_optical_source_config(config)

        # Decide whether density_matrices are actually provided.
        has_dms = False
        if density_matrices is not None:
            if not isinstance(density_matrices, Mapping):
                raise ParameterValidationError(
                    "density_matrices must be a mapping.",
                    param_name="density_matrices",
                    param_value=type(density_matrices).__name__,
                )
            has_dms = len(density_matrices) > 0

        if has_dms:
            return DensityMatrixSource.from_config(
                optical_config,
                density_matrices=density_matrices,
                security_metadata=security_metadata,
            )

        if optical_config.statistics_type == SourceStatisticsType.POISSON:
            return PoissonSource.from_config(
                optical_config,
                security_metadata=security_metadata,
            )

        return cls.from_config(optical_config, security_metadata=security_metadata)

    @classmethod
    def from_config(
        cls,
        config: OpticalSourceConfig,
        *,
        security_metadata: Optional[SourceSecurityMetadata] = None,
    ) -> "OpticalSource":
        """Build an instance from an :class:`OpticalSourceConfig`."""
        return cls(config=config, security_metadata=security_metadata)

    @classmethod
    def from_protocol_parameters(
        cls,
        params: ProtocolParameters,
        *,
        density_matrices: Optional[Mapping[str, Any]] = None,
        security_metadata: Optional[SourceSecurityMetadata] = None,
    ) -> "OpticalSource":
        """Build an instance from a :class:`ProtocolParameters` instance.

        Validates that ``params.optical`` is an :class:`OpticalSourceConfig`
        to avoid silent type coercion.
        """
        if not isinstance(params, ProtocolParameters):
            raise ParameterValidationError(
                "params must be a ProtocolParameters instance.",
                param_name="params",
                param_value=type(params).__name__,
            )
        if not isinstance(params.optical, OpticalSourceConfig):
            raise ParameterValidationError(
                "params.optical must be an OpticalSourceConfig instance.",
                param_name="params.optical",
                param_value=type(params.optical).__name__,
            )
        return cls.create(
            params.optical,
            density_matrices=density_matrices,
            security_metadata=security_metadata,
        )

    @classmethod
    def from_pulse_ensemble(
        cls,
        pulse_ensemble: PulseEnsembleConfig,
        *,
        source_rate: float,
        statistics_type: SourceStatisticsType = SourceStatisticsType.POISSON,
        error_model: SourceErrorModel = SourceErrorModel.RANDOM_GAUSSIAN,
        intensity_jitter: float = 0.0,
        modulation_index: float = 0.0,
        N_channels: int = 1,
        use_small_angle_approximation: bool = True,
        use_linear_modulation_approximation: bool = False,
        ideal_emission_probability: float = 1.0,
        adversarial_block_size: int = 1000,
        is_bidirectional: bool = False,
        mzm: Any = None,
        electrical_noise: Any = None,
        density_matrices: Optional[Mapping[str, Any]] = None,
        security_metadata: Optional[SourceSecurityMetadata] = None,
    ) -> "OpticalSource":
        """Build an instance from a :class:`PulseEnsembleConfig`.

        Pulse names are taken from the ensemble as-is (role mapping is the
        caller's responsibility).
        """
        config_kwargs: Dict[str, Any] = {
            "source_rate": source_rate,
            "pulse_configs": tuple(pulse_ensemble.pulses),
            "statistics_type": statistics_type,
            "error_model": error_model,
            "intensity_jitter": intensity_jitter,
            "modulation_index": modulation_index,
            "N_channels": N_channels,
            "use_small_angle_approximation": use_small_angle_approximation,
            "use_linear_modulation_approximation": use_linear_modulation_approximation,
            "ideal_emission_probability": ideal_emission_probability,
            "adversarial_block_size": adversarial_block_size,
            "is_bidirectional": is_bidirectional,
        }
        if mzm is not None:
            config_kwargs["mzm"] = mzm
        if electrical_noise is not None:
            config_kwargs["electrical_noise"] = electrical_noise

        config = OpticalSourceConfig(**config_kwargs)
        return cls.create(
            config,
            density_matrices=density_matrices,
            security_metadata=security_metadata,
        )

    @classmethod
    def from_intensity_config(
        cls,
        intensities: IntensityConfig,
        *,
        source_rate: float,
        statistics_type: SourceStatisticsType = SourceStatisticsType.POISSON,
        error_model: SourceErrorModel = SourceErrorModel.RANDOM_GAUSSIAN,
        intensity_jitter: float = 0.0,
        modulation_index: float = 0.0,
        N_channels: int = 1,
        use_small_angle_approximation: bool = True,
        use_linear_modulation_approximation: bool = False,
        ideal_emission_probability: float = 1.0,
        adversarial_block_size: int = 1000,
        is_bidirectional: bool = False,
        mzm: Any = None,
        electrical_noise: Any = None,
        density_matrices: Optional[Mapping[str, Any]] = None,
        security_metadata: Optional[SourceSecurityMetadata] = None,
    ) -> "OpticalSource":
        """Build an instance from an :class:`IntensityConfig`.

        Role-based naming is applied (first node = signal; zero-mu nodes =
        vacuum_i; others = decoy_i).
        """
        ensemble = PulseEnsembleConfig(
            pulses=list(_pulse_types_from_intensity_config(intensities))
        )
        return cls.from_pulse_ensemble(
            ensemble,
            source_rate=source_rate,
            statistics_type=statistics_type,
            error_model=error_model,
            intensity_jitter=intensity_jitter,
            modulation_index=modulation_index,
            N_channels=N_channels,
            use_small_angle_approximation=use_small_angle_approximation,
            use_linear_modulation_approximation=use_linear_modulation_approximation,
            ideal_emission_probability=ideal_emission_probability,
            adversarial_block_size=adversarial_block_size,
            is_bidirectional=is_bidirectional,
            mzm=mzm,
            electrical_noise=electrical_noise,
            density_matrices=density_matrices,
            security_metadata=security_metadata,
        )

    @classmethod
    def from_intensity_node(
        cls,
        node: IntensityNode,
        *,
        source_rate: float,
        statistics_type: SourceStatisticsType = SourceStatisticsType.POISSON,
        error_model: SourceErrorModel = SourceErrorModel.RANDOM_GAUSSIAN,
        intensity_jitter: float = 0.0,
        modulation_index: float = 0.0,
        N_channels: int = 1,
        use_small_angle_approximation: bool = True,
        use_linear_modulation_approximation: bool = False,
        ideal_emission_probability: float = 1.0,
        adversarial_block_size: int = 1000,
        is_bidirectional: bool = False,
        mzm: Any = None,
        electrical_noise: Any = None,
        density_matrices: Optional[Mapping[str, Any]] = None,
        security_metadata: Optional[SourceSecurityMetadata] = None,
    ) -> "OpticalSource":
        """Build an instance from a single :class:`IntensityNode` (signal only)."""
        intensities = IntensityConfig(signal=node, decoys=[])
        return cls.from_intensity_config(
            intensities,
            source_rate=source_rate,
            statistics_type=statistics_type,
            error_model=error_model,
            intensity_jitter=intensity_jitter,
            modulation_index=modulation_index,
            N_channels=N_channels,
            use_small_angle_approximation=use_small_angle_approximation,
            use_linear_modulation_approximation=use_linear_modulation_approximation,
            ideal_emission_probability=ideal_emission_probability,
            adversarial_block_size=adversarial_block_size,
            is_bidirectional=is_bidirectional,
            mzm=mzm,
            electrical_noise=electrical_noise,
            density_matrices=density_matrices,
            security_metadata=security_metadata,
        )

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        strict: bool = False,
        security_metadata: Optional[SourceSecurityMetadata] = None,
    ) -> "OpticalSource":
        """Build an instance from a serialized mapping.

        Parameters
        ----------
        data : dict
            Serialized source configuration. May contain a
            ``"density_matrices"`` key; if non-empty, a
            :class:`DensityMatrixSource` is returned (round-trip safe).
        strict : bool, default False
            If True, raise on unknown keys (helps catch typos in serialized
            configs). If False, unknown keys are silently ignored.
        security_metadata : optional
            Attached to the constructed source.

        Notes
        -----
        Density matrices are decoded from the JSON-safe
        ``{"real": [[...]], "imag": [[...]}`` format produced by
        :meth:`to_config_dict`. Direct complex-array payloads are also
        accepted (Python-only, not JSON-safe).

        If ``data["density_matrices"]`` is non-empty, this method dispatches
        to :meth:`DensityMatrixSource.from_dict` to preserve any
        ``missing_dm_policy`` and ensure the correct subclass is constructed.

        4th-review fix (issue 2.3): in strict mode, DM-specific keys
        (``density_matrices``, ``missing_dm_policy``,
        ``heterogeneous_dim_policy``) are rejected when the payload does
        not contain a non-empty ``density_matrices`` mapping (i.e., when
        a non-DM source would be constructed). The previous
        implementation accepted these keys silently, masking typos and
        accidental misconfiguration (e.g., a user setting
        ``missing_dm_policy`` on a Poisson source).
        """
        if not isinstance(data, Mapping):
            raise ParameterValidationError(
                "data must be a mapping.",
                param_name="data",
                param_value=type(data).__name__,
            )

        # If density_matrices are present and non-empty, dispatch to
        # DensityMatrixSource.from_dict to preserve missing_dm_policy and
        # ensure correct subclass construction.
        # Issue 1.6 in 3rd review: if density_matrices is present but NOT a
        # mapping (e.g., an integer or list), raise an error instead of
        # silently ignoring the invalid payload. The previous code only
        # dispatched when isinstance(raw_dms, Mapping), silently dropping
        # non-mapping payloads and constructing a non-DM source — masking
        # serialization corruption.
        raw_dms = data.get("density_matrices") if "density_matrices" in data else None
        has_dms = False
        if raw_dms is not None:
            if not isinstance(raw_dms, Mapping):
                raise ParameterValidationError(
                    "density_matrices payload must be a mapping (dict) from "
                    "pulse name to matrix data. Got a non-mapping payload of "
                    f"type {type(raw_dms).__name__}, which indicates a "
                    "corrupted or malformed serialized config.",
                    param_name="density_matrices",
                    param_value=type(raw_dms).__name__,
                )
            has_dms = len(raw_dms) > 0

        if strict:
            # 4th-review fix (issue 2.3): reject DM-specific keys when
            # constructing a non-DM source. DM-specific keys are only
            # valid when a non-empty density_matrices payload is present.
            if has_dms:
                # DM source: all known keys (base + DM) are allowed.
                unknown = set(data.keys()) - _KNOWN_SOURCE_CONFIG_KEYS
            else:
                # Non-DM source: only base keys are allowed. DM-specific
                # keys (density_matrices, missing_dm_policy,
                # heterogeneous_dim_policy) are rejected as unknown.
                unknown = set(data.keys()) - _KNOWN_BASE_CONFIG_KEYS
            if unknown:
                raise ParameterValidationError(
                    f"Unknown or DM-only keys in non-DM source config: "
                    f"{sorted(unknown)}. These keys are only valid when "
                    f"'density_matrices' is a non-empty mapping. Pass "
                    f"strict=False to ignore unknown keys.",
                    param_name="data",
                )

        if has_dms:
            return DensityMatrixSource.from_dict(
                data,
                strict=strict,
                security_metadata=security_metadata,
            )
        # Empty mapping or no density_matrices key: construct a non-DM source.
        # Note: an empty {} for density_matrices is treated as "not provided"
        # and falls through to construct a non-DM source (consistent with
        # create()'s treatment of empty mappings). In strict mode, the
        # empty-mapping case would have been rejected above as a DM-only key
        # in a non-DM config.

        config = _coerce_optical_source_config(data)
        return cls.create(
            config,
            density_matrices=None,
            security_metadata=security_metadata,
        )

    # ------------------------------------------------------------------
    # RNG & protocol-level utilities
    # ------------------------------------------------------------------

    @classmethod
    def validate_rng_seed(cls, seed: Any) -> None:
        """Validate a seed value for :class:`numpy.random.Generator`.

        Accepts:
          * Python ``int`` in ``[0, max(MAX_SEED_INT_PCG64, MAX_SEED_INT_UINT32)]``
          * :class:`numpy.integer` (same range)
          * :class:`numpy.random.SeedSequence` (passed through)
          * 1-D sequence of non-negative integers, each in
            ``[0, max(MAX_SEED_INT_PCG64, MAX_SEED_INT_UINT32)]`` (entropy
            for :class:`SeedSequence`)

        Notes
        -----
        The integer upper bound exists for compatibility with legacy
        ``RandomState``-style seeding; :class:`numpy.random.Generator` with
        PCG64 actually accepts arbitrary entropy via
        :class:`SeedSequence`, but we keep the bound for ints to surface
        accidental misuse (e.g., a stray float).

        4th-review fix (issue 3.4): array-like seeds are now bounds-checked
        element-wise against the same ``[0, max_seed]`` range as scalar
        integer seeds. The previous implementation only verified the dtype
        and dimensionality, silently accepting negative integers (which
        :class:`SeedSequence` would convert to unsigned via wrap-around,
        masking likely user errors such as ``-1`` to mean "no seed").
        """
        if isinstance(seed, np.random.SeedSequence):
            return
        # 5th-review fix (issue 4.1): reject bool via _is_integer.
        if _is_integer(seed):
            max_seed = max(MAX_SEED_INT_PCG64, MAX_SEED_INT_UINT32)
            seed_int = int(seed)
            if not (0 <= seed_int <= max_seed):
                raise ParameterValidationError(
                    f"integer seed must satisfy 0 <= seed <= {max_seed} "
                    f"(got {seed_int}).",
                    param_name="seed",
                    param_value=seed_int,
                )
            return
        if isinstance(seed, (list, tuple, np.ndarray)):
            # 4th-review fix (issue 3.4): bounds-check each element against
            # the same [0, max_seed] range as scalar integer seeds. This
            # catches negative-integral entropy (a common typo, e.g.,
            # passing ``-1`` intending "no seed") and overflow values that
            # SeedSequence would silently wrap to unsigned 32-bit.
            max_seed = max(MAX_SEED_INT_PCG64, MAX_SEED_INT_UINT32)

            if isinstance(seed, np.ndarray):
                arr = seed
                if arr.dtype.kind not in ("i", "u", "O"):
                    raise ParameterValidationError(
                        "array-like seed must contain integers (for SeedSequence entropy).",
                        param_name="seed",
                    )
                if arr.ndim != 1:
                    raise ParameterValidationError(
                        "array-like seed must be 1-dimensional.",
                        param_name="seed",
                    )
                # For integer dtypes, use vectorized comparison.
                if arr.dtype.kind in ("i", "u"):
                    if arr.size > 0:
                        if np.any(arr < 0):
                            bad = arr[arr < 0]
                            raise ParameterValidationError(
                                f"array-like seed contains negative integers: "
                                f"{bad.tolist()}. All entropy values must be in "
                                f"[0, {max_seed}].",
                                param_name="seed",
                            )
                        # 7th-review fix (issue III.1): apply the upper-bound
                        # check UNIFORMLY to BOTH signed ("i") and unsigned
                        # ("u") integer dtypes. The previous implementation
                        # restricted the check to unsigned dtypes via
                        # ``if arr.dtype.kind == "u" and np.any(arr > max_seed)``,
                        # with a comment claiming "For int64 arrays, max_seed
                        # = 2**63 - 1 (int64 max), so this is always
                        # satisfied". This reasoning was brittle: it relied
                        # on the specific value of ``MAX_SEED_INT_PCG64``
                        # (which happens to be 2**63 - 1 in the test stub
                        # but could be changed in production to a smaller
                        # value like 2**32 - 1, in which case int64 arrays
                        # could hold values exceeding ``max_seed``). The
                        # asymmetric check was also a code-clarity hazard:
                        # readers had to reason about why signed and unsigned
                        # dtypes were treated differently. The fix applies
                        # the upper-bound check uniformly, which is correct
                        # for all possible values of ``max_seed`` and
                        # eliminates the asymmetry. The check is a no-op
                        # when ``max_seed`` is large enough to cover the
                        # dtype's range (e.g., int64 with max_seed = 2**63-1),
                        # so the runtime cost is negligible in the common
                        # case. Note: the negative-value check above is
                        # similarly a no-op for unsigned dtypes (their
                        # values cannot be negative), but we keep it as
                        # defensive programming in case numpy changes its
                        # wrap-around behavior for unsigned arithmetic.
                        if np.any(arr > max_seed):
                            bad = arr[arr > max_seed]
                            raise ParameterValidationError(
                                f"array-like seed contains values > {max_seed}: "
                                f"{bad.tolist()}. All entropy values must be in "
                                f"[0, {max_seed}].",
                                param_name="seed",
                            )
                else:
                    # Object dtype: check element-wise.
                    for elem in arr.flat:
                        # 5th-review fix (issue 4.1): reject bool elements.
                        if not _is_integer(elem):
                            raise ParameterValidationError(
                                f"array-like seed must contain integers; got "
                                f"element of type {type(elem).__name__}.",
                                param_name="seed",
                            )
                        elem_int = int(elem)
                        if elem_int < 0:
                            raise ParameterValidationError(
                                f"array-like seed contains negative integers: "
                                f"{elem_int}. All entropy values must be in "
                                f"[0, {max_seed}].",
                                param_name="seed",
                            )
                        if elem_int > max_seed:
                            raise ParameterValidationError(
                                f"array-like seed contains values > {max_seed}: "
                                f"{elem_int}. All entropy values must be in "
                                f"[0, {max_seed}].",
                                param_name="seed",
                            )
            else:
                # List/tuple: validate element-wise WITHOUT numpy conversion.
                # numpy may silently upcast [1, 2**63] to float64, which
                # would fail the dtype check. We iterate directly over the
                # Python objects to preserve exact integer semantics.
                for elem in seed:
                    # 5th-review fix (issue 4.1): reject bool elements.
                    if not _is_integer(elem):
                        raise ParameterValidationError(
                            f"array-like seed must contain integers; got "
                            f"element of type {type(elem).__name__}.",
                            param_name="seed",
                        )
                    elem_int = int(elem)
                    if elem_int < 0:
                        raise ParameterValidationError(
                            f"array-like seed contains negative integers: "
                            f"{elem_int}. All entropy values must be in "
                            f"[0, {max_seed}].",
                            param_name="seed",
                        )
                    if elem_int > max_seed:
                        raise ParameterValidationError(
                            f"array-like seed contains values > {max_seed}: "
                            f"{elem_int}. All entropy values must be in "
                            f"[0, {max_seed}].",
                            param_name="seed",
                        )
            return
        raise ParameterValidationError(
            "seed must be int, numpy.integer, numpy.random.SeedSequence, "
            "or 1-D integer array-like.",
            param_name="seed",
            param_value=type(seed).__name__,
        )

    @classmethod
    def calculate_minimum_block_size(cls, expected_yield: float) -> int:
        """Utility: minimum block size for phase-estimation security.

        Notes
        -----
        This is a **protocol-level** utility kept on this class for
        backwards compatibility. It does not depend on any source state and
        may be moved to a protocol-level module in a future refactor.
        """
        if not is_finite_non_negative(expected_yield):
            raise ParameterValidationError(
                "expected_yield must be finite and non-negative.",
                param_name="expected_yield",
                param_value=expected_yield,
            )
        # Deferred import to avoid framework-layer constant leakage at
        # module-import time; the constant belongs to protocol/estimation
        # and is only needed when this utility is actually called.
        from .constants import MIN_SUCCESSFUL_Z1_BASIS_EVENTS_FOR_PHASE_EST

        return int(np.ceil(MIN_SUCCESSFUL_Z1_BASIS_EVENTS_FOR_PHASE_EST / (expected_yield + EPS)))

    # ------------------------------------------------------------------
    # Pulse lookup utilities
    # ------------------------------------------------------------------

    def pulse_names(self) -> Tuple[str, ...]:
        """Return the tuple of pulse-type names (in config order)."""
        return tuple(pc.name for pc in self.pulse_configs)

    def get_pulse_config_by_name(self, name: str) -> PulseTypeConfig:
        """Return the :class:`PulseTypeConfig` matching ``name``.

        4th-review fix (issue 4.3): uses the pre-computed
        ``_pulse_name_to_index`` dict for O(1) lookup, replacing the
        previous O(N) linear scan over ``pulse_configs``.
        """
        idx = self._pulse_name_to_index.get(name)
        if idx is None:
            raise ParameterValidationError(
                f"Pulse config with name {name!r} not found.",
                param_name="name",
                param_value=name,
            )
        return self.pulse_configs[idx]

    def get_pulse_index_by_name(self, name: str) -> int:
        """Return the index of the pulse type matching ``name``.

        4th-review fix (issue 4.3): uses the pre-computed
        ``_pulse_name_to_index`` dict for O(1) lookup, replacing the
        previous O(N) linear scan over ``pulse_configs``.
        """
        idx = self._pulse_name_to_index.get(name)
        if idx is None:
            raise ParameterValidationError(
                f"Pulse config with name {name!r} not found.",
                param_name="name",
                param_value=name,
            )
        return idx

    def get_pulse_probability_by_name(self, name: str) -> float:
        """Return the selection probability of the named pulse type."""
        return self.get_pulse_config_by_name(name).probability

    def get_mean_photon_number_by_name(self, name: str) -> float:
        """Return the configured mean photon number for the named pulse type."""
        return self.get_pulse_config_by_name(name).mean_photon_number

    def base_mean_photon_numbers(self) -> np.ndarray:
        """Return a *copy* of the cached base mean photon numbers."""
        return self._base_mus_cache.copy()

    def pulse_probabilities(self) -> np.ndarray:
        """Return a *copy* of the cached pulse-selection probabilities."""
        return self._probabilities_cache.copy()

    def number_probabilities_for_pulse(
        self,
        name: str,
        *,
        max_n: int = PN_TRUNCATION_DIM,
        include_emission_failure: bool = False,
    ) -> np.ndarray:
        """Return the photon-number probability vector P(n) for the named pulse.

        For Poisson and thermal sources, this computes P(n) analytically from
        the base mean photon number using :func:`poisson_pn_array` or
        :func:`thermal_pn_array`. For DensityMatrixSource, this method is
        overridden to return the cached DM diagonal.

        Parameters
        ----------
        name : str
            Pulse name.
        max_n : int, default PN_TRUNCATION_DIM
            Truncation dimension (array length). Only used for Poisson/thermal
            sources. DensityMatrixSource overrides this with its DM dimension.
        include_emission_failure : bool, default False
            If True, return the *unconditional* (post-emission-failure)
            probability vector, computed as the vacuum mixture
            ``p_emit * p + (1 - p_emit) * |0><0|``. If False (default),
            return the *conditional* (pre-emission-failure) probability
            vector.

        Returns
        -------
        np.ndarray
            Probability vector of shape (max_n,). For Poisson/thermal sources,
            the array is freshly computed; for DensityMatrixSource, it is a
            read-only view of the cached diagonal (see the override in that
            class).

        Notes
        -----
        This method unifies the analytical P(n) API across all source types.
        For Poisson/thermal sources, it delegates to the module-level helpers
        (:func:`poisson_pn_array`, :func:`thermal_pn_array`) which return
        raw NumPy arrays suitable for Numba-compiled loops. For
        DensityMatrixSource, it returns the cached diagonal directly.
        """
        mu = self.get_mean_photon_number_by_name(name)
        stats_type = self.statistics_type

        if stats_type == SourceStatisticsType.POISSON:
            probs = poisson_pn_array(mu, max_n)
        elif stats_type == SourceStatisticsType.THERMAL:
            probs = thermal_pn_array(mu, max_n)
        else:
            raise ParameterValidationError(
                f"number_probabilities_for_pulse not supported for "
                f"statistics_type={stats_type!r} on base OpticalSource. "
                f"Use DensityMatrixSource for density-matrix-based sampling.",
                param_name="statistics_type",
                param_value=stats_type,
            )

        if include_emission_failure:
            p_emit = clamp_probability(self.config.ideal_emission_probability)
            if not is_close(p_emit, 1.0):
                probs = probs * p_emit
                probs[0] += (1.0 - p_emit)

        return probs

    # ------------------------------------------------------------------
    # Pulse-index validation & sampling
    # ------------------------------------------------------------------

    def _validate_pulse_indices(
        self,
        pulse_indices: Union[int, np.integer, np.ndarray, Sequence[int]],
    ) -> Tuple[np.ndarray, bool]:
        """Validate ``pulse_indices`` and return ``(indices_1d_int, is_scalar)``.

        Validation:
          * Scalar input is detected via type (int, np.integer) OR 0-d array.
          * Array input must be 1-dimensional.
          * Integer dtype required. Float inputs are accepted **only** if
            they are exactly integer-valued (no tolerance-based coercion);
            otherwise they are rejected. This avoids both numerical edge
            cases and the per-element performance cost of tolerance checks
            on large arrays (issue 6 in 2nd review).
          * Boolean arrays are **rejected** (issue 1.2 in 3rd review):
            in NumPy, boolean arrays represent indexing masks, not positional
            indices. Silently casting ``[True, False, True]`` to ``[1, 0, 1]``
            would mask a likely user error (passing a mask instead of
            positional indices).
          * No NaN / inf.
          * Range: ``0 <= idx < len(pulse_configs)``. Float indices are
            range-checked **before** casting to int64 to prevent silent
            overflow (issue 1.3 in 3rd review).

        4th-review fixes (issues 3.1, 3.2):
          * **Issue 3.1 (redundant int64 bounds check):** The previous
            implementation checked float indices against the int64
            representable range BEFORE checking against ``[0, num_configs)``.
            This was logically backwards: any float exceeding ``num_configs``
            (a small integer, typically 2-5) is already out of range, so
            the int64 check is computationally redundant. Furthermore,
            doubles exceeding ``2^52`` lack fractional representation and
            would pass the ``np.array_equal(indices, rounded)`` exact-
            integer check, defeating the guard. The new implementation
            checks the much tighter ``[0, num_configs)`` range directly on
            the float values, which simultaneously catches overflow,
            out-of-range, and negative values with a single comparison.
          * **Issue 3.2 (inconsistent ``if n > 0`` guard):** The integer
            branch wrapped the bounds check in ``if n > 0:``, but the
            float branch did not. The guard is unnecessary because
            ``np.any(empty_array < 0)`` returns ``False`` (a no-op), so
            the check is safe on empty arrays. Both branches now run the
            bounds check unconditionally, eliminating the inconsistency.

        Returns
        -------
        indices : np.ndarray
            1-D int64 array of pulse indices.
        is_scalar : bool
            True if the original input was a scalar.
        """
        if pulse_indices is None:
            raise ParameterValidationError(
                "pulse_indices is None; expected int or array-like of int.",
                param_name="pulse_indices",
            )

        # 6th-review fix (issue 8): pre-check for bool content BEFORE calling
        # np.asarray. The previous implementation called
        # ``np.atleast_1d(np.asarray(pulse_indices))`` first, which silently
        # upcasts Python lists containing booleans (e.g., ``[True, 1, 0]``)
        # to dtype ``int64`` (with values ``[1, 1, 0]``), bypassing the
        # explicit bool-rejection logic below. By inspecting the raw input
        # for bool elements before numpy conversion, we ensure mixed-type
        # lists with booleans are rejected with a clear error message.
        # Scalar bool is already rejected by ``_is_integer`` above; this
        # check handles sequence inputs.
        #
        # 7th-review fix (issue II.1): extend the pre-check to ALL iterable
        # inputs, not just list/tuple. The previous implementation only
        # inspected list and tuple inputs, allowing other sequence types
        # (``deque``, generator expressions, custom sequences, or
        # ``np.array([True, 1, 0], dtype=object)``) to bypass the check.
        # When ``np.asarray`` is subsequently called on such inputs, boolean
        # elements are silently upcast to integers (``True -> 1``,
        # ``False -> 0``), yielding an integer-dtype array that bypasses
        # the bool-dtype rejection logic below. The fix uses an
        # object-dtype intermediate array to preserve the original Python
        # types during the bool check, covering any iterable input AND
        # object-dtype ndarrays (which preserve Python bool elements).
        # Scalar inputs (int, np.integer) are exempt (they're not iterable
        # and are handled by ``_is_integer`` above).
        if isinstance(pulse_indices, np.ndarray):
            # For ndarrays, only object-dtype arrays can contain Python bools
            # (typed bool arrays are caught by the dtype check below).
            if pulse_indices.dtype.kind == "O" and pulse_indices.ndim >= 1:
                for elem in pulse_indices.flat:
                    if isinstance(elem, bool):
                        raise ParameterValidationError(
                            "pulse_indices contains a Python bool element. "
                            "Boolean values represent indexing masks in NumPy, "
                            "not positional pulse indices. An object-dtype "
                            "ndarray containing bools would be silently "
                            "upcast to integers by np.asarray, masking a "
                            "likely user error. Pass an integer array of "
                            "positional indices instead. If you intended to "
                            "use boolean indexing, convert the mask to "
                            "indices explicitly via np.nonzero(mask).",
                            param_name="pulse_indices",
                            param_value="object-dtype ndarray with bool element",
                        )
        else:
            # Non-ndarray input: try to coerce to an object-dtype array to
            # preserve Python types. This consumes generators (callers
            # should pass materialized sequences) and rejects non-iterable
            # scalars via the TypeError handler.
            try:
                obj_arr = np.asarray(pulse_indices, dtype=object)
            except (TypeError, ValueError):
                obj_arr = None
            if obj_arr is not None and obj_arr.ndim >= 1:
                for elem in obj_arr.flat:
                    if isinstance(elem, bool):
                        raise ParameterValidationError(
                            "pulse_indices contains a Python bool element. "
                            "Boolean values represent indexing masks in NumPy, "
                            "not positional pulse indices. Any iterable "
                            "containing bools (list, tuple, deque, generator, "
                            "or object-dtype array) would be silently "
                            "upcast to integers by np.asarray, masking a "
                            "likely user error. Pass an integer array of "
                            "positional indices instead. If you intended to "
                            "use boolean indexing, convert the mask to "
                            "indices explicitly via np.nonzero(mask).",
                            param_name="pulse_indices",
                            param_value="iterable with bool element",
                        )

        # Detect scalars (issue 2.4: np.isscalar alone is unreliable).
        # 5th-review fix (issue 4.1): reject bool via _is_integer.
        is_scalar = _is_integer(pulse_indices)
        if isinstance(pulse_indices, np.ndarray) and pulse_indices.ndim == 0:
            is_scalar = True

        indices = np.atleast_1d(np.asarray(pulse_indices))

        # Must be 1D (issue 5.10).
        if indices.ndim != 1:
            raise ParameterValidationError(
                f"pulse_indices must be 1-dimensional, got shape {indices.shape}.",
                param_name="pulse_indices",
            )

        num_configs = len(self.pulse_configs)

        # Integer dtype check (issue 2.3, 2nd-review issue 6).
        # We use exact-equality coercion (no tolerance) for two reasons:
        #   (a) It is O(1) in constant factors and avoids the per-element
        #       ``np.abs(indices - np.round(indices)) > tol`` check that
        #       becomes slow on very large arrays.
        #   (b) Tolerance-based acceptance of "near-integer" floats can
        #       silently mask genuine off-by-one bugs at large index values,
        #       where floating-point precision is on the order of the
        #       tolerance.
        if np.issubdtype(indices.dtype, np.integer):
            # Native integer dtype: no coercion needed. Bounds check runs
            # unconditionally below (4th-review issue 3.2: removed the
            # inconsistent ``if n > 0:`` guard; np.any on an empty array
            # returns False, so the check is a safe no-op).
            pass
        elif np.issubdtype(indices.dtype, np.floating):
            # Reject non-finite (NaN/inf) immediately.
            if not np.all(np.isfinite(indices)):
                raise ParameterValidationError(
                    "pulse_indices contains NaN or non-finite values.",
                    param_name="pulse_indices",
                )
            # Exact-equality check: accept floats only if they are exactly
            # equal to their rounded values (e.g., 1.0, 2.0, but not 1.5
            # or 1.0000001). This is a single vectorized op (fast) and
            # unambiguous.
            rounded = np.rint(indices)
            if not np.array_equal(indices, rounded):
                raise ParameterValidationError(
                    "pulse_indices contains non-integer float values; "
                    "pass an integer array instead.",
                    param_name="pulse_indices",
                )
            # 4th-review fix (issue 3.1): the previous code checked the
            # int64 representable range here, which was both redundant
            # (the [0, num_configs) range check below is far tighter) and
            # ineffective for floats > 2^52 (which lack fractional
            # representation and pass the exact-integer check). We now
            # check the [0, num_configs) range directly on the float
            # values, which catches overflow, out-of-range, and negative
            # values in a single comparison. The cast to int64 happens
            # only after the range check, so overflow cannot occur.
            if np.any(indices < 0) or np.any(indices >= num_configs):
                bad = indices[(indices < 0) | (indices >= num_configs)]
                raise ParameterValidationError(
                    f"pulse_indices contains values out of range "
                    f"[0, {num_configs}) (number of pulse configs): "
                    f"{bad.tolist()}.",
                    param_name="pulse_indices",
                )
            indices = rounded.astype(np.int64)
        elif np.issubdtype(indices.dtype, np.bool_):
            # Issue 1.2 in 3rd review: reject boolean arrays instead of
            # silently casting them to integers. A boolean array
            # [True, False, True] is a mask, not a list of positional
            # indices. Silent coercion to [1, 0, 1] would mask a likely
            # user error.
            raise ParameterValidationError(
                "pulse_indices has boolean dtype. Boolean arrays represent "
                "indexing masks in NumPy, not positional pulse indices. "
                "Pass an integer array of positional indices instead. "
                "If you intended to use boolean indexing, convert the mask "
                "to indices explicitly via np.nonzero(mask).",
                param_name="pulse_indices",
                param_value=f"bool array of shape {indices.shape}",
            )
        else:
            raise ParameterValidationError(
                f"pulse_indices must be integer or integer-valued float, "
                f"got dtype {indices.dtype}.",
                param_name="pulse_indices",
            )

        # 4th-review fix (issue 3.2): unified bounds check for integer
        # arrays. Runs unconditionally (the previous ``if n > 0:`` guard
        # was unnecessary because np.any on an empty array returns False).
        # The float branch already range-checked above, but for integer
        # arrays we still need this check (negative values, out-of-range
        # positive values).
        if np.any(indices < 0):
            bad = indices[indices < 0]
            raise ParameterValidationError(
                f"pulse_indices contains negative values: {bad.tolist()}.",
                param_name="pulse_indices",
            )
        if np.any(indices >= num_configs):
            bad = indices[indices >= num_configs]
            raise ParameterValidationError(
                f"pulse_indices contains values >= {num_configs} "
                f"(number of pulse configs): {bad.tolist()}.",
                param_name="pulse_indices",
            )

        return indices.astype(np.int64, copy=False), is_scalar

    def sample_pulse_indices(self, rng: RNGType, num_samples: int) -> np.ndarray:
        """Sample ``num_samples`` pulse-type indices according to probabilities.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator.
        num_samples : int
            Number of samples. ``0`` is allowed and returns an empty array
            (issue 2.7).

        Returns
        -------
        np.ndarray
            1-D int64 array of pulse indices.
        """
        # 5th-review fix (issue 4.1): reject bool via _is_integer.
        if not _is_integer(num_samples):
            raise ParameterValidationError(
                "num_samples must be an integer.",
                param_name="num_samples",
                param_value=type(num_samples).__name__,
            )
        num_samples_int = int(num_samples)
        if num_samples_int < 0:
            raise ParameterValidationError(
                f"num_samples must be >= 0 (got {num_samples_int}).",
                param_name="num_samples",
            )
        if num_samples_int == 0:
            return np.array([], dtype=np.int64)
        return rng.choice(
            len(self.pulse_configs),
            size=num_samples_int,
            p=self._probabilities_cache,
        ).astype(np.int64, copy=False)

    def update_internal_tallies(
        self,
        num_pulses_generated: Optional[int] = None,
        *,
        num_pulses_attempted: Optional[int] = None,
    ) -> None:
        """Update security-metadata tallies for generated pulses.

        Two tally counters are supported:

          * ``num_pulses_attempted``: the number of pulses Alice *tried* to
            emit (i.e. the size of the requested pulse batch). This
            represents the number of pulses the source was asked to produce.
          * ``num_pulses_generated``: the number of pulses Alice *actually
            emitted* (i.e. post-emission-failure-thinning). Pulses that
            were requested but failed to emit (due to
            ``ideal_emission_probability < 1``) are NOT counted here.

        For backwards compatibility, the positional argument
        ``num_pulses_generated`` is interpreted as the number of emitted
        pulses. Callers that wish to report both attempted and emitted
        counts should pass both parameters as keyword arguments.

        Notes
        -----
        This split (issue 2 in 2nd review) is important for accurate
        security-metadata accounting: a source with ``p_emit = 0.5`` that
        is asked to emit 1000 pulses should report
        ``attempted = 1000, emitted ~= 500``, not ``emitted = 1000``.
        Treating failed-emission pulses as emitted would overcount Alice's
        transmission rate and bias downstream protocol analyses (e.g.,
        yield estimation, key-rate calculations).
        """
        if num_pulses_generated is None and num_pulses_attempted is None:
            raise ParameterValidationError(
                "update_internal_tallies requires at least one of "
                "num_pulses_generated or num_pulses_attempted.",
                param_name="num_pulses_generated",
            )

        if num_pulses_generated is not None:
            # 5th-review fix (issue 4.1): reject bool via _is_integer.
            if not _is_integer(num_pulses_generated):
                raise ParameterValidationError(
                    "num_pulses_generated must be an integer.",
                    param_name="num_pulses_generated",
                    param_value=type(num_pulses_generated).__name__,
                )
            n_emitted = int(num_pulses_generated)
            if n_emitted < 0:
                raise ParameterValidationError(
                    f"num_pulses_generated must be >= 0 (got {n_emitted}).",
                    param_name="num_pulses_generated",
                )
        else:
            n_emitted = None

        if num_pulses_attempted is not None:
            # 5th-review fix (issue 4.1): reject bool via _is_integer.
            if not _is_integer(num_pulses_attempted):
                raise ParameterValidationError(
                    "num_pulses_attempted must be an integer.",
                    param_name="num_pulses_attempted",
                    param_value=type(num_pulses_attempted).__name__,
                )
            n_attempted = int(num_pulses_attempted)
            if n_attempted < 0:
                raise ParameterValidationError(
                    f"num_pulses_attempted must be >= 0 (got {n_attempted}).",
                    param_name="num_pulses_attempted",
                )
        else:
            n_attempted = None

        # Cross-validate: emitted <= attempted (if both provided).
        if n_emitted is not None and n_attempted is not None:
            if n_emitted > n_attempted:
                raise ParameterValidationError(
                    f"num_pulses_generated ({n_emitted}) must be <= "
                    f"num_pulses_attempted ({n_attempted}).",
                    param_name="num_pulses_generated",
                )

        if self.security_metadata is not None:
            # 4th-review fix (issue 3.3): the previous implementation only
            # passed a single ``count_to_send`` value to
            # ``security_metadata.update_sent``, dropping the attempted
            # count when the emitted count was provided. This made the
            # ``num_pulses_attempted`` parameter effectively useless
            # whenever ``num_pulses_generated`` was also given (which is
            # the common case in generate_photons). The new implementation:
            #   * Always calls ``update_sent`` with the emitted count
            #     (or the attempted count, if emitted is unknown — for
            #     backwards compatibility with callers that only know the
            #     requested batch size).
            #   * Additionally calls ``update_attempted`` on the metadata
            #     object if it has that method (duck-typed), so security
            #     analyses that need the *attempted* count (e.g., for
            #     yield estimation accounting for emission failure) can
            #     access it. This is forward-compatible with future
            #     ``SourceSecurityMetadata`` extensions without requiring
            #     a breaking API change.
            count_to_send = (
                n_emitted if n_emitted is not None else n_attempted
            )
            if count_to_send is not None:
                self.security_metadata.update_sent(count_to_send)
            # Record the attempted count separately if the metadata object
            # supports it (duck-typed; the base SourceSecurityMetadata in
            # the test stub only has update_sent, so this is a no-op there
            # but works for production metadata objects that track both).
            # 5th-review fix (issue 4.2): use callable(getattr(...)) instead
            # of hasattr(...) so we reject non-callable attributes.
            # 6th-review fix (issue 7): when n_attempted != n_emitted AND the
            # metadata object does NOT implement update_attempted, log a
            # warning. Legacy SourceSecurityMetadata instances that only
            # implement update_sent permanently lose track of failed emission
            # cycles, which can introduce artificial clock skew into downstream
            # protocol timing, block-length accounting, and privacy
            # amplification analyses that rely on update_sent representing
            # total transmission clock cycles. The warning alerts users that
            # they should upgrade their metadata object to support
            # update_attempted for accurate accounting when p_emit < 1.
            has_update_attempted = callable(
                getattr(self.security_metadata, "update_attempted", None)
            )
            if n_attempted is not None and has_update_attempted:
                self.security_metadata.update_attempted(n_attempted)
            elif (
                n_attempted is not None
                and n_emitted is not None
                and n_attempted != n_emitted
                and not has_update_attempted
            ):
                logger.warning(
                    "update_internal_tallies: n_attempted (%d) != n_emitted "
                    "(%d) but the security_metadata object does not implement "
                    "'update_attempted'. Only update_sent(n_emitted) was "
                    "called, which means failed emission cycles (%d pulses) "
                    "are NOT recorded in the metadata. Downstream protocol "
                    "analyses that rely on update_sent representing total "
                    "transmission clock cycles will see artificial clock skew. "
                    "Upgrade SourceSecurityMetadata to implement "
                    "update_attempted(n_attempted) for accurate accounting "
                    "when ideal_emission_probability < 1.",
                    n_attempted, n_emitted, n_attempted - n_emitted,
                )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_config_dict(self) -> Dict[str, Any]:
        """Serialize the source to a JSON-safe dict.

        Notes
        -----
        The output includes ``statistics_type``, ``error_model`` (as enum
        values), ``pulse_configs``, ``pulse_ensemble``,
        ``intensity_config``, and ``security_metadata``. Subclasses extend
        this with subclass-specific fields.
        """
        data = self.config.to_dict()
        data["statistics_type"] = self.statistics_type.value
        data["error_model"] = self.error_model.value
        data["pulse_configs"] = [pc.to_dict() for pc in self.pulse_configs]
        data["pulse_ensemble"] = self.pulse_ensemble.to_dict()
        data["intensity_config"] = self.intensity_config.to_dict()
        data["security_metadata"] = (
            self.security_metadata.to_dict() if self.security_metadata else None
        )
        return sanitize_for_serialization(data)

    def to_dict(self) -> Dict[str, Any]:
        """Alias for :meth:`to_config_dict`."""
        return self.to_config_dict()

    # ------------------------------------------------------------------
    # Photon-number sampling internals
    # ------------------------------------------------------------------

    def _scm_scaling_factor(self) -> float:
        """SCM (subcarrier modulation) first-order sideband power fraction.

        Returns the fraction of total optical power that ends up in the
        first-order sideband(s) of a phase-modulated carrier. Used as a
        multiplicative scaling on the base mean photon number.

        Physics (issue 2.1 in 3rd review):
          Phase modulation of an optical carrier at frequency Omega with
          modulation index m expands via the Jacobi-Anger identity as::

              exp(i m sin(Omega t)) = sum_n J_n(m) exp(i n Omega t)

          This generates symmetric sidebands at n = +1, -1, +2, -2, ...
          Each first-order sideband (n = +1 and n = -1) carries J1(m)^2 of
          the total power (since J_{-1}(m) = -J_1(m), the power is the
          same). The total power in BOTH first-order sidebands is
          2 * J1(m)^2.

          The module-level constant ``SCM_NUM_SIDEBANDS`` (default 1)
          controls how many sidebands are counted:
            * ``SCM_NUM_SIDEBANDS = 1`` (default): single-sideband (SSB)
              filtering is assumed; only one sideband is detected/used.
              Scaling = J1(m)^2.
            * ``SCM_NUM_SIDEBANDS = 2``: double-sideband (DSB) detection;
              both sidebands contribute. Scaling = 2 * J1(m)^2.

        Semantics (issue 4.3, 4.5; 4th-review issue 1.1; 9th-review fix F-03):
          * If ``scm_enabled`` is ``False`` (default): SCM is **disabled**
            and the scaling is ``1.0``. The base ``mu`` is used directly
            without modification, interpreted as the **post-SCM** mean
            photon number. The ``modulation_index`` is ignored.
          * If ``scm_enabled`` is ``True``: SCM is **active**. The base
            ``mu`` is interpreted as the **total emitted pulse energy**,
            and the scaling factor gives the fraction in the first-order
            sideband(s):
              - linear approximation: ``SCM_NUM_SIDEBANDS * (m^2) / 4``
                (forced if ``use_linear_modulation_approximation`` is True)
              - small-angle approximation: same as above (if
                ``use_small_angle_approximation`` is True and
                ``m <= SCM_SMALL_ANGLE_LIMIT``)
              - exact: ``SCM_NUM_SIDEBANDS * J1(m)^2`` (otherwise)

        4th-review fix (issue 1.1 — discontinuity at zero limit):
          The previous implementation used ``m <= NUMERIC_REL_TOL`` (1e-9)
          as the "SCM disabled" threshold, returning ``1.0`` below it and
          ``~m^2/4`` above it. This created an unphysical ~30-order-of-
          magnitude discontinuity at the threshold (e.g., m=1e-9 → 1.0,
          m=1.0000001e-9 → ~2.5e-19), invalidating continuous parameter
          scans and gradient-based optimization of the modulation index.
          The new implementation uses **exact zero** (``m == 0.0``) as the
          only "SCM disabled" trigger. For any ``m > 0``, the physical
          sideband scaling is applied, which is continuous in ``m`` (the
          small-angle approximation ``m^2/4`` matches the exact ``J1(m)^2``
          to within ~2.5e-7 relative error at m=1e-3, and the relative
          error vanishes as m → 0). This preserves the "m=0 disables SCM"
          semantics while eliminating the discontinuity at non-zero m.

        6th-review note (issue 3 — semantic discontinuity at m=0):
          There remains an **intentional semantic discontinuity** at
          ``m == 0.0``: when ``m == 0.0``, the base ``mu`` is interpreted
          as the **post-SCM** mean photon number (SCM disabled, scaling =
          1.0); when ``m > 0``, the base ``mu`` is interpreted as the
          **total emitted pulse energy** (SCM enabled, scaling = sideband
          fraction). This is a physical configuration switch, not a
          continuous parameter: ``m == 0`` means "SCM is not part of the
          source", while ``m > 0`` means "SCM is active". As ``m → 0⁺``,
          the scaling factor approaches 0 (not 1.0), so there is an
          infinite relative jump at ``m == 0``. **Gradient-based
          optimization across ``m == 0`` is invalid** because the
          physical interpretation changes discontinuously. Users who need
          a continuous parameterization should use a separate boolean
          flag (e.g., ``scm_enabled``) to control whether SCM is active,
          keeping ``m > 0`` whenever SCM is enabled. The current API
          conflates the two into a single ``modulation_index`` field for
          backwards compatibility; a future refactor may split them.

        Precedence (issue 4.4):
          1. ``scm_enabled == False`` -> 1.0 (SCM disabled)
          2. ``use_linear_modulation_approximation`` -> forced linear
             (regardless of ``m``)
          3. ``use_small_angle_approximation`` and ``m <= limit`` ->
             small-angle approximation
          4. else -> exact Bessel

        Note on small-angle threshold (issue 2.2 in 3rd review):
          ``SCM_SMALL_ANGLE_LIMIT`` is set to 1e-3 (was 0.1). At m=1e-3,
          the relative error between (m^2)/4 and J1(m)^2 is ~2.5e-7 (from
          the J1(m) ~ m/2 - m^3/16 expansion), which is < 0.1% — a 1000x
          improvement over the old threshold of 0.1 (where the
          discontinuity was ~0.5%). This makes the first-derivative
          discontinuity at the threshold boundary negligible for
          gradient-based optimization of the modulation index.
        """
        # 9th-review fix (F-03): use the scm_enabled flag to determine
        # whether SCM is active, instead of the ``m == 0.0`` check.
        # The previous implementation conflated the physical SCM switch
        # with the continuous modulation_index parameter, creating a
        # semantic discontinuity at m=0 (where the interpretation of
        # mu switches between "post-SCM" and "total emitted energy").
        # The new scm_enabled flag decouples the switch from the
        # parameter, allowing m=0 to represent "SCM is active but the
        # modulation index is zero" (which gives scaling factor 0, not 1).
        # When scm_enabled=False, the base mu is interpreted as post-SCM
        # and the scaling factor is 1.0 (no modification).
        if not getattr(self, 'scm_enabled', SCM_ENABLED_DEFAULT):
            # SCM disabled: no scaling. mu is interpreted as post-SCM.
            return 1.0
        m = self.modulation_index

        # 7th-review fix (issue I.4): use the per-instance ``scm_num_sidebands``
        # attribute instead of the module-level ``SCM_NUM_SIDEBANDS`` constant.
        # This eliminates the thread-safety / race-condition hazard identified
        # in the 7th review, where modifying the module-level constant to
        # switch between SSB (n=1) and DSB (n=2) detection would affect all
        # sources in the process simultaneously. Each source now reads its
        # own per-instance value, defaulting to the module constant for
        # backwards compatibility.
        nsb = self.scm_num_sidebands

        if self.use_linear_modulation_approximation:
            # Forced linear approximation (numerical shortcut, regardless of m).
            return float(nsb * (m ** 2) / SCM_POWER_DIVISOR)

        if self.use_small_angle_approximation and m <= SCM_SMALL_ANGLE_LIMIT:
            # Physical small-angle approximation: J1(m) ~ m/2 for m << 1.
            # At m=1e-3, the relative error is ~2.5e-7 (negligible).
            return float(nsb * (m ** 2) / SCM_POWER_DIVISOR)

        return float(nsb * (jv(1, m)) ** 2)

    def _calculate_effective_mus(
        self, base_mus: np.ndarray, rng: RNGType, pulse_positions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply classical intensity imperfections to base mus.

        Order of operations (issue 2.3 in 3rd review — physical order):
          1. **Intensity jitter** (laser intensity fluctuation). Physically,
             pulse-to-pulse intensity jitter originates in the semiconductor
             laser diode *before* external modulation. Applying it first
             ensures that MZM electrical-driver noise and extinction-ratio
             leakage scale correctly with the jittered pulse energy.
          2. **MZM attenuation** (with electrical-driver voltage noise).
          3. **SCM sideband factor** (phase modulation, applied after
             intensity modulation).
          4. **Channel split**: ``mu / N_channels`` (energy-conserving
             uniform split, applied last since it represents the channel
             multiplexing stage after the source has emitted).

        For purely multiplicative models (linear MZM, multiplicative SCM),
        this order is mathematically equivalent to the previous order
        (MZM -> Split -> SCM -> Jitter). The reordering matters for
        non-multiplicative MZM models (e.g., with extinction-ratio leakage
        floors or additive electrical driver noise), where the previous
        order would incorrectly scale the MZM noise by the laser jitter.

        6th-review fix (issue 5): ``pulse_positions`` is an optional
        parameter that, when provided, is passed to
        :meth:`_apply_intensity_jitter` to map adversarial block factors
        to explicit physical pulse positions rather than sequential call
        order. This decouples the ADVERSARIAL_BLOCK error model from
        function invocation order, allowing non-sequential pulse access
        patterns to maintain correct block correlation.
        """
        # Step 1: Intensity jitter (laser fluctuation, before modulation).
        effective_mus = self._apply_intensity_jitter(base_mus, rng, pulse_positions)

        # Step 2: MZM attenuation (with electrical-driver voltage noise).
        effective_mus = self.mzm.apply(
            effective_mus, self._driver_voltage_noise_std, rng
        )

        # Step 3: SCM sideband factor.
        effective_mus = effective_mus * self._scm_scaling_factor()

        # Step 4: Channel split.
        if self.N_channels > 1:
            # Energy-conserving uniform split across N channels. Assumes
            # perfectly balanced channels with no extra loss or imbalance.
            # For unbalanced multiplexing, model the imbalance explicitly
            # downstream.
            effective_mus = effective_mus / float(self.N_channels)

        return effective_mus

    def calculate_effective_mus(
        self,
        pulse_indices: Optional[Union[int, np.ndarray, Sequence[int]]],
        rng: RNGType,
        num_samples: int = 1,
        *,
        pulse_positions: Optional[Union[np.ndarray, Sequence[int]]] = None,
    ) -> Union[np.ndarray, float]:
        """Compute effective mean photon numbers after all classical effects.

        Parameters
        ----------
        pulse_indices : int, array-like of int, or None
            If None, ``num_samples`` pulse indices are sampled randomly.
            Otherwise, the given indices are used.
        rng : numpy.random.Generator
            Random number generator (used for MZM noise, channel split
            randomness if any, and pulse sampling if ``pulse_indices is None``).
        num_samples : int, default 1
            Number of pulse indices to sample if ``pulse_indices is None``.
            Must be >= 0 (issue 1.5 in 3rd review: ``num_samples=0`` is
            allowed and returns an empty array, consistent with
            :meth:`sample_pulse_indices`).
        pulse_positions : optional array-like of int, keyword-only
            6th-review fix (issue 5): explicit physical pulse positions for
            the ADVERSARIAL_BLOCK error model. When provided, block IDs are
            computed from ``pulse_positions // adversarial_block_size``
            instead of sequential call order, decoupling block correlation
            from function invocation order. Must have the same length as
            ``pulse_indices`` (or ``num_samples`` if ``pulse_indices`` is
            None). Ignored for non-ADVERSARIAL_BLOCK error models.

        Returns
        -------
        np.ndarray or float
            Effective mean photon numbers. Scalar float if ``pulse_indices``
            was a scalar; otherwise 1-D float64 array.

        4th-review fix (issue 2.7 — side effects in analytical methods):
            The previous implementation mutated ``_adversarial_block_counter``
            and ``_adversarial_current_factor`` whenever the
            ``ADVERSARIAL_BLOCK`` error model was active, even though this
            method is documented as a read-only analytical utility for
            inspecting mean photon numbers. Inspecting the source would
            permanently alter its subsequent simulation output, violating
            the principle of least surprise. The new implementation saves
            and restores the adversarial-block state around the call, so
            ``calculate_effective_mus`` is now a true read-only operation.

        6th-review fix (issue 13): removed the redundant ``if num_samples < 0``
            check. ``sample_pulse_indices`` already validates ``num_samples``
            (rejects bool, requires integer, requires >= 0), so the check
            here was triplicated boilerplate. The same fix was applied to
            ``generate_photons``, ``generate_photons_dm``, and
            ``DensityMatrixSource.calculate_effective_mus``.
        """
        if pulse_indices is None:
            # 6th-review fix (issue 13): rely on sample_pulse_indices for
            # num_samples validation (rejects bool, requires integer >= 0).
            indices = self.sample_pulse_indices(rng, num_samples)
            is_scalar = False
        else:
            indices, is_scalar = self._validate_pulse_indices(pulse_indices)

        if len(indices) == 0:
            # Issue 2.5: scalar input cannot produce an empty array.
            if is_scalar:
                raise ParameterValidationError(
                    "Scalar pulse_indices produced an empty array; "
                    "this is a logical inconsistency.",
                    param_name="pulse_indices",
                )
            return np.array([], dtype=np.float64)

        # 4th-review fix (issue 2.7): save and restore the ADVERSARIAL_BLOCK
        # state so that this analytical/inspection method does not mutate
        # the source's simulation state. Previously, calling
        # calculate_effective_mus with ADVERSARIAL_BLOCK active would
        # permanently advance _adversarial_block_counter and overwrite
        # _adversarial_current_factor, corrupting subsequent generate_photons
        # calls. The save/restore pattern makes this method a true no-side-
        # effect analytical utility, matching its documented semantics.
        # 6th-review note (issue 5): when pulse_positions is provided, the
        # ADVERSARIAL_BLOCK model doesn't use the stateful counter, so the
        # save/restore is a no-op (but harmless).
        #
        # 7th-review fix (issue II.2): ALSO save and restore the RNG state.
        # The previous save/restore only covered the adversarial-block
        # counter and factor, but the call to ``_calculate_effective_mus``
        # passes the shared simulation RNG into ``_apply_intensity_jitter``
        # and ``mzm.apply``, both of which consume random numbers when
        # stochastic error models (RANDOM_GAUSSIAN, ADVERSARIAL_BLOCK) or
        # electrical noise models are active. Consuming RNG numbers during
        # a documented "read-only analytical utility" permanently alters
        # the generator's state, causing subsequent simulation calls to
        # ``generate_photons`` to diverge and destroying simulation
        # reproducibility. The fix saves ``rng.bit_generator.state`` before
        # the call and restores it in the ``finally`` block, so the RNG
        # stream is identical to what it would have been if
        # ``calculate_effective_mus`` had never been called. This is
        # safe because the analytical return value is computed and stored
        # in ``mus`` before the restore; the restore only affects
        # SUBSEQUENT RNG draws (which belong to the next simulation call).
        # NOTE: this save/restore is a measurable overhead (a dict
        # copy + bit_generator state set per call). Users who call
        # ``calculate_effective_mus`` in tight analytical loops on
        # deterministic configurations (no jitter, no MZM noise, no
        # adversarial blocks) may want to bypass it; we judge the
        # reproducibility guarantee to be worth the overhead for the
        # general case.
        saved_counter = self._adversarial_block_counter
        saved_factor = self._adversarial_current_factor
        saved_rng_state = rng.bit_generator.state
        try:
            # Issue 2.3 in 3rd review: _calculate_effective_mus now applies
            # intensity jitter internally (as the first step, physically
            # before MZM). We no longer call _apply_intensity_jitter
            # separately here.
            # 6th-review fix (issue 5): pass pulse_positions through.
            pp = None
            if pulse_positions is not None:
                pp = np.asarray(pulse_positions, dtype=np.int64)
                if pp.shape != (len(indices),):
                    raise ParameterValidationError(
                        f"pulse_positions must have shape ({len(indices)},), "
                        f"got {pp.shape}.",
                        param_name="pulse_positions",
                    )
            mus = self._calculate_effective_mus(
                self._base_mus_cache[indices], rng, pulse_positions=pp
            )
        finally:
            # Restore the state regardless of whether the call succeeded.
            self._adversarial_block_counter = saved_counter
            self._adversarial_current_factor = saved_factor
            # 7th-review fix (issue II.2): restore the RNG state so that
            # the analytical call leaves no trace on the shared RNG stream.
            rng.bit_generator.state = saved_rng_state
        return float(mus.item()) if is_scalar else mus

    def _apply_intensity_jitter(
        self, mus: np.ndarray, rng: RNGType, pulse_positions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply pulse-to-pulse intensity jitter to effective mus.

        Models multiplicative intensity fluctuation. Two error models:

        **RANDOM_GAUSSIAN** (issue 3.1, 3.2, 3.3; 2nd-review issue 3;
        3rd-review issues 2.4, 3.1; 4th-review issue 1.4):
          The model uses an additive Gaussian noise on the intensity:
          ``mu_new = mu * |1 + jitter * z|``, where ``z ~ N(0, 1)``. The
          reflection (``|.|``) is used instead of clipping (``max(0, .)``)
          because reflection's mean bias is second-order in ``P(X<0)``,
          while clipping's bias is first-order.

          9th-review fix (F-02): analytical mean-bias correction added. The reflection model mu_new = mu * |1 + jitter * z| has E[|1 + jitter * z|] > 1 (positive bias). The exact bias factor E[|1 + sigma * z|] = sigma * sqrt(2/pi) * exp(-1/(2*sigma^2)) + erf(1/(sigma * sqrt(2))) is now computed and divided out, making the corrected model mean-preserving (E[mu_new] = mu) to within floating-point precision. For small jitter (< 0.1), the correction is negligible (bias factor indistinguishable from 1.0 in double precision). For larger jitter, the correction removes the systematic positive bias that violated the mean-preserving assumption in decoy-state security analyses.

          4th-review fix (issue 1.4 — contradictory probability
          distribution): the previous implementation silently substituted
          a log-normal distribution for ``jitter >= GAUSSIAN_LIMIT_JITTER``
          (1e-3), violating the ``RANDOM_GAUSSIAN`` enum contract. The
          new implementation ALWAYS uses Gaussian (with reflection),
          matching the enum name. The reflection bias
          ``E[|X|] - E[X] = 2 * |E[X|X<0]| * P(X<0)`` grows with jitter
          (e.g., ~2% mean bias at jitter=0.5, ~16% at jitter=1.0); users
          who require strictly positive intensity fluctuations with exact
          mean preservation should request a dedicated ``RANDOM_LOGNORMAL``
          error model (not yet implemented in the ``SourceErrorModel``
          enum). The reflection bias is documented here so users can make
          an informed choice.

          Mathematical note (issue 3.1 in 3rd review): the previous
          docstring made the mathematical error of claiming
          ``E[|X|] = E[X]``. The correct relationship is
          ``E[|X|] > E[X]`` for any ``X ~ N(1, sigma^2)`` with
          ``sigma > 0``. For small jitter, ``P(X < 0) = Q(1/jitter)``
          is astronomically small (e.g., ``< 1e-300`` for
          ``jitter = 1e-3``), so the bias is effectively zero in double
          precision. For larger jitter, the bias becomes non-negligible
          and is the user's responsibility to account for.

          The artificial ``1e-4`` floor on ``jitter`` is **removed**: if
          ``jitter <= NUMERIC_ABS_TOL``, no jitter is applied.

        **ADVERSARIAL_BLOCK** (issue 3.4, 3.5; 3rd-review issue 2.6;
        4th-review issue 2.6):
          Block-wise lognormal multiplicative fluctuation. Pulses are
          grouped into consecutive blocks of size
          :attr:`adversarial_block_size`; all pulses within a block share
          the **same** fluctuation factor. This models an adversary who can
          manipulate source intensity in a correlated way over blocks.

          **Stateful block tracking (3rd-review fix):** Block boundaries
          are tracked **across calls** via ``_adversarial_block_counter``
          and ``_adversarial_current_factor``. The previous stateless
          design sampled a new lognormal factor per call, which broke
          block correlation when the simulation was chunked into small
          batches.

          **4th-review note (issue 2.6 — non-sequential pulse sampling):**
          Block correlation is tracked by *call order*, not by physical
          pulse position. When a caller passes specific, non-sequential,
          or repeated pulse indices to ``generate_photons`` or
          ``calculate_effective_mus``, the counter increments by the
          length of the input array, assigning non-sequential pulses to
          contiguous adversarial blocks. This was a known limitation:
          the source had no way to know the "true" physical pulse
          position unless the caller passed it explicitly.

          **6th-review fix (issue 5 — explicit pulse positions):**
          The optional ``pulse_positions`` parameter decouples block ID
          computation from call order. When provided (a 1-D array of
          non-negative integers specifying the physical pulse position of
          each entry in ``mus``), block IDs are computed directly from
          ``pulse_positions // block_size``, and the stateful counter is
          NOT advanced. This allows non-sequential access patterns (e.g.,
          querying decoy pulses separately from signal pulses) to maintain
          correct block correlation. When ``pulse_positions`` is ``None``
          (the default), the legacy call-order behavior is preserved for
          backwards compatibility. Callers that need block correlation
          tied to physical pulse positions should pass ``pulse_positions``
          explicitly to ``generate_photons`` / ``calculate_effective_mus``.

        Parameters
        ----------
        mus : np.ndarray
            1-D float64 array of effective mean photon numbers.
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        np.ndarray
            Jittered mean photon numbers (same shape as input).
        """
        if self.intensity_jitter <= NUMERIC_ABS_TOL:
            return mus

        mus = np.asarray(mus, dtype=np.float64).copy()

        if self.error_model == SourceErrorModel.RANDOM_GAUSSIAN:
            # 6th-review fix (issue 4): apply jitter to ALL strictly positive
            # pulses, not just those above NUMERIC_ABS_TOL. The previous
            # ``mus > NUMERIC_ABS_TOL`` mask silently bypassed jitter for
            # weak decoy states with 0 < mu < NUMERIC_ABS_TOL (e.g.,
            # mu = 1e-15), which is physically inconsistent: semiconductor
            # laser intensity fluctuations affect all emitted pulses
            # regardless of subsequent optical attenuation. The new ``> 0.0``
            # mask only excludes true vacuum pulses (mu == 0), for which
            # multiplicative jitter is a no-op (0 * anything = 0) and
            # consuming an RNG number would be wasteful.
            # 8th-review fix (F-10): ALWAYS consume N RNG draws for
            # RANDOM_GAUSSIAN, not just non-zero pulses. The previous
            # non_zero_mask = mus > 0.0 selectively drew RNG numbers,
            # desynchronizing the RNG stream across parameter sweeps
            # that change the decoy mixture (and thus vacuum fraction).
            # For vacuum pulses (mu == 0), the factor is irrelevant
            # (0 * anything = 0), but consuming the draw keeps the
            # stream consistent across all configurations.
            jitter = float(self.intensity_jitter)
            z = rng.standard_normal(size=mus.shape)  # ALWAYS N draws
            factor = 1.0 + jitter * z
            mus = mus * np.abs(factor)  # 0 * |factor| = 0 for vacuum
            # 9th-review fix (F-02): analytical bias correction.
            # E[|1 + sigma * z|] where z ~ N(0,1):
            #   = sigma * sqrt(2/pi) * exp(-1/(2*sigma^2))
            #     + erf(1/(sigma * sqrt(2)))
            # For sigma < NUMERIC_ABS_TOL, bias_factor = 1.0 (no correction).
            if jitter > NUMERIC_ABS_TOL:
                inv_sigma = 1.0 / jitter
                bias_factor = (
                    jitter * np.sqrt(2.0 / np.pi) * np.exp(-0.5 * inv_sigma * inv_sigma)
                    + scipy_erf(inv_sigma / np.sqrt(2.0))
                )
                mus = mus / bias_factor  # mean-preserving correction
            return mus

        if self.error_model == SourceErrorModel.ADVERSARIAL_BLOCK:
            # 9th-review fix (F-08): removed silent clamping of block_size
            # via max(1, ...). A block_size < 1 is a configuration error
            # and should be caught during validation, not silently coerced.
            block_size = int(self.adversarial_block_size)
            n = len(mus)
            if n == 0:
                return mus

            sigma_sq = np.log(1.0 + self.intensity_jitter ** 2)

            # 6th-review fix (issue 5): if pulse_positions is provided, use
            # them for block ID computation instead of the call-order counter.
            # This decouples the adversarial block correlation from function
            # invocation order, allowing non-sequential pulse access patterns
            # (e.g., querying decoy pulses separately from signal pulses, or
            # accessing non-sequential time slots) to maintain correct block
            # correlation. When pulse_positions is None, the legacy call-order
            # behavior is preserved.
            #
            # 7th-review fix (issue I.1): maintain a PERSISTENT cache mapping
            # block_id -> lognormal factor across calls when pulse_positions
            # is provided. The previous implementation sampled fresh factors
            # for ALL blocks on every call, which broke block correlation
            # across non-sequential access patterns: querying signal pulses in
            # Call 1 (positions [0..999], block 0) and decoy pulses in Call 2
            # (positions [10, 50, 100], also block 0) would assign independent,
            # uncorrelated fluctuation factors to pulses within the same
            # physical block — violating the definition of block-wise
            # correlation and contradicting the 6th-review fix's documented
            # claim that "supplying pulse_positions maintains correct block
            # correlation across non-sequential access patterns".
            #
            # The fix introduces ``_adversarial_block_factor_cache``, a dict
            # mapping block_id -> factor that persists across calls. For each
            # unique block_id in the current call:
            #   * If the block_id is already cached, reuse the cached factor
            #     (no RNG number consumed).
            #   * If the block_id is NOT cached, sample a fresh factor and
            #     store it in the cache for future calls.
            # This ensures that the same physical block always receives the
            # same fluctuation factor, regardless of how the caller chunks
            # the queries. The cache is keyed by block_id (int) and grows
            # monotonically; for very long simulations, the cache size is
            # bounded by (total_pulses / block_size), which is typically
            # manageable (e.g., 1M pulses / 1000-pulse blocks = 1000 entries).
            if pulse_positions is not None:
                pulse_positions = np.asarray(pulse_positions, dtype=np.int64)
                if pulse_positions.shape != (n,):
                    raise ParameterValidationError(
                        f"pulse_positions must have shape ({n},), got "
                        f"{pulse_positions.shape}.",
                        param_name="pulse_positions",
                    )
                if np.any(pulse_positions < 0):
                    raise ParameterValidationError(
                        "pulse_positions contains negative values; all "
                        "physical pulse positions must be >= 0.",
                        param_name="pulse_positions",
                    )
                block_ids = pulse_positions // block_size
                # Group by unique block ID and sample one factor per block.
                # 8th-review fix (F-29): preserve input order for block IDs.
                # np.unique sorts block_ids, making the RNG stream depend on
                # whether pulse_positions is sorted. Preserve insertion order
                # for reproducibility regardless of input ordering.
                seen_blocks = OrderedDict()
                for bid in block_ids:
                    bid_int = int(bid)
                    if bid_int not in seen_blocks:
                        seen_blocks[bid_int] = len(seen_blocks)
                unique_block_ids = np.array(list(seen_blocks.keys()), dtype=np.int64)
                inverse = np.array([seen_blocks[int(bid)] for bid in block_ids], dtype=np.int64)

                # 7th-review fix (issue I.1): look up cached factors and
                # sample fresh factors only for uncached blocks.
                cached_factors: List[float] = []
                uncached_block_positions: List[int] = []  # positions in unique_block_ids
                uncached_block_ids: List[int] = []
                for pos, bid in enumerate(unique_block_ids):
                    bid_int = int(bid)
                    if bid_int in self._adversarial_block_factor_cache:
                        cached_factors.append(
                            self._adversarial_block_factor_cache[bid_int]
                        )
                    else:
                        cached_factors.append(0.0)  # placeholder, filled below
                        uncached_block_positions.append(pos)
                        uncached_block_ids.append(bid_int)

                if uncached_block_ids:
                    # Sample fresh factors for the uncached blocks.
                    fresh = rng.lognormal(
                        mean=-0.5 * sigma_sq,
                        sigma=np.sqrt(sigma_sq),
                        size=len(uncached_block_ids),
                    )
                    for pos, bid_int, factor in zip(
                        uncached_block_positions, uncached_block_ids, fresh
                    ):
                        cached_factors[pos] = float(factor)
                        self._adversarial_block_factor_cache[bid_int] = float(factor)

                new_factors = np.array(cached_factors, dtype=np.float64)
                factors = new_factors[inverse]
                # Do NOT update _adversarial_block_counter or
                # _adversarial_current_factor when pulse_positions is
                # provided — the caller manages block assignment explicitly,
                # and the persistent cache (not the counter) is the source
                # of truth for block correlation in this mode.
                return mus * factors

            # 9th-review fix (F-05): legacy call-order mode is deprecated.
            # Block correlation tied to call order, not physical pulse position,
            # produces incorrect correlation structure for non-sequential or
            # chunked access patterns. Users should pass pulse_positions
            # explicitly.
            logger.warning(
                "ADVERSARIAL_BLOCK with pulse_positions=None: using "
                "deprecated legacy call-order tracking. Block correlation "
                "is tied to function call order, NOT physical pulse "
                "position, which produces incorrect correlation structure "
                "for non-sequential access patterns. Pass pulse_positions "
                "explicitly for correct physics. Legacy mode will be "
                "removed in a future version.",
            )

            # Legacy call-order tracking (original behavior).
            # Issue 2.6 in 3rd review: stateful block tracking across calls.
            # Compute global pulse indices (across all calls) and their
            # block IDs. This ensures block boundaries are respected
            # regardless of how many pulses are requested per call.
            #
            # 4th-review note (issue 2.6): block correlation is tracked
            # by call order, not physical pulse position. See the method
            # docstring above for the documented limitation.
            global_indices = np.arange(n) + self._adversarial_block_counter
            block_ids = global_indices // block_size

            # Unique block IDs in this call (sorted ascending).
            unique_block_ids, inverse = np.unique(block_ids, return_inverse=True)

            # 5th-review fix (issue 3.1): do NOT consume an RNG number for
            # the cached block. The previous implementation ALWAYS sampled
            # ``len(unique_block_ids)`` factors via ``rng.lognormal(...)``,
            # then overwrote the first factor with the cached value if the
            # first block was a continuation of the previous call's last
            # block. This wasted an RNG number for the cached block, which
            # desynchronized the RNG stream across different batch sizes:
            # the same simulation run with different chunk sizes would
            # produce different results because the number of "wasted" RNG
            # draws depended on how many times a call spanned a cached
            # block boundary.
            #
            # The fix: detect whether the first block is cached BEFORE
            # sampling. If so, sample only ``len(unique_block_ids) - 1``
            # factors (for the non-cached blocks) and prepend the cached
            # factor. If all blocks in this call are the cached block
            # (happens when n < block_size and the call continues the
            # cached block), sample 0 factors and use the cached factor
            # for all pulses.
            first_block_is_cached = False
            if self._adversarial_block_counter > 0:
                last_block_id = (self._adversarial_block_counter - 1) // block_size
                if unique_block_ids[0] == last_block_id:
                    first_block_is_cached = True

            if first_block_is_cached:
                # Sample only the non-cached blocks (all except the first).
                num_to_sample = len(unique_block_ids) - 1
                if num_to_sample > 0:
                    sampled = rng.lognormal(
                        mean=-0.5 * sigma_sq,
                        sigma=np.sqrt(sigma_sq),
                        size=num_to_sample,
                    )
                    new_factors = np.empty(len(unique_block_ids), dtype=np.float64)
                    new_factors[0] = self._adversarial_current_factor
                    new_factors[1:] = sampled
                else:
                    # All blocks in this call are the cached block.
                    new_factors = np.array(
                        [self._adversarial_current_factor], dtype=np.float64
                    )
            else:
                # No cached block; sample all factors.
                new_factors = rng.lognormal(
                    mean=-0.5 * sigma_sq,
                    sigma=np.sqrt(sigma_sq),
                    size=len(unique_block_ids),
                )

            # Map per-pulse block_ids to factors via the inverse index.
            factors = new_factors[inverse]

            # Cache the last factor and update the counter for the next call.
            self._adversarial_current_factor = float(new_factors[-1])
            self._adversarial_block_counter += n

            return mus * factors


        # 8th-review fix (F-09, F-75): RANDOM_LOGNORMAL error model.
        # A log-normal multiplicative model for intensity fluctuations
        # that preserves the mean photon number EXACTLY (E[mu_new] = mu).
        # Unlike RANDOM_GAUSSIAN (which uses |1 + sigma*z| and introduces
        # ~16% positive bias at sigma=1.0), log-normal is strictly positive
        # and mean-preserving. This branch will only be active once
        # RANDOM_LOGNORMAL is added to SourceErrorModel enum.
        if hasattr(SourceErrorModel, 'RANDOM_LOGNORMAL') and self.error_model == SourceErrorModel.RANDOM_LOGNORMAL:
            non_zero_mask = mus > 0.0
            if not np.any(non_zero_mask):
                return mus
            mu_nz = mus[non_zero_mask]
            sigma = float(self.intensity_jitter)
            sigma_sq = np.log(1.0 + sigma ** 2)
            factor = rng.lognormal(
                mean=-0.5 * sigma_sq,
                sigma=np.sqrt(sigma_sq),
                size=mu_nz.shape,
            )
            mus[non_zero_mask] = mu_nz * factor
            return mus
        # Unknown error model: no-op with a warning.
        logger.warning(
            "Unknown SourceErrorModel %r; no intensity jitter applied.",
            self.error_model,
        )
        return mus

    def _sample_poisson_or_thermal(self, mus: np.ndarray, rng: RNGType) -> np.ndarray:
        """Sample photon numbers from Poisson or thermal (geometric) statistics.

        For POISSON: ``n ~ Poisson(mu)``.
        For THERMAL: ``n ~ Geometric(p=1/(mu+1)) - 1`` (Bose-Einstein).
          * ``mu = 0`` -> ``p = 1`` -> ``n = 0`` (vacuum, exact).
          * ``mu > 0`` -> mean of (n-1) is ``mu``, variance is ``mu*(mu+1)``.

        Notes
        -----
        The previous implementation used ``Y1_SAFE_THRESHOLD`` to clamp
        ``mus`` away from zero in the thermal branch. This is **removed**
        (issue 3.9): for ``mu = 0`` the formula ``1/(0+1) = 1`` is exact
        and gives the correct vacuum distribution, so the clamp was
        unnecessary and would have biased vacuum pulses.
        """
        # 8th-review fix (F-07): validate finiteness of mus before sampling.
        # Extreme jitter or MZM noise can produce infinite effective mus,
        # which would crash rng.poisson(np.inf) or produce nonsensical
        # photon counts from rng.geometric(p=1e-15).
        mus_arr = np.asarray(mus, dtype=np.float64)
        if not np.all(np.isfinite(mus_arr)):
            bad = mus_arr[~np.isfinite(mus_arr)]
            raise ParameterValidationError(
                f'Effective mus contain non-finite values: {bad[:10].tolist()}. '
                f'This typically indicates extreme jitter, MZM noise, or a '
                f'numerical overflow in the intensity-imperfection pipeline.',
                param_name='mus',
            )
        # 9th-review fix (F-09 + F-10): raise errors for negative and
        # extreme mu instead of silently clamping. Clamping masked
        # configuration errors and violated the mean-preserving contract
        # (a clamped mu no longer represents the intended mean photon
        # number, breaking decoy-state security assumptions). The new
        # behavior forces users to investigate the root cause.
        if np.any(mus_arr > MAX_MU):
            bad = mus_arr[mus_arr > MAX_MU][:5]
            raise ParameterValidationError(
                f'Effective mus exceed MAX_MU={MAX_MU} '
                f'(max={bad.max():.6g}, values: {bad.tolist()}). '
                f'This typically indicates extreme jitter or MZM noise '
                f'that has pushed the mean photon number beyond the '
                f'sampling range. Reduce intensity_jitter or MZM noise, '
                f'or increase MAX_MU if the physics genuinely requires '
                f'larger counts.',
                param_name='mus',
            )
        if np.any(mus_arr < 0):
            bad = mus_arr[mus_arr < 0][:5]
            raise ParameterValidationError(
                f'Effective mus contain negative values '
                f'(min={bad.min():.6g}, values: {bad.tolist()}). '
                f'Mean photon numbers must be non-negative. Negative '
                f'values indicate numerical instability in the '
                f'intensity-imperfection pipeline (e.g., additive MZM '
                f'noise producing a negative effective intensity).',
                param_name='mus',
            )
        if self.statistics_type == SourceStatisticsType.POISSON:
            return rng.poisson(mus_arr).astype(np.int64, copy=False)

        # Thermal / Bose-Einstein statistics via geometric distribution.
        # p = 1/(mu+1) in (0, 1]; for mu=0, p=1 (geometric always returns 1,
        # so n-1=0 = vacuum).
        probs = 1.0 / (mus_arr + 1.0)
        # 6th-review fix (issue 10): clip to a STRICTLY POSITIVE lower bound.
        # The previous ``np.clip(probs, 0.0, 1.0)`` allowed ``probs = 0.0``
        # (e.g., from numerical underflow when ``mus_arr`` overflows to
        # ``inf``), which would crash ``rng.geometric(p=0.0)`` with a
        # ``ValueError`` (the geometric distribution requires p in (0, 1]).
        # The new lower bound ``1e-15`` (smallest positive float64 ~1e-308,
        # but 1e-15 is safely above underflow for thermal physics) ensures
        # the domain is respected. For finite ``mus_arr >= 0``, this clip
        # is a no-op since ``1/(mu+1)`` is already in (0, 1].
        probs = np.clip(probs, 1e-15, 1.0)
        return (rng.geometric(p=probs) - 1).astype(np.int64, copy=False)

    def generate_photons(
        self,
        alice_pulse_indices: Optional[Union[int, np.ndarray, Sequence[int]]],
        rng: RNGType,
        num_samples: int = 1,
        *,
        pulse_positions: Optional[Union[np.ndarray, Sequence[int]]] = None,
    ) -> Union[np.ndarray, int]:
        """Generate photon counts for the given (or sampled) pulse indices.

        Parameters
        ----------
        alice_pulse_indices : int, array-like of int, or None
            If None, ``num_samples`` pulse indices are sampled randomly.
        rng : numpy.random.Generator
            Random number generator.
        num_samples : int, default 1
            Number of pulse indices to sample if ``alice_pulse_indices is None``.
            Must be >= 0 (issue 1.5 in 3rd review: ``num_samples=0`` is
            allowed and returns an empty array, consistent with
            :meth:`sample_pulse_indices`).
        pulse_positions : optional array-like of int, keyword-only
            6th-review fix (issue 5): explicit physical pulse positions for
            the ADVERSARIAL_BLOCK error model. When provided, block IDs are
            computed from ``pulse_positions // adversarial_block_size``
            instead of sequential call order, decoupling block correlation
            from function invocation order. Must have the same length as
            ``alice_pulse_indices`` (or ``num_samples`` if ``alice_pulse_indices``
            is None). Ignored for non-ADVERSARIAL_BLOCK error models.

        Returns
        -------
        np.ndarray or int
            Photon counts. Scalar int if ``alice_pulse_indices`` was a
            scalar; otherwise 1-D int64 array.

        Notes
        -----
        Emission failure (``ideal_emission_probability < 1``) is modelled
        as a *thinning* (loss) operation: with probability
        ``1 - p_emit`` the photon count is set to 0; with probability
        ``p_emit`` the photon count is sampled from the configured
        distribution. This is the physically correct model for source
        emission failure (issue 3.6, 3.7, 3.8).

        Security-metadata tally (issue 2 in 2nd review):
          * ``num_pulses_attempted`` is recorded as the total number of
            pulses Alice was asked to emit (``len(indices)``).
          * ``num_pulses_generated`` is recorded as the number of pulses
            Alice *actually emitted* (post-thinning), i.e. the number of
            pulses whose ``emit_mask`` was True. Pulses that failed to emit
            are NOT counted as generated, which is critical for accurate
            yield / key-rate analyses downstream.

        6th-review fix (issue 13): removed the redundant ``if num_samples < 0``
        check. ``sample_pulse_indices`` already validates ``num_samples``.
        """
        if alice_pulse_indices is None:
            # 6th-review fix (issue 13): rely on sample_pulse_indices for
            # num_samples validation.
            indices = self.sample_pulse_indices(rng, num_samples)
            is_scalar = False
        else:
            indices, is_scalar = self._validate_pulse_indices(alice_pulse_indices)

        if len(indices) == 0:
            if is_scalar:
                raise ParameterValidationError(
                    "Scalar alice_pulse_indices produced an empty array; "
                    "this is a logical inconsistency.",
                    param_name="alice_pulse_indices",
                )
            return np.array([], dtype=np.int64)

        # Issue 2.3 in 3rd review: _calculate_effective_mus now applies
        # intensity jitter internally (as the first step, physically before
        # MZM). We no longer call _apply_intensity_jitter separately here.
        # 6th-review fix (issue 5): pass pulse_positions through.
        pp = None
        if pulse_positions is not None:
            pp = np.asarray(pulse_positions, dtype=np.int64)
            if pp.shape != (len(indices),):
                raise ParameterValidationError(
                    f"pulse_positions must have shape ({len(indices)},), "
                    f"got {pp.shape}.",
                    param_name="pulse_positions",
                )
        selected_mus = self._calculate_effective_mus(
            self._base_mus_cache[indices], rng, pulse_positions=pp
        )

        # 8th-review fix (F-41): validate finiteness of selected_mus.
        if not np.all(np.isfinite(selected_mus)):
            bad = selected_mus[~np.isfinite(selected_mus)]
            raise ParameterValidationError(
                f'Effective mus contain non-finite values: {bad[:10].tolist()}',
                param_name='effective_mus',
            )

        # Sample photon numbers from the configured distribution.
        photons = self._sample_poisson_or_thermal(selected_mus, rng)

        # Apply emission-failure thinning (issue 3.6, 3.7, 3.8, 8.1).
        # Track which pulses were actually emitted for accurate tally.
        # 6th-review fix (issue 6): ALWAYS consume ``n_attempted`` random
        # numbers from the RNG, even when ``p_emit`` is close to 0.0 or 1.0.
        # The previous implementation branched on ``is_close(p_emit, 1.0)``
        # and ``is_close(p_emit, 0.0)``, skipping the ``rng.random()`` call
        # in those branches. This desynchronized the RNG stream across
        # simulation runs whenever ``p_emit`` crossed the floating-point
        # tolerance boundary (e.g., from 1.0 - 1e-10 to 1.0 - 1e-8), making
        # parameter sweeps and gradient-based optimization non-reproducible.
        # The new implementation draws the mask unconditionally and applies
        # it via ``np.where``, preserving RNG stream consistency across
        # parameter changes.
        #
        # 7th-review fix (issue II.3): correct a misleading comment that
        # claimed "The ``is_close`` shortcuts are kept only for the
        # ``p_emit == 0.0`` exact-zero case". The actual code has NO
        # branching or shortcut for ``p_emit == 0.0``: the random number
        # draw (``rng.random(size=n_attempted) < p_emit``) occurs
        # unconditionally for ALL values of ``p_emit``, including exact
        # zero. The previous comment stated a false claim about the code's
        # execution. This fix removes the misleading statement and
        # documents the actual behavior: the RNG draw is ALWAYS performed
        # (no shortcuts, no branching), at the cost of one ``rng.random``
        # call per ``generate_photons`` invocation regardless of
        # ``p_emit``. The cost is negligible compared to the
        # ``_sample_poisson_or_thermal`` call above, and the benefit is
        # that the RNG stream is invariant under changes to ``p_emit``
        # (critical for parameter-sweep reproducibility).
        p_emit = clamp_probability(self.ideal_emission_probability)
        n_attempted = len(indices)
        # Unconditional RNG draw: preserves stream consistency.
        # No is_close branching, no shortcut for p_emit==0.0 or p_emit==1.0.
        emit_mask = rng.random(size=n_attempted) < p_emit
        photons = np.where(emit_mask, photons, 0).astype(np.int64, copy=False)
        n_emitted = int(np.count_nonzero(emit_mask))

        # Record tallies: attempted = all pulses requested, emitted = the
        # subset that actually emitted a (possibly zero-photon) pulse. The
        # "sent" tally on security_metadata is updated with the emitted
        # count (see update_internal_tallies).
        self.update_internal_tallies(
            num_pulses_generated=n_emitted,
            num_pulses_attempted=n_attempted,
        )
        return int(photons[0]) if is_scalar else photons


# ---------------------------------------------------------------------------
# DensityMatrixSource
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DensityMatrixSource(OpticalSource):
    """
    Source defined by density matrices in the Fock basis.

    For each pulse type, the diagonal of the density matrix defines the
    photon-number distribution used for sampling. Off-diagonal coherences
    are validated (Hermiticity, PSD, trace) but otherwise discarded by the
    sampler.

    Requirements
    ------------
    * The density matrix MUST be expressed in the Fock basis
      ``{|0>, |1>, |2>, ...}``. The sampler does not verify the basis.
    * The matrix MUST be Hermitian, PSD, and have trace 1.
    * The mean photon number computed from the diagonal
      ``sum_n n * p_n`` SHOULD match the configured
      ``mean_photon_number`` for the corresponding pulse (within
      ``SOURCE_DM_MEAN_RTOL`` relative tolerance). A mismatch raises a
      warning; the DM diagonal is used for sampling.

    Classical imperfections (issue 1 in 2nd review; 4th-review issues 1.2, 1.3)
    ---------------------------------------------------------------------------
    The classical intensity imperfections defined on
    :class:`OpticalSource` fall into two categories:

    **Loss processes (4th-review fix — now applied to DM):**
      * **Channel splitting** (``N_channels > 1``): a passive N-channel
        beam-splitter network transforms the DM diagonal via binomial
        Bernoulli loss with transmissivity ``η = 1/N``::

            p'_k = Σ_{n=k}^∞ p_n · C(n,k) · η^k · (1-η)^(n-k)

        This is applied to the cached ``_number_probs_table`` at
        construction time, so the DM diagonal already reflects the
        post-splitting photon-number distribution per output channel.
        (4th-review issue 1.2.)
      * **Emission failure thinning** (``ideal_emission_probability < 1``):
        source trigger failure transforms the emitted state into the
        statistical mixture
        ``ρ' = p_emit · ρ + (1 - p_emit) · |0><0|``, which on the diagonal
        becomes ``p'_0 = p_emit · p_0 + (1 - p_emit)`` and
        ``p'_k = p_emit · p_k`` for ``k > 0``. This is applied to the
        cached ``_number_probs_table`` at construction time.
        (4th-review issue 1.3.)

    **State-modifying processes (still NOT applied to DM):**
      * MZM attenuation / electrical-driver noise: depends on the MZM
        model (multiplicative or non-linear) and cannot be expressed as
        a simple diagonal transformation.
      * SCM sideband factor: phase modulation, applied in the frequency
        domain; the post-SCM photon-number distribution depends on the
        sideband filtering, which is not modeled here.
      * Intensity jitter: multiplicative stochastic fluctuation that
        changes the *mean* photon number; modeling it on the DM would
        require convolving the DM diagonal with the jitter distribution,
        which is not done automatically.

    For these state-modifying processes, the density matrix is assumed to
    already encode the emitted state (post-imperfection). When this source
    is constructed with non-default settings for MZM, SCM, or intensity
    jitter, a **warning is logged** at construction time to remind the
    user that these settings are silently ignored when sampling from the
    DM. Channel splitting and emission failure thinning are applied
    automatically (since they are well-defined loss processes on the DM
    diagonal); no warning is logged for these.

    If you need to model MZM, SCM, or intensity jitter on top of a
    density-matrix source, apply them explicitly to the DM before
    constructing the source (e.g., by convolving the DM diagonal with
    the appropriate noise model), or use a mixed
    :class:`OpticalSource` + post-hoc DM-based analysis.

    4th-review fix (issue 2.5 — DM source without DMs):
    ----------------------------------------------------
    Constructing a :class:`DensityMatrixSource` without density matrices
    (or with an empty ``{}`` mapping) now raises :class:`ParameterValidationError`
    instead of silently falling back to :meth:`OpticalSource.generate_photons`.
    The previous fallback violated class single-responsibility: a
    :class:`DensityMatrixSource` instance that internally produces Poisson
    or thermal statistics is a type-confusion bug waiting to happen.
    Users who want Poisson/thermal statistics should construct an
    :class:`OpticalSource` or :class:`PoissonSource` directly.

    Missing-DM policy
    -----------------
    By default (``missing_dm_policy="error"``), construction raises if any
    pulse config has no corresponding density matrix. Pass
    ``missing_dm_policy="vacuum"`` to silently use the vacuum state
    ``|0><0|`` for missing pulses (the legacy behaviour, now explicit).

    Dimensionality management (issue 5 in 2nd review; issue 2.7 in 3rd review)
    -----------------------------------------------------------------------
    Each pulse's DM may have a different Fock-basis dimension. The
    internal ``_rho_tensor`` is padded to ``max_dim`` (the largest DM
    dimension) by embedding each smaller DM in the top-left block of an
    otherwise-zero ``max_dim x max_dim`` matrix. This zero-padding is
    *physically meaningful*: it represents Fock states with zero
    probability (i.e., states not represented in the original truncated
    DM). After padding, the diagonal is renormalized in
    :meth:`_calculate_number_probabilities` to ensure the row sums to 1.

    **Heterogeneous dimension policy (issue 2.7 in 3rd review):**
    In decoy-state QKD, security proofs and linear-programming estimators
    require all pulse ensembles (signal + decoys) to span the same
    photon-number subspace. If a decoy DM is truncated at dimension N=5
    and zero-padded to N=10, its probability mass for Fock states |6>
    through |10> is exactly 0.0, while the signal DM may have positive
    probability for those states. This can make the decoy-state LP
    constraints ill-conditioned or insoluble.

    The ``heterogeneous_dim_policy`` field controls the behavior:
      * ``"warn"`` (default): log a warning and proceed with zero-padding.
        The warning explains the LP implication.
      * ``"error"``: raise ``ParameterValidationError`` on heterogeneous
        dimensions. Use this for strict security analyses where the LP
        solvability must be guaranteed.

    A **warning is logged** at construction time whenever the DMs have
    heterogeneous dimensions (under the ``"warn"`` policy), to alert the
    user that some pulse types are truncated at a smaller Fock dimension
    than others — this can affect multi-photon fraction calculations in
    decoy-state security analyses.
    """
    density_matrices: Optional[Dict[str, np.ndarray]] = field(default=None)
    missing_dm_policy: str = field(default="error")
    heterogeneous_dim_policy: str = field(default="warn")
    # 9th-review fix (F-06): opt-out flag for the strict check that raises
    # ParameterValidationError when DM sources are constructed with non-default
    # classical-imperfection settings. Set to True only if you have pre-applied
    # the effects to the DM and understand the implications for security proofs.
    ignore_classical_imperfections: bool = field(default=False, repr=False)

    _rho_tensor: np.ndarray = field(init=False, repr=False, compare=False)
    _number_probs_table: np.ndarray = field(init=False, repr=False, compare=False)
    # 7th-review fix (issue I.2): the ``_number_probs_table_pre_emission``
    # field has been REMOVED. Previously, this field stored the post-channel-
    # split, pre-emission-failure probability table for use in sampling,
    # while ``_number_probs_table`` stored the post-emission-failure table
    # for analytical queries. This created a semantic contradiction:
    # ``calculate_effective_mus(name)`` used the pre-emission table
    # (returning the conditional mean), while ``number_probabilities_for_pulse(name)``
    # and ``density_matrix_for_pulse(name)`` used the post-emission table
    # (returning the unconditional state, smaller by a factor of ``p_emit``).
    # The fix unifies the semantics: ``_number_probs_table`` now ALWAYS
    # represents the post-channel-split, pre-emission-failure state
    # (matching ``calculate_effective_mus``). Emission failure is applied
    # at sampling time as a post-computation masking operation in
    # ``generate_photons_dm``. Users who need the unconditional
    # (post-emission) state can request it via ``include_emission_failure=True``
    # on the analytical methods, which computes the vacuum mixture on demand.

    # ------------------------------------------------------------------
    # Construction (uses the hook pattern — see OpticalSource.__post_init__)
    # ------------------------------------------------------------------

    def _pre_init_hook(self) -> None:
        """Coerce DMs and validate policies BEFORE base validation.

        4th-review fix (issue 2.1): explicitly chains to the parent
        implementation via ``OpticalSource._pre_init_hook(self)`` at the
        start, per the cooperative-multilevel-dispatch contract documented
        on :meth:`OpticalSource._pre_init_hook`. We use the explicit
        class-reference form (not zero-argument ``super()``) because
        ``@dataclass(slots=True)`` in Python 3.12 breaks the ``__class__``
        cell that the implicit ``super()`` relies on (see the construction-
        strategy comment on :class:`OpticalSource`). This ensures that any
        future subclass of :class:`DensityMatrixSource` that overrides
        ``_pre_init_hook`` will preserve this method's DM-coercion and
        policy-validation logic by chaining via
        ``DensityMatrixSource._pre_init_hook(self)``.
        """
        OpticalSource._pre_init_hook(self)

        # Coerce DMs first (also normalizes empty mapping to None).
        if self.density_matrices is not None:
            self.density_matrices = _coerce_density_matrices(self.density_matrices)

        # Validate missing_dm_policy early so users see a clear error
        # even if base validation would also fail.
        if self.missing_dm_policy not in ("error", "vacuum"):
            raise ParameterValidationError(
                f"missing_dm_policy must be 'error' or 'vacuum' "
                f"(got {self.missing_dm_policy!r}).",
                param_name="missing_dm_policy",
                param_value=self.missing_dm_policy,
            )

        # Validate heterogeneous_dim_policy (issue 2.7 in 3rd review).
        if self.heterogeneous_dim_policy not in ("warn", "error"):
            raise ParameterValidationError(
                f"heterogeneous_dim_policy must be 'warn' or 'error' "
                f"(got {self.heterogeneous_dim_policy!r}).",
                param_name="heterogeneous_dim_policy",
                param_value=self.heterogeneous_dim_policy,
            )

    def _post_init_hook(self) -> None:
        """Build DM tensor and probability table after base validation.

        4th-review fix (issue 2.5): raises :class:`ParameterValidationError`
        if ``density_matrices`` is ``None`` (or was coerced to ``None``
        from an empty ``{}`` mapping). The previous fallback to
        :meth:`OpticalSource.generate_photons` violated class single-
        responsibility: a :class:`DensityMatrixSource` instance that
        internally produces Poisson/thermal statistics is a type-confusion
        bug waiting to happen. Users who want Poisson/thermal statistics
        should construct :class:`OpticalSource` or :class:`PoissonSource`
        directly.

        4th-review fix (issue 2.1): explicitly chains to the parent
        implementation via ``OpticalSource._post_init_hook(self)`` at the
        start (explicit class reference, not zero-argument ``super()``,
        because ``@dataclass(slots=True)`` breaks the implicit ``super()``
        in Python 3.12 — see :class:`OpticalSource` construction-strategy
        comment).
        """
        OpticalSource._post_init_hook(self)

        # 4th-review fix (issue 2.5): raise instead of fall back.
        if self.density_matrices is None:
            raise ParameterValidationError(
                "DensityMatrixSource requires non-empty density_matrices. "
                "An empty mapping ({}) is normalized to None and rejected. "
                "Use OpticalSource or PoissonSource directly if you want "
                "Poisson/thermal statistics without density-matrix sampling.",
                param_name="density_matrices",
            )

        # Validate DM names against pulse names.
        valid_names = set(self.pulse_names())
        extra_names = set(self.density_matrices.keys()) - valid_names
        if extra_names:
            raise ParameterValidationError(
                "density_matrices contains unknown pulse names.",
                param_name="density_matrices",
                context={"unknown_pulse_names": sorted(extra_names)},
            )

        # Validate each DM.
        for name, rho in self.density_matrices.items():
            try:
                _validate_density_matrix(np.asarray(rho, dtype=np.complex128))
            except ParameterValidationError:
                raise
            except (ValueError, np.linalg.LinAlgError) as exc:
                raise ParameterValidationError(
                    f"Invalid density matrix for pulse {name!r}: {exc}",
                    param_name="density_matrices",
                ) from exc

        # Check mu/DM consistency (issue 3.19, 4.6).
        for name, rho in self.density_matrices.items():
            diag = np.real(np.diag(rho))
            diag_nonneg = np.maximum(diag, 0.0)
            s = diag_nonneg.sum()
            if s > 0:
                diag_norm = diag_nonneg / s
            else:
                diag_norm = diag_nonneg
            mu_dm = float(np.sum(np.arange(len(diag_norm)) * diag_norm))
            pc = self.get_pulse_config_by_name(name)
            if not np.isclose(
                mu_dm,
                pc.mean_photon_number,
                rtol=SOURCE_DM_MEAN_RTOL,
                atol=SOURCE_DM_MEAN_ATOL,
            ):
                logger.warning(
                    "Density-matrix mean photon number (%.6e) for pulse %r "
                    "does not match config mean photon number (%.6e). "
                    "DM value will be used for sampling; config value is "
                    "metadata only.",
                    mu_dm, name, pc.mean_photon_number,
                )

        # Issue 1 in 2nd review: warn if (state-modifying) classical-
        # imperfection settings are non-default and would be silently
        # ignored by DM sampling. Channel splitting and emission failure
        # thinning are applied to the DM diagonal (4th-review issues 1.2,
        # 1.3) and are NOT warned about here.
        self._warn_classical_imperfections_ignored()

        # Build rho tensor with explicit dimension management
        # (issue 5 in 2nd review).
        self._build_rho_tensor()

        # 4th-review fixes (issues 1.2, 1.3): apply loss-process classical
        # imperfections (channel splitting + emission failure thinning) to
        # the cached probability table. These are well-defined diagonal
        # transformations on the photon-number distribution and are applied
        # AFTER _build_rho_tensor so they compose with the (already-
        # renormalized) probability table.
        self._apply_dm_loss_processes()

    def _warn_classical_imperfections_ignored(self) -> None:
        """Raise error if state-modifying classical-imperfection settings are non-default.

        9th-review fix (F-06): this method now raises
        ``ParameterValidationError`` instead of just logging a warning,
        because silently ignoring non-default MZM, SCM, or intensity
        jitter settings violates the mean-preserving contract assumed by
        decoy-state security proofs. Users who have pre-applied these
        effects to their density matrices can set
        ``ignore_classical_imperfections=True`` to opt out of the check.

        DensityMatrixSource samples from the DM diagonal directly and does
        NOT apply MZM modulation, SCM factor, or intensity jitter (these
        are state-modifying processes that cannot be expressed as a simple
        diagonal transformation). Channel splitting and emission-failure
        thinning ARE applied (4th-review issues 1.2, 1.3) because they are
        well-defined loss processes on the DM diagonal; no error is
        raised for those.

        If any of the state-modifying settings are non-default, the user
        may be expecting them to take effect — raise an error so the
        mis-modeling is caught at construction time, not silently ignored.
        """
        # 9th-review fix (F-06): opt-out via ignore_classical_imperfections.
        if self.ignore_classical_imperfections:
            return

        cfg = self.config
        warnings_list: List[str] = []

        # MZM: any non-identity behaviour. We can't introspect arbitrary
        # MZM objects, but if the driver voltage noise is non-zero, MZM
        # would normally attenuate/modulate. Use that as a heuristic.
        if self._driver_voltage_noise_std > 0:
            warnings_list.append(
                f"electrical_noise (voltage_std={self._driver_voltage_noise_std:.3e})"
            )

        # SCM factor (state-modifying: not applied to DM).
        # 9th-review fix (F-06): use the scm_enabled flag instead of
        # checking modulation_index > 0.0. When scm_enabled=True, the SCM
        # scaling factor would modify the photon-number distribution, but
        # DM sampling does not apply it.
        scm_en = getattr(self, 'scm_enabled', SCM_ENABLED_DEFAULT)
        if scm_en:
            warnings_list.append(
                f"scm_enabled=True (modulation_index={cfg.modulation_index:.3e})"
            )

        # Intensity jitter (state-modifying: not applied to DM).
        if cfg.intensity_jitter > NUMERIC_ABS_TOL:
            warnings_list.append(
                f"intensity_jitter={cfg.intensity_jitter:.3e} "
                f"(error_model={cfg.error_model.value!r})"
            )

        if warnings_list:
            raise ParameterValidationError(
                "DensityMatrixSource constructed with non-default "
                "state-modifying classical-imperfection settings that "
                "would be SILENTLY IGNORED because DM-diagonal sampling "
                "bypasses MZM modulation, SCM factor, and intensity "
                "jitter (these cannot be expressed as a diagonal "
                "transformation on the photon-number distribution). "
                "Silently ignoring these settings violates the "
                "mean-preserving contract assumed by decoy-state "
                "security proofs. Affected ignored settings: %s. "
                "Either (a) pre-apply the ignored effects to the DM "
                "before constructing the source and set "
                "ignore_classical_imperfections=True, or (b) use "
                "OpticalSource / PoissonSource if you want all "
                "classical effects applied at sampling time.",
                "; ".join(warnings_list),
            )

    def _apply_dm_loss_processes(self) -> None:
        """Apply channel-splitting loss to the DM diagonal and tensor.

        4th-review fix (issue 1.2):
          * **Channel splitting** (issue 1.2): a passive N-channel
            beam-splitter network transforms the photon-number
            distribution via binomial Bernoulli loss with
            transmissivity ``η = 1/N_channels``. The transformation is::

                p'_k = Σ_{n=k}^{D-1} p_n · C(n,k) · η^k · (1-η)^(n-k)

            where ``D`` is the Fock truncation dimension. This is
            applied per-row to ``_number_probs_table`` AND to
            ``_rho_tensor`` (via the full Kraus transformation) so that
            ``number_probabilities_for_pulse`` and
            ``density_matrix_for_pulse`` return physically consistent
            post-channel-split states.

        5th-review fixes:
          * **Issue 3.3 (vectorized binomial loss matrix):** the previous
            implementation used a local ``from math import comb`` and a
            pure-Python O(dim²) double loop to build the binomial loss
            matrix. The new implementation uses
            ``scipy.special.comb`` (vectorized) and NumPy broadcasting.

        6th-review fix (issue 1): applied the full Kraus beam-splitter
        transformation to ``_rho_tensor`` (not just the diagonal loss to
        ``_number_probs_table``), so that
        ``density_matrix_for_pulse(name)`` returns a post-loss DM
        consistent with ``number_probabilities_for_pulse(name)``.

        7th-review fixes (issues I.2, I.3):
          * **Issue I.2 (conditional vs. unconditional means):** the
            previous implementation applied emission failure thinning
            to ``_number_probs_table`` and ``_rho_tensor`` (the
            "post-emission" state), while
            :meth:`DensityMatrixSource.calculate_effective_mus` used
            ``_number_probs_table_pre_emission`` (the "pre-emission"
            state). This exposed mathematically contradictory expectation
            values for the same pulse: ``calculate_effective_mus(name)``
            returned the conditional mean (given emission success),
            while ``number_probabilities_for_pulse(name)`` and
            ``density_matrix_for_pulse(name)`` returned the unconditional
            state whose expectation value was smaller by a factor of
            ``p_emit``. The fix removes the emission failure thinning
            from ``_apply_dm_loss_processes`` entirely: both
            ``_number_probs_table`` and ``_rho_tensor`` now represent
            the **post-channel-split, pre-emission-failure** state
            (i.e., the conditional state given emission success),
            matching ``calculate_effective_mus``. Emission failure is
            applied at sampling time as a post-computation masking
            operation in :meth:`generate_photons_dm`, identical to the
            :meth:`OpticalSource.generate_photons` pattern. The removed
            ``_number_probs_table_pre_emission`` field was redundant
            with ``_number_probs_table`` after this fix, so it has been
            removed. Users who need the unconditional (post-emission)
            state can request it via the ``include_emission_failure=True``
            parameter on :meth:`number_probabilities_for_pulse` and
            :meth:`density_matrix_for_pulse`, which computes the
            vacuum mixture on demand.
          * **Issue I.3 (trace renormalization for _rho_tensor):** after
            the Kraus loop, the previous implementation did NOT
            renormalize ``_rho_tensor`` by its trace, allowing floating-
            point drift to accumulate in the off-diagonal coherences
            while ``_number_probs_table`` was forcibly renormalized to
            sum to exactly 1.0. This broke the exact mathematical
            correspondence between the diagonal of
            ``density_matrix_for_pulse(name)`` and the probability vector
            returned by ``number_probabilities_for_pulse(name)``. The
            fix adds a trace-renormalization step after the Kraus loop,
            matching the row-sum renormalization on
            ``_number_probs_table``.

        Notes
        -----
        Only channel splitting is applied here. Emission failure is
        applied at sampling time as a masking operation (matching
        :meth:`OpticalSource.generate_photons`), not as a pre-computed
        state transformation. The state-modifying parts (MZM, SCM,
        intensity jitter) are NOT applied here because they cannot be
        expressed as a diagonal transformation; see
        :meth:`_warn_classical_imperfections_ignored`.
        """
        # 9th-review fix (F-21): create writable copies instead of
        # toggling write flags. This avoids ValueError on arrays that
        # were allocated with writeable=False at allocation time.
        table = np.array(self._number_probs_table, copy=True)
        rho = np.array(self._rho_tensor, copy=True)
        try:

            # --- Channel splitting (binomial loss) ---
            # 4th-review fix (issue 1.2); 5th-review fix (issue 3.3: vectorized).
            N = self.config.N_channels
            if N > 1:
                eta = 1.0 / float(N)
                num_pulses, dim = self._number_probs_table.shape
                # Build the binomial loss matrix L[k, n] = C(n, k) * eta^k * (1-eta)^(n-k)
                # for n >= k, else 0. L is (dim, dim). Then p' = L @ p per row.
                #
                # 5th-review fix (issue 3.3): the previous implementation used
                # a local ``from math import comb`` and a pure-Python O(dim²)
                # double loop. The new implementation uses scipy.special.comb
                # (vectorized) and NumPy broadcasting to build L in a single
                # vectorized operation.
                n_idx = np.arange(dim, dtype=np.float64)[:, None]  # (dim, 1)
                k_idx = np.arange(dim, dtype=np.float64)[None, :]  # (1, dim)
                # Mask: 1 where k <= n, 0 elsewhere.
                mask = (k_idx <= n_idx)
                # Vectorized binomial coefficient C(n, k) = 0 where k > n.
                # scipy.special.comb with exact=False returns float64 and
                # handles k > n by returning 0.
                binom = scipy_comb(n_idx, k_idx, exact=False)
                # Vectorized powers: eta^k * (1-eta)^(n-k).
                # For k > n, n-k is negative; we zero those out via the mask.
                powers = np.where(
                    mask,
                    np.power(eta, k_idx) * np.power(1.0 - eta, n_idx - k_idx),
                    0.0,
                )
                # L[k, n] = C(n, k) * eta^k * (1-eta)^(n-k)
                # 6th-review fix (issue 11): removed the redundant double transpose.
                # The previous implementation defined ``L = (binom * powers).T``
                # and then applied it via ``self._number_probs_table @ L.T``.
                # Since ``(L.T).T = L = binom * powers``, this executed two
                # completely redundant array transposition operations. The new
                # implementation defines ``M = binom * powers`` (indexed as
                # ``M[n, k]``) and applies it directly via
                # ``self._number_probs_table @ M``, which is mathematically
                # identical and avoids both transpositions.
                M = binom * powers  # (dim, dim), indexed as M[n, k]
                # Apply: p'_k = sum_n M[n, k] * p_n  (per row).
                # (num_pulses, dim) @ (dim, dim) -> (num_pulses, dim)
                table = table @ M
                # Renormalize to absorb floating-point drift from the matrix
                # multiply (L is exactly stochastic in exact arithmetic, but
                # floating-point summation can introduce tiny drift).
                row_sums = table.sum(axis=1, keepdims=True)
                # Guard against zero rows (shouldn't happen since channel
                # splitting preserves the all-zero property, but be defensive).
                nonzero = row_sums[:, 0] > 0
                table[nonzero] = table[nonzero] / row_sums[nonzero]

                # 6th-review fix (issue 1): apply the full Kraus transformation
                # to _rho_tensor so that density_matrix_for_pulse returns a
                # post-loss DM consistent with number_probabilities_for_pulse.
                # The Kraus transformation for a beam-splitter loss channel with
                # transmissivity η is::
                #
                #   rho'_{m,m'} = Σ_k √(C(m+k,k)·C(m'+k,k))·(1−η)^k·η^((m+m')/2)·rho_{m+k, m'+k}
                #
                # which reduces to the binomial loss formula on the diagonal
                # (m = m'). We compute this with a loop over k (dim iterations)
                # and vectorized operations within each iteration, giving
                # O(dim^3 · num_pulses) total work — fast for typical dims (≤100).
                rho_new = np.zeros_like(rho)
                for k in range(dim):
                    n_max = dim - k  # m + k < dim => m < dim - k
                    if n_max <= 0:
                        break
                    m_idx = np.arange(n_max, dtype=np.float64)
                    # C(m+k, k) for m = 0, ..., n_max-1
                    comb_m = scipy_comb(m_idx + k, k, exact=False)
                    # η^(m/2) for the amplitude factor
                    eta_factor = np.power(eta, m_idx / 2.0)
                    # (1-η)^k for the loss factor
                    loss_factor = (1.0 - eta) ** k
                    # coeff[m, m'] = √(C(m+k,k)·C(m'+k,k)) · (1-η)^k · η^((m+m')/2)
                    coeff = (
                        np.sqrt(comb_m[:, None] * comb_m[None, :])
                        * loss_factor
                        * eta_factor[:, None]
                        * eta_factor[None, :]
                    )
                    # rho_shifted[pulse, m, m'] = rho[pulse, m+k, m'+k]
                    rho_shifted = rho[:, k:k + n_max, k:k + n_max]
                    rho_new[:, :n_max, :n_max] += coeff[None, :, :] * rho_shifted
                rho = rho_new

                # 7th-review fix (issue I.3): trace-renormalize _rho_tensor
                # after the Kraus loop to match the row-sum renormalization
                # applied to _number_probs_table above. The Kraus transformation
                # is exactly trace-preserving in exact arithmetic, but
                # floating-point summation can introduce tiny drift. Without
                # this renormalization, the diagonal of
                # density_matrix_for_pulse(name) would not exactly match the
                # probability vector returned by number_probabilities_for_pulse(name),
                # breaking the mathematical correspondence between the two
                # analytical methods and potentially corrupting downstream
                # linear-programming yield estimators that consume both.
                traces = np.trace(rho, axis1=1, axis2=2).real  # (num_pulses,)
                # Guard against zero traces (shouldn't happen for valid DMs,
                # but be defensive).
                nonzero_trace = traces > 0
                if np.any(nonzero_trace):
                    rho[nonzero_trace] = rho[nonzero_trace] / traces[nonzero_trace, None, None]

            # 7th-review fix (issue I.2): emission failure thinning is NO
            # LONGER applied to _number_probs_table or _rho_tensor here.
            # Previously, this method applied the vacuum mixture
            # ``p_emit · ρ + (1 − p_emit) · |0><0|`` to both arrays, while
            # ``calculate_effective_mus`` used ``_number_probs_table_pre_emission``
            # (the pre-emission state). This created a semantic contradiction:
            # ``calculate_effective_mus(name)`` returned the conditional mean
            # (given emission success), while ``number_probabilities_for_pulse(name)``
            # and ``density_matrix_for_pulse(name)`` returned the unconditional
            # state whose expectation was smaller by a factor of ``p_emit``.
            # The fix removes the emission failure application entirely:
            # both ``_number_probs_table`` and ``_rho_tensor`` now represent
            # the **post-channel-split, pre-emission-failure** state,
            # matching ``calculate_effective_mus``. Emission failure is
            # applied at sampling time as a post-computation masking
            # operation in ``generate_photons_dm`` (identical to the
            # ``OpticalSource.generate_photons`` pattern). The removed
            # ``_number_probs_table_pre_emission`` field was redundant with
            # ``_number_probs_table`` after this fix, so it has been removed.
            # Users who need the unconditional (post-emission) state can
            # request it via ``include_emission_failure=True`` on
            # ``number_probabilities_for_pulse`` / ``density_matrix_for_pulse``,
            # which computes the vacuum mixture on demand from the (cached)
            # pre-emission state.

        finally:
            # 9th-review fix (F-21): freeze the modified copies and assign
            # them back, instead of toggling flags on the original arrays.
            table.setflags(write=False)
            rho.setflags(write=False)
            object.__setattr__(self, '_number_probs_table', table)
            object.__setattr__(self, '_rho_tensor', rho)

    def calculate_effective_mus(
        self,
        pulse_indices: Optional[Union[int, np.ndarray, Sequence[int]]],
        rng: RNGType,
        num_samples: int = 1,
    ) -> Union[np.ndarray, float]:
        """Compute effective mean photon numbers from the DM diagonal.

        5th-review fix (issue 2.1): override the base-class
        :meth:`OpticalSource.calculate_effective_mus`, which applies the
        classical-imperfection pipeline (jitter, MZM, SCM, channel split).
        For :class:`DensityMatrixSource`, the classical-imperfection
        pipeline is NOT applied at sampling time (see class docstring);
        instead, channel splitting and emission failure thinning are
        pre-applied to the cached ``_number_probs_table`` at construction
        time, and the state-modifying imperfections (MZM, SCM, jitter) are
        assumed to be already encoded in the DM.

        This override returns the effective mean photon number computed
        from the cached ``_number_probs_table`` (which includes channel
        splitting and emission failure thinning)::

            mu_effective = Σ_n n · p'_n

        where ``p'_n`` is the post-loss photon-number probability for the
        selected pulse type. This is the *analytical* effective mu that a
        user inspecting the source would expect; it matches the
        *statistical* mean of the photons sampled by
        :meth:`generate_photons_dm` (which applies the same loss processes).

        Parameters
        ----------
        pulse_indices : int, array-like of int, or None
            If None, ``num_samples`` pulse indices are sampled randomly.
        rng : numpy.random.Generator
            Random number generator (used only for pulse sampling if
            ``pulse_indices is None``; NOT used for classical imperfections
            since they are pre-applied to the cached table).
        num_samples : int, default 1
            Number of pulse indices to sample if ``pulse_indices is None``.

        Returns
        -------
        np.ndarray or float
            Effective mean photon numbers. Scalar float if ``pulse_indices``
            was a scalar; otherwise 1-D float64 array.

        Notes
        -----
        Unlike the base-class implementation, this method does NOT mutate
        the adversarial-block state (since the ADVERSARIAL_BLOCK error
        model is a state-modifying process that is not applied to DM
        sources). It also does NOT call :meth:`_calculate_effective_mus`
        (which applies jitter, MZM, SCM, channel split), because those
        processes are either pre-applied (channel split) or silently
        ignored (jitter, MZM, SCM) for DM sources.

        6th-review fix (issue 2): use ``_number_probs_table_pre_emission``
        (post-channel-splitting, pre-emission-failure) instead of
        ``_number_probs_table`` (post-emission-failure). This aligns the
        semantics with :meth:`OpticalSource.calculate_effective_mus`,
        which returns the *conditional* mean (given emission success),
        i.e. the mean photon number BEFORE emission-failure thinning. The
        previous implementation returned the *unconditional* mean (post-
        thinning), which differed from the base-class return value by a
        factor of ``p_emit`` for sources with ``p_emit < 1``.

        7th-review fix (issue I.2): the ``_number_probs_table_pre_emission``
        field has been REMOVED (see :meth:`_apply_dm_loss_processes`).
        ``_number_probs_table`` now ALWAYS represents the post-channel-split,
        pre-emission-failure state, so this method reads it directly.
        Emission failure is applied at sampling time as a masking operation
        in :meth:`generate_photons_dm`, identical to the
        :meth:`OpticalSource.generate_photons` pattern.

        6th-review fix (issue 13): removed the redundant ``if num_samples < 0``
        check.
        """
        if pulse_indices is None:
            # 6th-review fix (issue 13): rely on sample_pulse_indices for
            # num_samples validation.
            indices = self.sample_pulse_indices(rng, num_samples)
            is_scalar = False
        else:
            indices, is_scalar = self._validate_pulse_indices(pulse_indices)

        if len(indices) == 0:
            if is_scalar:
                raise ParameterValidationError(
                    "Scalar pulse_indices produced an empty array; "
                    "this is a logical inconsistency.",
                    param_name="pulse_indices",
                )
            return np.array([], dtype=np.float64)

        # Compute effective mu from the cached probability table.
        # 7th-review fix (issue I.2): ``_number_probs_table`` now represents
        # the post-channel-split, pre-emission-failure state (the conditional
        # state given emission success), so the returned mean is the
        # conditional mean, matching ``OpticalSource.calculate_effective_mus``.
        probs = self._number_probs_table[indices]  # (n, max_dim)
        n_values = np.arange(probs.shape[1], dtype=np.float64)  # (max_dim,)
        mus = probs @ n_values  # (n,) — dot product gives Σ_n n*p_n per row

        return float(mus.item()) if is_scalar else mus

    def _build_rho_tensor(self) -> None:
        """Build ``_rho_tensor`` with explicit dimensionality management.

        Each DM is embedded in the top-left block of an
        ``max_dim x max_dim`` zero matrix. The zero-padding represents
        Fock states with zero probability — physically meaningful, not a
        memory-asymmetry bug (issue 5 in 2nd review). A warning is logged
        when DMs have heterogeneous dimensions, since this can affect
        multi-photon fraction calculations downstream.

        Issue 2.7 in 3rd review: the ``heterogeneous_dim_policy`` field
        controls whether heterogeneous dimensions raise an error (``"error"``)
        or just log a warning (``"warn"``, the default). The ``"error"``
        policy is recommended for strict decoy-state security analyses
        where LP constraint solvability must be guaranteed.

        5th-review fix (issue 2.3): the auto-generated vacuum state
        (``|0><0|``, dim=1) created by ``missing_dm_policy="vacuum"`` is
        now EXCLUDED from the heterogeneity check. The previous
        implementation included dim=1 in ``unique_dims``, which caused
        ``heterogeneous_dim_policy="error"`` to raise even when the user
        explicitly chose ``missing_dm_policy="vacuum"`` (a clear intent to
        use the vacuum state for missing pulses). The vacuum state is
        always compatible with any larger Fock dimension via zero-padding
        (it's the trivial |0><0| state, a subset of any Fock space), so
        it should not trigger the heterogeneity check. The check now only
        considers explicitly-provided DMs.
        """
        assert self.density_matrices is not None  # caller guarantees this

        # Collect dimensions for heterogeneity check.
        # 5th-review fix (issue 2.3): separate explicitly-provided DM dims
        # from auto-generated vacuum dims. The heterogeneity check only
        # considers explicitly-provided DMs; the auto-generated vacuum
        # state (dim=1) is always compatible via zero-padding.
        dims = {name: rho.shape[0] for name, rho in self.density_matrices.items()}
        # dims_for_check: only explicitly-provided DM dims (used for the
        # heterogeneity check and the "error"/"warn" policy decision).
        dims_for_check = dict(dims)  # copy; will NOT add vacuum dim=1
        unique_dims_for_check = set(dims_for_check.values())

        # all_dims: includes auto-generated vacuum dim=1, used for the
        # warning message (informational) and max_dim computation.
        all_dims = dict(dims)
        if self.missing_dm_policy == "vacuum":
            for pc in self.pulse_configs:
                if pc.name not in all_dims:
                    all_dims[pc.name] = 1

        if len(unique_dims_for_check) > 1:
            # Issue 2.7 in 3rd review: apply the heterogeneous_dim_policy.
            # The "error" policy raises to guarantee LP constraint solvability
            # in decoy-state security analyses; the "warn" policy (default)
            # logs a detailed warning explaining the LP implication.
            #
            # 5th-review fix (issue 2.3): the check uses
            # unique_dims_for_check (explicitly-provided DMs only), NOT
            # all_dims (which includes auto-generated vacuum dim=1). This
            # prevents the policy from triggering when the user explicitly
            # chose missing_dm_policy="vacuum" and the only "heterogeneity"
            # is between the explicitly-provided DMs and the auto-generated
            # vacuum state.
            dims_str = ", ".join(
                f"{n!r}: {d}" for n, d in sorted(dims_for_check.items())
            )
            if self.heterogeneous_dim_policy == "error":
                raise ParameterValidationError(
                    f"Density matrices have heterogeneous Fock dimensions "
                    f"({dims_str}). In decoy-state QKD, this can break LP "
                    f"constraints: a decoy DM truncated at dimension N has "
                    f"zero probability for Fock states |n> with n >= N, "
                    f"while the signal DM may have positive probability for "
                    f"those states, making the yield-bounding LP "
                    f"ill-conditioned or insoluble. Set "
                    f"heterogeneous_dim_policy='warn' to allow (with a "
                    f"warning), or ensure all DMs have the same dimension.",
                    param_name="density_matrices",
                )
            elif self.heterogeneous_dim_policy == "warn":
                logger.warning(
                    "Density matrices have heterogeneous Fock dimensions "
                    "(%s). Smaller DMs are zero-padded to max_dim=%d. "
                    "This padding represents zero-probability Fock states "
                    "(physically meaningful) but means different pulse "
                    "types have different truncation of the photon-number "
                    "distribution. WARNING: in decoy-state QKD, this can "
                    "break LP constraints — a decoy DM truncated at "
                    "dimension N has zero probability for Fock states "
                    "|n> with n >= N, while the signal DM may have "
                    "positive probability for those states, making the "
                    "yield-bounding LP ill-conditioned or insoluble. "
                    "Verify that multi-photon tails are not artificially "
                    "suppressed for the smaller DMs, or set "
                    "heterogeneous_dim_policy='error' to enforce uniform "
                    "dimensions.",
                    dims_str,
                    max(unique_dims_for_check),
                )

        # max_dim: the largest dimension across all DMs (including
        # auto-generated vacuum, though vacuum dim=1 never affects the max).
        max_dim = max(all_dims.values()) if all_dims else 1
        num_configs = len(self.pulse_configs)
        rho_tensor = np.zeros((num_configs, max_dim, max_dim), dtype=np.complex128)

        for idx, pc in enumerate(self.pulse_configs):
            if pc.name in self.density_matrices:
                rho = np.asarray(self.density_matrices[pc.name], dtype=np.complex128)
                dim = rho.shape[0]
                # Embed the DM in the top-left block of the max_dim x max_dim
                # tensor slot. Zero-padding outside the block represents
                # zero-probability Fock states |n> for n >= dim.
                rho_tensor[idx, :dim, :dim] = rho
            else:
                # Missing DM: apply policy (issue 1.5, 8.3).
                if self.missing_dm_policy == "error":
                    raise ParameterValidationError(
                        f"No density matrix provided for pulse {pc.name!r}. "
                        "Pass missing_dm_policy='vacuum' to use |0><0| "
                        "for missing pulses.",
                        param_name="density_matrices",
                        context={"missing_pulse_name": pc.name},
                    )
                elif self.missing_dm_policy == "vacuum":
                    logger.warning(
                        "No density matrix for pulse %r; using vacuum state |0><0|.",
                        pc.name,
                    )
                    # Vacuum: |0><0| = a 1x1 matrix [[1.0]], embedded in
                    # the top-left corner. All other Fock states have
                    # zero probability.
                    rho_tensor[idx, 0, 0] = 1.0
                # (other policies already rejected in _pre_init_hook)

        # 4th-review fix (issue 4.4): freeze the rho tensor as read-only.
        # The probability table is also frozen here; subsequent
        # transformations (channel splitting, emission-failure thinning)
        # in _apply_dm_loss_processes will temporarily make it writable
        # and re-freeze it. Returning read-only views from
        # number_probabilities_for_pulse and density_matrix_for_pulse
        # is safe because the underlying arrays cannot be mutated.
        rho_tensor.setflags(write=False)
        self._rho_tensor = rho_tensor
        self._number_probs_table = self._calculate_number_probabilities(rho_tensor)
        # _calculate_number_probabilities returns a freshly-allocated
        # writable array; freeze it now (it will be temporarily unfrozen
        # by _apply_dm_loss_processes if needed).
        self._number_probs_table.setflags(write=False)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: OpticalSourceConfig,
        *,
        density_matrices: Optional[Mapping[str, Any]] = None,
        missing_dm_policy: str = "error",
        heterogeneous_dim_policy: str = "warn",
        security_metadata: Optional[SourceSecurityMetadata] = None,
    ) -> "DensityMatrixSource":
        """Build a :class:`DensityMatrixSource` from config + DMs."""
        return cls(
            config=config,
            density_matrices=_coerce_density_matrices(density_matrices),
            missing_dm_policy=missing_dm_policy,
            heterogeneous_dim_policy=heterogeneous_dim_policy,
            security_metadata=security_metadata,
        )

    @classmethod
    def from_protocol_parameters(
        cls,
        params: ProtocolParameters,
        *,
        density_matrices: Optional[Mapping[str, Any]] = None,
        missing_dm_policy: str = "error",
        heterogeneous_dim_policy: str = "warn",
        security_metadata: Optional[SourceSecurityMetadata] = None,
    ) -> "DensityMatrixSource":
        """Build from :class:`ProtocolParameters`."""
        if not isinstance(params, ProtocolParameters):
            raise ParameterValidationError(
                "params must be a ProtocolParameters instance.",
                param_name="params",
                param_value=type(params).__name__,
            )
        if not isinstance(params.optical, OpticalSourceConfig):
            raise ParameterValidationError(
                "params.optical must be an OpticalSourceConfig instance.",
                param_name="params.optical",
                param_value=type(params.optical).__name__,
            )
        return cls.from_config(
            params.optical,
            density_matrices=density_matrices,
            missing_dm_policy=missing_dm_policy,
            heterogeneous_dim_policy=heterogeneous_dim_policy,
            security_metadata=security_metadata,
        )

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        strict: bool = False,
        missing_dm_policy: Optional[str] = None,
        heterogeneous_dim_policy: Optional[str] = None,
        security_metadata: Optional[SourceSecurityMetadata] = None,
    ) -> "DensityMatrixSource":
        """Build from a serialized dict (round-trip safe with :meth:`to_dict`).

        Density matrices are decoded from the JSON-safe
        ``{"real": [[...]], "imag": [[...]}`` format.

        If ``missing_dm_policy`` is not explicitly passed, it is read from
        ``data["missing_dm_policy"]`` if present, else defaults to ``"error"``.

        If ``heterogeneous_dim_policy`` is not explicitly passed, it is read
        from ``data["heterogeneous_dim_policy"]`` if present, else defaults
        to ``"warn"`` (issue 2.7 in 3rd review).
        """
        if not isinstance(data, Mapping):
            raise ParameterValidationError(
                "data must be a mapping.",
                param_name="data",
                param_value=type(data).__name__,
            )

        if strict:
            unknown = set(data.keys()) - _KNOWN_SOURCE_CONFIG_KEYS
            if unknown:
                raise ParameterValidationError(
                    f"Unknown keys in source config: {sorted(unknown)}. "
                    "Pass strict=False to ignore unknown keys.",
                    param_name="data",
                )

        # Resolve missing_dm_policy: explicit kwarg wins, else read from data.
        if missing_dm_policy is None:
            missing_dm_policy = data.get("missing_dm_policy", "error")

        # Resolve heterogeneous_dim_policy (issue 2.7 in 3rd review).
        if heterogeneous_dim_policy is None:
            heterogeneous_dim_policy = data.get(
                "heterogeneous_dim_policy", "warn"
            )

        config = _coerce_optical_source_config(data)

        dm_data: Optional[Dict[str, np.ndarray]] = None
        if "density_matrices" in data:
            dm_data = _decode_density_matrices_from_serialized(data["density_matrices"])

        return cls.from_config(
            config,
            density_matrices=dm_data,
            missing_dm_policy=missing_dm_policy,
            heterogeneous_dim_policy=heterogeneous_dim_policy,
            security_metadata=security_metadata,
        )

    # ------------------------------------------------------------------
    # DM-specific methods
    # ------------------------------------------------------------------

    def _calculate_number_probabilities(self, rho_tensor: np.ndarray) -> np.ndarray:
        """Extract photon-number probabilities from the DM diagonal.

        This is **non-destructive** (issue 3.11, 3.12, 3.13, 8.5):
          * No ``ENTROPY_PROB_CLAMP`` substitution of small probabilities.
          * No ``DEFAULT_POISSON_TAIL_THRESHOLD`` thresholding.
          * No ``LP_CONSTRAINT_VIOLATION_TOL``-driven renormalization.
          * Only light clipping of tiny negative values (from floating-point
            eigenvalue drift).

        Issue 3.2 in 3rd review: renormalization is **always** applied
        after clipping, regardless of the drift magnitude. The previous
        code only renormalized if the row sum deviated from 1.0 by more
        than ``SOURCE_NORMALIZATION_TOL`` (1e-10). This could leave the
        row summing to e.g. 1.0 + 0.5e-10 (within tolerance) after
        clipping tiny negative values, causing
        :meth:`number_probabilities_for_pulse` to return an unnormalized
        distribution. While :meth:`generate_photons_dm` performs defensive
        per-sample renormalization, direct callers of
        :meth:`number_probabilities_for_pulse` would receive the
        unnormalized distribution. The fix ensures the cached
        ``_number_probs_table`` always sums to exactly 1.0 (within
        floating-point precision).

        The returned table preserves the physical photon-number
        distribution. Entropy / log-safety clamping must be applied by
        downstream entropy computations, not here.
        """
        num_pulses = rho_tensor.shape[0]
        max_dim = rho_tensor.shape[1]
        probs = np.zeros((num_pulses, max_dim), dtype=np.float64)

        for idx in range(num_pulses):
            diag = np.real(np.diag(rho_tensor[idx])).copy()
            # Clip tiny negative values from floating-point drift.
            diag = np.maximum(diag, 0.0)

            s = diag.sum()
            if s <= 0:
                # 7th-review fix (issue III.3): RAISE instead of logging
                # and continuing. A density matrix whose diagonal sums to
                # <= 0 is mathematically impossible — it violates the
                # positive-semidefinite and trace-1 requirements that
                # ``_validate_density_matrix`` already enforces at
                # construction time. The previous implementation logged
                # the error and left an all-zero row in the cached
                # ``_number_probs_table``, allowing invalid state to
                # persist until runtime sampling (where it would raise
                # a less clear error from the zero-row check in
                # ``generate_photons_dm``). Catching the condition here,
                # at construction time, surfaces the error at the
                # earliest possible point with the most context.
                raise ParameterValidationError(
                    f"Density matrix diagonal for pulse index {idx} sums "
                    f"to {s:.6e} (non-positive). A valid density matrix "
                    f"in the Fock basis must have a non-negative diagonal "
                    f"that sums to 1.0 (positive-semidefinite and trace-1 "
                    f"requirements). The input density matrix is corrupted "
                    f"or invalid; cannot proceed with photon-number "
                    f"sampling. Re-validate the density matrix before "
                    f"constructing the source.",
                    param_name="density_matrices",
                    param_value=f"pulse_index={idx}, diagonal_sum={s:.6e}",
                )

            # Issue 3.2 in 3rd review: ALWAYS renormalize after clipping,
            # not just when the drift exceeds SOURCE_NORMALIZATION_TOL.
            # This guarantees the cached probability table sums to exactly
            # 1.0 (within floating-point precision), which is critical for
            # direct callers of number_probabilities_for_pulse and for
            # decoy-state LP constraint satisfaction.
            if not np.isclose(s, 1.0, atol=SOURCE_NORMALIZATION_TOL):
                logger.debug(
                    "Renormalizing DM diagonal for pulse index %d (sum=%.6e).",
                    idx, s,
                )
            # Unconditional renormalization (even for tiny drift).
            diag = diag / s

            # 6th-review fix (issue 12): direct row assignment (see comment above).
            probs[idx] = diag

        return probs

    def number_probabilities_for_pulse(
        self,
        name: str,
        *,
        include_emission_failure: bool = False,
    ) -> np.ndarray:
        """Return the photon-number probability vector for the named pulse.

        Parameters
        ----------
        name : str
            Pulse name.
        include_emission_failure : bool, default False
            If True, return the *unconditional* (post-emission-failure)
            probability vector, computed as the vacuum mixture
            ``p_emit · p + (1 − p_emit) · |0><0|``. If False (default),
            return the *conditional* (pre-emission-failure) probability
            vector, which is the cached state used for sampling and for
            :meth:`calculate_effective_mus`.

        4th-review fix (issue 4.4): returns a direct read-only view of the
        internal cached row instead of a copy. The internal
        ``_number_probs_table`` is frozen as read-only at construction
        time (see :meth:`_build_rho_tensor`), so the returned view cannot
        be mutated by the caller. This eliminates the per-call heap
        allocation that the previous ``.copy()`` incurred, which was a
        measurable overhead in tight analytical loops (e.g., per-pulse-
        type yield estimation over long simulations).

        7th-review fix (issue I.2): the default semantics changed from
        "post-emission-failure" (unconditional state) to
        "pre-emission-failure" (conditional state). The previous
        implementation cached the post-emission-failure table as
        ``_number_probs_table`` and returned it by default, while
        :meth:`calculate_effective_mus` used a separate pre-emission
        table. This created a contradiction where the same pulse had
        different expectation values depending on which analytical
        method was called. The fix unifies the semantics: the default
        is now the pre-emission-failure state (matching
        :meth:`calculate_effective_mus`), and the post-emission-failure
        state is available via ``include_emission_failure=True``.

        If you need a writable copy (e.g., to experiment with a modified
        distribution without affecting the source), call ``.copy()``
        explicitly on the returned view.
        """
        idx = self.get_pulse_index_by_name(name)
        if not include_emission_failure:
            return self._number_probs_table[idx]
        # Compute the post-emission-failure state on demand.
        p_emit = clamp_probability(self.config.ideal_emission_probability)
        if is_close(p_emit, 1.0):
            return self._number_probs_table[idx]
        pre = self._number_probs_table[idx]
        post = pre * p_emit
        post[0] += (1.0 - p_emit)
        # 8th-review fix (F-19): make the on-demand computed array read-only
        # to match the documented contract "returns a read-only view".
        post.setflags(write=False)
        return post

    def density_matrix_for_pulse(
        self,
        name: str,
        *,
        include_emission_failure: bool = False,
    ) -> np.ndarray:
        """Return the density matrix for the named pulse.

        Parameters
        ----------
        name : str
            Pulse name.
        include_emission_failure : bool, default False
            If True, return the *unconditional* (post-emission-failure)
            density matrix, computed as the vacuum mixture
            ``p_emit · ρ + (1 − p_emit) · |0><0|``. If False (default),
            return the *conditional* (pre-emission-failure) density
            matrix, which is the cached state.

        4th-review fix (issue 4.4): returns a direct read-only view of the
        internal cached slice instead of a copy. The internal
        ``_rho_tensor`` is frozen as read-only at construction time (see
        :meth:`_build_rho_tensor`), so the returned view cannot be
        mutated by the caller. This eliminates the per-call heap
        allocation that the previous ``.copy()`` incurred.

        7th-review fix (issue I.2): the default semantics changed from
        "post-emission-failure" (unconditional state) to
        "pre-emission-failure" (conditional state). See
        :meth:`number_probabilities_for_pulse` for the rationale.

        If you need a writable copy, call ``.copy()`` explicitly on the
        returned view.
        """
        idx = self.get_pulse_index_by_name(name)
        if not include_emission_failure:
            return self._rho_tensor[idx]
        # Compute the post-emission-failure state on demand.
        p_emit = clamp_probability(self.config.ideal_emission_probability)
        if is_close(p_emit, 1.0):
            return self._rho_tensor[idx]
        pre = self._rho_tensor[idx]
        post = pre * p_emit
        # Add (1 - p_emit) to the [0, 0] element (vacuum state |0><0|).
        # post is a fresh array (from the multiplication above), so it's
        # writable; we can assign directly.
        post[0, 0] += (1.0 - p_emit)
        # 8th-review fix (F-20): make the on-demand computed DM read-only
        # to match the documented contract "returns a read-only view".
        post.setflags(write=False)
        return post

    def generate_photons_dm(
        self,
        alice_pulse_indices: Optional[Union[int, np.ndarray, Sequence[int]]],
        rng: RNGType,
        num_samples: int = 1,
    ) -> Union[np.ndarray, int]:
        """Generate photon counts by sampling from DM diagonals.

        Notes
        -----
        Loss-process classical imperfections (channel splitting) are
        pre-applied to the cached ``_number_probs_table`` at construction
        time (4th-review issue 1.2; 7th-review issue I.2 — the field is
        now ``_number_probs_table`` itself, no longer a separate
        ``_number_probs_table_pre_emission`` field). Emission failure
        thinning is applied as post-computation masking at sampling time
        (5th-review issue 2.2), enabling accurate tracking of
        ``num_pulses_generated`` (emitted) vs
        ``num_pulses_attempted``. The DM is assumed to already encode
        the state-modifying imperfections (MZM, SCM, intensity jitter).

        Sampling uses the inverse-CDF method with a vectorized
        ``searchsorted``-style approach: ``photons = (cdf < u).sum(axis=1)``.

        4th-review fixes (issues 4.1, 4.2):
          * **Issue 4.1 (memory):** the previous implementation allocated
            full 2D arrays (``cdf``, ``cdf < u``) of shape
            ``(len(indices), max_dim)``. For a standard QKD simulation
            batch of 10^7 pulses and a Fock truncation dimension of 100,
            this single line allocated multiple 8 GB floating-point
            arrays and a 1 GB boolean matrix simultaneously, causing
            severe memory bloat and cache thrashing. The new
            implementation processes the batch in chunks, limiting peak
            memory to ``_DM_SAMPLING_CHUNK_BYTES`` (default 80 MB) per
            chunk. The chunk size is automatically reduced for larger
            Fock dimensions to respect the memory budget.
          * **Issue 4.2 (redundant renormalization):** the previous
            implementation called ``selected_probs / row_sums`` on every
            generation call, even though ``_number_probs_table`` is
            immutably cached and pre-normalized to sum to exactly 1.0
            during construction in :meth:`_calculate_number_probabilities`
            (and re-normalized after the loss-process transformations in
            :meth:`_apply_dm_loss_processes`). The new implementation
            skips the per-call division; the zero-row check is preserved
            (it requires the row sums but not the division), and the CDF
            is built directly from the pre-normalized table.

        5th-review fixes:
          * **Issue 3.2 (per-chunk RNG):** the previous implementation
            pre-allocated ``u_all = rng.random(size=n_pulses)`` before the
            chunk loop. For a batch of 10^7 pulses, this single array
            consumes 80 MB — exactly the per-chunk memory budget. The
            pre-allocation therefore doubled the peak memory, defeating
            the purpose of chunking. The new implementation generates
            uniform random numbers per-chunk (``rng.random(size=chunk_len)``),
            keeping peak memory within the chunk budget.
          * **Issue 2.2 (emission failure tally):** the previous
            implementation sampled from the post-loss table (which
            already included emission failure thinning) and reported
            ``num_pulses_generated == num_pulses_attempted``. This was
            incorrect when ``p_emit < 1``: it conflated "emitted with 0
            photons" and "failed to emit" in the tally. The new
            implementation samples from the pre-emission table and
            applies emission failure as post-computation masking
            (matching :meth:`OpticalSource.generate_photons`), tracking
            the emitted count from the mask. The post-loss table
            (``_number_probs_table``) is still used for analytical queries
            (``number_probabilities_for_pulse``).
        """
        # 4th-review fix (issue 2.5): density_matrices is now guaranteed
        # non-None at construction time (the constructor raises if it is
        # None or empty). This branch is kept as a defensive guard against
        # post-construction mutation (which is impossible for read-only
        # fields, but defensive programming is cheap here).
        if self.density_matrices is None:
            raise ParameterValidationError(
                "DensityMatrixSource.density_matrices is None; this should "
                "have been rejected at construction time (4th-review issue "
                "2.5). The instance may have been corrupted post-construction.",
                param_name="density_matrices",
            )

        if alice_pulse_indices is None:
            # 6th-review fix (issue 13): rely on sample_pulse_indices for
            # num_samples validation.
            indices = self.sample_pulse_indices(rng, num_samples)
            is_scalar = False
        else:
            indices, is_scalar = self._validate_pulse_indices(alice_pulse_indices)

        if len(indices) == 0:
            if is_scalar:
                raise ParameterValidationError(
                    "Scalar alice_pulse_indices produced an empty array; "
                    "this is a logical inconsistency.",
                    param_name="alice_pulse_indices",
                )
            return np.array([], dtype=np.int64)

        # 7th-review fix (issue I.2): sample from ``_number_probs_table``
        # (which is now the post-channel-split, pre-emission-failure state)
        # so that emission failure can be applied as post-computation
        # masking, enabling accurate emitted-vs-attempted tally tracking.
        # The previous implementation used ``_number_probs_table_pre_emission``
        # (a separate field); the 7th-review fix removed that field and
        # made ``_number_probs_table`` ALWAYS represent the pre-emission
        # state, unifying the semantics across all analytical methods.
        selected_probs = self._number_probs_table[indices]
        max_dim = selected_probs.shape[1]

        # 4th-review fix (issue 4.2): compute row sums only for the zero-row
        # check; skip the per-call renormalization since the table is
        # pre-normalized at construction time.
        row_sums = selected_probs.sum(axis=1)

        # Issue 3.15: explicit error for all-zero rows (corrupted DM).
        zero_rows = row_sums == 0.0
        if np.any(zero_rows):
            bad_indices = np.where(zero_rows)[0].tolist()
            raise ParameterValidationError(
                f"Photon-number distribution is all-zero for pulse indices "
                f"{bad_indices}. This indicates a corrupted or missing "
                f"density matrix. Check construction logs for warnings.",
                param_name="density_matrices",
            )

        # 4th-review fix (issue 4.1): chunked inverse-CDF sampling to limit
        # peak memory. For a batch of N pulses with Fock dimension D, the
        # previous implementation allocated:
        #   * cdf: N*D*8 bytes (float64)
        #   * cdf < u: N*D bytes (bool)
        #   * u: N*8 bytes
        # For N=10^7 and D=100, this is ~8 GB + ~1 GB + ~80 MB. The chunked
        # approach limits peak memory to _DM_SAMPLING_CHUNK_BYTES per chunk
        # (default 80 MB), at the cost of a Python-level loop over chunks
        # (typically 1-100 iterations for realistic batch sizes).
        n_pulses = len(indices)
        # Compute chunk size: how many rows can we process within the memory
        # budget? Each row needs D*8 bytes for cdf + D bytes for the bool
        # mask + 8 bytes for u. Use D*9 as a conservative per-row estimate.
        bytes_per_row = max(1, max_dim * 9 + 8)
        chunk_size = max(1, _DM_SAMPLING_CHUNK_BYTES // bytes_per_row)
        # Cap chunk_size at n_pulses to avoid allocating oversized chunks.
        chunk_size = min(chunk_size, n_pulses)

        photons = np.empty(n_pulses, dtype=np.int64)

        # 5th-review fix (issue 3.2): generate uniform random numbers
        # PER-CHUNK instead of pre-allocating u_all. The previous
        # implementation allocated ``u_all = rng.random(size=n_pulses)``
        # before the chunk loop. For n_pulses = 10^7, this single array
        # consumes 80 MB — exactly the per-chunk memory budget. The
        # pre-allocation therefore doubled the peak memory, defeating the
        # purpose of chunking. The per-chunk approach keeps peak memory
        # within the chunk budget. The slight overhead of multiple rng
        # calls is negligible compared to the memory savings.
        for start in range(0, n_pulses, chunk_size):
            end = min(start + chunk_size, n_pulses)
            chunk_probs = selected_probs[start:end]  # view, no copy
            # Build CDF per row directly from the pre-normalized table
            # (4th-review issue 4.2: skip the per-call renormalization).
            chunk_cdf = np.cumsum(chunk_probs, axis=1)
            # Guard against floating-point round-off in the last column.
            chunk_cdf[:, -1] = 1.0
            # Generate uniform random numbers for this chunk only (5th-review
            # issue 3.2).
            # 6th-review fix (issue 9): ensure u > 0 to handle zero-probability
            # Fock states correctly. The previous implementation used
            # ``chunk_u = rng.random(size=...)`` directly, which can return
            # exactly 0.0 (with probability ~2^-53). When ``chunk_cdf[0] == 0``
            # (i.e., p_0 = 0) and ``u == 0.0``, the strict inequality
            # ``chunk_cdf < chunk_u[:, None]`` evaluates to ``False`` for all
            # columns, causing ``.sum(axis=1)`` to return 0 — sampling the
            # vacuum state despite ``p_0 = 0``. The fix clamps u to a tiny
            # positive value (``np.finfo(np.float64).tiny ~ 2.2e-308``),
            # which is negligible compared to any physical probability but
            # strictly positive, ensuring the inverse-CDF correctly skips
            # zero-probability states.
            chunk_u = np.maximum(
                rng.random(size=end - start),
                np.finfo(np.float64).tiny,
            )
            # Inverse-CDF sampling: photons[i] = number of cdf values
            # strictly less than u[i]. Vectorized over the chunk.
            # 8th-review fix (F-13): replace (cdf < u).sum(axis=1) with
            # np.searchsorted for O(N log D) memory-efficient sampling.
            # The previous boolean-matrix approach allocated N x D booleans
            # per chunk (~1 GB for N=10^7, D=100). searchsorted uses O(N) memory.
            # 9th-review fix (F-11): replaced per-row Python loop with
            # vectorized boolean-matrix approach. The chunk memory budget
            # already limits the chunk size, so N x D bytes is safe within
            # each chunk for typical QKD dimensions (D <= 100).
            # This is ~100-1000x faster than the per-row loop for typical
            # batch sizes (10^3-10^5 pulses).
            chunk_photons = (chunk_cdf < chunk_u[:, None]).sum(axis=1).astype(np.int64)
            photons[start:end] = chunk_photons

        # Clamp to valid Fock-index range (defensive; cdf[:,-1]=1.0 ensures
        # photons stay within [0, max_dim-1] for u in [0, 1)).
        np.clip(photons, 0, max_dim - 1, out=photons)

        # 5th-review fix (issue 2.2): apply emission failure as
        # post-computation masking (matching OpticalSource.generate_photons),
        # so that the emitted-vs-attempted tally can be tracked correctly.
        # Previously, emission failure was pre-mixed into the probability
        # table and the tally reported num_pulses_generated == num_pulses_attempted,
        # which conflated "emitted with 0 photons" and "failed to emit".
        # 6th-review fix (issue 6): ALWAYS consume ``n_attempted`` random
        # numbers from the RNG, even when ``p_emit`` is close to 0.0 or 1.0,
        # to preserve RNG stream consistency across parameter sweeps (see
        # the matching fix in OpticalSource.generate_photons).
        p_emit = clamp_probability(self.ideal_emission_probability)
        n_attempted = len(indices)
        # Unconditional RNG draw: preserves stream consistency.
        emit_mask = rng.random(size=n_attempted) < p_emit
        photons = np.where(emit_mask, photons, 0).astype(np.int64, copy=False)
        n_emitted = int(np.count_nonzero(emit_mask))

        # Record tallies: attempted = all pulses requested, emitted = the
        # subset that actually emitted a (possibly zero-photon) pulse.
        self.update_internal_tallies(
            num_pulses_generated=n_emitted,
            num_pulses_attempted=n_attempted,
        )
        return int(photons[0]) if is_scalar else photons

    def generate_photons(
        self,
        alice_pulse_indices: Optional[Union[int, np.ndarray, Sequence[int]]],
        rng: RNGType,
        num_samples: int = 1,
    ) -> Union[np.ndarray, int]:
        """Generate photons (dispatches to :meth:`generate_photons_dm`)."""
        return self.generate_photons_dm(
            alice_pulse_indices=alice_pulse_indices,
            rng=rng,
            num_samples=num_samples,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_config_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dict (round-trip safe with :meth:`from_dict`).

        Density matrices are encoded in the JSON-safe
        ``{"real": [[...]], "imag": [[...]}`` format to preserve complex
        values through JSON serialization (issue 5.12, 5.13).

        5th-review fix (issue 4.3): re-sanitize after adding subclass keys.
        The parent's ``to_config_dict`` calls ``sanitize_for_serialization``
        and returns a JSON-safe dict. The subclass then adds more keys
        (``density_matrices``, ``missing_dm_policy``,
        ``heterogeneous_dim_policy``). Although these specific keys are
        already JSON-safe (encoded DMs are lists of floats, policies are
        strings), the parent's post-condition — "the returned dict is
        JSON-safe" — is violated in principle because the subclass mutates
        the dict after sanitization. If a future subclass or refactor adds
        non-JSON-safe values (e.g., a raw ``np.ndarray`` or a
        ``complex128``), the post-condition would be silently violated.
        We re-sanitize to guarantee the post-condition, which is cheap
        (the dict is small and ``sanitize_for_serialization`` is
        idempotent).
        """
        data = OpticalSource.to_config_dict(self)
        data["density_matrices"] = _encode_density_matrices_for_serialization(
            self.density_matrices
        )
        data["missing_dm_policy"] = self.missing_dm_policy
        data["heterogeneous_dim_policy"] = self.heterogeneous_dim_policy
        # Re-sanitize to guarantee the JSON-safe post-condition.
        return sanitize_for_serialization(data)

    def to_dict(self) -> Dict[str, Any]:
        """Alias for :meth:`to_config_dict`."""
        return self.to_config_dict()


# ---------------------------------------------------------------------------
# PoissonSource
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PoissonSource(OpticalSource):
    """
    Convenience subclass for standard attenuated-laser (Poisson) modelling.

    Identical to :class:`OpticalSource` with ``statistics_type == POISSON``;
    provided for explicit typing and convenience.
    """

    def _post_init_hook(self) -> None:
        """Validate POISSON statistics type after base validation.

        Uses the hook pattern (see :meth:`OpticalSource.__post_init__`)
        rather than overriding ``__post_init__`` directly, so base
        validation is invoked automatically without fragile explicit
        parent calls.

        4th-review fix (issue 2.1): explicitly chains to the parent
        implementation via ``OpticalSource._post_init_hook(self)`` at the
        start (explicit class reference, not zero-argument ``super()``,
        because ``@dataclass(slots=True)`` breaks the implicit ``super()``
        in Python 3.12 — see :class:`OpticalSource` construction-strategy
        comment). Multi-level subclasses of :class:`PoissonSource` should
        chain via ``PoissonSource._post_init_hook(self)``.
        """
        OpticalSource._post_init_hook(self)
        if self.statistics_type != SourceStatisticsType.POISSON:
            raise ParameterValidationError(
                "PoissonSource requires config.statistics_type == "
                "SourceStatisticsType.POISSON "
                f"(got {self.statistics_type!r}).",
                param_name="statistics_type",
            )

    @classmethod
    def from_config(
        cls,
        config: OpticalSourceConfig,
        *,
        security_metadata: Optional[SourceSecurityMetadata] = None,
    ) -> "PoissonSource":
        """Build from :class:`OpticalSourceConfig` (must be POISSON)."""
        if config.statistics_type != SourceStatisticsType.POISSON:
            raise ParameterValidationError(
                "PoissonSource requires config.statistics_type == "
                "SourceStatisticsType.POISSON "
                f"(got {config.statistics_type!r}).",
                param_name="statistics_type",
            )
        return cls(config=config, security_metadata=security_metadata)

    @classmethod
    def from_protocol_parameters(
        cls,
        params: ProtocolParameters,
        *,
        security_metadata: Optional[SourceSecurityMetadata] = None,
    ) -> "PoissonSource":
        """Build from :class:`ProtocolParameters`."""
        if not isinstance(params, ProtocolParameters):
            raise ParameterValidationError(
                "params must be a ProtocolParameters instance.",
                param_name="params",
                param_value=type(params).__name__,
            )
        if not isinstance(params.optical, OpticalSourceConfig):
            raise ParameterValidationError(
                "params.optical must be an OpticalSourceConfig instance.",
                param_name="params.optical",
                param_value=type(params.optical).__name__,
            )
        return cls.from_config(params.optical, security_metadata=security_metadata)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        strict: bool = False,
        security_metadata: Optional[SourceSecurityMetadata] = None,
    ) -> "PoissonSource":
        """Build from a serialized dict (PoissonSource does not accept DMs).

        4th-review fixes (issues 2.3, 2.4):
          * **Issue 2.3:** in strict mode, DM-specific keys
            (``density_matrices``, ``missing_dm_policy``,
            ``heterogeneous_dim_policy``) are rejected because
            :class:`PoissonSource` is not a DM source.
          * **Issue 2.4:** the ``density_matrices`` payload is now checked
            with ``isinstance(..., Mapping)`` (matching
            :meth:`OpticalSource.from_dict`) instead of a bare truthiness
            test. The previous code raised a custom rejection error for
            non-mapping truthy objects (e.g., a boolean or integer),
            whereas :meth:`OpticalSource.from_dict` raised a parameter-
            type validation error. The new implementation uses the same
            type check as :meth:`OpticalSource.from_dict` for consistency.
        """
        if not isinstance(data, Mapping):
            raise ParameterValidationError(
                "data must be a mapping.",
                param_name="data",
                param_value=type(data).__name__,
            )

        if strict:
            # 4th-review fix (issue 2.3): PoissonSource is not a DM source,
            # so DM-specific keys are rejected in strict mode.
            unknown = set(data.keys()) - _KNOWN_BASE_CONFIG_KEYS
            if unknown:
                raise ParameterValidationError(
                    f"Unknown or DM-only keys in PoissonSource config: "
                    f"{sorted(unknown)}. PoissonSource does not accept "
                    f"DM-specific keys (density_matrices, missing_dm_policy, "
                    f"heterogeneous_dim_policy). Use DensityMatrixSource "
                    f"if you need DM sampling. Pass strict=False to ignore "
                    f"unknown keys.",
                    param_name="data",
                )

        config = _coerce_optical_source_config(data)

        # 4th-review fix (issue 2.4): use isinstance(..., Mapping) instead
        # of a bare truthiness test, matching OpticalSource.from_dict's
        # behavior. A non-mapping truthy payload (e.g., 42 or True) now
        # raises the same parameter-type validation error as
        # OpticalSource.from_dict, instead of the custom "does not accept
        # density_matrices" rejection.
        if "density_matrices" in data:
            raw_dms = data["density_matrices"]
            if raw_dms is not None:
                if not isinstance(raw_dms, Mapping):
                    raise ParameterValidationError(
                        "density_matrices payload must be a mapping (dict) "
                        "from pulse name to matrix data. Got a non-mapping "
                        f"payload of type {type(raw_dms).__name__}, which "
                        "indicates a corrupted or malformed serialized config.",
                        param_name="density_matrices",
                        param_value=type(raw_dms).__name__,
                    )
                if len(raw_dms) > 0:
                    raise ParameterValidationError(
                        "PoissonSource.from_dict does not accept non-empty "
                        "density_matrices; use OpticalSource.from_dict or "
                        "DensityMatrixSource.from_dict.",
                        param_name="density_matrices",
                    )

        return cls.from_config(config, security_metadata=security_metadata)
