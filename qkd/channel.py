# qkd/channel.py
# -*- coding: utf-8 -*-
"""
Models a simple quantum channel, focusing on fiber optic attenuation.

This module reuses :class:`AttenuationConfig` from ``qkd.datatypes`` instead of
redefining an attenuation configuration structure locally.

Channel model
-------------
The channel is a **pure scalar attenuation** channel. Given a fiber of
length ``L`` (km) and an attenuation coefficient ``alpha`` (dB/km), the
total channel loss and transmittance are::

    L_total = alpha * L                      [dB]
    T       = 10 ** (-L_total / 10)          [dimensionless, in (0, 1]]

This is the standard Beer--Lambert form used for optical-fibre QKD
(see e.g. Lo--Ma--Chen 2005 for decoy-state BB84).

Wavelength field
----------------
An optional ``wavelength_nm`` field (F-15) allows recording the operating
wavelength for provenance and consistency-checking. When set, a
physical-suspicion warning is emitted if ``alpha`` is inconsistent with
the typical attenuation range for that wavelength (e.g. at 1550 nm,
alpha should be ~0.15--0.25 dB/km). The field is NOT used in equality
comparison unless both channels have a wavelength set.

Limitations
-----------
This model is intentionally minimal. It does **not** model any of the
following physics, and results that depend on them are not trustworthy
without external composition (see :meth:`FiberChannel.limitations` for
a programmatic query interface, review S-6):

* **Wavelength dependence** of ``alpha``. Real fibres exhibit
  ``alpha ~= 0.2 dB/km`` at 1550 nm, ``0.35 dB/km`` at 1310 nm, and
  ``2--3 dB/km`` at 850 nm. The scalar ``alpha`` here is whatever the
  caller supplies; cross-wavelength comparisons are not validated
  (review S-4). The optional ``wavelength_nm`` field (F-15) emits a
  physical-suspicion warning on inconsistency but does NOT enforce a
  specific alpha.
* **Coupling, splice, and connector losses** (typically 0.5--3 dB
  total in real links). These must be added by the caller (review S-5).
* **Chromatic dispersion** (pulse broadening ``~ D(lambda) * L * Delta_lambda``).
* **Polarisation-mode dispersion** (differential group delay).
* **Polarisation-dependent loss** (PDL).
* **Time dependence** (thermal / mechanical fluctuations of ``alpha``
  and polarisation drift). A Monte-Carlo simulation that samples
  channel parameters across time should NOT use a single static
  :class:`FiberChannel` instance -- construct a fresh instance per
  time step (review S-7).
* **Non-uniform attenuation** along the fibre (splices, repair
  sections, heterogeneous fibre types).
* **Nonlinear optical effects** (self-phase modulation, Raman /
  Brillouin scattering).
* **Free-space / satellite links** (review C-16, S-1). The fibre
  Beer--Lambert model is NOT physically valid for free-space channels,
  which have a fundamentally different loss budget (geometric,
  atmospheric, pointing). ``MAX_DISTANCE_KM`` was lowered from
  50,000 km to 10,000 km (F-16) to reflect per-segment fibre-realistic
  distances. Multi-segment chains must be composed via
  :meth:`cascade`. Satellite QKD users should use a separate
  ``FreeSpaceChannel`` class (not yet implemented).

Detector / coupling composition
-------------------------------
The transmittance returned by :class:`FiberChannel` is ``eta_channel``
**only**. In a QKD link budget, the overall detection efficiency is::

    eta = eta_channel * eta_detector * eta_coupling

Typical SNSPD detector efficiencies are ``eta_detector ~ 0.1 -- 0.3``.
Users who plug ``channel.transmittance`` directly into the decoy-state
gain formula ``Q_mu = sum_n Y_n * mu^n * e^-mu / n!`` with
``Y_n = 1 - (1 - eta)^n + Y_0`` will underestimate total loss by the
detector and coupling efficiencies, which can shift the predicted
key-rate curve by 50--100 km. The composition MUST be performed by the
caller. A subclass :class:`AttenuationOnlyFiberChannel` is provided to
make this restriction visible at the type-check level via ``isinstance``
(review C-41, S-2).

Exception hierarchy
-------------------
A :class:`ChannelError` base exception (F-41) is defined locally in
this module and re-exported. Upstream should eventually make both
:class:`ConfigurationError` and :class:`ParameterValidationError`
inherit from it. For now, it serves as a common catch target for
channel-related errors.

Strict mode
-----------
Strict mode is implemented as a :class:`contextvars.ContextVar` and is
therefore thread-safe and reproducible (review A-3). Set
``QKD_STRICT=1`` (or ``true``/``yes``/``on``, case-insensitive) in the
environment, or call :func:`set_strict_mode` ``(True)`` to enable it.

In strict mode this module REJECTS configurations that would otherwise
be silently approximated:

* Transmittance numerical underflow to ``0.0`` (raises instead of
  warning).
* ``AttenuationConfig`` contract violations (raises instead of
  warning).
* Subnormal transmittance values (raises instead of warning,
  F-subnormal).

Physical-suspicion warnings
---------------------------
Warnings about physically suspicious configurations (e.g. sub-millimetre
fibre, transmittance underflow, contract violations) are emitted via
:func:`warnings.warn` with the :class:`PhysicalSuspicionWarning`
category (review C-15). This ensures the warnings appear by default,
unlike the previous ``logger.warning`` calls which were silent unless
logging was explicitly configured.
"""
from __future__ import annotations

__version__ = "4.2.1"
import contextlib
import contextvars
import copy  # F-44: moved import copy to module level (was lazy import in __deepcopy__)
import dataclasses
import importlib.metadata
import logging
import math
import os
import sys
import types
import warnings
from collections.abc import Hashable
from dataclasses import dataclass, field, replace as _dc_replace
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

# ---------------------------------------------------------------------------
# Hard dependencies (review A-2): no fallback imports.
# ``qkd.exceptions``, ``qkd.constants`` and ``qkd.datatypes`` MUST be
# importable. If they are not, raise ImportError with a clear message
# rather than silently degrading equality / validation semantics.
# ---------------------------------------------------------------------------
from .exceptions import ParameterValidationError, ConfigurationError
from .constants import is_close, is_finite_non_negative
from .datatypes import AttenuationConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# F-41: ChannelError base exception. Defined locally and re-exported.
# Upstream should eventually make ConfigurationError and
# ParameterValidationError inherit from this. For now, it serves as a
# common catch target for channel-related errors.
# ---------------------------------------------------------------------------
class ChannelError(Exception):
    """Base exception for all channel-related errors.

    F-02 fix (review channel_review_8): The dual-exception contract
    (ConfigurationError + ParameterValidationError without a common
    base) means callers who ``except ConfigurationError`` will silently
    miss ``ParameterValidationError``, accepting invalid configs. The
    :data:`CHANNEL_ERRORS` tuple is the only safe catch mechanism until
    the hierarchy is fixed upstream.

    F-H2: The upstream ``qkd.exceptions`` module MUST be updated so
    that both :class:`ConfigurationError` and
    :class:`ParameterValidationError` inherit from :class:`ChannelError`.
    Once that is done, catching ``ChannelError`` alone will catch both
    exception types, and :data:`CHANNEL_ERRORS` can be deprecated.

    **Required upstream change** (F-H2, F-02): In ``qkd.exceptions``,
    modify the class hierarchy so that ``ConfigurationError`` and
    ``ParameterValidationError`` both inherit from ``ChannelError``
    (which should be defined in ``qkd.exceptions``, not locally here).
    Then ``qkd.channel.ChannelError`` can be removed or made to
    inherit from the upstream version. :data:`CHANNEL_ERRORS` can be
    deprecated with a ``DeprecationWarning`` once the hierarchy is fixed.
    """
    pass

# P-08: Convenience tuple for catching all channel-related errors.
# Since ConfigurationError and ParameterValidationError do NOT inherit
# from ChannelError yet, this tuple allows a single ``except`` clause
# to catch both. When upstream aligns the hierarchy, catching
# ChannelError alone will suffice and this tuple can be deprecated.
CHANNEL_ERRORS: Tuple[Type[Exception], ...] = (ConfigurationError, ParameterValidationError)

# F-02 fix (review channel_review_8): Runtime check that the exception
# hierarchy is properly set up. If ConfigurationError and
# ParameterValidationError both inherit from ChannelError, then
# CHANNEL_ERRORS can be deprecated. Until then, emit a warning at
# import time to remind developers that the upstream fix is needed.
if not (
    issubclass(ConfigurationError, ChannelError)
    and issubclass(ParameterValidationError, ChannelError)
):
    logger.debug(
        "F-02: ConfigurationError and ParameterValidationError do NOT "
        "inherit from ChannelError. Callers MUST use CHANNEL_ERRORS "
        "to catch both exception types. Fix qkd.exceptions to make "
        "both inherit from ChannelError."
    )


__all__ = [
    # Core class & alias
    "FiberChannel",
    "AttenuationOnlyFiberChannel",
    # Re-exported dependencies (so users can import from one place)
    "AttenuationConfig",
    "ParameterValidationError",
    "ConfigurationError",
    # F-41: ChannelError base exception
    "ChannelError",
    "CHANNEL_ERRORS",  # P-08: convenience catch-all tuple
    "is_close",
    "is_finite_non_negative",
    # Strict-mode API
    "get_strict_mode",
    "set_strict_mode",
    "reset_strict_mode",
    # Validate-on-construction API (F-30)
    "get_validate_on_construction",
    "set_validate_on_construction",
    "reset_validate_on_construction",
    # Warning category (review C-15)
    "PhysicalSuspicionWarning",
    # Constants
    "MAX_DISTANCE_KM",
    "MIN_MEANINGFUL_DISTANCE_KM",
    "WARNING_DISTANCE_KM",
    "MAX_FIBER_LOSS_DB_KM",
    "TRANSMITTANCE_SUBNORMAL_THRESHOLD",
    "TRANSMITTANCE_UNDERFLOW_THRESHOLD",  # deprecated alias
    "UNITS",
    "VALIDATED_UNITS",  # P-12
    "METADATA_UNITS",  # P-12
    # Wavelength-related constants (F-15)
    "TYPICAL_WAVELENGTH_ALPHA_RANGES",
    "WAVELENGTH_CHECK_TOLERANCE_NM",
    # Type aliases (review A-10)
    "Transmittance",
    "Distance",
    "Loss",
    "Numeric",
    # F-08: new name for loss_fraction_per_km
    "single_km_power_loss_fraction",
    # ── Simulation helpers (vectorized + composition) ──
    # Vectorized transmittance computation for bulk distance sweeps
    "transmittance_array",
    "transmittance_scalar",
    # Factory that reads from the main_optimized config dict format
    "build_attenuation_config_from_sim_config",
    # Link-budget composition helper
    "link_efficiency",
    # Lightweight bundle of channel-derived quantities for simulation use
    "ChannelSimParams",
    "build_channel_sim_params",
]

# ---------------------------------------------------------------------------
# Type aliases (review A-10).
# Documentation-only; runtime type is plain ``float``.
# Review C-50 (low severity) notes that ``NewType`` would provide
# stricter type-checking but would break backward compatibility with
# existing call sites that pass plain ``float``. Kept as aliases.
# ---------------------------------------------------------------------------
Transmittance = float
Distance = float
Loss = float
Numeric = Union[int, float]

T_FiberChannel = TypeVar("T_FiberChannel", bound="FiberChannel")

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Smallest *normal* positive float (~2.225e-308). Values strictly below
#: this are subnormal (denormal) but have not yet underflowed to zero.
#: The next step below the smallest subnormal (~5e-324) is 0.0.
TRANSMITTANCE_SUBNORMAL_THRESHOLD: float = sys.float_info.min
#: Deprecated alias (review C-58).
TRANSMITTANCE_UNDERFLOW_THRESHOLD: float = TRANSMITTANCE_SUBNORMAL_THRESHOLD

#: Maximum supported fibre distance, in km (review C-16, S-1, F-16).
#:
#: **F-16 fix: Raised from 1,000 km to 10,000 km.** The model
#: ``T = 10^(-alpha*L/10)`` is a per-segment Beer--Lambert law.
#: Multi-segment chains must be composed via :meth:`cascade`. The
#: previous limit of 1,000 km excluded amplified links; 10,000 km
#: covers ultra-long-haul amplified fibre links (per segment).
#: For satellite QKD, use a separate ``FreeSpaceChannel`` class
#: (not yet implemented).
MAX_DISTANCE_KM: float = 10_000.0

#: Minimum distance below which a warning (or, in strict mode, an error)
#: is emitted. 1 mm (1e-6 km) so that legitimate 0.5--2 m lab patch
#: fibres do not trip the warning.
WARNING_DISTANCE_KM: float = 1e-6
#: Deprecated alias preserved for backwards compatibility; new code
#: should use :data:`WARNING_DISTANCE_KM`.
MIN_MEANINGFUL_DISTANCE_KM: float = WARNING_DISTANCE_KM

#: Maximum supported fibre attenuation coefficient, in dB/km (F-17).
#:
#: **F-17 fix: Raised from 10 dB/km to 100 dB/km.** The previous limit
#: excluded specialty fibre (e.g. highly attenuating fibre for
#: filtering or attenuation purposes). 100 dB/km covers all known
#: practical and specialty fibre types.
MAX_FIBER_LOSS_DB_KM: float = 100.0

#: Units registry. Read-only (review A-4): mutation attempts
#: raise :class:`TypeError`.
#:
#: F-15: Added ``wavelength_nm`` entry.
#:
#: Dual-purpose: the first two entries are validated by
#: :meth:`FiberChannel.validate_parameter`; the latter entries are
#: metadata-only (used by :meth:`FiberChannel.to_config_dict` and
#: :meth:`FiberChannel.__repr__`). Review C-59 (low severity) suggests
#: splitting into ``VALIDATED_UNITS`` and ``METADATA_UNITS``; kept as a
#: single registry for backward compatibility.
# P-12: Split UNITS into VALIDATED_UNITS and METADATA_UNITS.
# VALIDATED_UNITS contains entries that are validated by
# :meth:`FiberChannel.validate_parameter`; METADATA_UNITS
# contains entries that are metadata-only (used by
# :meth:`FiberChannel.to_config_dict` and
# :meth:`FiberChannel.__repr__`). UNITS is preserved as a
# combined registry for backward compatibility.
VALIDATED_UNITS: types.MappingProxyType = types.MappingProxyType({
    "distance_km": "km",
    "fiber_loss_db_km": "dB/km",
})

METADATA_UNITS: types.MappingProxyType = types.MappingProxyType({
    "total_loss_db": "dB",
    "transmittance": "dimensionless",
    "wavelength_nm": "nm",  # F-15
})

UNITS: types.MappingProxyType = types.MappingProxyType({
    **dict(VALIDATED_UNITS),
    **dict(METADATA_UNITS),
})

# ---------------------------------------------------------------------------
# F-15: Wavelength consistency check ranges
# ---------------------------------------------------------------------------
#: Typical alpha ranges for standard fibre at common QKD wavelengths.
#: Keys are wavelengths in nm; values are ``(alpha_min, alpha_max)``
#: tuples in dB/km.
#: C-7 fix (review channel_review_8): Wrapped in
#: ``types.MappingProxyType`` to prevent runtime mutation. A test or
#: notebook that mutates this dict could change validation behavior for
#: all subsequent constructions, causing silent validation drift.
TYPICAL_WAVELENGTH_ALPHA_RANGES: types.MappingProxyType = types.MappingProxyType({
    1550.0: (0.15, 0.25),
    1310.0: (0.30, 0.40),
    850.0: (2.0, 3.0),
    # P-14: Added common QKD wavelengths.
    780.0: (2.5, 4.0),    # short-wavelength free-space / VCSEL
    1064.0: (0.5, 1.0),   # Nd:YAG wavelength
    1625.0: (0.15, 0.25), # L-band telecom (similar to 1550)
})

#: Tolerance (in nm) for matching a wavelength to a known range key.
#: If ``wavelength_nm`` is within this tolerance of a key in
#: :data:`TYPICAL_WAVELENGTH_ALPHA_RANGES`, the alpha consistency
#: check is applied.
WAVELENGTH_CHECK_TOLERANCE_NM: float = 50.0

# F-H1 fix: Wavelength range validation boundaries.
# Typical QKD fiber wavelengths are in the 380--2000 nm range
# (UV-visible through near-IR). Values outside this range are
# physically meaningless for fiber QKD and are rejected in
# strict mode or warned in non-strict mode.
_MIN_QKD_WAVELENGTH_NM: float = 380.0
_MAX_QKD_WAVELENGTH_NM: float = 2000.0

# F-L2 fix: Shared singleton for the empty MappingProxyType used as
# default_factory for _metadata_extra. Previously, a new
# MappingProxyType({}) was created per instance, but since the empty
# proxy is immutable, a shared singleton is cheaper.
_EMPTY_METADATA_EXTRA = types.MappingProxyType({})

# ---------------------------------------------------------------------------
# Repr formatting thresholds
# ---------------------------------------------------------------------------

#: Below this transmittance, ``__repr__`` switches to scientific notation.
_REPR_TRANSMITTANCE_SCIENTIFIC_THRESHOLD: float = 1e-4

#: P-13: Lowered from 1e4 to 1e3 so the scientific-notation branch
#: fires for realistic configurations. At ``alpha=0.2``, ``L=500``
#: gives ``total_loss_db = 100`` dB, well below the old threshold of
#: 1e4. At ``L=5000``, ``total_loss_db = 1000``, which is above 1e3.
#: The new threshold ensures realistic long-haul links use scientific
#: notation in ``__repr__``.
_REPR_LARGE_LOSS_THRESHOLD: float = 1e3

# ---------------------------------------------------------------------------
# Numerical-stability constants (review C-11, C-17, C-26, C-60)
# ---------------------------------------------------------------------------

#: Named constant for the underflow absolute tolerance in
#: :meth:`FiberChannel.validate` (review C-11, C-60). Previously a
#: magic constant ``1e-323``; now derived from :data:`sys.float_info.min`
#: (the smallest *normal* positive float) for self-documentation and
#: robustness against future Python float-format changes.
_TRANSMITTANCE_UNDERFLOW_ABS_TOL: float = sys.float_info.min

#: Tighter tolerance for the :class:`AttenuationConfig` contract check
#: (review C-17). Previously used :func:`is_close` with default
#: ``rel_tol=1e-9``, which for ``total_loss_db = 1e5`` dB allowed
#: discrepancies of ~1e-4 dB -- effectively a no-op for long links.
#: F-9: These are now used everywhere that needs a contract-level
#: tolerance, including :meth:`from_config_dict` three-key consistency
#: check and :meth:`validate`.
#:
#: F-M1 note (review channel_review_6): ``_CONTRACT_CHECK_ABS_TOL = 1e-12``
#: is extremely tight as an absolute tolerance. For normal-sized products
#: (e.g. alpha*L ~ 10--1e3), the relative tolerance dominates and the
#: check is comfortably loose. For tiny products near 1e-12, the absolute
#: tolerance is equal to the product, making the check very loose. The
#: concern is for products in the range 1e-14 to 1e-12, where the absolute
#: tolerance is tight relative to the product but the relative tolerance
#: has vanished. However, both paths typically compute the same product
#: with the same operands, so the practical risk is low. If this becomes
#: an issue, consider relaxing to 1e-10.
_CONTRACT_CHECK_REL_TOL: float = 1e-12
_CONTRACT_CHECK_ABS_TOL: float = 1e-12

#: C-4 fix (review channel_review_8): Single named constant for
#: "is total loss effectively zero?" used in :meth:`from_total_loss`.
#: Previously, L2844 used ``is_close(total, 0.0)`` with default
#: ``abs_tol=0.0``, while L2880 used ``is_close(total, 0.0,
#: abs_tol=_CONTRACT_CHECK_ABS_TOL=1e-12)``. For ``0 < L_tot < 1e-12``
#: dB, the two branches disagreed. Now both use this single constant,
#: ensuring consistent zero-loss semantics everywhere.
_ZERO_LOSS_ABS_TOL: float = 1e-12

# ---------------------------------------------------------------------------
# Module-level key sets for :meth:`FiberChannel.from_config_dict`
# (review C-48: moved from per-call computation to module-level constants).
# ---------------------------------------------------------------------------

_REQUIRED_KEYS_CANONICAL: frozenset = frozenset({"distance_km", "fiber_loss_db_km"})
_REQUIRED_KEYS_TOTAL_LOSS: frozenset = frozenset({"distance_km", "total_loss_db"})
# F-15: wavelength_nm is a known optional key, not an unknown key.
_IGNORED_KEYS: frozenset = frozenset({"_metadata", "wavelength_nm"})
_KNOWN_OPTIONAL_KEYS: frozenset = frozenset({"wavelength_nm"})

# ---------------------------------------------------------------------------
# Strict-mode flag (review A-3)
# ---------------------------------------------------------------------------

#: Thread-local strict-mode flag. Defaults to the value of the
#: ``QKD_STRICT`` environment variable, parsed case-insensitively
#: against ``("1", "true", "yes", "on")``. Use :func:`get_strict_mode`,
#: :func:`set_strict_mode`, and :func:`reset_strict_mode` to access it.
#:
#: Review C-51 (low severity) notes that the default is fixed at import
#: time; if the environment variable changes after import, the default
#: does not update. :func:`set_strict_mode` can override at runtime.
#: Review C-52 (low severity) notes that ``ContextVar`` is thread-local
#: / async-local, which is good for reproducibility but makes global
#: strict mode harder to enforce.
_STRICT_MODE: "contextvars.ContextVar[bool]" = contextvars.ContextVar(
    "QKD_STRICT",
    default=os.environ.get("QKD_STRICT", "").lower()
    in ("1", "true", "yes", "on"),
)


def get_strict_mode() -> bool:
    """Return the current strict-mode flag for this context.

    The flag is stored in a :class:`contextvars.ContextVar` and is
    therefore thread-safe and async-safe (review A-3).
    """
    return _STRICT_MODE.get()


def set_strict_mode(enabled: bool) -> contextvars.Token:
    """Enable or disable strict mode for the current context.

    Returns a :class:`contextvars.Token` that can be passed to
    :func:`reset_strict_mode` to restore the previous value. This
    makes scoped strictness easy::

        token = set_strict_mode(True)
        try:
            ch = FiberChannel(AttenuationConfig(2000.0, 0.2))
        finally:
            reset_strict_mode(token)
    """
    return _STRICT_MODE.set(bool(enabled))


def reset_strict_mode(token: contextvars.Token) -> None:
    """Reset strict mode to its previous value using a token from :func:`set_strict_mode`."""
    _STRICT_MODE.reset(token)


# ---------------------------------------------------------------------------
# C-10 fix: ContextVar flag to skip transmittance health check in
# __post_init__ when from_total_loss will provide the override-based
# value. Set before cls() call in from_total_loss; __post_init__
# checks it and skips the health check on the old (alpha*L-based)
# transmittance. The single correct health check is performed in
# from_total_loss on the final override-based transmittance.
# ---------------------------------------------------------------------------
_SKIP_TRANSMITTANCE_HEALTH: "contextvars.ContextVar[bool]" = contextvars.ContextVar(
    "_SKIP_TRANSMITTANCE_HEALTH", default=False,
)


# ---------------------------------------------------------------------------
# F-30: Validate-on-construction flag (ContextVar, like strict mode)
# ---------------------------------------------------------------------------
#: Thread-local flag for automatic validation at end of ``__post_init__``.
#: Default ``False`` preserves current behaviour. Set to ``True`` for
#: extra assurance in critical pipelines.
_VALIDATE_ON_CONSTRUCTION: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "QKD_VALIDATE_ON_CONSTRUCTION",
    default=False,
)


def get_validate_on_construction() -> bool:
    """Return the current validate-on-construction flag for this context.

    F-30: Converted from mutable ClassVar to ContextVar for thread safety.
    """
    return _VALIDATE_ON_CONSTRUCTION.get()


def set_validate_on_construction(enabled: bool) -> contextvars.Token:
    """Enable or disable validate-on-construction for the current context.

    F-30: Returns a :class:`contextvars.Token` for scoped usage,
    analogous to :func:`set_strict_mode`.
    """
    return _VALIDATE_ON_CONSTRUCTION.set(bool(enabled))


def reset_validate_on_construction(token: contextvars.Token) -> None:
    """Reset validate-on-construction to its previous value.

    F-30: Use the token returned by :func:`set_validate_on_construction`.
    """
    _VALIDATE_ON_CONSTRUCTION.reset(token)


@contextlib.contextmanager
def _optional_strict(strict: Optional[bool]) -> Iterator[None]:
    """Context manager that temporarily sets strict mode if ``strict`` is not ``None``.

    Used by :meth:`FiberChannel.with_distance` / :meth:`FiberChannel.with_loss`
    to honour an explicit ``strict=`` override.
    """
    if strict is None:
        yield
        return
    token = set_strict_mode(bool(strict))
    try:
        yield
    finally:
        reset_strict_mode(token)


# ---------------------------------------------------------------------------
# Physical-suspicion warning category (review C-15)
# ---------------------------------------------------------------------------


class PhysicalSuspicionWarning(UserWarning):
    """Warning emitted when a channel configuration is physically suspicious.

    These warnings were previously emitted via ``logger.warning``, which
    is silent by default unless logging is configured. Using
    :func:`warnings.warn` with this category ensures the warnings appear
    by default (review C-15), so users see physically suspicious
    configurations (e.g. 1-nm fibre with non-zero alpha) without having
    to configure logging.
    """


def _warn_physical_suspicion(msg: str, *, stacklevel: int = 2) -> None:
    """Emit a :class:`PhysicalSuspicionWarning`.

    H-8 fix (review channel_review_8): Previously, this function called
    both ``warnings.warn`` and ``logger.warning``, so every suspicious
    configuration produced two messages. This caused log noise and
    could lead to users filtering both, missing real warnings. Now,
    only ``warnings.warn`` is called. Users who configured logging
    can set up a ``warnings`` -> ``logging`` filter if they want
    structured logging; this is documented in the module-level docs.
    Production users who want structured logging should use Python's
    ``logging.captureWarnings(True)`` which routes all ``warnings.warn``
    calls through the logging system automatically.

    Review C-15.
    """
    warnings.warn(msg, PhysicalSuspicionWarning, stacklevel=stacklevel)


# ---------------------------------------------------------------------------
# Shared numeric coercion helper (review A-5)
# ---------------------------------------------------------------------------


def _coerce_numeric(
    value: Any,
    *,
    param_name: str,
) -> float:
    """Coerce ``value`` to ``float`` with a clear error message.

    Catches ``ValueError``, ``TypeError`` and the rare ``OverflowError``
    that some numeric types (e.g. :class:`decimal.Decimal` with huge
    exponents) can raise on ``float()``.

    **C-5 fix:** ``bool`` values are explicitly rejected.
    ``float(True)`` would silently produce ``1.0``, masking bugs where
    JSON-parsed ``0``/``1`` round-trip as booleans. The
    :meth:`validate_parameter` docstring claims "Booleans are explicitly
    rejected"; this fix makes that claim true for all factory paths.

    **F-33 fix:** After ``float(value)`` succeeds, if the original type
    was not ``int``, ``float``, or ``numpy`` numeric, a DEBUG log
    warning is emitted about precision loss from downcasting
    :class:`decimal.Decimal` / :class:`fractions.Fraction` inputs.

    Note on strings (review C-19): this helper still accepts numeric
    strings (e.g. ``"100"``) for backward compatibility with
    :meth:`from_total_loss`, :meth:`with_distance`, and :meth:`with_loss`.
    :meth:`from_tuple` and :meth:`from_config_dict` add explicit string
    guards to catch config bugs where a string is passed instead of a
    number.
    """
    # C-5: reject bool before float() silently converts True -> 1.0.
    if isinstance(value, bool):
        raise ParameterValidationError(
            f"Parameter {param_name!r} must be a real number, not a bool "
            f"(got {value!r}). Booleans are explicitly rejected to catch "
            f"JSON-parsing bugs where 0/1 round-trip as booleans.",
            param_name=param_name,
            param_value=value,
        )
    original_type = type(value)
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as e:
        raise ParameterValidationError(
            f"Parameter {param_name!r} must be a real number convertible to "
            f"float (got {value!r} of type {type(value).__name__}).",
            param_name=param_name,
            param_value=value,
            cause=e,
        ) from e
    # F-33: warn about precision loss for non-builtin/non-numpy types.
    if original_type.__module__ not in ("builtins", "numpy"):
        logger.debug(
            "Parameter %r was coerced from %s (module %s) to float, "
            "which may lose precision for Decimal/Fraction inputs.",
            param_name,
            original_type.__name__,
            original_type.__module__,
        )
    return result


# ---------------------------------------------------------------------------
# Frozen sentinel for the lazy ``_transmittance`` field
# ---------------------------------------------------------------------------


class _Unset:
    """Singleton sentinel meaning 'derived field not yet computed'."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return "<unset>"

    def __bool__(self) -> bool:
        return False


_UNSET: Union[_Unset, float] = _Unset()

# ---------------------------------------------------------------------------
# F-37 / F-M3 fix: Module-level flag for one-time deprecation warning in
# ``linear_loss_per_km``. Uses threading.Lock for thread safety.
# Previously, the bare ``bool`` flag was not thread-safe; two threads
# could both see it as False and both emit the DeprecationWarning.
# ---------------------------------------------------------------------------
import threading

_LINEAR_LOSS_LOCK = threading.Lock()
_LINEAR_LOSS_WARNED: bool = False


# ===========================================================================
# FiberChannel
# ===========================================================================


@dataclass(frozen=True, slots=True)
class FiberChannel:
    """Models a simple optical fiber channel defined by attenuation.

    This class is immutable and wraps an :class:`AttenuationConfig`
    instance from :mod:`qkd.datatypes` to represent the underlying
    channel attenuation settings.

    Equality semantics
    ------------------
    Equality (``==``) is **tolerance-based** and uses
    :func:`qkd.constants.is_close` on the two defining scalar fields
    ``distance_km`` and ``fiber_loss_db_km`` only. Tolerance-based
    equality is **non-transitive**: ``a == b`` and ``b == c`` does not
    imply ``a == c``. Therefore instances are intentionally
    **unhashable** (``__hash__`` is set to ``None`` after class
    decoration) and must not be used as dict keys or set members.

    **C-20 note:** Equality compares only the defining scalars
    (``distance_km``, ``fiber_loss_db_km``); it does NOT compare
    ``_total_loss_db_override``. Two channels built via
    :meth:`from_total_loss` with different ``total_loss_db`` inputs but
    the same resulting ``(L, alpha)`` will compare equal even though
    their :attr:`total_loss_db` properties differ. For full semantic
    comparison including the override, use :meth:`equals_semantic`.

    For regression tests requiring exact equality, use
    :meth:`equals_exact` (bit-for-bit on defining scalars) or
    :meth:`equals_semantic` (includes ``_total_loss_db_override``).

    F-15: When both instances have ``wavelength_nm`` set, equality also
    requires ``is_close`` on the wavelength. If either instance has
    ``wavelength_nm = None``, the wavelength dimension is ignored.

    NaN behaviour
    -------------
    If any defining scalar field is ``NaN`` (which can only happen via
    monkey-patching or pickle corruption -- :class:`AttenuationConfig`
    rejects ``NaN`` at construction), :meth:`__eq__` returns ``False``
    (because :func:`is_close` returns ``False`` for ``NaN`` operands)
    rather than raising. This is the standard Python idiom for
    tolerance-based equality.

    Subclassing
    -----------
    ``__eq__`` uses :func:`isinstance(other, FiberChannel)`, which
    means a subclass instance with extra fields compares equal to a
    base instance if the two defining scalars match. This is
    intentional covariance. Subclasses that add fields SHOULD override
    :meth:`__eq__` and :meth:`equals_exact` to include the new fields.

    Pickling
    --------
    Explicit :meth:`__getstate__` / :meth:`__setstate__` are provided
    so that the ``init=False`` derived field ``_transmittance``
    survives pickling. :meth:`__setstate__` calls :meth:`validate` at
    the end to catch corruption (review C-10, C-32). F-2: raises
    :class:`ConfigurationError` if ``"attenuation"`` is missing from
    the state dict. F-3: calls :meth:`_check_transmittance_health`
    after restoring fields.

    Construction-time provenance
    ----------------------------
    The ``_construction_path`` field (review C-6) is now an ``init=True``
    field, so factories can pass it before ``__post_init__`` runs. This
    ensures warnings emitted during ``__post_init__`` include the
    correct construction path (previously always ``"direct"``).

    The ``_strict_mode_at_construction`` field (review C-1) records the
    strict-mode value at construction time (not serialisation time),
    so reproducibility metadata is not falsified when strict mode is
    toggled between construction and serialisation.
    """

    attenuation: AttenuationConfig

    # C-6: ``_construction_path`` is now ``init=True`` so factories can
    # pass it before ``__post_init__`` runs. Previously it was
    # ``init=False`` and set AFTER ``__post_init__``, causing warnings
    # emitted during ``__post_init__`` to always show ``"direct"``.
    _construction_path: str = field(default="direct", repr=False)

    # F-11: ``_total_loss_db_override`` is now ``init=False`` (was
    # ``init=True``). This prevents it from appearing in the
    # ``__init__`` signature and from being carried over by
    # ``dataclasses.replace`` in ``with_distance``/``with_loss``.
    # Factory methods (``from_total_loss``, ``_fast_construct``) use
    # two-phase construction: build the instance first, then set the
    # override via ``object.__setattr__``, then re-compute
    # ``_transmittance`` from the override using the
    # ``_transmittance_from_loss_db`` helper.
    _total_loss_db_override: Optional[float] = field(
        init=False, default=None, repr=False
    )

    # Sentinel default so the slot is always initialised even if
    # ``__post_init__`` raises partway through. The sentinel ``_UNSET``
    # is type-safe (no valid transmittance value aliases with it) and
    # is overwritten on successful construction.
    _transmittance: Any = field(init=False, repr=False, default=_UNSET)

    # C-1: Records the strict-mode value at construction time (not
    # serialisation time). Emitted in :meth:`to_config_dict`
    # ``_metadata.strict_mode`` so reproducibility metadata is not
    # falsified when strict mode is toggled between construction and
    # serialisation.
    _strict_mode_at_construction: bool = field(init=False, repr=False, default=False)

    # F-15: Optional wavelength in nanometers. When set, a consistency
    # check is performed against typical alpha ranges for the wavelength.
    wavelength_nm: Optional[float] = field(default=None, repr=False)

    # Unknown ``_metadata`` sub-keys preserved through round-trip.
    # Populated by :meth:`from_config_dict`; merged into the output of
    # :meth:`to_config_dict`.
    # P-07: Changed from mutable dict to MappingProxyType to enforce
    # immutability on the frozen dataclass. All writes now wrap the
    # dict in MappingProxyType before setting via object.__setattr__.
    # F-L2 fix: Use shared module-level singleton instead of creating
    # a new MappingProxyType per instance (review channel_review_6).
    # The empty MappingProxyType is immutable, so a shared singleton
    # is cheaper and avoids per-instance overhead.
    _metadata_extra: Mapping[str, Any] = field(
        init=False, repr=False, default_factory=lambda: _EMPTY_METADATA_EXTRA
    )

    # C-8 fix (review channel_review_8): __hash__ = None defined inside
    # the class body. Since @dataclass(frozen=True, eq=True) generates a
    # __hash__ based on fields, we must explicitly set it to None to
    # make the class unhashable (required because __eq__ is
    # tolerance-based and non-transitive, violating the hash contract).
    # NOTE: When slots=True, the dataclass decorator rebuilds the class
    # and may overwrite in-body __hash__. The post-decoration override
    # (below, after the class definition) is the reliable mechanism.
    # This in-body definition serves as a defensive marker and
    # documentation; the post-decoration line is the actual enforcement.
    __hash__ = None  # type: ignore[assignment,method-assign]

    # ------------------------------------------------------------------
    # __post_init__ -- decomposed into named sub-methods (review A-1)
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Validate inputs and pre-compute derived state.

        The work is decomposed into named helper methods so each step
        is unit-testable in isolation (review A-1).

        C-1: Records the construction-time strict mode in
        ``_strict_mode_at_construction`` before any validation runs,
        so warnings emitted during validation reflect the correct
        strict-mode state.

        F-11: Since ``_total_loss_db_override`` is now ``init=False``,
        it defaults to ``None`` during ``__post_init__``. Factory
        methods that need the override set it AFTER construction via
        ``object.__setattr__``, then re-compute ``_transmittance``.

        F-15: ``wavelength_nm`` consistency is checked during
        ``__post_init__``.
        """
        # C-1: Record the construction-time strict mode FIRST, before
        # any validation that may depend on or report strict-mode state.
        object.__setattr__(self, "_strict_mode_at_construction", get_strict_mode())

        self._check_type()
        self._validate_parameter_ranges()
        self._check_meaningful_alpha()  # F-04 fix
        self._check_meaningful_distance()
        self._check_total_loss_finite()
        self._check_attenuation_config_contract()
        self._check_wavelength_consistency()  # F-15
        self._validate_wavelength_range()  # F-H1 fix: wavelength range validation
        transmittance = self._compute_transmittance()
        # C-10 fix: Skip the health check in __post_init__ if the
        # ContextVar flag is set. from_total_loss sets this flag
        # before construction and clears it afterwards. This prevents
        # contradictory warnings from the old (alpha*L-based) and new
        # (override-based) transmittance values.
        if not _SKIP_TRANSMITTANCE_HEALTH.get():
            self._check_transmittance_health(transmittance)
        # ``object.__setattr__`` is the canonical way to set a field on
        # a frozen dataclass. The pattern is fragile if more derived
        # fields are added (review C-49, A-8) -- in that case, switch
        # to ``default_factory`` or a custom ``__init__``.
        object.__setattr__(self, "_transmittance", transmittance)

        # F-30: Use ContextVar accessor instead of ClassVar.
        if get_validate_on_construction():
            self.validate()

    # --- sub-methods --------------------------------------------------

    def _check_type(self) -> None:
        """Type-check ``attenuation`` at the boundary.

        Without this guard, passing a non-:class:`AttenuationConfig`
        object produces :class:`AttributeError` deep inside validation
        rather than a clear :class:`TypeError` at construction.
        """
        if not isinstance(self.attenuation, AttenuationConfig):
            raise TypeError(
                f"attenuation must be an AttenuationConfig, got "
                f"{type(self.attenuation).__name__}"
            )

    def _validate_parameter_ranges(self) -> None:
        """Range-validate ``distance_km`` and ``fiber_loss_db_km``."""
        self.validate_parameter(
            "distance_km", self.attenuation.fiber_length, MAX_DISTANCE_KM
        )
        self.validate_parameter(
            "fiber_loss_db_km",
            self.attenuation.attenuation_coefficient,
            MAX_FIBER_LOSS_DB_KM,
        )
        # F-L5 / N-4 fix: Check for negative zero in defining scalars.
        # IEEE 754: -0.0 == 0.0 is True, but -0.0 is semantically
        # suspicious and can propagate through computations producing
        # unexpected negative-zero results (e.g. -0.0 * 0.2 = -0.0).
        for name, value in [
            ("distance_km", self.attenuation.fiber_length),
            ("fiber_loss_db_km", self.attenuation.attenuation_coefficient),
        ]:
            if value == 0.0 and math.copysign(1.0, value) < 0:
                msg = (
                    f"Parameter {name} is negative zero (-0.0). "
                    f"While IEEE 754 treats -0.0 == 0.0, negative zero "
                    f"may mask an upstream sign error and propagates "
                    f"unexpectedly through multiplication."
                )
                if get_strict_mode():
                    raise ParameterValidationError(
                        msg,
                        param_name=name,
                        param_value=value,
                    )
                _warn_physical_suspicion(msg, stacklevel=4)

    # ------------------------------------------------------------------
    # F-04 fix: Minimum-alpha physical suspicion check
    # ------------------------------------------------------------------
    def _check_meaningful_alpha(self) -> None:
        """Warn (non-strict) or reject (strict) on physically suspicious alpha.

        F-04 fix (review channel_review_8): Direct construction silently
        accepts ``alpha=0`` with ``distance>0``, creating a lossless fiber
        over non-zero distance (physically impossible). The
        :meth:`from_total_loss` path already warns about this (F-20), but
        direct construction did not. This method ensures consistency
        across construction paths.

        Checks:
        (a) alpha=0 with distance>0 --- lossless fiber over non-zero
            distance is physically impossible.
        (b) alpha < 0.05 dB/km without wavelength justification ---
            suspiciously low for any real fiber.
        (c) alpha > 5 dB/km without wavelength justification ---
            suspiciously high for standard fiber (but valid for
            specialty fiber / POF).
        """
        alpha = self.attenuation.attenuation_coefficient
        distance = self.attenuation.fiber_length

        # (a) alpha=0 with distance>0: lossless fiber over non-zero
        # distance is physically impossible. This is flagged in
        # from_total_loss (F-20) but was missing in direct construction.
        if alpha == 0.0 and distance > 0.0:
            msg = (
                f"fiber_loss_db_km=0 with distance_km={distance:.6g} > 0 "
                f"creates a lossless fiber over non-zero distance, which is "
                f"physically impossible. No real fiber has zero attenuation "
                f"over any finite length. "
                f"[construction_path={self._construction_path}]"
            )
            if get_strict_mode():
                raise ParameterValidationError(
                    msg,
                    param_name="fiber_loss_db_km",
                    param_value=alpha,
                    context={
                        "distance_km": distance,
                        "construction_path": self._construction_path,
                    },
                )
            _warn_physical_suspicion(msg, stacklevel=4)

        # (b) alpha < 0.05 dB/km without wavelength justification:
        # suspiciously low for any real fiber. The lowest realistic
        # telecom fiber alpha is ~0.15 dB/km at 1550 nm. Values below
        # 0.05 are likely configuration errors.
        elif 0.0 < alpha < 0.05 and self.wavelength_nm is None:
            msg = (
                f"fiber_loss_db_km={alpha:.6g} is below 0.05 dB/km "
                f"without wavelength_nm set. The lowest realistic telecom "
                f"fiber attenuation is ~0.15 dB/km at 1550 nm. Values "
                f"below 0.05 dB/km are likely configuration errors or "
                f"specialty ultra-low-loss fiber that should be documented "
                f"with wavelength_nm. "
                f"[construction_path={self._construction_path}]"
            )
            if get_strict_mode():
                raise ParameterValidationError(
                    msg,
                    param_name="fiber_loss_db_km",
                    param_value=alpha,
                    context={
                        "distance_km": distance,
                        "construction_path": self._construction_path,
                    },
                )
            _warn_physical_suspicion(msg, stacklevel=4)

        # (c) alpha > 5 dB/km without wavelength justification:
        # suspiciously high for standard fiber. Valid for POF at 650 nm
        # (~3 dB/km) or specialty high-attenuation fiber, but should
        # be documented with wavelength_nm.
        elif alpha > 5.0 and self.wavelength_nm is None:
            msg = (
                f"fiber_loss_db_km={alpha:.6g} is above 5 dB/km "
                f"without wavelength_nm set. Standard telecom fiber has "
                f"alpha < 0.25 dB/km at 1550 nm. Values above 5 dB/km "
                f"are typical of plastic optical fiber (POF) at 650 nm "
                f"or specialty high-attenuation fiber, and should be "
                f"documented with wavelength_nm. "
                f"[construction_path={self._construction_path}]"
            )
            if get_strict_mode():
                raise ParameterValidationError(
                    msg,
                    param_name="fiber_loss_db_km",
                    param_value=alpha,
                    context={
                        "distance_km": distance,
                        "construction_path": self._construction_path,
                    },
                )
            _warn_physical_suspicion(msg, stacklevel=4)

    def _check_meaningful_distance(self) -> None:
        """Warn (non-strict) or reject (strict) on physically tiny distances.

        F-38 fix: Removed the ``alpha > 0.0`` guard. Sub-mm distances
        are now warned regardless of alpha. If alpha=0, the message
        notes that it's a "physically tiny but lossless" configuration
        (less suspicious). If alpha>0, it's more suspicious.

        C-6: Uses ``self._construction_path`` which is now set BEFORE
        ``__post_init__`` runs (it is an ``init=True`` field), so the
        correct factory path is included in the warning message.

        C-15: Emits via :func:`warnings.warn` with
        :class:`PhysicalSuspicionWarning` so the warning appears by
        default.
        """
        if 0.0 < self.attenuation.fiber_length < WARNING_DISTANCE_KM:
            # F-38: warn for sub-mm distance regardless of alpha.
            if self.attenuation.attenuation_coefficient > 0.0:
                msg = (
                    f"distance_km={self.attenuation.fiber_length!r} is below the "
                    f"advisory minimum of {WARNING_DISTANCE_KM} km (1 mm) "
                    f"with non-zero alpha={self.attenuation.attenuation_coefficient:.6g} "
                    f"[construction_path={self._construction_path}]. The model "
                    f"is mathematically scale-invariant but the configuration "
                    f"is physically suspicious."
                )
            else:
                # F-38: alpha=0 case is less suspicious but still worth noting.
                msg = (
                    f"distance_km={self.attenuation.fiber_length!r} is below the "
                    f"advisory minimum of {WARNING_DISTANCE_KM} km (1 mm) "
                    f"with alpha=0 (physically tiny but lossless configuration) "
                    f"[construction_path={self._construction_path}]. The model "
                    f"is mathematically scale-invariant but the configuration "
                    f"is unusual."
                )
            if get_strict_mode():
                raise ParameterValidationError(
                    msg,
                    param_name="distance_km",
                    param_value=self.attenuation.fiber_length,
                )
            _warn_physical_suspicion(msg, stacklevel=4)

    def _check_total_loss_finite(self) -> None:
        """Verify that the effective ``total_loss_db`` is finite.

        F-27 fix: Uses ``self._effective_total_loss_db()`` (from F-31)
        instead of ``self.attenuation.total_loss_db``, so the override
        is respected.

        If the effective value is non-finite, the error message points
        the finger at :class:`AttenuationConfig` (or the override)
        rather than at the upstream user inputs.
        """
        # F-27: use _effective_total_loss_db() to respect override.
        total_db = self._effective_total_loss_db()
        if not math.isfinite(total_db):
            raise ParameterValidationError(
                f"Effective total_loss_db is not finite "
                f"({total_db!r}); this is an AttenuationConfig or "
                f"override bug, not a channel-level validation error. "
                f"Inputs were "
                f"distance_km={self.attenuation.fiber_length!r}, "
                f"fiber_loss_db_km={self.attenuation.attenuation_coefficient!r}, "
                f"_total_loss_db_override={self._total_loss_db_override!r}.",
                param_name="total_loss_db",
                param_value=total_db,
                context={
                    "distance_km": self.attenuation.fiber_length,
                    "fiber_loss_db_km": self.attenuation.attenuation_coefficient,
                    "source": "AttenuationConfig_or_override",
                },
            )

    def _check_attenuation_config_contract(self) -> None:
        """Verify ``total_loss_db == fiber_length * attenuation_coefficient``.

        C-5 fix (review channel_review_8): Previously, this check was
        **skipped entirely** when ``_total_loss_db_override`` was set.
        A wildly inconsistent override (e.g. L=100, alpha=0.2,
        override=1e6) passed every check silently, producing a channel
        whose ``total_loss_db`` had no relation to ``alpha * L``.
        Now, when the override is set, we still check that the override
        is not wildly inconsistent with ``alpha * L``: if the relative
        discrepancy exceeds a generous tolerance, we warn (non-strict)
        or raise (strict). The tolerance is deliberately loose (1e-6)
        because the override is expected to differ from ``alpha * L``
        by a few ULP, but a factor-of-1000 discrepancy should not
        pass silently.

        When the override is NOT set, the original contract check is
        performed with the tighter tolerance
        ``rel_tol=1e-12, abs_tol=1e-12``.

        C-15: Emits via :func:`warnings.warn` with
        :class:`PhysicalSuspicionWarning` so the warning appears by
        default.
        """
        if self._total_loss_db_override is not None:
            # C-5 fix: Even when override is set, check that the override
            # is not wildly inconsistent with alpha * L. A few ULP
            # discrepancy is expected, but a factor-of-1000 discrepancy
            # should not pass silently.
            override = self._total_loss_db_override
            expected = (
                self.attenuation.fiber_length
                * self.attenuation.attenuation_coefficient
            )
            # Skip if expected is 0 (L=0, alpha=0 case) -- override of 0
            # is trivially consistent.
            if expected == 0.0 and override == 0.0:
                return
            # Use a generous relative tolerance for the override check.
            # The override is expected to differ from alpha*L by a few
            # ULP, but a discrepancy of >1e-6 relative should not pass
            # silently. For expected=0 but override!=0, use abs_tol.
            if not is_close(override, expected, rel_tol=1e-6,
                           abs_tol=_CONTRACT_CHECK_ABS_TOL):
                msg = (
                    f"Total-loss override ({override:.6g} dB) is wildly "
                    f"inconsistent with alpha*L ({expected:.6g} dB). "
                    f"Relative discrepancy: "
                    f"{abs(override - expected) / max(1.0, abs(expected)):e}. "
                    f"The stored total_loss_db will have no relation to "
                    f"alpha * L, which may mislead downstream code that "
                    f"trusts attenuation.total_loss_db for literature "
                    f"comparison."
                )
                if get_strict_mode():
                    raise ParameterValidationError(
                        msg,
                        param_name="total_loss_db",
                        param_value=override,
                        context={
                            "expected_total_loss_db": expected,
                            "actual_total_loss_db_override": override,
                            "relative_discrepancy": abs(override - expected) / max(1.0, abs(expected)),
                        },
                    )
                _warn_physical_suspicion(msg, stacklevel=4)
            return

        total_db = self.attenuation.total_loss_db
        expected_total_db = (
            self.attenuation.fiber_length * self.attenuation.attenuation_coefficient
        )
        # C-17 / F-9: tighter tolerance using named constants.
        if is_close(
            total_db,
            expected_total_db,
            rel_tol=_CONTRACT_CHECK_REL_TOL,
            abs_tol=_CONTRACT_CHECK_ABS_TOL,
        ):
            return

        short_msg = (
            f"AttenuationConfig contract violation: total_loss_db={total_db!r} "
            f"!= fiber_length * attenuation_coefficient={expected_total_db!r}. "
            f"FiberChannel assumes L_total = alpha * L."
        )
        if get_strict_mode():
            raise ParameterValidationError(
                short_msg,
                param_name="total_loss_db",
                param_value=total_db,
                context={
                    "expected_total_loss_db": expected_total_db,
                    "actual_total_loss_db": total_db,
                },
            )
        _warn_physical_suspicion(short_msg, stacklevel=4)
        logger.debug(
            "Full contract-violation context: if AttenuationConfig "
            "intentionally uses a different model (e.g. adds coupling "
            "loss), the transmittance computed here will silently "
            "include that extra loss. Enable strict mode to reject this."
        )

    # ------------------------------------------------------------------
    # F-15: Wavelength consistency check
    # ------------------------------------------------------------------
    def _check_wavelength_consistency(self) -> None:
        """Check that ``alpha`` is consistent with ``wavelength_nm``.

        F-15: When ``wavelength_nm`` is set, look up the typical alpha
        range for the nearest known wavelength. If ``alpha`` is outside
        that range, emit :class:`PhysicalSuspicionWarning` (non-strict)
        or raise :class:`ParameterValidationError` (strict).

        If ``wavelength_nm`` is ``None``, the check is skipped.
        """
        if self.wavelength_nm is None:
            return

        wl = self.wavelength_nm
        alpha = self.attenuation.attenuation_coefficient

        # Find the closest known wavelength.
        closest_wl: Optional[float] = None
        min_diff: float = float("inf")
        for known_wl in TYPICAL_WAVELENGTH_ALPHA_RANGES:
            diff = abs(wl - known_wl)
            if diff < min_diff:
                min_diff = diff
                closest_wl = known_wl

        if closest_wl is None or min_diff > WAVELENGTH_CHECK_TOLERANCE_NM:
            # Wavelength not close to any known range; skip check.
            return

        alpha_min, alpha_max = TYPICAL_WAVELENGTH_ALPHA_RANGES[closest_wl]
        if not (alpha_min <= alpha <= alpha_max):
            msg = (
                f"wavelength_nm={wl:.1f} (closest known: {closest_wl:.0f} nm) "
                f"typically has alpha in [{alpha_min:.2f}, {alpha_max:.2f}] dB/km, "
                f"but this channel has alpha={alpha:.6g} dB/km. The configuration "
                f"is physically suspicious."
            )
            if get_strict_mode():
                raise ParameterValidationError(
                    msg,
                    param_name="fiber_loss_db_km",
                    param_value=alpha,
                    context={
                        "wavelength_nm": wl,
                        "closest_known_wavelength": closest_wl,
                        "typical_alpha_range": (alpha_min, alpha_max),
                    },
                )
            _warn_physical_suspicion(msg, stacklevel=4)

    # ------------------------------------------------------------------
    # F-H1 fix: Wavelength range validation
    # ------------------------------------------------------------------
    def _validate_wavelength_range(self) -> None:
        """Validate that ``wavelength_nm`` is in a physically plausible range.

        F-H1 / F-M6 fix: Previously, ``wavelength_nm`` accepted any
        finite float, including negative values, zero, and extremely
        large values (e.g. 1e9 nm = 1 m). The consistency check only
        fires for wavelengths matching a known QKD band, so unusual
        values like -500 nm or 0 nm were silently accepted without
        any warning.

        This method validates that ``wavelength_nm`` is:
        1. Finite and positive (negative, zero, or non-finite values
           are rejected unconditionally).
        2. Within the typical QKD fiber range [380, 2000] nm
           (UV-visible through near-IR). Values outside this range
           emit a :class:`PhysicalSuspicionWarning` (non-strict) or
           raise :class:`ParameterValidationError` (strict).

        If ``wavelength_nm`` is ``None``, the check is skipped.
        """
        if self.wavelength_nm is None:
            return
        wl = self.wavelength_nm
        if not math.isfinite(wl) or wl <= 0:
            raise ParameterValidationError(
                f"wavelength_nm must be a finite positive number (got {wl!r}).",
                param_name="wavelength_nm",
                param_value=wl,
            )
        if wl < _MIN_QKD_WAVELENGTH_NM or wl > _MAX_QKD_WAVELENGTH_NM:
            msg = (
                f"wavelength_nm={wl:.1f} is outside the typical QKD fiber "
                f"range [{_MIN_QKD_WAVELENGTH_NM}, {_MAX_QKD_WAVELENGTH_NM}] nm."
            )
            if get_strict_mode():
                raise ParameterValidationError(
                    msg,
                    param_name="wavelength_nm",
                    param_value=wl,
                )
            _warn_physical_suspicion(msg, stacklevel=4)

    def _compute_transmittance(self) -> float:
        """Compute ``T = 10 ** (-L_total / 10)`` (Beer--Lambert).

        Uses :func:`math.pow` instead of the ``**`` operator for
        robustness against platform-specific ``OverflowError`` on
        extreme exponents (review N-7). Uses the override
        ``total_loss_db`` when set (review C-2).

        F-31: Uses ``self._effective_total_loss_db()`` instead of
        inline override lookup, and ``self._transmittance_from_loss_db``
        instead of inline ``math.pow``.

        The zero-loss branch uses exact equality ``total_db == 0.0``
        (review N-3): the previous ``is_close(total_db, 0.0)`` with
        ``abs_tol=1e-12`` treated ``L_total < 1e-12`` dB as zero,
        silently zeroing out a non-zero loss for ``alpha=0.2`` and
        ``L = 5e-12`` km. Exact equality lets ``math.pow`` produce a
        value arbitrarily close to 1.0 for tiny non-zero ``L_total``,
        which is the correct physical behaviour.
        """
        # F-31: use helper method instead of inline override lookup.
        total_db = self._effective_total_loss_db()
        # F-31: use helper instead of inline computation.
        return self._transmittance_from_loss_db(total_db)

    # ------------------------------------------------------------------
    # F-31: DRY helpers -- _effective_total_loss_db and
    # _transmittance_from_loss_db
    # ------------------------------------------------------------------
    def _effective_total_loss_db(self) -> float:
        """Return the effective total loss in dB, respecting the override.

        F-31: Replaces the 5 inline ``_total_loss_db_override if ...
        else attenuation.total_loss_db`` pattern sites with a single
        helper method. Sites replaced:
        - ``_compute_transmittance``
        - ``_check_transmittance_health`` (NaN branch, underflow/range branches)
        - ``_check_total_loss_finite`` (F-27)
        - ``_fast_construct``
        """
        if self._total_loss_db_override is not None:
            return self._total_loss_db_override
        return self.attenuation.total_loss_db

    @staticmethod
    def _transmittance_from_loss_db(total_db: float) -> float:
        """Compute transmittance from total loss in dB.

        F-31: Replaces the repeated ``math.pow(10.0, -total_db / 10.0)``
        pattern (or ``1.0`` if ``total_db == 0.0``) in
        ``_compute_transmittance``, ``validate``, ``_fast_construct``,
        and ``from_total_loss`` (two-phase construction re-computation).

        Uses ``math.pow`` for robustness against platform-specific
        ``OverflowError`` (review N-7).

        Uses exact equality ``total_db == 0.0`` for the zero-loss
        branch (review N-3).
        """
        if total_db == 0.0:
            # F-31 / P-06: check for negative zero. ``-0.0 == 0.0``
            # is True in IEEE 754, so the zero-loss branch activates.
            # ``math.pow(10.0, -0.0/10.0)`` returns 1.0, which is
            # physically correct (no loss). However, negative zero may
            # mask an upstream sign error, so emit a warning.
            if math.copysign(1.0, total_db) < 0:
                _warn_physical_suspicion(
                    f"_transmittance_from_loss_db: total_db is negative "
                    f"zero (-0.0). math.pow(10.0, -0.0/10.0) returns 1.0 "
                    f"(physically correct for zero loss), but negative "
                    f"zero may mask an upstream sign error in the "
                    f"computation that produced total_db.",
                    stacklevel=2,
                )
            return 1.0
        return math.pow(10.0, -total_db / 10.0)

    def _check_transmittance_health(self, transmittance: float) -> None:
        """Detect NaN, underflow, subnormal, and out-of-range transmittance.

        Off-by-one fix on the subnormal threshold (review N-4):
        ``<=`` would include :data:`sys.float_info.min` itself, which is
        normal; ``<`` is the correct comparison for subnormals only.

        F-6 fix: Uses ``self.total_loss_db`` (the property, which
        respects the override) in ALL error messages, including the NaN
        branch, underflow branch, and range validation branch. Previously
        used ``self.attenuation.total_loss_db`` (rounded alpha*L) even
        when ``_total_loss_db_override`` was set.

        F-31: Uses ``self._effective_total_loss_db()`` for the local
        ``total_db`` variable instead of inline override lookup.

        C-15: Emits underflow warnings via :func:`warnings.warn` with
        :class:`PhysicalSuspicionWarning` so they appear by default.

        F-subnormal: Subnormal transmittance is now promoted from
        DEBUG log to ``_warn_physical_suspicion`` (PhysicalSuspicionWarning),
        consistent with the underflow case.
        """
        # NaN: reject unconditionally.
        if math.isnan(transmittance):
            # F-6: use self.total_loss_db (property) in error message.
            raise ParameterValidationError(
                f"Transmittance computed as NaN for distance_km="
                f"{self.attenuation.fiber_length!r}, fiber_loss_db_km="
                f"{self.attenuation.attenuation_coefficient!r}, total_loss_db="
                f"{self.total_loss_db!r}. Inputs likely contain NaN.",
                param_name="transmittance",
                param_value=transmittance,
                context={
                    "distance_km": self.attenuation.fiber_length,
                    "fiber_loss_db_km": self.attenuation.attenuation_coefficient,
                    "total_loss_db": self.total_loss_db,
                },
            )

        # P-06: Detect negative-zero transmittance. ``-0.0 == 0.0``
        # is True in IEEE 754, so the underflow branch below treats
        # ``-0.0`` the same as ``+0.0``. However, ``-0.0`` as
        # transmittance is physically suspicious: it can only arise
        # from a negative-zero total_loss_db (via math.pow(10.0,
        # 0.0/10.0) = 1.0, not -0.0) or from upstream computation
        # errors. Explicitly flag it.
        if transmittance == 0.0 and math.copysign(1.0, transmittance) < 0:
            neg_zero_msg = (
                f"Transmittance is negative zero (-0.0) for "
                f"distance_km={self.attenuation.fiber_length:.6g}, "
                f"fiber_loss_db_km={self.attenuation.attenuation_coefficient:.6g}. "
                f"Negative-zero transmittance is physically equivalent to "
                f"+0.0 but may mask an upstream sign error."
            )
            if get_strict_mode():
                raise ParameterValidationError(
                    neg_zero_msg,
                    param_name="transmittance",
                    param_value=transmittance,
                    context={
                        "distance_km": self.attenuation.fiber_length,
                        "fiber_loss_db_km": self.attenuation.attenuation_coefficient,
                        "total_loss_db": self.total_loss_db,
                    },
                )
            _warn_physical_suspicion(neg_zero_msg, stacklevel=4)

        # F-31: use _effective_total_loss_db() for local total_db.
        total_db = self._effective_total_loss_db()

        # Underflow: exact-zero transmittance with non-zero total_db.
        # NOTE: ``total_db > 0.0`` is ``False`` for ``-0.0`` (negative
        # zero), which is correct -- there is nothing to underflow for
        # ``-0.0`` (review N-? / C-56).
        if transmittance == 0.0 and total_db > 0.0:
            # F-6: use self.total_loss_db in the message.
            underflow_msg = (
                f"Transmittance underflowed to exactly 0.0 for "
                f"distance_km={self.attenuation.fiber_length:.6g}, "
                f"fiber_loss_db_km={self.attenuation.attenuation_coefficient:.6g} "
                f"(total_loss_db={self.total_loss_db:.6g}). QKD results in this regime "
                f"are numerically invalid (gain, yield, QBER, and key-rate "
                f"formulas will silently produce zero or NaN)."
            )
            if get_strict_mode():
                raise ParameterValidationError(
                    underflow_msg,
                    param_name="transmittance",
                    param_value=transmittance,
                    context={
                        "distance_km": self.attenuation.fiber_length,
                        "fiber_loss_db_km": self.attenuation.attenuation_coefficient,
                        "total_loss_db": self.total_loss_db,
                    },
                )
            _warn_physical_suspicion(underflow_msg, stacklevel=4)
        # F-subnormal: Subnormal transmittance promoted from DEBUG log
        # to _warn_physical_suspicion, consistent with underflow.
        elif 0.0 < transmittance < TRANSMITTANCE_SUBNORMAL_THRESHOLD:
            subnormal_msg = (
                f"Transmittance is subnormal (below smallest normal float) "
                f"for distance_km={self.attenuation.fiber_length:.6g}, "
                f"fiber_loss_db_km={self.attenuation.attenuation_coefficient:.6g} "
                f"(T={transmittance:.3e}). Numerical precision may be degraded."
            )
            if get_strict_mode():
                raise ParameterValidationError(
                    f"Transmittance {transmittance!r} is subnormal (below "
                    f"{TRANSMITTANCE_SUBNORMAL_THRESHOLD}). Strict mode rejects "
                    f"this configuration; numerical precision is degraded.",
                    param_name="transmittance",
                    param_value=transmittance,
                )
            _warn_physical_suspicion(subnormal_msg, stacklevel=4)

        # Range validation. We explicitly do NOT clamp silently: if a
        # non-physical value slipped past :class:`AttenuationConfig`
        # validation, raise rather than normalise. The only legitimate
        # out-of-range value is the underflow-to-zero case handled
        # above, which is allowed (with warning) in non-strict mode.
        if not (0.0 <= transmittance <= 1.0):
            # F-6: use self.total_loss_db in error message.
            raise ParameterValidationError(
                f"Transmittance {transmittance!r} is out of the physical range "
                f"[0.0, 1.0]. This indicates an upstream bug (e.g. negative "
                f"fiber_length or attenuation_coefficient accepted by "
                f"AttenuationConfig). total_loss_db={self.total_loss_db!r}.",
                param_name="transmittance",
                param_value=transmittance,
                context={
                    "distance_km": self.attenuation.fiber_length,
                    "fiber_loss_db_km": self.attenuation.attenuation_coefficient,
                    "total_loss_db": self.total_loss_db,
                },
            )

    # ------------------------------------------------------------------
    # Parameter validation helper
    # ------------------------------------------------------------------
    def validate_parameter(self, name: str, value: float, upper_bound: float) -> None:
        """Validate a single numeric parameter.

        Parameters
        ----------
        name:
            Parameter name; must be a key in :data:`VALIDATED_UNITS`
            (P-12) for the units suffix to appear in the error message.
            Names in :data:`METADATA_UNITS` are also accepted for
            backward compatibility but are not range-validated here.
        value:
            Parameter value to validate.
        upper_bound:
            Inclusive upper bound. F-L7 fix (review channel_review_6):
            must be positive; a negative ``upper_bound`` would silently
            disable validation since all non-negative values would pass
            the ``value > upper_bound`` check.

        Raises
        ------
        ParameterValidationError
            If ``value`` is not finite and non-negative, or if it
            exceeds ``upper_bound``. Booleans are explicitly rejected
            by :func:`is_finite_non_negative` (and by
            :func:`_coerce_numeric` on factory paths, review C-5).
        """
        # F-L7 fix: assert upper_bound is positive. A negative
        # upper_bound silently disables the range check (all
        # non-negative values pass), which would mask validation bugs.
        if upper_bound <= 0:
            raise ParameterValidationError(
                f"validate_parameter: upper_bound for {name!r} must be "
                f"positive (got {upper_bound!r}). A negative or zero "
                f"upper_bound silently disables range validation.",
                param_name=name,
                param_value=upper_bound,
            )
        # P-12: Prefer VALIDATED_UNITS for the units suffix; fall back
        # to UNITS for backward compatibility with any parameter name.
        units = VALIDATED_UNITS.get(name, UNITS.get(name, ""))

        if not is_finite_non_negative(value):
            raise ParameterValidationError(
                f"Parameter {name!r} must be a finite and non-negative number "
                f"(got {value!r}). Booleans are explicitly rejected.",
                param_name=name,
                param_value=value,
            )

        if value > upper_bound:
            raise ParameterValidationError(
                f"Parameter {name!r} ({value!r}) exceeds the physical limit of "
                f"{upper_bound} {units}.",
                param_name=name,
                param_value=value,
                context={"limit": upper_bound, "units": units},
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def distance_km(self) -> Distance:
        """Fiber distance in km."""
        return self.attenuation.fiber_length

    @property
    def fiber_loss_db_km(self) -> Loss:
        """Fiber attenuation coefficient ``alpha`` in dB/km."""
        return self.attenuation.attenuation_coefficient

    @property
    def transmittance(self) -> Transmittance:
        """Total channel transmittance ``T`` in ``[0, 1]``.

        Computed once at construction time as
        ``T = math.pow(10.0, -alpha * L / 10)`` (Beer--Lambert) and
        cached in ``_transmittance``. The cached value is NOT
        recomputed on reload -- if the floating-point environment
        changes (e.g. ``fenv`` rounding mode), the stored value may
        differ from a freshly recomputed one. Use :meth:`validate` to
        detect such drift.

        C-2 fix: For channels built via :meth:`from_total_loss`, the
        transmittance is now computed from the original
        ``total_loss_db`` override (not the rounded ``alpha * L``),
        eliminating the ULP-level inconsistency with
        :attr:`total_loss_db`.

        F-11: Since ``_total_loss_db_override`` is ``init=False``, the
        override is set after initial construction in
        :meth:`from_total_loss`, and ``_transmittance`` is re-computed
        from the override.

        This is ``eta_channel`` ONLY; it does NOT include detector or
        coupling efficiencies. See the module docstring for the
        composition requirement.
        """
        return self._transmittance

    @property
    def total_loss_db(self) -> Loss:
        """Total channel loss in decibels (dB).

        Returns the :meth:`from_total_loss` override when set
        (review C-3), else ``attenuation.total_loss_db``.

        F-31: Uses ``self._effective_total_loss_db()`` internally.
        """
        return self._effective_total_loss_db()

    @property
    def is_lossless(self) -> bool:
        """``True`` iff the channel has effectively zero loss.

        P-03: Uses :func:`is_close` on :attr:`total_loss_db` against
        ``0.0`` with an explicit ``abs_tol=_CONTRACT_CHECK_ABS_TOL``
        (``1e-12``). Previously, this relied on the default ``abs_tol``
        of :func:`qkd.constants.is_close`, which may be ``0.0``.
        With ``abs_tol=0.0``, ``is_close(2e-16, 0.0)`` returns
        ``False``, so the docstring example
        ``fiber_length=1e-15, alpha=0.2, total_loss_db=2e-16`` would
        NOT be classified as lossless. The explicit ``abs_tol=1e-12``
        ensures that any ``total_loss_db`` below ``1e-12`` dB is
        considered effectively zero, which is consistent with the
        physical interpretation (``1e-12`` dB is well below any
        measurable loss).

        C-27 note: :attr:`is_lossless` and :attr:`is_exactly_lossless`
        can disagree for tiny non-zero ``total_db``.
        :attr:`is_lossless` uses :func:`is_close` (tolerance-based,
        ``abs_tol=1e-12``), while :attr:`is_exactly_lossless` uses
        exact equality on the defining scalars. For
        ``fiber_length=1e-15, alpha=0.2, total_loss_db=2e-16``:
        ``is_lossless=True`` (since ``2e-16 < 1e-12``) but
        ``is_exactly_lossless=False``. This is intentional; use the
        property that matches your semantics.

        For strict (exact-zero) semantics, use :attr:`is_exactly_lossless`.
        """
        return is_close(self.total_loss_db, 0.0, abs_tol=_CONTRACT_CHECK_ABS_TOL)

    @property
    def is_exactly_lossless(self) -> bool:
        """``True`` iff at least one defining scalar field is exactly zero.

        P-10: Fixed docstring -- previously stated ``True iff the
        defining scalar fields are exactly zero``, which would imply
        BOTH must be zero. The actual code uses ``or``, so only ONE
        needs to be zero. This is physically correct: if either
        ``distance_km == 0`` or ``alpha == 0``, total loss is zero.

        Strict counterpart to :attr:`is_lossless`. Useful for tests
        that need to distinguish ``total_loss_db == 0`` from
        ``total_loss_db`` merely rounding to zero under tolerance.

        C-27 note: See :attr:`is_lossless` for the documented
        disagreement between the two properties for tiny non-zero
        ``total_db``.
        """
        return (
            self.attenuation.fiber_length == 0.0
            or self.attenuation.attenuation_coefficient == 0.0
        )

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    def single_km_power_loss_fraction(self) -> float:
        """Fraction of optical power lost in exactly **one kilometre** of fibre.

        F-08 fix (review channel_review_8): Renamed from
        ``loss_fraction_per_km`` to ``single_km_power_loss_fraction``
        to reduce the cognitive trap of multiplying by distance.
        The name ``loss_fraction_per_km`` naturally invites
        ``fraction * distance`` to estimate total loss, but this is
        WRONG because the quantity does NOT compose linearly::

            ell_total(L) = 1 - (1 - ell) ** L   !=   ell * L

        The new name ``single_km_power_loss_fraction`` makes it clear
        that this is the loss fraction for a SINGLE kilometre, not a
        rate that composes per km. The old name is preserved as a
        deprecated alias.

        For small ``alpha``, the naive formulation
        ``1.0 - 10.0 ** (-alpha / 10.0)`` suffers catastrophic
        cancellation: at ``alpha = 1e-6`` dB/km, ``10.0 ** (-1e-7)``
        is ``0.99999976...``, and subtracting from 1.0 leaves only
        ~6 significant digits. The stable formulation uses
        :func:`math.expm1`::

            ell = -math.expm1(-alpha * math.log(10) / 10.0)

        which preserves full precision for small ``alpha``.
        """
        alpha = self.fiber_loss_db_km
        if alpha == 0.0:
            return 0.0
        return -math.expm1(-alpha * math.log(10.0) / 10.0)

    def loss_fraction_per_km(self) -> float:
        """Deprecated alias for :meth:`single_km_power_loss_fraction`.

        .. deprecated:: 7.0.0
           Use :meth:`single_km_power_loss_fraction` instead. The name
           ``loss_fraction_per_km`` invites multiplication by distance,
           which produces WRONG total-loss estimates. The new name makes
           the non-linear composition nature explicit.

        F-08 fix (review channel_review_8): This method now delegates
        to :meth:`single_km_power_loss_fraction` and emits a
        DeprecationWarning on every call.
        """
        warnings.warn(
            "loss_fraction_per_km is deprecated; use "
            "single_km_power_loss_fraction instead. The name "
            "loss_fraction_per_km invites multiplication by distance, "
            "which produces WRONG total-loss estimates. "
            "single_km_power_loss_fraction makes the non-linear "
            "composition nature explicit.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.single_km_power_loss_fraction()

    def linear_loss_per_km(self) -> float:
        """Return the first-order linear approximation of per-km loss.

        .. deprecated:: 6.0.0
           Use :meth:`loss_fraction_per_km` for the exact (non-linear)
           per-km power-loss fraction. This method now returns the
           actual linear (first-order Taylor) approximation for
           backward semantic compatibility (review C-40).

        C-40 fix: Previously this method returned the same non-linear
        value as :meth:`loss_fraction_per_km`, which was semantically
        misleading -- users who saw the name ``linear_loss_per_km``
        would use the returned fraction as a linear per-km loss rate,
        producing grossly wrong total-loss estimates for long links.
        This method now returns the true first-order Taylor expansion
        of ``1 - 10^(-alpha/10)`` for small ``alpha``::

            ell_linear = alpha * ln(10) / 10

        which IS linear in ``alpha`` and composes linearly with
        distance (as a first-order approximation). For ``alpha = 0.2``
        dB/km, ``ell_linear ~ 0.0461``, which is close to the exact
        ``ell ~ 0.0451`` but composes correctly: ``ell_linear * L``
        gives the first-order approximation of the total dB loss
        fraction (valid for small ``alpha * L``).

        F-10: If the computed value exceeds 1.0 (which happens for
        ``alpha > 4.34 dB/km``), a :class:`PhysicalSuspicionWarning`
        is emitted noting that the first-order approximation is
        unphysical. The value is returned unclamped (clamping would
        be misleading for a deprecated approximation method).

        F-37: The deprecation warning is emitted only once per process
        using a module-level flag, rather than on every call.

        For the exact per-km fraction, use :meth:`loss_fraction_per_km`.
        """
        # F-37 / F-M3 fix: emit deprecation warning only once per process.
        # Use threading.Lock for thread safety; previously the bare
        # bool flag allowed duplicate warnings in multi-threaded use.
        global _LINEAR_LOSS_WARNED
        with _LINEAR_LOSS_LOCK:
            if not _LINEAR_LOSS_WARNED:
                warnings.warn(
                    "FiberChannel.linear_loss_per_km() is deprecated; use "
                    "loss_fraction_per_km() for the exact per-km power-loss "
                    "fraction. This method now returns the first-order Taylor "
                    "approximation (alpha * ln(10) / 10) which is truly linear "
                    "in alpha. See the docstring for details (review C-40).",
                    DeprecationWarning,
                    stacklevel=2,
                )
                _LINEAR_LOSS_WARNED = True

        alpha = self.fiber_loss_db_km
        # C-40: first-order Taylor expansion of 1 - 10^(-alpha/10).
        # This is truly linear in alpha and composes linearly with
        # distance (as a first-order approximation).
        val = alpha * math.log(10.0) / 10.0
        # C-6 fix (review channel_review_8): For alpha > 4.34 dB/km
        # (where 10/log(10) ≈ 4.34), the linear approximation exceeds
        # 1.0, which is unphysical for a "loss fraction". Previously
        # this only warned but returned the unphysical value, allowing
        # downstream code to compose nonsensical total-loss estimates.
        # Now, in strict mode, this raises ParameterValidationError.
        # In non-strict mode, the value is clamped to 1.0 and a
        # PhysicalSuspicionWarning is emitted, since a loss fraction
        # > 1.0 violates the physical meaning of "fraction".
        if val > 1.0:
            msg = (
                f"linear_loss_per_km() returned {val:.6g} > 1.0 for "
                f"alpha={alpha:.6g} dB/km. The first-order Taylor "
                f"approximation is unphysical at this attenuation; "
                f"use single_km_power_loss_fraction() for the exact "
                f"value."
            )
            if get_strict_mode():
                raise ParameterValidationError(
                    msg,
                    param_name="fiber_loss_db_km",
                    param_value=alpha,
                    context={
                        "linear_loss_value": val,
                        "alpha_threshold": 10.0 / math.log(10.0),
                    },
                )
            _warn_physical_suspicion(msg, stacklevel=2)
            val = 1.0  # clamp to physical range [0, 1]
        return val

    # ------------------------------------------------------------------
    # Physics / scope introspection (review S-6)
    # ------------------------------------------------------------------
    @classmethod
    def limitations(cls) -> List[str]:
        """Return a list of unsupported physics, for programmatic checks.

        Review S-6. Downstream key-rate code can iterate over this
        list to verify that the channel model is consistent with the
        assumptions of the chosen key-rate formula.
        """
        return [
            "wavelength_dependence",
            "coupling_splice_connector_loss",
            "chromatic_dispersion",
            "polarisation_mode_dispersion",
            "polarisation_dependent_loss",
            "time_dependence",
            "non_uniform_attenuation",
            "nonlinear_optical_effects",
            "free_space_satellite_links",
        ]

    # ------------------------------------------------------------------
    # Multi-node composition (review S-8)
    # ------------------------------------------------------------------
    def cascade(self, other: "FiberChannel") -> "FiberChannel":
        """Return a **loss-equivalent** channel for ``self`` followed by ``other``.

        F-07 fix (review channel_review_8): The cascaded channel has
        ``alpha = total_loss / total_distance``, which is an
        **arithmetic mean** that does NOT correspond to any physical
        fiber type. For example, cascading a 100 km segment at
        0.2 dB/km with a 10 km segment at 3.0 dB/km produces
        alpha_avg ~ 0.47 dB/km, which matches no standard fiber at
        any wavelength. The resulting channel is a **loss-equivalent**
        composite, not a single physical fiber. Users should NOT treat
        cascaded channels as real fiber links in literature comparisons.

        Total transmittance is the product of per-link transmittances;
        equivalently, total loss in dB is the sum. Per-link parameters
        are not preserved on the result; only the aggregate
        ``(L, alpha)`` is recorded (review S-8).

        C-33 fix: ``_metadata_extra`` from both inputs is merged into
        the result (with ``other`` taking precedence on key collision).

        F-5 fix: Uses ``cls(...)`` and ``cls.from_total_loss(...)``
        instead of ``FiberChannel(...)`` and
        ``FiberChannel.from_total_loss(...)``, so subclasses produce
        instances of the subclass type.

        F-15: ``wavelength_nm`` is NOT carried over in cascade (the
        resulting channel's wavelength is undefined since it may span
        heterogeneous links).

        Raises
        ------
        TypeError
            If ``other`` is not a :class:`FiberChannel`.
        """
        if not isinstance(other, FiberChannel):
            raise TypeError(
                f"cascade requires a FiberChannel argument, got "
                f"{type(other).__name__}"
            )
        total_loss_db = self.total_loss_db + other.total_loss_db
        total_distance = self.distance_km + other.distance_km
        if total_distance == 0.0:
            # Both links are zero-distance; the result is the identity
            # channel. We cannot recover a specific alpha, so use the
            # convention alpha=0 (F-8: consistent with from_total_loss
            # alpha=0 convention for (0,0)).
            # F-5: use cls(...) instead of FiberChannel(...).
            instance = cls(
                AttenuationConfig(
                    fiber_length=0.0,
                    attenuation_coefficient=0.0,
                ),
                _construction_path="cascade",
            )
        else:
            # F-5: use cls.from_total_loss(...) instead of
            # FiberChannel.from_total_loss(...).
            instance = cls.from_total_loss(
                total_distance,
                total_loss_db,
                _construction_path="cascade",
            )
        # C-33: Merge _metadata_extra from both inputs.
        merged_metadata: Dict[str, Any] = dict(self._metadata_extra)
        merged_metadata.update(other._metadata_extra)
        # P-07: wrap merged dict in MappingProxyType for immutability.
        if merged_metadata:
            object.__setattr__(
                instance, "_metadata_extra",
                types.MappingProxyType(merged_metadata)
            )
        # F-07 fix (review channel_review_8): Warn if the cascaded
        # average alpha is outside typical fiber ranges. The cascaded
        # alpha is an arithmetic mean and does NOT correspond to any
        # physical fiber. Users may mistakenly treat it as a real fiber.
        if total_distance > 0.0:
            avg_alpha = total_loss_db / total_distance
            # Check against typical fiber alpha ranges
            # Telecom fiber: 0.15-0.35 dB/km (1310-1550 nm)
            # POF: 0.1-3 dB/km (visible wavelengths)
            # Suspicious: outside 0.05-5 dB/km without wavelength
            if avg_alpha < 0.05 or avg_alpha > 5.0:
                msg = (
                    f"cascade: The cascaded average alpha="
                    f"{avg_alpha:.6g} dB/km is outside typical fiber "
                    f"ranges (0.05--5 dB/km). This is a loss-equivalent "
                    f"composite, not a physical fiber. The average "
                    f"alpha does not correspond to any standard fiber "
                    f"type at any wavelength. Do NOT use this channel "
                    f"in literature comparisons as a real fiber link."
                )
                if get_strict_mode():
                    raise ParameterValidationError(
                        msg,
                        param_name="fiber_loss_db_km",
                        param_value=avg_alpha,
                        context={
                            "total_loss_db": total_loss_db,
                            "total_distance": total_distance,
                            "is_composite_equivalent": True,
                        },
                    )
                _warn_physical_suspicion(msg, stacklevel=2)
        return instance

    @classmethod
    def _cascade_unchecked(
        cls,
        channels: Sequence["FiberChannel"],
    ) -> "FiberChannel":
        """Cascade multiple channels WITHOUT validation (review C-29).

        INTERNAL USE ONLY. Skips all validation and contract checks.
        The caller is responsible for ensuring all channels are valid.
        Use only in performance-critical Monte Carlo loops where
        inputs are trusted.

        Uses :meth:`_fast_construct` internally to bypass per-channel
        validation.

        F-29: Checks ``total_distance > MAX_DISTANCE_KM`` after
        summing distances. Raises :class:`ParameterValidationError`
        if exceeded.
        """
        if not channels:
            # F-L8 fix: Use ParameterValidationError instead of ValueError
            # for consistent exception contract on invalid inputs.
            raise ParameterValidationError(
                "_cascade_unchecked requires at least one channel",
                param_name="channels",
                param_value=channels,
            )
        # F-C1 fix: Type-check FIRST before accessing attributes.
        # Previously, math.fsum accessed ch.total_loss_db and ch.distance_km
        # before the isinstance guard, causing AttributeError (not TypeError)
        # for non-FiberChannel elements.
        for ch in channels:
            if not isinstance(ch, FiberChannel):
                raise TypeError(
                    f"Expected FiberChannel, got {type(ch).__name__}"
                )
        # F-35: Use math.fsum for numerically stable accumulation.
        # For thousands of channels, naive summation can accumulate
        # significant floating-point error, potentially causing
        # false-positive rejection at the MAX_DISTANCE_KM boundary.
        # Now safe to access attributes since type-check is done above.
        total_loss = math.fsum(ch.total_loss_db for ch in channels)
        total_distance = math.fsum(ch.distance_km for ch in channels)
        metadata_extra: Dict[str, Any] = {}
        for ch in channels:
            # C-33: Merge _metadata_extra from all inputs.
            metadata_extra.update(ch._metadata_extra)
        # F-29: check total_distance against MAX_DISTANCE_KM.
        if total_distance > MAX_DISTANCE_KM:
            raise ParameterValidationError(
                f"Cascade total distance {total_distance:.6g} km exceeds "
                f"MAX_DISTANCE_KM={MAX_DISTANCE_KM} km.",
                param_name="distance_km",
                param_value=total_distance,
                context={"total_loss_db": total_loss},
            )
        if total_distance == 0.0:
            instance = cls._fast_construct(
                AttenuationConfig(0.0, 0.0),
                _construction_path="cascade",
            )
        else:
            alpha = total_loss / total_distance
            # P-04: Check computed alpha against MAX_FIBER_LOSS_DB_KM.
            # Previously, _cascade_unchecked bypassed this validation,
            # allowing unrealistic attenuation coefficients in batch
            # contexts (review C-3).
            if alpha > MAX_FIBER_LOSS_DB_KM:
                raise ParameterValidationError(
                    f"_cascade_unchecked: computed alpha={alpha:.6g} dB/km "
                    f"exceeds MAX_FIBER_LOSS_DB_KM={MAX_FIBER_LOSS_DB_KM} "
                    f"dB/km. The cascaded channel's effective attenuation "
                    f"coefficient is physically unrealistic.",
                    param_name="fiber_loss_db_km",
                    param_value=alpha,
                    context={
                        "total_loss_db": total_loss,
                        "total_distance_km": total_distance,
                        "max_fiber_loss_db_km": MAX_FIBER_LOSS_DB_KM,
                    },
                )
            instance = cls._fast_construct(
                AttenuationConfig(total_distance, alpha),
                _construction_path="cascade",
                _total_loss_db_override=total_loss,
            )
        # P-07: wrap merged dict in MappingProxyType for immutability.
        if metadata_extra:
            object.__setattr__(
                instance, "_metadata_extra",
                types.MappingProxyType(metadata_extra)
            )
        return instance

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def as_tuple(self) -> Tuple[Distance, Loss]:
        """Return the channel's *defining* parameters as a tuple.

        Returns only the two scalar inputs ``(distance_km,
        fiber_loss_db_km)``. Derived state (``total_loss_db``,
        ``transmittance``) is NOT included; use :meth:`to_config_dict`
        for a self-describing representation.
        """
        return (self.distance_km, self.fiber_loss_db_km)

    @classmethod
    def from_tuple(
        cls: Type[T_FiberChannel], t: Tuple[Numeric, Numeric]
    ) -> T_FiberChannel:
        """Inverse of :meth:`as_tuple`.

        Parameters
        ----------
        t:
            A 2-element sequence ``(distance_km, fiber_loss_db_km)``.

        C-18 fix: Accepts any 2-element sequence (tuple, list, etc.),
        not just :class:`tuple`. Strings and bytes are rejected
        explicitly to avoid iterating character-by-character.

        C-19 fix: Strings are explicitly rejected to catch config bugs
        where a string is passed instead of a number (consistent with
        :meth:`from_config_dict`).

        Notes
        -----
        Asymmetric with :meth:`from_config_dict`: this method does NOT
        accept a third ``_metadata`` element. Use
        :meth:`from_config_dict` if you need to round-trip metadata.

        Inputs are coerced to :class:`float`; this silently downcasts
        :class:`decimal.Decimal` and :class:`fractions.Fraction`
        inputs. Pass pre-coerced floats if you need to preserve exact
        arithmetic upstream.
        """
        # C-18: accept any 2-element sequence (not just tuple).
        # C-19: reject strings/bytes explicitly.
        if isinstance(t, (str, bytes)):
            raise ParameterValidationError(
                f"from_tuple requires a 2-element sequence "
                f"(distance_km, fiber_loss_db_km); got a "
                f"{type(t).__name__}. Strings/bytes are rejected to "
                f"avoid iterating character-by-character.",
                param_name="t",
                param_value=t,
            )
        if not (hasattr(t, "__len__") and hasattr(t, "__getitem__")):
            raise ParameterValidationError(
                f"from_tuple requires a 2-element sequence "
                f"(distance_km, fiber_loss_db_km); got {t!r} of type "
                f"{type(t).__name__}.",
                param_name="t",
                param_value=t,
            )
        if len(t) != 2:
            raise ParameterValidationError(
                f"from_tuple requires a 2-element sequence "
                f"(distance_km, fiber_loss_db_km); got a sequence of "
                f"length {len(t)}: {t!r}.",
                param_name="t",
                param_value=t,
                context={"expected_length": 2, "actual_length": len(t)},
            )
        distance, loss = t[0], t[1]
        # C-19: reject strings explicitly.
        for _name, _val in (("distance_km", distance), ("fiber_loss_db_km", loss)):
            if isinstance(_val, str):
                raise ParameterValidationError(
                    f"Parameter {_name!r} must be a real number, not a str "
                    f"(got {_val!r}). Pass a numeric type to avoid masking "
                    f"config bugs.",
                    param_name=_name,
                    param_value=_val,
                )
        attenuation = AttenuationConfig(
            fiber_length=_coerce_numeric(distance, param_name="distance_km"),
            attenuation_coefficient=_coerce_numeric(loss, param_name="fiber_loss_db_km"),
        )
        # C-6: pass _construction_path before __post_init__ runs.
        return cls(
            attenuation=attenuation,
            _construction_path="from_tuple",
        )

    def to_config_dict(self) -> Dict[str, Any]:
        """Serialise the channel configuration to a self-describing dict.

        The dict contains:

        * ``distance_km`` -- fibre length in km.
        * ``fiber_loss_db_km`` -- attenuation coefficient in dB/km.
        * ``total_loss_db`` -- (only when ``_total_loss_db_override`` is
          set, review C-3) the original total loss in dB. This ensures
          :meth:`from_config_dict` can prefer the ``total_loss_db`` form
          and preserve round-trip precision for :meth:`from_total_loss`
          channels.
        * ``wavelength_nm`` -- (F-15) only when set.
        * ``_metadata`` -- reproducibility metadata:
          - module version,
          - ``class`` -- concrete class name (F-14),
          - units,
          - model identifier,
          - derived state (total_loss_db, transmittance),
          - ``strict_mode`` -- the **construction-time** strict-mode flag
            (review C-1, previously the serialisation-time value),
          - construction path,
          - provenance hint,
          - ``limits`` -- the validation limits (review C-22):
            ``max_distance_km`` and ``max_fiber_loss_db_km``.

        Notes
        -----
        * ``_metadata.transmittance`` is rounded to 15 significant
          figures to avoid spurious cross-platform diffs from ULP
          noise (review C-24, F-L6 fix). Previously 12 sig figs,
          which for transmittance near 1.0 (e.g. T = 1 - 2.3e-14)
          yielded 1.0, losing loss information. 15 sig figs preserves
          more precision while still avoiding ULP noise. The rounded
          value is NOT the actual cached value; use
          ``instance.transmittance`` for the exact value. The rounding
          uses ``float(f"{x:.15g}")`` because :func:`round` rounds to
          decimal places (not significant figures), which would not
          achieve the desired diff stability for very large or very
          small transmittance values.
        * The returned dict is a plain :class:`dict`; mutations to
          derived fields (``total_loss_db``, ``transmittance``) are
          silently ignored by :meth:`from_config_dict`.
        * Unknown ``_metadata`` sub-keys that were preserved through
          :meth:`from_config_dict` are merged back into the output
          under ``_metadata.extra``.
        """
        # C-24: Round transmittance to 15 sig figs for diff stability.
        # F-L6 fix (review channel_review_6): Previously used 12 sig figs,
        # which for transmittance very close to 1.0 (e.g. T = 1 - 2.3e-14)
        # yielded 1.0, losing the loss information. 15 sig figs preserves
        # more precision while still avoiding spurious cross-platform diffs
        # from ULP noise. Documented explicitly: the rounded value is not
        # the actual cached value.
        if isinstance(self._transmittance, (int, float)) and math.isfinite(
            float(self._transmittance)
        ):
            transmittance_meta: Any = float(f"{float(self._transmittance):.15g}")
        else:
            transmittance_meta = self._transmittance

        metadata: Dict[str, Any] = {
            "version": __version__,
            # F-14: record the concrete class name.
            "class": self.__class__.__name__,
            "model": "pure_attenuation",
            "units": dict(UNITS),
            "total_loss_db": self.total_loss_db,
            "transmittance": transmittance_meta,
            # C-1: construction-time strict mode, not serialisation-time.
            "strict_mode": self._strict_mode_at_construction,
            "construction_path": self._construction_path,
            "provenance": "FiberChannel.to_config_dict",
            # C-22: include validation limits for cross-version
            # reproducibility.
            "limits": {
                "max_distance_km": MAX_DISTANCE_KM,
                "max_fiber_loss_db_km": MAX_FIBER_LOSS_DB_KM,
            },
        }
        # P-07: _metadata_extra is now a MappingProxyType; dict()
        # creates a mutable copy for serialisation.
        if self._metadata_extra:
            metadata["extra"] = dict(self._metadata_extra)
        # F-15: include wavelength_nm in metadata if set.
        if self.wavelength_nm is not None:
            metadata["wavelength_nm"] = self.wavelength_nm
        result: Dict[str, Any] = {
            "distance_km": self.distance_km,
            "fiber_loss_db_km": self.fiber_loss_db_km,
            "_metadata": metadata,
        }
        # C-3: Emit total_loss_db as a top-level key when the override
        # is set, so from_config_dict can prefer it and preserve
        # round-trip precision for from_total_loss channels.
        if self._total_loss_db_override is not None:
            result["total_loss_db"] = self._total_loss_db_override
        # F-15: Emit wavelength_nm as a top-level key if set.
        if self.wavelength_nm is not None:
            result["wavelength_nm"] = self.wavelength_nm
        return result

    @classmethod
    def from_config(
        cls: Type[T_FiberChannel],
        attenuation: AttenuationConfig,
    ) -> T_FiberChannel:
        """Create a :class:`FiberChannel` directly from an :class:`AttenuationConfig`.

        This is a thin convenience factory retained for backwards
        compatibility (review C-7, C-39, A-7).

        F-40: A :class:`DeprecationWarning` is now emitted, advising
        users to use the constructor directly. Removal target: version
        7.0.0.

        C-7 fix: Sets ``_construction_path = "from_config"`` so
        provenance tracking is complete. Previously, ``from_config``-
        built instances were indistinguishable from direct-construction
        instances.

        C-39 note: This method is a no-op wrapper around the
        constructor. It exists for backward compatibility and for
        users who prefer the factory-method style. There is no
        functional difference between ``FiberChannel(attenuation)`` and
        ``FiberChannel.from_config(attenuation)`` except for the
        ``_construction_path`` metadata.
        """
        # F-40: deprecation warning with removal target.
        warnings.warn(
            "FiberChannel.from_config() is deprecated and will be "
            "removed in version 7.0.0. Use FiberChannel(attenuation) "
            "directly, which is functionally identical except for the "
            "_construction_path metadata.",
            DeprecationWarning,
            stacklevel=2,
        )
        # C-7: set _construction_path for provenance tracking.
        return cls(
            attenuation=attenuation,
            _construction_path="from_config",
        )

    @classmethod
    def from_config_dict(
        cls: Type[T_FiberChannel],
        config: Union[Mapping[str, Any], AttenuationConfig],
    ) -> T_FiberChannel:
        """Create a :class:`FiberChannel` from a configuration mapping.

        Accepted input shapes
        ---------------------
        * An :class:`AttenuationConfig` instance -- equivalent to
          calling the constructor directly.
        * A :class:`Mapping` with **any** of:
          - ``distance_km`` AND ``fiber_loss_db_km`` (the canonical
            pair), OR
          - ``distance_km`` AND ``total_loss_db`` (alternative form);
            in this case ``alpha`` is derived by division and the
            round-trip caveat from :meth:`from_total_loss` applies, OR
          - All three keys (review C-12): consistency is validated via
            :func:`is_close`, and the ``total_loss_db`` form is
            preferred (review C-3) to preserve round-trip precision.

        Optional keys
        -------------
        * ``wavelength_nm`` -- (F-15) optional wavelength in nm. If
          present, it is passed to the constructor.
        * ``_metadata`` -- ignored on input except for model, version,
          units, transmittance, class, strict_mode, limits, and
          total_loss_db validation (review C-13, C-14, F-13, F-14,
          F-23); regenerated on output by :meth:`to_config_dict`.
          Unknown sub-keys are preserved through round-trip.

        C-3 fix: When ``total_loss_db`` is present, the
        ``total_loss_db`` form is preferred (via :meth:`from_total_loss`)
        so that round-trip precision is preserved for channels
        originally built via :meth:`from_total_loss`.

        C-12 fix: When all three keys are present, consistency is
        validated via :func:`is_close`.

        F-9: The three-key consistency check now uses
        ``_CONTRACT_CHECK_REL_TOL`` and ``_CONTRACT_CHECK_ABS_TOL``
        (named constants) instead of hardcoded ``1e-9`` and ``1e-12``.

        F-13: ``_metadata.strict_mode`` and ``_metadata.limits`` are
        now validated after constructing the instance.

        F-14: ``_metadata.class`` is validated if present.

        F-23: ``_metadata.total_loss_db`` is validated if present.

        F-24: ``_metadata.units`` is validated if present (even if not
        a Mapping -- a non-Mapping emits a warning).

        C-13 fix: If ``_metadata.transmittance`` is present, it is
        compared to the reconstructed channel's transmittance via
        :func:`is_close`. On mismatch, a warning is emitted (non-strict)
        or :class:`ConfigurationError` is raised (strict).

        C-14 fix: If ``_metadata.units`` is present, it is compared to
        :data:`UNITS`. On mismatch, a warning is emitted (non-strict)
        or :class:`ConfigurationError` is raised (strict).

        Unknown keys
        ------------
        Unknown top-level keys are rejected to prevent typos like
        ``distance_km_typo`` from being silently ignored.

        Bool / string rejection
        -----------------------
        ``bool`` values are explicitly rejected to catch JSON-parsing
        bugs where ``0``/``1`` round-trip as booleans. ``str`` values
        are also explicitly rejected: ``float("1.0")`` succeeds, but
        accepting strings masks config bugs where a string is passed
        instead of a number.

        C-34 fix: Uses :func:`_coerce_numeric` instead of bare
        :func:`float` for consistent type handling and error messages
        across all factory methods. This also catches
        :class:`OverflowError` from extreme :class:`decimal.Decimal`
        inputs (review C-4).

        Exception contract
        -------------------
        * :class:`ConfigurationError` is raised for config-level
          errors (bad keys, bad types, inconsistent keys, metadata
          mismatch in strict mode).
        * :class:`ParameterValidationError` is raised for value-level
          errors (negative distance, etc.) and propagates unwrapped.
          Callers catching only :class:`ConfigurationError` will miss
          :class:`ParameterValidationError` -- this is documented
          rather than wrapped to preserve the original error type.
        """
        # Accept AttenuationConfig directly.
        if isinstance(config, AttenuationConfig):
            return cls(
                attenuation=config,
                _construction_path="from_config_dict",
            )

        # Type-check first, then key-check.
        if not isinstance(config, Mapping):
            raise ConfigurationError(
                f"Configuration must be a mapping or AttenuationConfig, "
                f"but got {type(config).__name__}.",
                code="invalid_config_type",
                context={"got_type": type(config).__name__},
            )

        # Use ``set(config)`` (which uses ``__iter__``) rather than
        # ``set(config.keys())`` so we work with exotic Mappings
        # (review C-47: documented as working).
        config_keys = set(config)

        has_a = _REQUIRED_KEYS_CANONICAL.issubset(config_keys)
        has_b = _REQUIRED_KEYS_TOTAL_LOSS.issubset(config_keys)

        if not (has_a or has_b):
            raise ConfigurationError(
                "Configuration must contain either "
                "{'distance_km', 'fiber_loss_db_km'} or "
                "{'distance_km', 'total_loss_db'}.",
                code="missing_keys",
                context={
                    "provided_keys": sorted(config_keys),
                    "required_alternatives": [
                        sorted(_REQUIRED_KEYS_CANONICAL),
                        sorted(_REQUIRED_KEYS_TOTAL_LOSS),
                    ],
                },
            )

        # C-3 / C-12: Determine which form to use.
        # C-3: Prefer the total_loss_db form when present (for
        #      round-trip precision).
        # C-12: If both forms are present, validate consistency and
        #      still prefer the total_loss_db form.
        if has_b:
            use_total_loss = True
            verify_consistency = has_a  # validate if canonical is also present
            if has_a:
                # Both forms present; accept all three keys.
                used_keys = _REQUIRED_KEYS_CANONICAL | _REQUIRED_KEYS_TOTAL_LOSS
            else:
                used_keys = _REQUIRED_KEYS_TOTAL_LOSS
        else:  # has_a only
            used_keys = _REQUIRED_KEYS_CANONICAL
            use_total_loss = False
            verify_consistency = False

        # F-15: wavelength_nm is a known optional key.
        extra_keys = config_keys - used_keys - _IGNORED_KEYS
        if extra_keys:
            raise ConfigurationError(
                f"Configuration contains unknown keys: {sorted(extra_keys)}.",
                code="unknown_keys",
                context={
                    "unknown_keys": sorted(extra_keys),
                    "accepted_keys": sorted(used_keys),
                    "ignored_keys": sorted(_IGNORED_KEYS),
                },
            )

        # Validate _metadata.model, _metadata.version, _metadata.units.
        metadata = config.get("_metadata")
        if isinstance(metadata, Mapping):
            model = metadata.get("model")
            if model is not None and model != "pure_attenuation":
                msg = (
                    f"_metadata.model={model!r} does not match the channel "
                    f"model 'pure_attenuation'. The config may have been "
                    f"serialised from a different channel implementation."
                )
                if get_strict_mode():
                    raise ConfigurationError(
                        msg,
                        code="model_mismatch",
                        context={"expected": "pure_attenuation", "actual": model},
                    )
                _warn_physical_suspicion(msg, stacklevel=2)
            version = metadata.get("version")
            if version is not None and version != __version__:
                logger.warning(
                    f"_metadata.version={version!r} does not match the "
                    f"installed channel version {__version__!r}. Behaviour "
                    f"may differ; check the changelog."
                )
            # C-14 / F-24: Validate _metadata.units.
            meta_units = metadata.get("units")
            if meta_units is not None:
                # F-24: If meta_units is present but not a Mapping,
                # emit warning.
                if not isinstance(meta_units, Mapping):
                    _warn_physical_suspicion(
                        f"_metadata.units is present but not a Mapping "
                        f"(got {type(meta_units).__name__}). Unit "
                        f"validation skipped; mismatch could lead to "
                        f"gross misinterpretation.",
                        stacklevel=2,
                    )
                elif dict(meta_units) != dict(UNITS):
                    msg = (
                        f"_metadata.units={dict(meta_units)!r} does not match "
                        f"the expected units registry {dict(UNITS)!r}. Unit "
                        f"mismatch could lead to gross misinterpretation "
                        f"(e.g. treating metres as kilometres)."
                    )
                    if get_strict_mode():
                        raise ConfigurationError(
                            msg,
                            code="units_mismatch",
                            context={
                                "expected": dict(UNITS),
                                "actual": dict(meta_units),
                            },
                        )
                    _warn_physical_suspicion(msg, stacklevel=2)

        # Bool / string rejection.
        for key in used_keys:
            value = config[key]
            if isinstance(value, bool):
                raise ConfigurationError(
                    f"Configuration value for {key!r} must be a real number, "
                    f"not a bool (got {value!r}).",
                    code="bool_rejected",
                    context={key: value},
                )
            if isinstance(value, str):
                raise ConfigurationError(
                    f"Configuration value for {key!r} must be a real number, "
                    f"not a str (got {value!r}). Pass a numeric type to "
                    f"avoid masking config bugs.",
                    code="str_rejected",
                    context={key: value},
                )

        # F-15: Extract optional wavelength_nm.
        wavelength_nm_value: Optional[float] = None
        if "wavelength_nm" in config:
            wl_raw = config["wavelength_nm"]
            if isinstance(wl_raw, bool):
                raise ConfigurationError(
                    f"Configuration value for 'wavelength_nm' must be a "
                    f"real number or None, not a bool (got {wl_raw!r}).",
                    code="bool_rejected",
                    context={"wavelength_nm": wl_raw},
                )
            if isinstance(wl_raw, str):
                raise ConfigurationError(
                    f"Configuration value for 'wavelength_nm' must be a "
                    f"real number or None, not a str (got {wl_raw!r}).",
                    code="str_rejected",
                    context={"wavelength_nm": wl_raw},
                )
            wavelength_nm_value = _coerce_numeric(
                wl_raw, param_name="wavelength_nm"
            )

        # Build the channel.
        # C-34: use _coerce_numeric instead of bare float() for
        #       consistent type handling.
        # C-4: _coerce_numeric catches OverflowError (from extreme
        #      Decimal inputs) in addition to ValueError/TypeError.
        try:
            distance = _coerce_numeric(
                config["distance_km"], param_name="distance_km"
            )
            if use_total_loss:
                total_loss = _coerce_numeric(
                    config["total_loss_db"], param_name="total_loss_db"
                )
                if verify_consistency:
                    # C-12: Both forms present; validate consistency.
                    # F-9: Use _CONTRACT_CHECK_REL_TOL / _CONTRACT_CHECK_ABS_TOL
                    # instead of hardcoded 1e-9 / 1e-12.
                    loss = _coerce_numeric(
                        config["fiber_loss_db_km"],
                        param_name="fiber_loss_db_km",
                    )
                    expected_total = distance * loss
                    if not is_close(
                        total_loss,
                        expected_total,
                        rel_tol=_CONTRACT_CHECK_REL_TOL,
                        abs_tol=_CONTRACT_CHECK_ABS_TOL,
                    ):
                        raise ConfigurationError(
                            f"Configuration has both (distance_km, "
                            f"fiber_loss_db_km) and total_loss_db, but they "
                            f"are inconsistent: distance_km * fiber_loss_db_km "
                            f"= {expected_total!r} != total_loss_db = "
                            f"{total_loss!r}.",
                            code="inconsistent_keys",
                            context={
                                "distance_km": distance,
                                "fiber_loss_db_km": loss,
                                "total_loss_db": total_loss,
                                "expected_total_loss_db": expected_total,
                            },
                        )
                instance = cls.from_total_loss(
                    distance,
                    total_loss,
                    _construction_path="from_config_dict",
                    wavelength_nm=wavelength_nm_value,  # F-C3 fix: forward wavelength
                )
            else:
                loss = _coerce_numeric(
                    config["fiber_loss_db_km"],
                    param_name="fiber_loss_db_km",
                )
                attenuation = AttenuationConfig(
                    fiber_length=distance,
                    attenuation_coefficient=loss,
                )
                # F-C3 fix: pass wavelength_nm through to constructor so
                # __post_init__ runs the consistency check naturally.
                instance = cls(
                    attenuation=attenuation,
                    _construction_path="from_config_dict",
                    wavelength_nm=wavelength_nm_value,
                )
        except ParameterValidationError:
            # Documented dual-exception contract: let validation errors
            # propagate unwrapped so callers can catch them by their
            # actual type. (Review C-55 suggests this is redundant, but
            # it is kept as a safety guard in case ParameterValidationError
            # is a subclass of ValueError/TypeError/OverflowError.)
            raise
        except (ValueError, TypeError, OverflowError) as e:
            # C-4: Added OverflowError to the except tuple.
            raise ConfigurationError(
                f"Configuration values must be convertible to float; "
                f"failed while building the channel. distance_km="
                f"{config.get('distance_km')!r}, "
                f"fiber_loss_db_km={config.get('fiber_loss_db_km')!r}, "
                f"total_loss_db={config.get('total_loss_db')!r}.",
                code="value_conversion_failed",
                context={
                    "distance_km": config.get("distance_km"),
                    "fiber_loss_db_km": config.get("fiber_loss_db_km"),
                    "total_loss_db": config.get("total_loss_db"),
                },
                cause=e,
            ) from e

        # F-C2/F-C3 fix: wavelength_nm is now passed through the
        # constructor (and from_total_loss) so __post_init__ handles
        # the consistency check naturally. The old post-construction
        # object.__setattr__ bypass has been removed.
        # For channels where wavelength was already passed to the
        # constructor, this section is a no-op. However, for backward
        # compatibility with any code path that did NOT pass wavelength
        # through (e.g. AttenuationConfig direct construction), we
        # still handle the case where wavelength_nm was not set in the
        # constructor but needs to be set now.
        if wavelength_nm_value is not None and instance.wavelength_nm is None:
            object.__setattr__(instance, "wavelength_nm", wavelength_nm_value)
            # F-C2 fix: Re-run consistency check after setting wavelength.
            instance._check_wavelength_consistency()

        # P-09: Delegate post-build metadata validation and preservation
        # to extracted helper methods for testability and maintainability.
        # C-9 fix: pass cls as owner_cls (renamed from cls to clarify
        # that this is an explicit positional arg, not implicit class binding).
        cls._validate_config_metadata_post_build(owner_cls=cls, instance=instance, metadata=metadata)
        cls._preserve_unknown_metadata(instance, metadata)
        return instance

    # ------------------------------------------------------------------
    # P-09: Extracted helper methods from from_config_dict
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_config_metadata_post_build(
        owner_cls: Type[T_FiberChannel],
        instance: T_FiberChannel,
        metadata: Optional[Any],
    ) -> None:
        """Validate config metadata against the constructed channel instance.

        P-09: Extracted from :meth:`from_config_dict` for testability.
        Validates transmittance, class name, strict_mode, limits, and
        total_loss_db consistency between the metadata and the instance.

        C-9 fix (review channel_review_8): This method is a
        ``@staticmethod`` (not ``@classmethod``). The first parameter
        was previously named ``cls``, which was misleading -- it
        suggested it was the implicit class binding of a ``@classmethod``
        call. Renamed to ``owner_cls`` to clarify that this is an
        explicit positional argument, not an implicit class binding.
        A future maintainer who converts it to ``@classmethod`` would
        need to remove the explicit ``owner_cls`` argument at call
        sites; the rename makes that transformation more visible.
        """
        if not isinstance(metadata, Mapping):
            return

        # C-13: Validate _metadata.transmittance consistency.
        meta_transmittance = metadata.get("transmittance")
        if (
            meta_transmittance is not None
            and isinstance(meta_transmittance, (int, float))
            and isinstance(instance.transmittance, (int, float))
        ):
            if not is_close(
                float(meta_transmittance),
                float(instance.transmittance),
                rel_tol=1e-9,
            ):
                msg = (
                    f"_metadata.transmittance={meta_transmittance!r} does "
                    f"not match the reconstructed channel's transmittance="
                    f"{instance.transmittance!r}. Reproducibility "
                    f"metadata may be inconsistent with the actual channel."
                )
                if get_strict_mode():
                    raise ConfigurationError(
                        msg,
                        code="transmittance_mismatch",
                        context={
                            "metadata_transmittance": meta_transmittance,
                            "actual_transmittance": instance.transmittance,
                        },
                    )
                _warn_physical_suspicion(msg, stacklevel=2)

        # F-14: Validate _metadata.class if present.
        meta_class = metadata.get("class")
        if meta_class is not None and meta_class != owner_cls.__name__:
            msg = (
                f"_metadata.class={meta_class!r} does not match the "
                f"current class {owner_cls.__name__!r}. The config may have "
                f"been serialised from a different subclass."
            )
            if get_strict_mode():
                raise ConfigurationError(
                    msg,
                    code="class_mismatch",
                    context={"expected": owner_cls.__name__, "actual": meta_class},
                )
            _warn_physical_suspicion(msg, stacklevel=2)

        # F-13: Validate _metadata.strict_mode if present.
        meta_strict = metadata.get("strict_mode")
        if meta_strict is not None and isinstance(meta_strict, bool):
            if meta_strict != instance._strict_mode_at_construction:
                msg = (
                    f"_metadata.strict_mode={meta_strict!r} differs from "
                    f"the instance's construction-time strict mode "
                    f"{instance._strict_mode_at_construction!r}. "
                    f"The config may have been serialised in a "
                    f"different strict-mode context."
                )
                _warn_physical_suspicion(msg, stacklevel=2)

        # F-13: Validate _metadata.limits if present.
        meta_limits = metadata.get("limits")
        if meta_limits is not None and isinstance(meta_limits, Mapping):
            current_limits = {
                "max_distance_km": MAX_DISTANCE_KM,
                "max_fiber_loss_db_km": MAX_FIBER_LOSS_DB_KM,
            }
            if dict(meta_limits) != current_limits:
                msg = (
                    f"_metadata.limits={dict(meta_limits)!r} differs "
                    f"from the current limits {current_limits!r}. "
                    f"Validation thresholds may have changed between "
                    f"serialisation and deserialisation."
                )
                _warn_physical_suspicion(msg, stacklevel=2)

        # F-23: Validate _metadata.total_loss_db if present.
        meta_total_loss = metadata.get("total_loss_db")
        if (
            meta_total_loss is not None
            and isinstance(meta_total_loss, (int, float))
            and math.isfinite(float(meta_total_loss))
        ):
            if not is_close(
                float(meta_total_loss),
                instance.total_loss_db,
                rel_tol=_CONTRACT_CHECK_REL_TOL,
                abs_tol=_CONTRACT_CHECK_ABS_TOL,
            ):
                msg = (
                    f"_metadata.total_loss_db={meta_total_loss!r} does "
                    f"not match the reconstructed channel's "
                    f"total_loss_db={instance.total_loss_db!r}. "
                    f"Reproducibility metadata may be inconsistent."
                )
                _warn_physical_suspicion(msg, stacklevel=2)

    @staticmethod
    def _preserve_unknown_metadata(
        instance: T_FiberChannel,
        metadata: Optional[Any],
    ) -> None:
        """Preserve unknown _metadata sub-keys on the channel instance.

        P-09: Extracted from :meth:`from_config_dict` for testability.
        Unknown sub-keys are stored in ``_metadata_extra`` so they
        round-trip through :meth:`to_config_dict`.

        P-07: ``_metadata_extra`` is now a ``MappingProxyType`` for
        immutability; the preserved dict is wrapped before setting.
        """
        if not isinstance(metadata, Mapping):
            return

        _KNOWN_METADATA_KEYS = {
            "version",
            "model",
            "class",
            "units",
            "total_loss_db",
            "transmittance",
            "strict_mode",
            "construction_path",
            "provenance",
            "limits",
            "wavelength_nm",
            "extra",
        }
        preserved = {
            k: v for k, v in metadata.items() if k not in _KNOWN_METADATA_KEYS
        }
        if preserved:
            object.__setattr__(
                instance, "_metadata_extra",
                types.MappingProxyType(preserved)
            )

    @classmethod
    def from_total_loss(
        cls: Type[T_FiberChannel],
        distance_km: Numeric,
        total_loss_db: Numeric,
        *,
        _construction_path: str = "from_total_loss",
        wavelength_nm: Optional[float] = None,  # F-C3 fix
    ) -> T_FiberChannel:
        """Create a :class:`FiberChannel` from a total loss value and distance.

        Parameters
        ----------
        distance_km:
            Fibre length in km. Must be finite and non-negative. Zero
            is now **accepted** when total_loss_db is also zero (F-8).
        total_loss_db:
            Total channel loss in dB. Must be finite and non-negative.
        _construction_path:
            Internal parameter for provenance tracking. Factories that
            delegate to :meth:`from_total_loss` (e.g.
            :meth:`from_config_dict`) pass their own path.
        wavelength_nm:
            F-C3 fix: Optional wavelength in nm. When set, the
            wavelength consistency check is performed during
            ``__post_init__``. Previously, this parameter was missing,
            so :meth:`from_config_dict` could not forward the wavelength
            through, and it was set post-construction via
            ``object.__setattr__`` (bypassing the consistency check).

        Notes
        -----
        Computes ``alpha = total_loss_db / distance_km`` and delegates
        to the standard constructor. The original ``total_loss_db`` is
        preserved as an override so that :attr:`total_loss_db`
        round-trips exactly (review C-3).

        F-11: Since ``_total_loss_db_override`` is now ``init=False``,
        the override is set after construction via
        ``object.__setattr__``, and ``_transmittance`` is re-computed
        from the override using ``_transmittance_from_loss_db``.

        C-6 fix: ``_construction_path`` is passed to the constructor
        before ``__post_init__`` runs, so warnings emitted during
        ``__post_init__`` include the correct factory path.

        Zero-distance handling
        ----------------------
        F-8: ``from_total_loss(0, 0)`` is now accepted with the
        convention ``alpha=0``, producing an identity (lossless)
        channel. This matches the ``cascade()`` convention for the
        same ``(0, 0)`` case. Previously, this raised
        ``ParameterValidationError`` with an "ambiguous" message.
        ``distance_km=0`` with ``total_loss_db > 0`` still raises,
        since non-zero loss over zero distance is physically
        impossible.

        F-20: When ``distance > 0`` and ``total_loss_db == 0``
        (or within tolerance of 0), a
        :class:`PhysicalSuspicionWarning` is emitted noting the
        suspicious zero-loss configuration.

        F-21: Distance is checked against ``MAX_DISTANCE_KM``
        before computing alpha, so that out-of-range distances are
        caught early.

        Round-trip precision
        --------------------
        For non-zero ``L``, ``alpha = L_tot / L`` is well-conditioned
        for most inputs. The recomputed ``L_tot' = alpha * L`` differs
        from the input ``L_tot`` by at most a few ULP for
        well-conditioned ratios, but can differ by much more for
        ill-conditioned ones (e.g. ``L=3, L_tot=1`` gives
        ``alpha = 0.333...``, ``L_tot' = 0.999...`` -- error ~3e-16).
        The override mechanism hides this discrepancy for
        :attr:`total_loss_db` and :attr:`transmittance`, but the
        underlying ``alpha`` field on :class:`AttenuationConfig` is
        still the rounded value.

        F-43: Fixed docstring typo -- ``:attr:`transmittance`}`` was
        ``:attr:`transmittance}`` (unmatched brace).
        """
        coerced_distance = _coerce_numeric(distance_km, param_name="distance_km")
        coerced_total_loss = _coerce_numeric(total_loss_db, param_name="total_loss_db")

        if not is_finite_non_negative(coerced_distance):
            raise ParameterValidationError(
                f"distance_km must be finite and non-negative (got "
                f"{coerced_distance!r}).",
                param_name="distance_km",
                param_value=coerced_distance,
            )
        if not is_finite_non_negative(coerced_total_loss):
            raise ParameterValidationError(
                f"total_loss_db must be finite and non-negative (got "
                f"{coerced_total_loss!r}).",
                param_name="total_loss_db",
                param_value=coerced_total_loss,
            )

        # F-8: Accept (0, 0) with alpha=0 convention instead of raising
        # "ambiguous". Only raise if distance=0 but total_loss > 0.
        if coerced_distance == 0.0:
            # C-4 fix: use single named constant _ZERO_LOSS_ABS_TOL
            # instead of default abs_tol=0.0, consistent with the
            # zero-loss check at L2889.
            if not is_close(coerced_total_loss, 0.0, abs_tol=_ZERO_LOSS_ABS_TOL):
                raise ParameterValidationError(
                    f"distance_km=0 requires total_loss_db=0 (got "
                    f"{coerced_total_loss!r}); cannot have non-zero loss "
                    f"over zero distance.",
                    param_name="total_loss_db",
                    param_value=coerced_total_loss,
                )
            # F-8: (0, 0) is accepted with alpha=0 convention.
            attenuation = AttenuationConfig(
                fiber_length=0.0,
                attenuation_coefficient=0.0,
            )
            # No override needed for (0, 0) -- alpha*L = 0 exactly.
            # F-C3 fix: pass wavelength_nm through to constructor.
            return cls(
                attenuation=attenuation,
                _construction_path=_construction_path,
                wavelength_nm=wavelength_nm,
            )

        # F-21: Check distance against MAX_DISTANCE_KM before computing alpha.
        if coerced_distance > MAX_DISTANCE_KM:
            raise ParameterValidationError(
                f"distance_km={coerced_distance!r} exceeds MAX_DISTANCE_KM="
                f"{MAX_DISTANCE_KM}.",
                param_name="distance_km",
                param_value=coerced_distance,
            )

        # F-20 / P-05: Warn when distance > 0 and total_loss_db == 0 (or
        # within tolerance of 0) -- suspicious zero-loss configuration.
        # In strict mode, this now RAISES ParameterValidationError instead
        # of only warning, consistent with the strict-mode contract
        # (P-05). A lossless channel over non-zero distance is physically
        # suspicious and strict mode should reject it.
        # C-4 fix: use _ZERO_LOSS_ABS_TOL (same value as _CONTRACT_CHECK_ABS_TOL
        # but with a different semantic name) for consistent zero-loss semantics.
        if is_close(coerced_total_loss, 0.0, abs_tol=_ZERO_LOSS_ABS_TOL):
            msg = (
                f"from_total_loss: distance_km={coerced_distance:.6g} > 0 "
                f"but total_loss_db={coerced_total_loss:.6g} is zero "
                f"(or within tolerance). This produces alpha=0, a "
                f"lossless channel over non-zero distance, which is "
                f"physically suspicious."
            )
            if get_strict_mode():
                raise ParameterValidationError(
                    msg,
                    param_name="total_loss_db",
                    param_value=coerced_total_loss,
                    context={
                        "distance_km": coerced_distance,
                        "total_loss_db": coerced_total_loss,
                    },
                )
            _warn_physical_suspicion(msg, stacklevel=2)

        # Non-zero distance: compute alpha = L_tot / L.
        fiber_loss_db_km = coerced_total_loss / coerced_distance

        # Explicit finiteness check on the computed alpha.
        if not math.isfinite(fiber_loss_db_km):
            raise ParameterValidationError(
                f"Calculated fiber_loss_db_km is not finite "
                f"({fiber_loss_db_km!r}). Check for overflow from "
                f"small distance_km relative to total_loss_db.",
                param_name="total_loss_db",
                param_value=coerced_total_loss,
                context={"calculated_loss_per_km": fiber_loss_db_km},
            )

        if fiber_loss_db_km > MAX_FIBER_LOSS_DB_KM:
            raise ParameterValidationError(
                f"Calculated fiber_loss_db_km ({fiber_loss_db_km:.6g}) exceeds "
                f"the limit of {MAX_FIBER_LOSS_DB_KM} dB/km. Check if "
                f"distance_km is too small relative to total_loss_db.",
                param_name="total_loss_db",
                param_value=coerced_total_loss,
                context={"calculated_loss_per_km": fiber_loss_db_km},
            )

        attenuation = AttenuationConfig(
            fiber_length=coerced_distance,
            attenuation_coefficient=fiber_loss_db_km,
        )
        # F-11: Two-phase construction. Since _total_loss_db_override
        # is init=False, we construct the instance first (transmittance
        # computed from alpha*L in __post_init__), then set the override
        # via object.__setattr__, then re-compute _transmittance from
        # the override.
        # F-C3 fix: pass wavelength_nm through to constructor so
        # __post_init__ runs the consistency check.
        # C-10 fix: Use a ContextVar flag to skip the transmittance
        # health check in __post_init__ for this construction path.
        # __post_init__ normally runs _check_transmittance_health on
        # the OLD transmittance (computed from alpha*L), which will be
        # replaced by the override-based value. This could emit
        # contradictory warnings (old value fine, new value underflowed)
        # or raise in strict mode on a value that will be replaced.
        # The ContextVar flag is set before cls() is called and
        # cleared in __post_init__ after the skip, so it doesn't
        # pollute the dataclass schema and works correctly with
        # thread/async isolation.
        _skip_health_token = _SKIP_TRANSMITTANCE_HEALTH.set(True)
        try:
            instance = cls(
                attenuation=attenuation,
                _construction_path=_construction_path,
                wavelength_nm=wavelength_nm,
            )
        finally:
            _SKIP_TRANSMITTANCE_HEALTH.reset(_skip_health_token)
        object.__setattr__(instance, "_total_loss_db_override", coerced_total_loss)
        # Re-compute _transmittance from the override (exact value).
        new_t = cls._transmittance_from_loss_db(coerced_total_loss)
        object.__setattr__(
            instance,
            "_transmittance",
            new_t,
        )
        # F-03 fix (review channel_review_7): Call _check_transmittance_health
        # on the NEW transmittance value. C-10 fix: This is now the ONLY
        # health check for from_total_loss-built channels, since __post_init__
        # was skipped via the _SKIP_TRANSMITTANCE_HEALTH ContextVar.
        # Previously, both __post_init__ (on old alpha*L value) and
        # from_total_loss (on new override value) ran the check, which
        # could emit contradictory warnings. Now, the single correct
        # check runs here on the final override-based transmittance.
        instance._check_transmittance_health(new_t)
        return instance

    # ------------------------------------------------------------------
    # Convenience constructors for parameter sweeps.
    # ------------------------------------------------------------------
    def with_distance(
        self,
        distance_km: Numeric,
        *,
        strict: Optional[bool] = None,
    ) -> "FiberChannel":
        """Return a copy of this channel with ``distance_km`` replaced.

        Uses :func:`dataclasses.replace` so subclasses with extra
        fields are preserved correctly.

        C-8 fix: ``_metadata_extra`` is manually copied from ``self``
        to the new instance (``dataclasses.replace`` resets
        ``init=False`` fields to their defaults). ``_construction_path``
        is set to ``"with_distance"``.

        F-1 / F-11: After ``_dc_replace``, ``_total_loss_db_override``
        is explicitly reset to ``None`` via ``object.__setattr__``,
        preventing the stale override from the source instance from
        being carried over.

        P-01: ``_strict_mode_at_construction`` is NO LONGER preserved
        from the source instance. ``__post_init__`` sets the correct
        construction-time value, which reflects the ``_optional_strict``
        override when ``strict`` is passed. The previous F-18 override
        falsified provenance metadata when strict mode was temporarily
        changed.

        F-15: ``wavelength_nm`` is carried over to the new instance.

        F-M2 fix (review channel_review_6): Note that
        ``_total_loss_db_override`` is reset to ``None``. If the source
        channel was built via :meth:`from_total_loss`, the override
        precision is **lost**. ``total_loss_db`` on the result comes
        from ``alpha * L``, which may differ from the source's
        ``total_loss_db`` by a few ULP. For channels where round-trip
        precision of ``total_loss_db`` is critical, avoid using
        ``with_distance`` / ``with_loss`` on ``from_total_loss``-built
        channels; instead, reconstruct via :meth:`from_total_loss` with
        the new distance.

        Parameters
        ----------
        distance_km:
            New fibre length in km.
        strict:
            Optional override for strict mode during the new
            instance's construction. ``None`` inherits the current
            context's strict mode.
        """
        d = _coerce_numeric(distance_km, param_name="distance_km")
        with _optional_strict(strict):
            new_instance = _dc_replace(
                self,
                attenuation=AttenuationConfig(
                    fiber_length=d,
                    attenuation_coefficient=self.fiber_loss_db_km,
                ),
                _construction_path="with_distance",
            )
        # C-8 / P-07: Manually copy _metadata_extra as a
        # MappingProxyType (dataclasses.replace resets init=False
        # fields to their defaults). Using MappingProxyType enforces
        # immutability consistent with the frozen dataclass contract.
        object.__setattr__(
            new_instance, "_metadata_extra",
            types.MappingProxyType(dict(self._metadata_extra))
        )
        # F-1 / F-11: Explicitly reset _total_loss_db_override to None
        # to prevent stale override from the source instance.
        object.__setattr__(new_instance, "_total_loss_db_override", None)
        # F-05 fix (review channel_review_8): Warn when with_distance is
        # called on a channel that was built via from_total_loss. The
        # override preserves exact total_loss_db; resetting it means
        # total_loss_db on the result comes from alpha*L which may differ
        # by a few ULP. This can cause non-reproducible results in QKD
        # simulations where round-trip precision of total_loss_db matters.
        if self._total_loss_db_override is not None:
            _warn_physical_suspicion(
                f"with_distance: source channel was built via from_total_loss "
                f"with _total_loss_db_override={self._total_loss_db_override!r}. "
                f"The override is reset to None on the new instance, meaning "
                f"total_loss_db will come from alpha*L (may differ by a few "
                f"ULP from the source). For channels where round-trip "
                f"precision of total_loss_db is critical, reconstruct via "
                f"from_total_loss with the new distance instead.",
                stacklevel=3,
            )
        # P-01: Removed stale override of _strict_mode_at_construction.
        # __post_init__ already set the correct value (reflecting the
        # _optional_strict override), so copying the SOURCE instance's
        # value falsifies provenance metadata when strict mode was
        # temporarily changed by the ``strict`` parameter.
        return new_instance

    def with_loss(
        self,
        fiber_loss_db_km: Numeric,
        *,
        strict: Optional[bool] = None,
    ) -> "FiberChannel":
        """Return a copy of this channel with ``fiber_loss_db_km`` replaced.

        Uses :func:`dataclasses.replace`. See :meth:`with_distance` for
        the ``strict`` parameter and the C-8 fix for metadata
        preservation.

        F-1 / F-11: After ``_dc_replace``, ``_total_loss_db_override``
        is explicitly reset to ``None`` via ``object.__setattr__``.

        P-01: ``_strict_mode_at_construction`` is NO LONGER preserved
        from the source instance. ``__post_init__`` sets the correct
        construction-time value, which reflects the ``_optional_strict``
        override when ``strict`` is passed.

        F-15: ``wavelength_nm`` is carried over to the new instance.

        F-M2 fix (review channel_review_6): Note that
        ``_total_loss_db_override`` is reset to ``None``. If the source
        channel was built via :meth:`from_total_loss`, the override
        precision is **lost**. ``total_loss_db`` on the result comes
        from ``alpha * L``, which may differ from the source's
        ``total_loss_db`` by a few ULP. For channels where round-trip
        precision of ``total_loss_db`` is critical, avoid using
        ``with_distance`` / ``with_loss`` on ``from_total_loss``-built
        channels; instead, reconstruct via :meth:`from_total_loss` with
        the new loss value.
        """
        a = _coerce_numeric(fiber_loss_db_km, param_name="fiber_loss_db_km")
        with _optional_strict(strict):
            new_instance = _dc_replace(
                self,
                attenuation=AttenuationConfig(
                    fiber_length=self.distance_km,
                    attenuation_coefficient=a,
                ),
                _construction_path="with_loss",
            )
        # C-8 / P-07: Manually copy _metadata_extra as MappingProxyType.
        object.__setattr__(
            new_instance, "_metadata_extra",
            types.MappingProxyType(dict(self._metadata_extra))
        )
        # F-1 / F-11: Explicitly reset _total_loss_db_override to None.
        object.__setattr__(new_instance, "_total_loss_db_override", None)
        # F-05 fix (review channel_review_8): Same warning as with_distance.
        if self._total_loss_db_override is not None:
            _warn_physical_suspicion(
                f"with_loss: source channel was built via from_total_loss "
                f"with _total_loss_db_override={self._total_loss_db_override!r}. "
                f"The override is reset to None on the new instance, meaning "
                f"total_loss_db will come from alpha*L (may differ by a few "
                f"ULP from the source). For channels where round-trip "
                f"precision of total_loss_db is critical, reconstruct via "
                f"from_total_loss with the new loss value instead.",
                stacklevel=3,
            )
        # P-01: Removed stale override of _strict_mode_at_construction.
        # __post_init__ already set the correct value (reflecting the
        # _optional_strict override), so copying the SOURCE instance's
        # value falsifies provenance metadata when strict mode was
        # temporarily changed by the ``strict`` parameter.
        return new_instance

    # ------------------------------------------------------------------
    # Fast construction for hot loops (review C-28)
    # ------------------------------------------------------------------
    @classmethod
    def _fast_construct(
        cls,
        attenuation: AttenuationConfig,
        *,
        _construction_path: str = "fast_construct",
        _total_loss_db_override: Optional[float] = None,
        _wavelength_nm: Optional[float] = None,  # F-15
    ) -> "FiberChannel":
        """Construct a channel WITHOUT validation, for hot loops (review C-28).

        INTERNAL USE ONLY. Skips all validation and contract checks.
        The caller is responsible for ensuring the inputs are valid.
        Use only in performance-critical Monte Carlo loops where
        inputs are trusted (e.g. when the inputs were already validated
        upstream).

        Not subject to ``__debug__`` guards; the caller assumes all
        risk. The constructed instance still has consistent derived
        state (``_transmittance``, ``_strict_mode_at_construction``)
        because those are computed from the inputs without validation.

        F-12: After computing transmittance ``t``, checks
        ``math.isfinite(t)`` and ``0.0 <= t <= 1.0``. Raises
        :class:`ParameterValidationError` if not.

        P-15: Added ``isinstance`` type check for the ``attenuation``
        parameter. Previously, passing a non-AttenuationConfig would
        produce ``AttributeError`` deep inside the computation rather
        than a clear ``TypeError`` at construction.

        F-31: Uses ``_transmittance_from_loss_db`` helper instead of
        inline ``math.pow``.
        """
        # P-15: type check for attenuation parameter.
        if not isinstance(attenuation, AttenuationConfig):
            raise TypeError(
                f"_fast_construct requires an AttenuationConfig, got "
                f"{type(attenuation).__name__}. Skipped validation "
                f"is only safe for trusted AttenuationConfig inputs."
            )
        instance = object.__new__(cls)
        object.__setattr__(instance, "attenuation", attenuation)
        object.__setattr__(instance, "_construction_path", _construction_path)
        object.__setattr__(
            instance, "_total_loss_db_override", _total_loss_db_override
        )
        object.__setattr__(
            instance, "_strict_mode_at_construction", get_strict_mode()
        )
        # P-07 / F-L2: use shared singleton MappingProxyType for immutability.
        object.__setattr__(
            instance, "_metadata_extra", _EMPTY_METADATA_EXTRA
        )
        # F-15: set wavelength_nm.
        object.__setattr__(instance, "wavelength_nm", _wavelength_nm)
        # F-31: use _effective_total_loss_db() pattern.
        total_db = (
            _total_loss_db_override
            if _total_loss_db_override is not None
            else attenuation.total_loss_db
        )
        # F-31: use _transmittance_from_loss_db helper.
        t = cls._transmittance_from_loss_db(total_db)
        # F-12: check that t is finite and in [0, 1].
        if not math.isfinite(t):
            raise ParameterValidationError(
                f"_fast_construct: computed transmittance {t!r} is not "
                f"finite. Inputs: distance_km={attenuation.fiber_length!r}, "
                f"fiber_loss_db_km={attenuation.attenuation_coefficient!r}, "
                f"_total_loss_db_override={_total_loss_db_override!r}.",
                param_name="transmittance",
                param_value=t,
            )
        if not (0.0 <= t <= 1.0):
            raise ParameterValidationError(
                f"_fast_construct: computed transmittance {t!r} is out of "
                f"the physical range [0.0, 1.0]. Inputs: "
                f"distance_km={attenuation.fiber_length!r}, "
                f"fiber_loss_db_km={attenuation.attenuation_coefficient!r}, "
                f"_total_loss_db_override={_total_loss_db_override!r}.",
                param_name="transmittance",
                param_value=t,
            )
        # F-M9 fix (review channel_review_6): Add lightweight subnormal
        # check. Previously, _fast_construct skipped all warnings,
        # including subnormal warnings. Instances built via
        # _fast_construct could have subnormal transmittance without
        # any warning, then fail validate() when called later. At
        # minimum, log a DEBUG warning for subnormal transmittance.
        if 0.0 < t < TRANSMITTANCE_SUBNORMAL_THRESHOLD:
            logger.debug(
                "_fast_construct: computed transmittance %r is subnormal "
                "(below smallest normal float). Numerical precision may "
                "be degraded. Inputs: distance_km=%r, "
                "fiber_loss_db_km=%r.",
                t,
                attenuation.fiber_length,
                attenuation.attenuation_coefficient,
            )
        object.__setattr__(instance, "_transmittance", t)
        return instance

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        # C-9 / C-36: guard against _transmittance == _UNSET so repr
        # on a partially-constructed instance does not raise.
        t = self._transmittance
        if not isinstance(t, (int, float)):
            trans_str = "<unset>"
        else:
            if math.isnan(float(t)):
                trans_fmt = ".4e"
            elif 0.0 < float(t) < _REPR_TRANSMITTANCE_SCIENTIFIC_THRESHOLD:
                trans_fmt = ".4e"
            else:
                trans_fmt = ".4f"
            trans_str = f"{t:{trans_fmt}}"
        # F-42: Use .6g for total_loss_db instead of .2f/.2e.
        loss_str = f"{self.total_loss_db:.6g}"
        # Use ``:.6g`` for distance and alpha to avoid misleading
        # fixed-point output for very small or very large values.
        parts = [
            f"distance_km={self.distance_km:.6g}",
            f"fiber_loss_db_km={self.fiber_loss_db_km:.6g}",
            f"total_loss_db={loss_str}",
            f"transmittance={trans_str}",
        ]
        # F-15: include wavelength_nm if set.
        if self.wavelength_nm is not None:
            parts.append(f"wavelength_nm={self.wavelength_nm:.1f}")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    def __str__(self) -> str:
        """Concise human-readable summary."""
        t = self._transmittance
        if not isinstance(t, (int, float)):
            t_str = "<unset>"
        elif math.isnan(float(t)):
            t_str = "nan"
        # C-37: distinguish +0.0 (underflowed) from -0.0 (not underflowed).
        elif t == 0.0 and math.copysign(1.0, t) > 0:
            t_str = "0.0 (underflowed)"
        elif t == 0.0 and math.copysign(1.0, t) < 0:
            t_str = "-0.0"
        elif 0.0 < t < 1e-4:
            t_str = f"{t:.3e}"
        else:
            t_str = f"{t:.6g}"
        # Use ``:.6g`` for distance and alpha to preserve precision in logs.
        base = (
            f"FiberChannel(L={self.distance_km:.6g} km, "
            f"alpha={self.fiber_loss_db_km:.6g} dB/km, "
            f"T={t_str})"
        )
        # F-15: include wavelength_nm if set.
        if self.wavelength_nm is not None:
            base += f", wl={self.wavelength_nm:.1f} nm"
        return base

    def __eq__(self, other: Any) -> bool:
        """Tolerance-based equality on defining scalars, transmittance, and optional wavelength (F-15).

        Compares ``distance_km``, ``fiber_loss_db_km``, and
        ``transmittance`` via :func:`is_close`. The transmittance
        comparison ensures that channels built via
        :meth:`from_total_loss` with different ``total_loss_db``
        inputs (but the same resulting ``(L, alpha)``) do NOT
        compare equal, since their :attr:`transmittance` properties
        differ at the ULP level.

        F-02 fix (review channel_review_7): Previously, ``__eq__``
        compared only ``distance_km`` and ``fiber_loss_db_km``,
        ignoring ``_total_loss_db_override``. Two channels with
        different overrides but the same ``(L, alpha)`` compared
        equal, even though their transmittance values differed.
        Now, ``transmittance`` is compared with ``is_close`` to
        catch this case. For full semantic comparison including
        the override value itself, use :meth:`equals_semantic`.

        F-15: When both instances have ``wavelength_nm`` set,
        ``is_close`` is also applied to the wavelength. If either
        instance has ``wavelength_nm = None``, the wavelength
        dimension is ignored.

        ``isinstance`` (covariant) is intentional; subclasses with
        extra fields SHOULD override this method to include their
        fields.

        Returns ``NotImplemented`` for non-:class:`FiberChannel`
        operands, letting Python fall back to identity comparison.

        NaN fields make :func:`is_close` return ``False``, so NaN
        channels compare unequal to everything (including themselves)
        -- this is the standard Python idiom.
        """
        if not isinstance(other, FiberChannel):
            return NotImplemented
        if not (
            is_close(self.distance_km, other.distance_km)
            and is_close(self.fiber_loss_db_km, other.fiber_loss_db_km)
        ):
            return False
        # F-02 fix (review channel_review_7): Compare transmittance to
        # catch channels with different _total_loss_db_override but the
        # same (L, alpha). Two channels built via from_total_loss with
        # different total_loss_db inputs will have different transmittance
        # values (at the ULP level), and is_close will detect this.
        # C-1 fix (review channel_review_8): Use a tolerance scaled to
        # the transmittance magnitude instead of
        # abs_tol=_TRANSMITTANCE_UNDERFLOW_ABS_TOL (= sys.float_info.min).
        # For subnormal transmittances (T < 2.22e-308), two values can
        # differ by a factor of ~10^90 and still be "close" with
        # abs_tol=sys.float_info.min, which silently passes wrong
        # transmittances in regression tests on long-distance links.
        # The fix: compare total_loss_db (which is well-conditioned) for
        # subnormal T, and use a scaled abs_tol for normal T.
        a_t = self.transmittance
        b_t = other.transmittance
        # For subnormal transmittance, compare total_loss_db instead,
        # since it is well-conditioned (no underflow/precision loss).
        if (0.0 < abs(a_t) < TRANSMITTANCE_SUBNORMAL_THRESHOLD
                or 0.0 < abs(b_t) < TRANSMITTANCE_SUBNORMAL_THRESHOLD):
            if not is_close(self.total_loss_db, other.total_loss_db,
                            rel_tol=1e-9):
                return False
        elif not is_close(
            a_t, b_t,
            abs_tol=max(sys.float_info.min, 1e-15 * max(abs(a_t), abs(b_t))),
        ):
            return False
        # F-15: compare wavelength if both are set.
        if self.wavelength_nm is not None and other.wavelength_nm is not None:
            if not is_close(self.wavelength_nm, other.wavelength_nm):
                return False
        return True

    def equals_exact(self, other: Any) -> bool:
        """Exact (bit-for-bit) equality on the defining scalar fields.

        Use this for regression tests where tolerance-based
        :meth:`__eq__` would be flaky near tolerance boundaries.

        C-20 note: Compares the defining scalars
        (``distance_km``, ``fiber_loss_db_km``) and
        ``transmittance`` (F-02 fix). For full semantic comparison
        including the override, use :meth:`equals_semantic`.

        C-21 note: Returns ``False`` (not ``NotImplemented``) for
        non-:class:`FiberChannel` operands. This differs from
        :meth:`__eq__`, which returns ``NotImplemented``. The
        difference is intentional: :meth:`equals_exact` is a regular
        method (not an operator), so returning ``NotImplemented``
        would be unusual and require callers to explicitly handle it.

        F-15: ``wavelength_nm`` is compared exactly when both are set.
        If one is ``None`` and the other is not, they differ.
        """
        if not isinstance(other, FiberChannel):
            return False
        if not (
            self.distance_km == other.distance_km
            and self.fiber_loss_db_km == other.fiber_loss_db_km
        ):
            return False
        # F-02 fix (review channel_review_7): Compare transmittance
        # exactly to catch channels with different _total_loss_db_override.
        if self.transmittance != other.transmittance:
            return False
        # F-15: exact wavelength comparison.
        if self.wavelength_nm is None and other.wavelength_nm is None:
            return True
        if self.wavelength_nm is None or other.wavelength_nm is None:
            return False
        return self.wavelength_nm == other.wavelength_nm

    def equals_semantic(self, other: Any) -> bool:
        """Semantic equality: defining scalars, override, and wavelength.

        C-20 fix: This is the full semantic comparison that includes
        ``_total_loss_db_override``. Two channels that compare equal
        under :meth:`__eq__` may differ in ``_total_loss_db_override``,
        which affects :attr:`total_loss_db` and :attr:`transmittance`.

        F-25: Uses ``is_close`` for ``_total_loss_db_override``
        comparison instead of exact ``==``, with the same tolerance
        as :meth:`__eq__`.

        F-15: ``wavelength_nm`` is compared semantically (``is_close``
        when both set, ``None`` vs ``None`` matches, ``None`` vs
        non-``None`` differs).

        F-M7 fix (review channel_review_6): Previously, this method
        used exact ``==`` on defining scalars (``distance_km``,
        ``fiber_loss_db_km``) but ``is_close`` on the override. Two
        channels differing by 1 ULP in alpha would fail
        ``equals_semantic`` (exact comparison) but pass ``__eq__``
        (tolerance comparison). This asymmetry was confusing because
        the name ``equals_semantic`` suggested a deeper comparison
        but was actually stricter than ``__eq__`` on primitives.
        Now uses ``is_close`` on defining scalars too, making
        ``equals_semantic`` consistently tolerance-based on all
        fields, while ``equals_exact`` remains bit-for-bit.

        Returns ``False`` for non-:class:`FiberChannel` operands
        (consistent with :meth:`equals_exact`).
        """
        if not isinstance(other, FiberChannel):
            return False
        # F-M7 fix: use is_close on defining scalars, not exact ==.
        if not (
            is_close(self.distance_km, other.distance_km)
            and is_close(self.fiber_loss_db_km, other.fiber_loss_db_km)
        ):
            return False
        # F-25: use is_close for _total_loss_db_override comparison.
        if self._total_loss_db_override is None and other._total_loss_db_override is None:
            pass  # both None: match
        elif self._total_loss_db_override is None or other._total_loss_db_override is None:
            return False  # one has override, other doesn't
        elif not is_close(self._total_loss_db_override, other._total_loss_db_override):
            return False
        # F-15: wavelength comparison.
        if self.wavelength_nm is None and other.wavelength_nm is None:
            pass  # both None: match
        elif self.wavelength_nm is None or other.wavelength_nm is None:
            return False  # one has wavelength, other doesn't
        elif not is_close(self.wavelength_nm, other.wavelength_nm):
            return False
        return True

    # ------------------------------------------------------------------
    # Pickling support
    # ------------------------------------------------------------------
    def __getstate__(self) -> Dict[str, Any]:
        """Return a picklable state dict.

        Explicit ``__getstate__`` / ``__setstate__`` are required
        because the class is a frozen dataclass with ``slots=True``
        and ``init=False`` derived fields; the default pickle protocol
        for such classes can lose the derived fields.

        F-15: ``wavelength_nm`` is included in the state dict.
        """
        return {
            "attenuation": self.attenuation,
            "_transmittance": self._transmittance,
            "_total_loss_db_override": self._total_loss_db_override,
            "_construction_path": self._construction_path,
            "_strict_mode_at_construction": self._strict_mode_at_construction,
            # P-07: serialize as mutable dict (MappingProxyType is not
            # directly picklable in all Python versions).
            "_metadata_extra": dict(self._metadata_extra),
            "wavelength_nm": self.wavelength_nm,  # F-15
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state from a pickled dict.

        F-2: Raises :class:`ConfigurationError` if ``"attenuation"``
        is not in ``state`` (previously ``state.get("attenuation")``
        returned ``None`` on missing keys, causing downstream
        ``AttributeError``).

        F-01 fix (review channel_review_7): The standalone
        ``_check_transmittance_health`` call was removed because it
        was unprotected by the try/except block below. In strict mode
        it raised ``ParameterValidationError`` instead of
        ``ConfigurationError``, breaking the documented contract.
        ``validate()`` (below) already calls
        ``_check_transmittance_health`` internally, so the standalone
        call was both redundant and buggy.

        F-15: ``wavelength_nm`` is restored from the state dict.

        C-10 fix: Uses :meth:`state.get` with defaults for all OTHER
        fields (previously used ``state["attenuation"]`` and
        ``state["_transmittance"]`` directly, which raised
        :class:`KeyError` on missing keys).

        C-32 fix: Calls :meth:`validate` at the end to catch corruption
        introduced by pickling. In strict mode, validation failure
        raises; in non-strict mode, a warning is logged.
        """
        # F-2: raise ConfigurationError if "attenuation" not in state.
        if "attenuation" not in state:
            raise ConfigurationError(
                f"Pickled state is missing required key 'attenuation'. "
                f"The state dict may be corrupted or from an incompatible "
                f"version. Available keys: {sorted(state.keys())}.",
                code="missing_attenuation",
                context={"available_keys": sorted(state.keys())},
            )
        object.__setattr__(self, "attenuation", state["attenuation"])
        object.__setattr__(
            self, "_transmittance", state.get("_transmittance", _UNSET)
        )
        object.__setattr__(
            self, "_total_loss_db_override", state.get("_total_loss_db_override")
        )
        object.__setattr__(
            self,
            "_construction_path",
            state.get("_construction_path", "direct"),
        )
        object.__setattr__(
            self,
            "_strict_mode_at_construction",
            state.get("_strict_mode_at_construction", False),
        )
        object.__setattr__(
            self, "_metadata_extra",
            types.MappingProxyType(dict(state.get("_metadata_extra", {})))
        )
        # F-15: restore wavelength_nm.
        object.__setattr__(
            self, "wavelength_nm", state.get("wavelength_nm")
        )
        # F-01 fix (review channel_review_7): Removed the standalone
        # _check_transmittance_health call here. Previously, this call
        # was UNPROTECTED by the try/except below, so in strict mode it
        # raised ParameterValidationError instead of ConfigurationError,
        # breaking the documented contract that "unpickling in strict
        # mode raises ConfigurationError, not ParameterValidationError".
        # Since validate() (below) already calls _check_transmittance_health
        # internally, the standalone call was redundant as well as buggy.
        # C-32: call validate() to catch corruption, but only if the
        # instance is fully constructed (attenuation and _transmittance
        # are present).
        # H-17 fix (review channel_review_8): Previously,
        # __setstate__ caught ALL ParameterValidationError and
        # re-raised as ConfigurationError. This was misleading for
        # range violations (e.g. distance_km > MAX_DISTANCE_KM) which
        # are NOT strict-mode-specific. Now, we only catch
        # transmittance-health-related ParameterValidationErrors (which
        # are strict-mode-specific) and convert those to
        # ConfigurationError. Range violations are allowed to propagate
        # as ParameterValidationError so the error message is accurate.
        # F-M8 fix (review channel_review_6): Unpickling a valid channel
        # serialized in non-strict mode can fail in strict mode if the
        # channel has underflowed transmittance. This breaks
        # cross-environment pickle compatibility. Now, instead of
        # re-raising ParameterValidationError in strict mode, convert it
        # to a ConfigurationError with a clear message about
        # strict-mode mismatch, so callers can handle it appropriately.
        if (
            self.attenuation is not None
            and isinstance(self._transmittance, (int, float))
        ):
            try:
                self.validate()
            except ParameterValidationError as e:
                # H-17 fix: Distinguish strict-mode-specific
                # transmittance-health errors from range-violation
                # errors. Transmittance-health errors (underflow,
                # subnormal, NaN) are strict-mode-specific and should
                # be wrapped in ConfigurationError. Range violations
                # (distance_km > MAX, alpha > MAX) are NOT
                # strict-mode-specific and should propagate as
                # ParameterValidationError with their original message.
                is_strict_mode_error = (
                    "transmittance" in str(e).lower()
                    or "subnormal" in str(e).lower()
                    or "underflow" in str(e).lower()
                    or "nan" in str(e).lower()
                )
                if is_strict_mode_error and get_strict_mode():
                    raise ConfigurationError(
                        f"Pickled instance failed strict-mode validation: {e}. "
                        f"The instance was likely serialized in non-strict "
                        f"mode and contains values (e.g. underflowed "
                        f"transmittance) that are rejected in strict mode. "
                        f"Either unpickle in non-strict mode or reconstruct "
                        f"the channel from config in strict mode.",
                        code="strict_mode_pickle_mismatch",
                        context={
                            "validation_error": str(e),
                            "strict_mode_at_unpickle": True,
                        },
                    ) from e
                elif is_strict_mode_error:
                    logger.warning(
                        "Pickled instance failed validation: %s. The instance "
                        "may be corrupted or was pickled from an incompatible "
                        "version.",
                        e,
                    )
                else:
                    # Range violations: propagate as-is so the error
                    # message is accurate (not misleadingly labeled as
                    # "strict-mode pickle mismatch").
                    raise

    def __copy__(self) -> "FiberChannel":
        """Frozen dataclass is immutable; ``copy.copy`` returns ``self``.

        Review C-43: documented behaviour. Both :meth:`__copy__` and
        :meth:`__deepcopy__` return ``self`` because the class is
        immutable.
        """
        return self

    def __deepcopy__(self, memo: Dict[int, Any]) -> "FiberChannel":
        """Frozen dataclass is immutable; ``copy.deepcopy`` returns ``self``.

        C-42 fix: Defensive check -- if the wrapped
        :class:`AttenuationConfig` is not :class:`Hashable` (i.e. a
        future version becomes mutable), deep-copy it instead of
        returning ``self``. This prevents silent state sharing if
        :class:`AttenuationConfig` changes.

        F-44: ``import copy`` moved to module level (was lazy import).

        Review C-43: documented behaviour for the immutable case.
        """
        # C-42: defensive deep-copy if AttenuationConfig becomes mutable.
        if not isinstance(self.attenuation, Hashable):
            new_attenuation = copy.deepcopy(self.attenuation, memo)
            instance = self.__class__(
                attenuation=new_attenuation,
                _construction_path=self._construction_path,
            )
            # F-11: set override via object.__setattr__ (init=False).
            if self._total_loss_db_override is not None:
                object.__setattr__(
                    instance, "_total_loss_db_override", self._total_loss_db_override
                )
                object.__setattr__(
                    instance,
                    "_transmittance",
                    self.__class__._transmittance_from_loss_db(
                        self._total_loss_db_override
                    ),
                )
            # F-15: copy wavelength_nm.
            if self.wavelength_nm is not None:
                object.__setattr__(instance, "wavelength_nm", self.wavelength_nm)
            # P-07: copy _metadata_extra as MappingProxyType.
            if self._metadata_extra:
                object.__setattr__(
                    instance, "_metadata_extra",
                    types.MappingProxyType(dict(self._metadata_extra))
                )
            return instance
        # Immutable AttenuationConfig; return self (existing behaviour).
        return self

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Re-validate instance state against channel-level constraints.

        On a frozen dataclass the defining fields cannot change, so
        the parameter-range checks are technically a no-op for valid
        instances. However, this method additionally checks **derived
        state consistency**:

        * ``total_loss_db == fiber_length * attenuation_coefficient``
          (the :class:`AttenuationConfig` contract) -- skipped when
          ``_total_loss_db_override`` is set.
        * ``transmittance`` matches ``math.pow(10.0, -total_loss_db / 10.0)``
          (or 1.0 if ``total_loss_db == 0``).

        These checks catch corruption caused by e.g. pickling bugs,
        monkey-patching, or upstream :class:`AttenuationConfig` changes.

        F-3: Calls ``self._check_transmittance_health(self._transmittance)``
        to explicitly check NaN, underflow, subnormal, and range
        (previously this was not called by validate).

        F-9: Uses ``_CONTRACT_CHECK_REL_TOL`` and
        ``_CONTRACT_CHECK_ABS_TOL`` for the AttenuationConfig contract
        check, consistent with ``_check_attenuation_config_contract``.

        C-11 / C-60 fix: The underflow ``abs_tol`` uses the named
        constant :data:`_TRANSMITTANCE_UNDERFLOW_ABS_TOL` (=
        :data:`sys.float_info.min`) instead of the magic constant
        ``1e-323``.

        C-26 fix: The zero-loss branch now uses exact equality
        ``self.total_loss_db == 0.0`` (instead of
        ``is_close(self.total_loss_db, 0.0)``) for consistency with
        :meth:`_compute_transmittance`. Previously, the two code paths
        disagreed on what "zero loss" means, which could cause
        false-positive inconsistency for tiny non-zero ``total_db``.

        F-31: Uses ``_transmittance_from_loss_db`` helper for
        recomputing expected transmittance.

        When to call
        ------------
        This method is NOT called automatically by ``__post_init__``
        (it would be redundant -- the same checks already run there).
        Call it manually after any operation that might corrupt
        derived state: deserialisation from untrusted sources,
        monkey-patching, or after upgrading :mod:`qkd.datatypes` to a
        version with a different contract.

        F-30: Set ``set_validate_on_construction(True)`` to call
        :meth:`validate` at the end of every ``__post_init__`` for
        extra assurance (uses ContextVar instead of ClassVar).
        """
        self.validate_parameter(
            "distance_km", self.distance_km, MAX_DISTANCE_KM
        )
        self.validate_parameter(
            "fiber_loss_db_km", self.fiber_loss_db_km, MAX_FIBER_LOSS_DB_KM
        )

        # Derived-state consistency: total_loss_db vs alpha * L.
        # Skipped when _total_loss_db_override is set (the override is
        # by construction inconsistent with alpha * L up to IEEE-754
        # rounding).
        if self._total_loss_db_override is None:
            expected_total_db = (
                self.attenuation.fiber_length
                * self.attenuation.attenuation_coefficient
            )
            # F-9: use _CONTRACT_CHECK tolerances instead of default is_close.
            if not is_close(
                self.total_loss_db,
                expected_total_db,
                rel_tol=_CONTRACT_CHECK_REL_TOL,
                abs_tol=_CONTRACT_CHECK_ABS_TOL,
            ):
                raise ParameterValidationError(
                    f"Derived state inconsistency: total_loss_db="
                    f"{self.total_loss_db!r} does not match fiber_length * "
                    f"attenuation_coefficient={expected_total_db!r}.",
                    param_name="total_loss_db",
                    param_value=self.total_loss_db,
                    context={
                        "expected": expected_total_db,
                        "actual": self.total_loss_db,
                    },
                )

        # F-31: use _transmittance_from_loss_db helper.
        expected_transmittance = self._transmittance_from_loss_db(self.total_loss_db)

        # Allow the underflow-to-zero case (numerical artifact, not
        # corruption) but log a warning so any corruption that happens
        # to produce 0.0 is at least visible.
        if expected_transmittance == 0.0 and self.transmittance == 0.0:
            logger.warning(
                "validate(): both expected and stored transmittance are "
                "0.0 (underflow). This may be a legitimate numerical "
                "artifact OR corruption that happens to produce 0.0; "
                "manually inspect instance %r if corruption is suspected.",
                self,
            )
        elif not is_close(
            self.transmittance,
            expected_transmittance,
            rel_tol=1e-9,
            # P-02: Always use _TRANSMITTANCE_UNDERFLOW_ABS_TOL as the
            # absolute tolerance, even when expected_transmittance > 0.0.
            # The previous conditional ``abs_tol=0.0 if ... > 0.0``
            # caused false-positive validation failures for subnormal
            # transmittance values (C-11, C-60). Subnormal floats have
            # very few significant digits; a zero abs_tol makes any
            # subnormal comparison fail even if the value is numerically
            # correct. Using sys.float_info.min (the smallest normal
            # float) as abs_tol accommodates the precision limits of
            # subnormal IEEE-754 values.
            abs_tol=_TRANSMITTANCE_UNDERFLOW_ABS_TOL,
        ):
            raise ParameterValidationError(
                f"Derived state inconsistency: transmittance="
                f"{self.transmittance!r} does not match recomputed value "
                f"{expected_transmittance!r}.",
                param_name="transmittance",
                param_value=self.transmittance,
                context={
                    "expected": expected_transmittance,
                    "actual": self.transmittance,
                },
            )

        # F-3: explicitly call _check_transmittance_health.
        self._check_transmittance_health(self._transmittance)

        # F-03 fix (review channel_review_8): Explicit subnormal transmittance
        # WARNING in validate(). Previously, validate() used
        # abs_tol=sys.float_info.min for the transmittance comparison,
        # which trivially passes any subnormal value (a subnormal wrong by
        # a factor of 2 still passes because |a-b| < sys.float_info.min
        # for any subnormal a, b). Now, in addition to the consistency
        # check, we explicitly warn about subnormal transmittance and
        # its implications for QKD numerical accuracy.
        if 0.0 < self.transmittance < TRANSMITTANCE_SUBNORMAL_THRESHOLD:
            subnormal_warning = (
                f"validate(): Transmittance {self.transmittance:.3e} is subnormal "
                f"(below smallest normal float ~{TRANSMITTANCE_SUBNORMAL_THRESHOLD:.3e}). "
                f"Subnormal floats have as few as 1-2 significant mantissa bits. "
                f"QKD gain/yield formulas using this transmittance will produce "
                f"wildly inaccurate results. Total loss: {self.total_loss_db:.6g} dB."
            )
            if get_strict_mode():
                raise ParameterValidationError(
                    subnormal_warning,
                    param_name="transmittance",
                    param_value=self.transmittance,
                    context={
                        "total_loss_db": self.total_loss_db,
                        "is_subnormal": True,
                    },
                )
            _warn_physical_suspicion(subnormal_warning, stacklevel=2)

        logger.debug("Instance %r passed validation.", self)


# ---------------------------------------------------------------------------
# Post-decoration override of ``__hash__`` (review C-31)
# ---------------------------------------------------------------------------
# When ``@dataclass(frozen=True, slots=True)`` is used, the decorator
# rebuilds the class with ``__slots__`` and (because ``frozen=True``
# and ``eq=True`` by default) generates a ``__hash__`` based on the
# fields. An explicit ``__hash__ = None`` written inside the class
# body is LOST during slot rebuilding, so it cannot be used to make
# the class unhashable. We therefore override ``__hash__`` AFTER
# decoration. This makes ``hash(FiberChannel(...))`` raise
# ``TypeError: unhashable type``, which is required because the
# tolerance-based ``__eq__`` is non-transitive and therefore
# incompatible with the ``a == b => hash(a) == hash(b)`` contract.
#
# C-31: A unit test (T-21) should assert that ``hash(FiberChannel(...))``
# raises ``TypeError`` to catch regressions if CPython changes the
# dataclass slot-rebuilding behaviour.
FiberChannel.__hash__ = None  # type: ignore[assignment,method-assign]


# ---------------------------------------------------------------------------
# C-41 / F-4 fix: AttenuationOnlyFiberChannel is now a proper subclass
# (not just an alias) with ``__slots__ = ()`` (F-4: previously had bare
# ``pass`` body, no ``__slots__ = ()``, which could cause slot
# conflicts in future subclassing).
# ---------------------------------------------------------------------------
class AttenuationOnlyFiberChannel(FiberChannel):
    """Strict type marker for attenuation-only fiber channels (review C-41, S-2).

    This subclass is behaviourly identical to :class:`FiberChannel`
    but allows ``isinstance`` checks to distinguish attenuation-only
    channels from future ``FullFiberChannel`` subclasses that include
    coupling loss, detector efficiency, etc.

    F-4: ``__slots__ = ()`` added to prevent slot conflicts in
    future subclassing (previously had bare ``pass`` body).

    C-2 fix (review channel_review_8): ``__eq__`` now requires
    ``type(self) == type(other)`` so that a ``FiberChannel`` and an
    ``AttenuationOnlyFiberChannel`` with the same scalars do NOT
    compare equal. This preserves the type-level restriction that
    ``isinstance`` checks provide, making equality-based containers
    (sets, dicts) correctly distinguish the two types. ``__hash__``
    remains ``None`` (inherited from the base class) since the
    tolerance-based ``__eq__`` is non-transitive.

    Usage::

        ch = AttenuationOnlyFiberChannel(AttenuationConfig(100.0, 0.2))
        isinstance(ch, AttenuationOnlyFiberChannel)  # True
        isinstance(ch, FiberChannel)                 # True

        # A future FullFiberChannel would NOT be an AttenuationOnlyFiberChannel:
        # full_ch = FullFiberChannel(...)
        # isinstance(full_ch, AttenuationOnlyFiberChannel)  # False

    Downstream decoy-state modules can use this to verify that the
    channel model is consistent with the assumptions of the chosen
    key-rate formula. A downstream module that takes
    :class:`FiberChannel` can check
    ``isinstance(ch, AttenuationOnlyFiberChannel)`` and warn if the
    channel might be a future ``FullFiberChannel`` with additional
    loss components.

    Note: Previously this was just an alias
    (``AttenuationOnlyFiberChannel = FiberChannel``), which provided
    no type-level restriction. Existing code that constructs
    ``FiberChannel(...)`` directly will produce base-class instances,
    NOT :class:`AttenuationOnlyFiberChannel` instances. To benefit
    from the type distinction, construct via
    ``AttenuationOnlyFiberChannel(...)``.
    """

    __slots__ = ()  # F-4

    # C-2 fix: Override __eq__ to require exact type match.
    # Without this, a FiberChannel and an AttenuationOnlyFiberChannel
    # with the same scalars compare equal, defeating the type-level
    # restriction for equality-based containers (sets, dicts).
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, AttenuationOnlyFiberChannel):
            return NotImplemented
        return super().__eq__(other)


# ---------------------------------------------------------------------------
# Simulation helpers: vectorized transmittance, config-dict adapter,
# link-budget composition, and ChannelSimParams bundle.
#
# These functions bridge the gap between the pure-attenuation channel
# model (above) and the simulation driver (main_optimized.py).  They
# provide:
#
# 1. **Vectorized speed** — ``transmittance_array`` and
#    ``transmittance_scalar`` compute Beer-Lambert transmittance for
#    bulk distance arrays using NumPy, without creating individual
#    FiberChannel objects per distance point.  The scalar variant is
#    Numba-``@njit``-compatible (pure math, no Python objects).
#
# 2. **Modular coupling** — ``build_attenuation_config_from_sim_config``
#    reads from the ``main_optimized.py`` nested config dict format
#    (``config["channel"]["fiber_loss_db_km"]``) and returns an
#    ``AttenuationConfig``.  ``ChannelSimParams`` bundles all
#    channel-derived quantities needed by the simulation loop in a
#    single frozen dataclass, so main_optimized.py never needs to
#    read from the raw config dict for channel parameters.
#
# 3. **Parameter consistency** — ``link_efficiency`` computes the
#    overall detection probability ``eta = eta_channel * eta_detector
#    * eta_coupling`` in one place, ensuring the Rogers et al. (2007)
#    analytic SBR cross-check and the decoy-state gain formula use the
#    same composition.  ``ChannelSimParams`` freezes the channel's
#    transmittance, total loss, and dispersion at construction time,
#    so downstream code cannot accidentally read stale config-dict
#    values.
# ---------------------------------------------------------------------------

import numpy as np


# ---------------------------------------------------------------------------
# Vectorized Beer-Lambert transmittance
# ---------------------------------------------------------------------------

def transmittance_array(
    distances_km: np.ndarray,
    fiber_loss_db_km: float,
) -> np.ndarray:
    """Compute transmittance for an array of distances (NumPy-vectorized).

    This is the Beer-Lambert law::

        T(L) = 10 ** (-alpha * L / 10)

    applied element-wise to the entire distance array, without creating
    individual :class:`FiberChannel` objects.  The result is a NumPy
    array of transmittance values that can be passed directly to
    compiled loops (Numba ``@njit`` functions, vectorised detector
    simulations, etc.).

    Parameters
    ----------
    distances_km:
        1-D array of fibre lengths in km.  Must be non-negative and
        finite; NaN / Inf values will propagate as NaN transmittance.
    fiber_loss_db_km:
        Attenuation coefficient in dB/km.  Must be non-negative.

    Returns
    -------
    np.ndarray
        Transmittance array of the same shape as ``distances_km``.
        Each element is in ``[0.0, 1.0]`` for valid inputs.

    Notes
    -----
    - For ``fiber_loss_db_km == 0.0``, returns an array of ``1.0``
      (lossless channel for all distances).
    - For ``distances_km == 0.0``, returns ``1.0`` (zero-distance
      channel has no loss).
    - Underflow to ``0.0`` can occur for very large ``alpha * L``
      products (total loss > ~700 dB).  This matches the scalar
      ``FiberChannel._transmittance_from_loss_db`` behaviour.
    - No validation warnings are emitted (the vectorised path is
      designed for hot loops where per-point warnings would be
      prohibitively expensive).  Use :class:`FiberChannel` for
      validated single-point computation with full diagnostic output.
    """
    distances_km = np.asarray(distances_km, dtype=np.float64)
    if fiber_loss_db_km == 0.0:
        return np.ones_like(distances_km)
    total_loss_db = fiber_loss_db_km * distances_km
    # NumPy's power handles underflow gracefully (returns 0.0 for
    # very large exponents), matching math.pow behaviour.
    return np.float_power(10.0, -total_loss_db / 10.0)


def transmittance_scalar(distance_km: float, fiber_loss_db_km: float) -> float:
    """Compute transmittance for a single distance (Numba-@njit-compatible).

    This is a pure-math scalar version of the Beer-Lambert law::

        T(L) = 10 ** (-alpha * L / 10)

    It uses only ``math.pow`` and basic arithmetic — no Python objects,
    no dataclass fields, no logging.  This makes it safe to call from
    inside a Numba ``@njit`` compiled loop where Python object access
    would trigger fallback-to-object-mode overhead.

    Parameters
    ----------
    distance_km:
        Fibre length in km.  Must be non-negative and finite.
    fiber_loss_db_km:
        Attenuation coefficient in dB/km.  Must be non-negative.

    Returns
    -------
    float
        Channel transmittance in ``[0.0, 1.0]``.

    Notes
    -----
    - For ``fiber_loss_db_km == 0.0``, returns ``1.0`` (lossless).
    - For ``distance_km == 0.0``, returns ``1.0`` (zero distance).
    - No validation or warnings are emitted.  Use :class:`FiberChannel`
      for validated computation with full diagnostics.
    - Equivalent to ``FiberChannel._transmittance_from_loss_db(alpha * L)``.
    """
    if fiber_loss_db_km == 0.0 or distance_km == 0.0:
        return 1.0
    total_db = fiber_loss_db_km * distance_km
    return math.pow(10.0, -total_db / 10.0)


# ---------------------------------------------------------------------------
# Config-dict adapter: build AttenuationConfig from main_optimized format
# ---------------------------------------------------------------------------

def build_attenuation_config_from_sim_config(
    distance_km: float,
    config: Dict[str, Any],
) -> AttenuationConfig:
    """Build an :class:`AttenuationConfig` from the simulation config dict.

    This replaces the ``build_attenuation_config`` function that was
    previously defined in ``main_optimized.py``, centralising the
    config-dict-to-AttenuationConfig adapter in ``channel.py`` so that
    all channel parameter handling lives in one module.

    The config dict is expected to have the structure::

        config["channel"]["fiber_loss_db_km"] = 0.2

    which is the format used by ``DEFAULT_CONFIG`` in
    ``main_optimized.py``.

    Parameters
    ----------
    distance_km:
        Fibre length in km for this simulation point.
    config:
        The nested simulation config dict.  Must contain a
        ``"channel"`` sub-dict with a ``"fiber_loss_db_km"`` key.

    Returns
    -------
    AttenuationConfig
        A validated attenuation configuration that can be passed to
        ``FiberChannel(attenuation)`` or ``FiberChannel._fast_construct``.

    Raises
    ------
    ConfigurationError
        If ``config["channel"]`` is missing or ``fiber_loss_db_km`` is
        not a valid non-negative number.
    """
    channel_section = config.get("channel")
    if channel_section is None:
        raise ConfigurationError(
            "Config dict must contain a 'channel' section.",
            context={"available_sections": sorted(config.keys())},
        )
    raw_alpha = channel_section.get("fiber_loss_db_km")
    if raw_alpha is None:
        raise ConfigurationError(
            "Config dict 'channel' section must contain 'fiber_loss_db_km'.",
            context={"channel_keys": sorted(channel_section.keys())},
        )
    try:
        alpha = float(raw_alpha)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"channel.fiber_loss_db_km must be a number, got {raw_alpha!r}.",
            context={"fiber_loss_db_km": raw_alpha},
            cause=exc,
        ) from exc
    if alpha < 0.0:
        raise ParameterValidationError(
            f"channel.fiber_loss_db_km must be non-negative, got {alpha}.",
            param_name="fiber_loss_db_km",
            param_value=alpha,
        )
    if alpha > MAX_FIBER_LOSS_DB_KM:
        raise ParameterValidationError(
            f"channel.fiber_loss_db_km={alpha} exceeds "
            f"MAX_FIBER_LOSS_DB_KM={MAX_FIBER_LOSS_DB_KM}.",
            param_name="fiber_loss_db_km",
            param_value=alpha,
        )
    return AttenuationConfig(
        fiber_length=float(distance_km),
        attenuation_coefficient=alpha,
    )


# ---------------------------------------------------------------------------
# Link-budget composition helper
# ---------------------------------------------------------------------------

def link_efficiency(
    channel_transmittance: float,
    detector_efficiency: float,
    coupling_efficiency: float = 1.0,
) -> float:
    """Compute overall detection probability: eta = eta_ch * eta_det * eta_coup.

    In a QKD link budget, the overall detection efficiency is the
    product of three independent factors::

        eta = eta_channel * eta_detector * eta_coupling

    This composition must be performed by the caller (as documented in
    the module docstring).  This helper centralises the multiplication
    so that both the Rogers et al. (2007) analytic SBR cross-check and
    the decoy-state gain formula use the same composition, avoiding
    parameter inconsistency.

    Parameters
    ----------
    channel_transmittance:
        The fibre transmittance ``T = 10 ** (-alpha * L / 10)`` from
        :attr:`FiberChannel.transmittance` or :func:`transmittance_scalar`.
    detector_efficiency:
        The single-detector efficiency (e.g. ``det_eff_d0``).
        Typical SNSPD values: 0.1--0.3.  Typical SPAD values: 0.05--0.15.
    coupling_efficiency:
        Additional coupling / splice / connector efficiency.
        Defaults to 1.0 (no coupling loss).  Typical real-world
        values: 0.5--0.9 (0.5--3 dB total coupling/splice loss).

    Returns
    -------
    float
        The overall link efficiency in ``[0.0, 1.0]`` for valid inputs.

    Notes
    -----
    - For zero transmittance or zero detector efficiency, returns ``0.0``.
    - No validation warnings are emitted.  Use :class:`FiberChannel` and
      :class:`SinglePhotonDetector` for validated single-point computation.
    - This is the ``L`` parameter in the Rogers et al. (2007) notation
      (Eq. 8), representing the probability that a photon emitted by
      Alice is detected at Bob.
    """
    result = float(channel_transmittance) * float(detector_efficiency) * float(coupling_efficiency)
    # Clamp to [0, 1] for physical validity.  Negative inputs would
    # produce a negative result, which is unphysical.
    if result < 0.0:
        return 0.0
    if result > 1.0:
        return 1.0
    return result


# ---------------------------------------------------------------------------
# ChannelSimParams: frozen bundle of channel-derived quantities
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ChannelSimParams:
    """Frozen bundle of channel-derived quantities for simulation use.

    This dataclass captures all channel-related values that the
    simulation driver needs at runtime:

    - ``transmittance`` — the Beer-Lambert channel transmittance
    - ``total_loss_db`` — the total channel loss in dB
    - ``fiber_loss_db_km`` — the attenuation coefficient
    - ``distance_km`` — the fibre length for this simulation point
    - ``dispersion_parameter_ps_nm_km`` — chromatic dispersion (optional,
      from config dict, not modeled by FiberChannel)

    By freezing these values at construction time (from a validated
    :class:`FiberChannel` or from :func:`transmittance_scalar`), we
    ensure that downstream simulation code (finite-key scaling, Rogers
    analytic SBR, detector simulation) always reads from the same
    consistent snapshot.  No stale config-dict reads are possible.

    Construction
    ------------
    Use the factory :func:`build_channel_sim_params` rather than
    constructing directly — the factory validates inputs and fills
    default values from the config dict.
    """
    transmittance: float
    total_loss_db: float
    fiber_loss_db_km: float
    distance_km: float
    dispersion_parameter_ps_nm_km: float = 0.0
    wavelength_nm: Optional[float] = None


def build_channel_sim_params(
    channel: FiberChannel,
    config: Dict[str, Any],
) -> ChannelSimParams:
    """Build :class:`ChannelSimParams` from a validated FiberChannel and config dict.

    This is the **single approved path** for main_optimized.py to
    extract all channel-derived quantities for the simulation loop.
    It reads dispersion from the config dict (since FiberChannel does
    not model chromatic dispersion) but takes transmittance, loss, and
    attenuation from the validated FiberChannel object.

    Parameters
    ----------
    channel:
        A validated :class:`FiberChannel` for the current distance point.
    config:
        The simulation config dict (for dispersion and optional
        wavelength).

    Returns
    -------
    ChannelSimParams
        A frozen bundle of all channel-derived quantities.
    """
    channel_section = config.get("channel", {})
    dispersion = float(channel_section.get("dispersion_parameter_ps_nm_km", 0.0))

    # Optional wavelength: read from FiberChannel if set, else from config.
    wavelength = channel.wavelength_nm
    if wavelength is None:
        wavelength = channel_section.get("wavelength_nm")
        if wavelength is not None:
            wavelength = float(wavelength)

    return ChannelSimParams(
        transmittance=channel.transmittance,
        total_loss_db=channel.total_loss_db,
        fiber_loss_db_km=channel.fiber_loss_db_km,
        distance_km=channel.distance_km,
        dispersion_parameter_ps_nm_km=dispersion,
        wavelength_nm=wavelength,
    )


# ---------------------------------------------------------------------------
# Module-level STRICT_MODE back-compat shim (review C-30, F-01 fix)
# ---------------------------------------------------------------------------
# Legacy code may read ``channel.STRICT_MODE``. Because the underlying
# storage is now a :class:`contextvars.ContextVar`, we provide a
# module-level ``__getattr__`` (PEP 562, Python 3.7+) that proxies
# reads through :func:`get_strict_mode` and emits a
# :class:`DeprecationWarning` on access. New code should call
# :func:`get_strict_mode` / :func:`set_strict_mode` directly.
#
# F-01 fix (review channel_review_8): The previous _ChannelModule
# approach (replacing sys.modules[__name__] with a custom module class)
# was removed because it is extremely fragile and breaks:
#   (a) pickle -- module references may not be pickle-compatible
#   (b) pytest -- module reloading for test isolation may fail
#   (c) introspection -- inspect.getmodule() may reference wrong object
#   (d) type checkers -- mypy, pylint treat modules as ModuleType
#
# The silent-no-op bug (channel.STRICT_MODE = True not working) is
# accepted as a documented limitation of PEP 562, which does NOT support
# module-level __setattr__. Users MUST use set_strict_mode() to change
# strict mode at runtime. This is the lesser evil compared to the
# module-replacement hack which affects ALL module-level operations.
#
def __getattr__(name: str) -> Any:
    """Module-level attribute access for deprecated ``STRICT_MODE`` (PEP 562).

    F-01 fix (review channel_review_8): Reverted from _ChannelModule
    back to PEP 562 __getattr__. The _ChannelModule approach was more
    fragile than the silent-no-op bug it was trying to fix.

    IMPORTANT: PEP 562 does NOT support module-level __setattr__.
    Setting ``channel.STRICT_MODE = ...`` directly will silently
    create a new module attribute WITHOUT changing the strict-mode
    ContextVar. Always use :func:`set_strict_mode` to change strict
    mode at runtime.
    """
    if name == "STRICT_MODE":
        warnings.warn(
            "qkd.channel.STRICT_MODE is deprecated; use "
            "qkd.channel.get_strict_mode() / set_strict_mode() instead. "
            "Setting channel.STRICT_MODE = ... is a NO-OP (use "
            "set_strict_mode() instead); PEP 562 does not support "
            "module-level __setattr__.",
            DeprecationWarning,
            stacklevel=2,
        )
        return get_strict_mode()
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


# ---------------------------------------------------------------------------
# F-7: Check that STRICT_MODE has not been shadowed by a direct
# assignment. If ``channel.STRICT_MODE = True`` was executed, it
# silently created a module attribute that shadows __getattr__
# without toggling the ContextVar. Emit a RuntimeWarning.
# ---------------------------------------------------------------------------
if "STRICT_MODE" in globals():
    warnings.warn(
        "channel.STRICT_MODE has been shadowed by a direct assignment "
        "and no longer reflects the underlying ContextVar. Use "
        "set_strict_mode() instead.",
        RuntimeWarning,
        stacklevel=1,
    )
