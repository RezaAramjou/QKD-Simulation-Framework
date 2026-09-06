# qkd/__init__.py
# -*- coding: utf-8 -*-
"""
qkd — Quantum Key Distribution simulation framework.

This package provides a modular and extensible framework for simulating
Quantum Key Distribution (QKD) protocols.

Public, stable API (highest-level objects):
- QKDParams: The high-level parameter container.
- Protocol classes (e.g., BB84DecoyProtocol, MDIQKDProtocol).
- Component classes (e.g., PoissonSource, FiberChannel, ThresholdDetector).
- SimulationResults: A data container for simulation outputs.
- __version__: The package version string.

This package lazily imports heavy submodules on attribute access to keep import
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# Expose version information dynamically from package metadata (PEP 621)
# This is the modern, standard way to handle package versioning.
try:
    # Available in Python 3.8+
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Fallback for older Python versions or environments, requires `importlib-metadata` backport
    try:
        from importlib_metadata import version, PackageNotFoundError  # type: ignore
    except ImportError:
        # If all else fails, provide placeholders that are compatible with mypy.
        def version(distribution_name: str) -> str:
            """Placeholder version function."""
            return "0.0.0+unknown"

        class PackageNotFoundError(Exception):  # type: ignore[no-redef]
            """Placeholder exception for when importlib.metadata is not available."""
            pass

# Define the public API of the package. Using a tuple makes it immutable.
# This list should be the single source of truth for what is considered stable.
__all__ = (
    "QKDParams",
    "BB84DecoyProtocol",
    "MDIQKDProtocol",
    "PoissonSource",
    "FiberChannel",
    "ThresholdDetector",
    "SimulationResults",
    "__version__",
)

# Set the __version__ attribute for runtime access (e.g., qkd.__version__)
try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # This happens if the package is not installed correctly (e.g., editable mode)
    __version__ = "0.0.0+unknown"


# For static type checkers (like Mypy), we import the symbols directly.
# The `if TYPE_CHECKING:` block is not executed at runtime, so it has zero
# performance cost and avoids circular import errors.
if TYPE_CHECKING:
    from .channel import FiberChannel
    from .datatypes import SimulationResults
    from .detectors import ThresholdDetector
    from .params import QKDParams
    from .protocols import BB84DecoyProtocol, MDIQKDProtocol
    from .sources import PoissonSource


# Lazy attribute resolution using PEP 562 (__getattr__ on modules).
# This is the core of the lazy-loading mechanism. It maps the public attribute
# names to the modules where they can be found.
_LAZY_EXPORTS = {
    "BB84DecoyProtocol": ".protocols",
    "MDIQKDProtocol": ".protocols",
    "PoissonSource": ".sources",
    "FiberChannel": ".channel",
    "ThresholdDetector": ".detectors",
    "SimulationResults": ".datatypes",
    "cli": ".cli",
}


def __getattr__(name: str) -> Any:
    """
    Lazy-load public attributes to reduce import-time cost and prevent circular imports.

    This function is called by the Python interpreter only when an attribute
    is accessed that is not already in the module's globals.
    """
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    # Import the submodule where the attribute is defined.
    # The `package=__name__` argument makes it a relative import.
    module = importlib.import_module(_LAZY_EXPORTS[name], package=__name__)

    # Get the attribute from the now-imported submodule.
    value = getattr(module, name)

    # Cache the attribute in the package's globals dictionary. This ensures
    # that `__getattr__` is only called once for each attribute.
    globals()[name] = value

    return value


def __dir__() -> list[str]:
    """
    Provide a complete list of module attributes for tools like `dir()` and auto-completion.
    """
    return sorted(__all__)


# SCM Backward Compat Hook
from . import scm_shim
