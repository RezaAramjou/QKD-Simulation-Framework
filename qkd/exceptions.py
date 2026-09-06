# -*- coding: utf-8 -*-

"""
Custom, research-grade exceptions for the QKD simulation framework.

This module defines a comprehensive, well-structured exception hierarchy for the
framework. It is designed to be:
  - Hierarchical: A single base exception (`QKDException`) allows for catching all
    framework-specific errors.
  - Context-Rich: Exceptions carry structured, typed context beyond a simple
    message, aiding programmatic handling and debugging.
  - Serializable: A `.to_dict()` method provides a JSON-safe representation for
    structured logging, telemetry, and reproducible error reporting.
  - Stable: A machine-readable `ErrorCode` enum provides stable identifiers for
    automated error handling.
  - User-Friendly: Enhanced string representations, clear docstrings, and
    convenience factories (e.g., for SciPy results) improve developer experience.
  - Robust: Exceptions are picklable for use across multiprocessing boundaries,
    with careful sanitization of attached diagnostic data.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, Optional, Mapping, Type
from .utils.utils import sanitize_for_serialization


__version__ = "1.1.0"

# Exported names for the module's public API
__all__ = [
    "QKDException",
    "ParameterValidationError",
    "ConfigurationError",
    "QKDSimulationError",
    "LPFailureError",
    "SimulationInterruptedError",
    "ErrorCode",
]


class ErrorCode(str, Enum):
    """
    Stable, machine-readable error codes for QKD exceptions.
    Using an Enum ensures consistency and prevents typos.
    """
    # General Errors
    UNCATEGORIZED = "ERR_UNCATEGORIZED"
    INTERRUPTED = "ERR_INTERRUPTED"

    # Configuration and Parameter Errors
    CONFIG = "ERR_CONFIG"
    PARAM_VALIDATION = "ERR_PARAM_VALIDATION"

    # Runtime Simulation Errors
    SIMULATION = "ERR_SIMULATION"
    LP_SOLVER_FAILED = "ERR_LP_SOLVER_FAILED"


class QKDException(Exception):
    """
    Base class for all QKD framework exceptions.

    This class provides core functionality inherited by all other custom exceptions,
    including structured context, error codes, serialization, and logging helpers.

    Attributes:
        message (str): Human-readable error message.
        code (ErrorCode): Machine-readable error code.
        context (Dict[str, Any]): Additional structured context for logging.
    """

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode = ErrorCode.UNCATEGORIZED,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.context = {} if context is None else context

        if cause:
            self.__cause__ = cause

    def __str__(self) -> str:
        """Provides a clear, informative string representation."""
        return f"{self.__class__.__name__} ({self.code.value}): {self.message}"

    def __repr__(self) -> str:
        """Provides an unambiguous representation for developers."""
        context_keys = f", context_keys={list(self.context.keys())!r}" if self.context else ""
        return f"{self.__class__.__name__}({self.message!r}, code={self.code!r}{context_keys})"

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a JSON-serializable dictionary representing the exception.
        This is ideal for structured logging and telemetry.
        """
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "code": self.code.value,
            "context": sanitize_for_serialization(self.context),
            "cause": repr(self.__cause__) if self.__cause__ else None,
        }

    def log(self, logger: logging.Logger, level: int = logging.ERROR) -> None:
        """
        Logs the exception's structured data using the provided logger.

        Args:
            logger: The logger instance to use.
            level: The logging level (e.g., logging.ERROR, logging.WARNING).
        """
        logger.log(level, self.message, extra={"qkd_error": self.to_dict()})


class ParameterValidationError(QKDException, ValueError):
    """
    Raised when a simulation parameter or user input is invalid.
    Inherits from ValueError for semantic compatibility.

    Example:
        if not 0 <= attenuation <= 1:
            raise ParameterValidationError(
                "Attenuation must be between 0 and 1",
                param_name="attenuation",
                param_value=attenuation
            )
    """

    def __init__(
        self,
        message: str,
        *,
        param_name: Optional[str] = None,
        param_value: Any = None,
        code: ErrorCode = ErrorCode.PARAM_VALIDATION,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ):
        ctx = context or {}
        if param_name is not None:
            ctx.setdefault("param_name", param_name)
        if param_value is not None:
            ctx.setdefault("param_value", param_value)
        super().__init__(message, code=code, context=ctx, cause=cause)


class ConfigurationError(QKDException):
    """
    Raised for errors in configuration, such as missing files,
    incompatible components, or structural problems in config files.
    """

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode = ErrorCode.CONFIG,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ):
        super().__init__(message, code=code, context=context, cause=cause)


class QKDSimulationError(QKDException, RuntimeError):
    """
    Generic runtime error during simulation execution.
    Inherits from RuntimeError for semantic compatibility.
    """

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode = ErrorCode.SIMULATION,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ):
        super().__init__(message, code=code, context=context, cause=cause)


class LPFailureError(QKDSimulationError):
    """
    Raised when a linear programming (LP) solver fails to find a solution.
    This exception captures structured diagnostics from the solver.

    Attributes:
        status (Optional[Any]): The raw status from the solver (e.g., int or str).
        solver_message (Optional[str]): The human-readable message from the solver.
        diagnostics (Dict[str, Any]): A sanitized dictionary of solver diagnostics.
    """

    def __init__(
        self,
        message: str,
        *,
        status: Any = None,
        solver_message: Optional[str] = None,
        diagnostics: Optional[Mapping[str, Any]] = None,
        code: ErrorCode = ErrorCode.LP_SOLVER_FAILED,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ):
        ctx = context or {}
        ctx.setdefault("lp_status", status)
        ctx.setdefault("lp_solver_message", solver_message)
        ctx.setdefault("lp_diagnostics", diagnostics)

        super().__init__(message, code=code, context=ctx, cause=cause)
        self.status = status
        self.solver_message = solver_message
        self.diagnostics = sanitize_for_serialization(diagnostics or {})

    @classmethod
    def from_scipy_result(
        cls: Type[LPFailureError],
        result: Any,
        **kwargs: Any,
    ) -> LPFailureError:
        """
        Convenience factory to create an LPFailureError from a SciPy result object.
        `scipy.optimize.linprog` returns an object with `status`, `message`, etc.
        """
        # Extract attributes safely, providing defaults if they don't exist.
        status = getattr(result, "status", "unknown")
        solver_message = getattr(result, "message", "No message provided.")

        # Create a diagnostics dict from the result object's attributes.
        diagnostics = result.__dict__ if hasattr(result, "__dict__") else {}

        message = f"LP solver failed with status '{status}': {solver_message}"

        return cls(
            message,
            status=status,
            solver_message=solver_message,
            diagnostics=diagnostics,
            **kwargs,
        )


class SimulationInterruptedError(QKDSimulationError):
    """
    Raised when a simulation is intentionally interrupted (e.g., by Ctrl+C).
    This allows for graceful shutdown logic to distinguish from unexpected errors.
    """

    def __init__(
        self,
        message: str = "Simulation interrupted by user.",
        *,
        code: ErrorCode = ErrorCode.INTERRUPTED,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ):
        super().__init__(message, code=code, context=context, cause=cause)
