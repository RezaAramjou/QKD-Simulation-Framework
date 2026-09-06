# -*- coding: utf-8 -*-
"""
Input/Output and serialization helper functions.

This module is the single source of truth for file I/O, serialization, and
deserialization across the QKD simulation package. It provides:

- :func:`save_results_json`: atomic, durable save of a ``SimulationResults``
  object to JSON (optionally gzipped, with optional SHA256 metadata).
- :func:`safe_json_dumps`: best-effort JSON serialization with a TypeError
  fallback, suitable for logging / error-context rendering.
- :func:`parse_json_strict`: JSON deserialization with mandatory type
  validation and consistent ``ConfigurationError`` mapping.
- :func:`open_atomic_text`: context manager for atomic, durable text-file
  writes (used for streaming CSV output and any other text artifacts).

All public helpers share the same robustness guarantees: path validation,
parent-directory creation, atomic temp-file write + rename, fsync on POSIX,
secure temp-file permissions, and consistent OSError-to-QKD-exception mapping.
"""
import os
import json
import tempfile
import logging
import gzip
import io
import hashlib
import errno
from contextlib import contextmanager
from typing import Any, Dict, IO, Optional, Union, TYPE_CHECKING

# TYPE_CHECKING block for type hints
if TYPE_CHECKING:
    from .datatypes import SimulationResults

# --- IMPROVEMENT: Centralized Imports from the constants module ---
# Import the canonical serialization function and constants utility.
# This eliminates redundant code and ensures consistency.
from .constants import as_dict

# --- IMPROVEMENT: Custom exceptions for consistent error handling ---
from .exceptions import (
    ConfigurationError,
    ParameterValidationError,
    QKDSimulationError,
)

logger = logging.getLogger(__name__)

# --- Version bumped to reflect significant improvements ---
__version__ = "3.1.0"  # Added safe_json_dumps, parse_json_strict, open_atomic_text

__all__ = [
    "save_results_json",
    "safe_json_dumps",
    "parse_json_strict",
    "open_atomic_text",
]


def save_results_json(
    results: "SimulationResults",
    path: str,
    *,
    overwrite: bool = True,
    compress: bool = False,
    sort_keys: bool = True,
    return_metadata: bool = False,
) -> Union[str, Dict[str, Any]]:
    """
    Atomically and securely writes a results object and simulation constants to a JSON file.

    This function implements a robust save operation by:
    1.  Embedding simulation constants from the `constants` module for reproducibility.
    2.  Using the object's dedicated `.to_dict()` method for safe serialization.
    3.  Using `tempfile.mkstemp` for atomic temporary file creation with secure permissions.
    4.  Writing to a temporary file in the same directory as the target path.
    5.  Correctly handling gzip compression with binary file objects.
    6.  Ensuring file and directory metadata are synced to disk for durability on POSIX.
    7.  Cleaning up the temporary file on any error.
    8.  Offering optional gzip compression, a no-overwrite policy, and metadata return.

    Args:
        results: The SimulationResults object to serialize and save.
        path: The final destination file path.
        overwrite: If False, raises FileExistsError if the destination path already exists.
        compress: If True, saves the file with gzip compression (and a .gz extension).
        sort_keys: If True, sorts dictionary keys for deterministic output.
        return_metadata: If True, returns a dictionary with path, size, and SHA256 hash.

    Returns:
        The absolute path to the saved file, or a dictionary with file metadata.

    Raises:
        ConfigurationError: If `overwrite` is False and path exists, or for permission errors.
        ParameterValidationError: If the path is a directory or `results` contains unserializable data.
        QKDSimulationError: Wraps any underlying I/O or other unexpected errors.

    Note on Concurrency (TOCTOU):
        When `overwrite=False`, a race condition (Time-of-check to time-of-use) exists.
        Another process could create the file between the existence check and the final
        `os.replace`. For strict non-overwrite guarantees, application-level locking
        is recommended.
    """
    full_path = os.path.abspath(path)
    
    if compress and not full_path.endswith(".gz"):
        full_path += ".gz"

    if not overwrite and (os.path.exists(full_path) or os.path.islink(full_path)):
        raise ConfigurationError(
            "Destination path exists and overwrite is False.",
            context={"path": full_path}
        )

    if os.path.isdir(full_path):
        raise ParameterValidationError(
            "Target path is a directory, not a file.",
            param_name="path",
            param_value=full_path
        )

    dir_path = os.path.dirname(full_path) or os.getcwd()
    os.makedirs(dir_path, exist_ok=True)

    suffix = ".json.gz.tmp" if compress else ".json.tmp"
    
    tmp_fd = -1
    tmp_path = None
    try:
        # --- IMPROVEMENT: Create a structured payload for reproducibility ---
        # The top-level object now includes the simulation constants that
        # were used to generate the results. The SimulationResults object
        # is serialized using its own to_dict() method for correctness.
        full_payload = {
            "simulation_constants": as_dict(),
            "results": results.to_dict()
        }

        tmp_fd, tmp_path = tempfile.mkstemp(prefix=".", suffix=suffix, dir=dir_path)

        if os.name == "posix":
            try:
                os.fchmod(tmp_fd, 0o600)
            except OSError:
                try:
                    os.chmod(tmp_path, 0o600)
                except OSError:
                    logger.debug("Failed to set secure permissions on temp file", exc_info=True)

        if compress:
            with os.fdopen(tmp_fd, "wb") as binary_f:
                tmp_fd = -1
                with gzip.GzipFile(fileobj=binary_f, mode="wb") as gz_f:
                    with io.TextIOWrapper(gz_f, encoding="utf-8") as text_writer:
                        json.dump(full_payload, text_writer, indent=4, ensure_ascii=False, sort_keys=sort_keys)
                os.fsync(binary_f.fileno())
        else:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                tmp_fd = -1
                json.dump(full_payload, f, indent=4, ensure_ascii=False, sort_keys=sort_keys)
                f.flush()
                os.fsync(f.fileno())

        os.replace(tmp_path, full_path)
        tmp_path = None

        if os.name == "posix":
            dir_fd = -1
            try:
                dir_fd = os.open(dir_path, os.O_RDONLY)
                os.fsync(dir_fd)
            except OSError as e:
                logger.warning(f"Could not fsync directory {dir_path}: {e}")
            finally:
                if dir_fd != -1:
                    os.close(dir_fd)

        file_size_bytes = os.path.getsize(full_path)
        logger.info(f"Results saved to file: {full_path} ({file_size_bytes / 1024:.2f} KB)")
        
        if return_metadata:
            sha256 = hashlib.sha256()
            with open(full_path, 'rb') as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
            return {
                "path": full_path,
                "size_bytes": file_size_bytes,
                "sha256": sha256.hexdigest(),
            }
        return full_path

    except TypeError as e:
        # This now catches errors from the object's to_dict() method.
        raise ParameterValidationError(
            "Failed to serialize the results object to JSON. It may contain an unsupported data type.",
            param_name="results",
            cause=e,
        ) from e
    except OSError as e:
        if e.errno == errno.EACCES:
            raise ConfigurationError(
                f"Permission denied while trying to write to '{dir_path}'. Check directory permissions.",
                context={"path": full_path},
                cause=e,
            ) from e
        elif e.errno == errno.ENOSPC:
            raise QKDSimulationError(
                "Failed to save results: No space left on device.",
                context={"path": full_path},
                cause=e,
            ) from e
        else:
            raise QKDSimulationError(
                f"An operating system error occurred during file save: {e.strerror}",
                context={"path": full_path, "errno": e.errno},
                cause=e,
            ) from e
    except Exception as e:
        logger.error(f"Failed to save results to {path}", exc_info=True)
        raise QKDSimulationError(
            f"An unexpected error occurred while saving results to {path}",
            cause=e,
            context={"path": path},
        ) from e
    finally:
        if tmp_fd != -1:
            os.close(tmp_fd)
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError as e:
                logger.warning(f"Failed to remove temporary file {tmp_path} during cleanup: {e}")


# =============================================================================
# Generic I/O helpers (single source of truth for non-SimulationResults I/O)
# =============================================================================
# These helpers exist so that other modules (e.g., main_optimized.py) never
# need to roll their own file-write / JSON-serialize / JSON-parse logic and
# can instead route every I/O concern through this module. Each helper shares
# the same atomic-write, error-mapping, and POSIX-durability guarantees as
# ``save_results_json`` above.


def safe_json_dumps(data: Any) -> str:
    """
    Best-effort JSON serialization with a TypeError fallback.

    Attempts to serialize *data* to JSON. If *data* contains objects that
    ``json`` cannot natively encode (e.g., numpy arrays, custom classes
    without a ``default`` hook), the function falls back to ``str(data)``
    rather than raising. This makes it safe to use inside logging or
    error-reporting code paths, where a secondary ``TypeError`` from the
    serializer would mask the original error.

    Args:
        data: The object to serialize.

    Returns:
        A JSON string (or, on fallback, the result of ``str(data)`` wrapped
        in a JSON string).
    """
    try:
        return json.dumps(data, ensure_ascii=False)
    except TypeError:
        return json.dumps(str(data), ensure_ascii=False)


def parse_json_strict(
    raw: str,
    *,
    expected_type: type = dict,
    error_message: Optional[str] = None,
) -> Any:
    """
    Parse *raw* as JSON and validate that the result is an instance of
    *expected_type*.

    Centralizes JSON deserialization + type validation + error wrapping so
    that callers (e.g., CLI argument parsers reading a ``--sweep-json``
    argument) get the same consistent ``ConfigurationError`` mapping
    regardless of where the JSON originated.

    Args:
        raw: The JSON string to parse.
        expected_type: The type the parsed value must be an instance of.
            Defaults to ``dict``.
        error_message: Optional override for the error message raised when
            parsing or type validation fails.

    Returns:
        The parsed JSON value (guaranteed to be an instance of
        *expected_type*).

    Raises:
        ConfigurationError: If *raw* is not valid JSON or the parsed value
            is not an instance of *expected_type*.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            error_message or "Input string is not valid JSON.",
            context={"raw_prefix": raw[:200]} if isinstance(raw, str) else None,
            cause=exc,
        ) from exc

    if not isinstance(parsed, expected_type):
        raise ConfigurationError(
            error_message
            or f"JSON input must decode to a {expected_type.__name__}, "
            f"got {type(parsed).__name__}.",
            context={"expected_type": expected_type.__name__,
                     "actual_type": type(parsed).__name__},
        )

    return parsed


@contextmanager
def open_atomic_text(
    path: str,
    mode: str = "w",
    *,
    encoding: str = "utf-8",
    newline: str = "",
    overwrite: bool = True,
    fsync: bool = True,
) -> "IO[str]":
    """
    Context manager that opens a text file for writing with atomic-rename
    semantics.

    Mirrors the robustness guarantees of :func:`save_results_json`:

    - Validates that the target path is not a directory.
    - Creates parent directories as needed.
    - Writes to a temp file in the same directory as the target.
    - On clean exit, fsyncs the file (and the parent directory on POSIX)
      and atomically renames the temp file to the target path.
    - On any exception inside the ``with`` block, the temp file is removed
      and the target path is left untouched. The original exception is
      re-raised unchanged (we never wrap user code exceptions).
    - ``OSError`` raised by our own setup / commit code is mapped to
      ``ConfigurationError`` / ``QKDSimulationError`` for consistent error
      handling.

    Intended as a drop-in replacement for ``open(path, "w", ...)`` in
    streaming-write scenarios (e.g., writing CSV rows as they arrive from a
    multiprocessing pool)::

        with open_atomic_text(csv_filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    Args:
        path: The final destination file path.
        mode: Write-style file mode (``"w"``, ``"x"``, or ``"a"``). Defaults
            to ``"w"``.
        encoding: Text encoding passed to ``os.fdopen``. Defaults to
            ``"utf-8"``.
        newline: Newline handling passed to ``os.fdopen``. Defaults to
            ``""`` (no translation), which is the correct value for
            ``csv.writer`` on all platforms.
        overwrite: If ``False``, raises ``ConfigurationError`` when the
            destination path already exists.
        fsync: If ``True`` (default), fsyncs the file and the parent
            directory on POSIX before the atomic rename.

    Yields:
        A writable text file object.

    Raises:
        ConfigurationError: If ``overwrite`` is False and the path exists,
            or for permission errors.
        ParameterValidationError: If the path is a directory, or *mode* is
            not a write-style mode.
        QKDSimulationError: Wraps any other underlying I/O error from this
            function's own setup / commit code.
    """
    if "w" not in mode and "x" not in mode and "a" not in mode:
        raise ParameterValidationError(
            "open_atomic_text() requires a write-style mode ('w', 'x', or 'a').",
            param_name="mode",
            param_value=mode,
        )

    full_path = os.path.abspath(path)

    if not overwrite and (os.path.exists(full_path) or os.path.islink(full_path)):
        raise ConfigurationError(
            "Destination path exists and overwrite is False.",
            context={"path": full_path},
        )

    if os.path.isdir(full_path):
        raise ParameterValidationError(
            "Target path is a directory, not a file.",
            param_name="path",
            param_value=full_path,
        )

    dir_path = os.path.dirname(full_path) or os.getcwd()
    os.makedirs(dir_path, exist_ok=True)

    def _wrap_oserror(exc: OSError) -> QKDSimulationError:
        if exc.errno == errno.EACCES:
            return ConfigurationError(
                f"Permission denied while trying to write to '{dir_path}'. "
                f"Check directory permissions.",
                context={"path": full_path},
                cause=exc,
            )
        if exc.errno == errno.ENOSPC:
            return QKDSimulationError(
                "Failed to write file: No space left on device.",
                context={"path": full_path},
                cause=exc,
            )
        return QKDSimulationError(
            f"An operating system error occurred during file write: {exc.strerror}",
            context={"path": full_path, "errno": exc.errno},
            cause=exc,
        )

    tmp_fd = -1
    tmp_path: Optional[str] = None
    file_obj: Optional[IO[str]] = None
    fd_owned_by_us = True  # whether we still own tmp_fd (False once os.fdopen takes it)
    try:
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(prefix=".", suffix=".tmp", dir=dir_path)
        except OSError as e:
            raise _wrap_oserror(e) from e

        if os.name == "posix":
            try:
                os.fchmod(tmp_fd, 0o600)
            except OSError:
                try:
                    os.chmod(tmp_path, 0o600)
                except OSError:
                    logger.debug(
                        "Failed to set secure permissions on temp file",
                        exc_info=True,
                    )

        try:
            file_obj = os.fdopen(tmp_fd, mode, newline=newline, encoding=encoding)
            fd_owned_by_us = False  # fd is now owned by file_obj
        except OSError as e:
            raise _wrap_oserror(e) from e

        # ---- yield: user code runs here. Any exception propagates through
        # the finally below unchanged (we never wrap user exceptions).
        yield file_obj

        # ---- clean exit: flush, fsync, close, atomic rename. These are our
        # own I/O operations, so OSError here IS wrapped.
        try:
            file_obj.flush()
            if fsync:
                os.fsync(file_obj.fileno())
            file_obj.close()
            file_obj = None
        except OSError as e:
            raise _wrap_oserror(e) from e

        try:
            os.replace(tmp_path, full_path)
            tmp_path = None  # committed
        except OSError as e:
            raise _wrap_oserror(e) from e

        if fsync and os.name == "posix":
            dir_fd = -1
            try:
                dir_fd = os.open(dir_path, os.O_RDONLY)
                os.fsync(dir_fd)
            except OSError as e:
                logger.warning(f"Could not fsync directory {dir_path}: {e}")
            finally:
                if dir_fd != -1:
                    os.close(dir_fd)

        file_size_bytes = os.path.getsize(full_path)
        logger.info(
            f"File saved atomically: {full_path} ({file_size_bytes / 1024:.2f} KB)"
        )
    finally:
        # Best-effort cleanup. We suppress secondary OSError here so we never
        # mask the original exception (whether it's a wrapped QKD exception,
        # an OSError from setup/commit, or a user exception from inside the
        # `with` block).
        if file_obj is not None and not file_obj.closed:
            try:
                file_obj.close()
            except OSError:
                pass
        if fd_owned_by_us and tmp_fd != -1:
            try:
                os.close(tmp_fd)
            except OSError:
                pass
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError as e:
                logger.warning(
                    f"Failed to remove temporary file {tmp_path} during cleanup: {e}"
                )
