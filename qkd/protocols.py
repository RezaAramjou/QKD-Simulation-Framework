# qkd/protocols.py
# -*- coding: utf-8 -*-
"""
Implements Quantum Key Distribution (QKD) protocols with a focus on correctness,
robustness, and clear separation of concerns.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any

import numpy as np
from numpy.random import Generator, PCG64, SeedSequence
from numpy.typing import NDArray

from .sources import PoissonSource, OpticalSource
from . import exceptions
from . import constants
from .datatypes import DoubleClickPolicy, PulseTypeConfig
from .utils import math as qkd_math

__version__ = "8.2.2"

__all__ = [
    "BB84DecoyProtocol",
    "MDIQKDProtocol",
    "B92Protocol",
    "RedundantTransmissionProtocol",
    "Protocol",
    "BB84PreparedStates",
    "MDIPreparedStates",
    "B92PreparedStates",
    "DetectionResults",
    "SiftingResults",
    "make_worker_rngs",
    "sample_for_parameter_estimation",
]

logger = logging.getLogger(__name__)

BitArray = NDArray[np.uint8]
IndexArray = NDArray[np.int64]
MaskArray = NDArray[np.bool_]
RNG = Generator

Z_BASIS: int = 0
X_BASIS: int = 1
DETECTOR_0_KEY: str = "click0"
DETECTOR_1_KEY: str = "click1"


def _validate_and_freeze_array(
    arr: np.ndarray,
    name: str,
    expected_dtype: type,
    expected_ndim: int = 1,
) -> np.ndarray:
    """Validates array properties, coerces type, and makes it immutable."""
    dtype_obj = np.dtype(expected_dtype)
    arr = np.ascontiguousarray(arr)

    if not np.issubdtype(arr.dtype, dtype_obj):
        # Validate values on the *original* array before coercing dtype,
        # to prevent silent truncation/wrapping of invalid scientific inputs.
        if dtype_obj == np.uint8:
            if not np.all((arr == 0) | (arr == 1)):
                raise exceptions.ParameterValidationError(
                    f"Array '{name}' contains values outside {{0, 1}} before "
                    f"uint8 coercion. Rejecting to prevent silent data corruption.",
                    param_name=name,
                )
        arr = arr.astype(dtype_obj)

    if arr.ndim != expected_ndim:
        raise exceptions.ParameterValidationError(
            f"Array '{name}' has incorrect dimensions.",
            param_name=name,
            context={
                "expected_ndim": expected_ndim,
                "actual_ndim": arr.ndim,
                "shape": arr.shape,
            },
        )

    arr.flags.writeable = False
    return arr


def _validate_binary_array(arr: np.ndarray, name: str) -> None:
    """Ensures an array contains only binary values 0 and 1."""
    if arr.dtype.kind == 'f':
        if not np.all(np.isfinite(arr)):
            raise exceptions.ParameterValidationError(
                f"Array '{name}' contains NaN or inf values.",
                param_name=name,
            )
        if not np.all((arr == 0.0) | (arr == 1.0)):
            raise exceptions.ParameterValidationError(
                f"Array '{name}' contains non-binary floating-point values.",
                param_name=name,
            )
    if not np.all((arr == 0) | (arr == 1)):
        raise exceptions.ParameterValidationError(
            "Bit and basis arrays must contain only 0s and 1s.",
            param_name=name,
        )


def _normalized_source_probabilities(source: OpticalSource) -> np.ndarray:
    """Returns normalized pulse-type probabilities from source config."""
    probs = np.array([pc.probability for pc in source.pulse_configs], dtype=float)
    prob_sum = float(np.sum(probs))
    if prob_sum <= 0.0:
        raise exceptions.ConfigurationError(
            "The sum of pulse probabilities in the source must be positive."
        )
    probs /= prob_sum
    return probs


def make_worker_rngs(master_seed: Union[int, SeedSequence], num_workers: int) -> List[RNG]:
    """Creates a list of independent, high-quality RNGs for parallel processing."""
    if isinstance(master_seed, Generator):
        raise exceptions.ConfigurationError(
            "Cannot create worker RNGs from a Generator instance. Pass an int or SeedSequence."
        )
    if not isinstance(num_workers, int) or num_workers < 1:
        raise exceptions.ParameterValidationError(
            "num_workers must be a positive integer.",
            param_name="num_workers",
            param_value=num_workers,
        )

    if isinstance(master_seed, int):
        seed_seq = SeedSequence(master_seed)
    elif isinstance(master_seed, SeedSequence):
        seed_seq = master_seed
    elif isinstance(master_seed, np.ndarray):
        seed_seq = SeedSequence(np.ascontiguousarray(master_seed))
    else:
        raise exceptions.ConfigurationError(
            f"master_seed must be an int, SeedSequence, or np.ndarray, "
            f"got {type(master_seed).__name__}."
        )
    child_seeds = seed_seq.spawn(num_workers)
    return [Generator(PCG64(s)) for s in child_seeds]


def sample_for_parameter_estimation(
    sifting_results: "SiftingResults",
    rng: RNG,
    sample_fraction: float,
) -> MaskArray:
    """Randomly samples a fraction of sifted bits for parameter estimation."""
    if not constants.is_valid_probability(sample_fraction):
        raise exceptions.ParameterValidationError(
            "sample_fraction must be a valid probability in the range [0, 1].",
            param_name="sample_fraction",
            param_value=sample_fraction,
        )

    sifted_indices = np.flatnonzero(sifting_results.sifted_mask)
    num_sifted = len(sifted_indices)
    num_samples = int(round(num_sifted * sample_fraction))
    # Guarantee at least one sample when fraction > 0 and sifted > 0,
    # to prevent silent discard of the entire estimation request.
    if num_samples == 0 and sample_fraction > 0 and num_sifted > 0:
        num_samples = 1

    if num_samples <= 0 or num_sifted == 0:
        return np.zeros(sifting_results.num_pulses, dtype=bool)

    if num_samples > num_sifted:
        num_samples = num_sifted

    sampled_indices = rng.choice(sifted_indices, size=num_samples, replace=False)
    param_estimation_mask = np.zeros(sifting_results.num_pulses, dtype=bool)
    param_estimation_mask[sampled_indices] = True
    return param_estimation_mask


@dataclass(frozen=True)
class BB84PreparedStates:
    """Immutable dataclass for states prepared in a BB84 protocol run."""
    num_pulses: int
    alice_bits: BitArray
    alice_bases: BitArray
    alice_pulse_type_indices: IndexArray
    bob_bases: BitArray

    def __post_init__(self) -> None:
        arrays = {
            "alice_bits": (self.alice_bits, np.uint8),
            "alice_bases": (self.alice_bases, np.uint8),
            "alice_pulse_type_indices": (self.alice_pulse_type_indices, np.int64),
            "bob_bases": (self.bob_bases, np.uint8),
        }

        for name, (arr, dtype) in arrays.items():
            if arr.shape[0] != self.num_pulses:
                raise exceptions.ParameterValidationError(
                    f"Array '{name}' length mismatch with num_pulses.",
                    param_name=name,
                    context={
                        "expected_length": self.num_pulses,
                        "actual_length": arr.shape[0],
                    },
                )
            validated_arr = _validate_and_freeze_array(arr, name, dtype)
            object.__setattr__(self, name, validated_arr)

        _validate_binary_array(self.alice_bits, "alice_bits")
        _validate_binary_array(self.alice_bases, "alice_bases")
        _validate_binary_array(self.bob_bases, "bob_bases")

        if np.any(self.alice_pulse_type_indices < 0):
            raise exceptions.ParameterValidationError(
                "alice_pulse_type_indices must be non-negative.",
                param_name="alice_pulse_type_indices",
            )


@dataclass(frozen=True)
class B92PreparedStates:
    """Immutable dataclass for states prepared in a B92 protocol run."""
    num_pulses: int
    alice_bits: BitArray
    alice_pulse_type_indices: IndexArray
    bob_bases: BitArray

    def __post_init__(self) -> None:
        arrays = {
            "alice_bits": (self.alice_bits, np.uint8),
            "alice_pulse_type_indices": (self.alice_pulse_type_indices, np.int64),
            "bob_bases": (self.bob_bases, np.uint8),
        }

        for name, (arr, dtype) in arrays.items():
            if arr.shape[0] != self.num_pulses:
                raise exceptions.ParameterValidationError(
                    f"Array '{name}' length mismatch with num_pulses.",
                    param_name=name,
                    context={
                        "expected_length": self.num_pulses,
                        "actual_length": arr.shape[0],
                    },
                )
            validated_arr = _validate_and_freeze_array(arr, name, dtype)
            object.__setattr__(self, name, validated_arr)

        _validate_binary_array(self.alice_bits, "alice_bits")
        _validate_binary_array(self.bob_bases, "bob_bases")

        if np.any(self.alice_pulse_type_indices < 0):
            raise exceptions.ParameterValidationError(
                "alice_pulse_type_indices must be non-negative.",
                param_name="alice_pulse_type_indices",
            )


@dataclass(frozen=True)
class MDIPreparedStates:
    """Immutable dataclass for states prepared in an MDI-QKD protocol run."""
    num_pulses: int
    alice_bits: BitArray
    alice_bases: BitArray
    alice_pulse_type_indices: IndexArray
    bob_bits: BitArray
    bob_bases: BitArray
    bob_pulse_type_indices: IndexArray

    def __post_init__(self) -> None:
        arrays = {
            "alice_bits": (self.alice_bits, np.uint8),
            "alice_bases": (self.alice_bases, np.uint8),
            "alice_pulse_type_indices": (self.alice_pulse_type_indices, np.int64),
            "bob_bits": (self.bob_bits, np.uint8),
            "bob_bases": (self.bob_bases, np.uint8),
            "bob_pulse_type_indices": (self.bob_pulse_type_indices, np.int64),
        }

        for name, (arr, dtype) in arrays.items():
            if arr.shape[0] != self.num_pulses:
                raise exceptions.ParameterValidationError(
                    f"Array '{name}' length mismatch with num_pulses.",
                    param_name=name,
                    context={
                        "expected_length": self.num_pulses,
                        "actual_length": arr.shape[0],
                    },
                )
            validated_arr = _validate_and_freeze_array(arr, name, dtype)
            object.__setattr__(self, name, validated_arr)

        _validate_binary_array(self.alice_bits, "alice_bits")
        _validate_binary_array(self.alice_bases, "alice_bases")
        _validate_binary_array(self.bob_bits, "bob_bits")
        _validate_binary_array(self.bob_bases, "bob_bases")

        if np.any(self.alice_pulse_type_indices < 0):
            raise exceptions.ParameterValidationError(
                "alice_pulse_type_indices must be non-negative.",
                param_name="alice_pulse_type_indices",
            )
        if np.any(self.bob_pulse_type_indices < 0):
            raise exceptions.ParameterValidationError(
                "bob_pulse_type_indices must be non-negative.",
                param_name="bob_pulse_type_indices",
            )


@dataclass(frozen=True)
class DetectionResults:
    """Immutable dataclass for detection outcomes."""
    num_pulses: int
    click0: MaskArray
    click1: MaskArray
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.click0.shape[0] != self.num_pulses:
            raise exceptions.ParameterValidationError(
                "Detection array 'click0' length must match num_pulses.",
                param_name="click0",
                context={"expected": self.num_pulses, "actual": self.click0.shape[0]},
            )
        if self.click1.shape[0] != self.num_pulses:
            raise exceptions.ParameterValidationError(
                "Detection array 'click1' length must match num_pulses.",
                param_name="click1",
                context={"expected": self.num_pulses, "actual": self.click1.shape[0]},
            )

        object.__setattr__(self, "click0", _validate_and_freeze_array(self.click0, "click0", np.bool_))
        object.__setattr__(self, "click1", _validate_and_freeze_array(self.click1, "click1", np.bool_))


@dataclass(frozen=True)
class SiftingResults:
    """Structured, immutable return type for sifting results."""
    num_pulses: int
    sifted_mask: MaskArray
    error_mask: MaskArray
    sifted_alice_pulse_type_indices: IndexArray
    sifted_bob_pulse_type_indices: Optional[IndexArray] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sifted_mask", _validate_and_freeze_array(self.sifted_mask, "sifted_mask", np.bool_))
        object.__setattr__(self, "error_mask", _validate_and_freeze_array(self.error_mask, "error_mask", np.bool_))
        object.__setattr__(
            self,
            "sifted_alice_pulse_type_indices",
            _validate_and_freeze_array(self.sifted_alice_pulse_type_indices, "sifted_alice_pulse_type_indices", np.int64),
        )

        if self.sifted_bob_pulse_type_indices is not None:
            object.__setattr__(
                self,
                "sifted_bob_pulse_type_indices",
                _validate_and_freeze_array(self.sifted_bob_pulse_type_indices, "sifted_bob_pulse_type_indices", np.int64),
            )

        for arr_name, arr in [("sifted_mask", self.sifted_mask), ("error_mask", self.error_mask)]:
            if arr.shape[0] != self.num_pulses:
                raise exceptions.ParameterValidationError(
                    f"Mask '{arr_name}' has invalid shape or length.",
                    param_name=arr_name,
                    context={
                        "expected_shape": (self.num_pulses,),
                        "actual_shape": arr.shape,
                    },
                )

        num_sifted = int(np.count_nonzero(self.sifted_mask))
        if self.sifted_alice_pulse_type_indices.shape[0] != num_sifted:
            raise exceptions.ParameterValidationError(
                "sifted_alice_pulse_type_indices length must equal number of sifted events.",
                param_name="sifted_alice_pulse_type_indices",
                context={
                    "expected_length": num_sifted,
                    "actual_length": self.sifted_alice_pulse_type_indices.shape[0],
                },
            )

        if self.sifted_bob_pulse_type_indices is not None and self.sifted_bob_pulse_type_indices.shape[0] != num_sifted:
            raise exceptions.ParameterValidationError(
                "sifted_bob_pulse_type_indices length must equal number of sifted events.",
                param_name="sifted_bob_pulse_type_indices",
                context={
                    "expected_length": num_sifted,
                    "actual_length": self.sifted_bob_pulse_type_indices.shape[0],
                },
            )

        if np.any(self.sifted_alice_pulse_type_indices < 0):
            raise exceptions.ParameterValidationError(
                "sifted_alice_pulse_type_indices must be non-negative.",
                param_name="sifted_alice_pulse_type_indices",
            )

        if self.sifted_bob_pulse_type_indices is not None and np.any(self.sifted_bob_pulse_type_indices < 0):
            raise exceptions.ParameterValidationError(
                "sifted_bob_pulse_type_indices must be non-negative.",
                param_name="sifted_bob_pulse_type_indices",
            )

        if np.any(self.error_mask & ~self.sifted_mask):
            raise exceptions.QKDSimulationError(
                "Internal logic error: error_mask contains True values outside of sifted_mask.",
                context={"num_invalid_errors": int(np.count_nonzero(self.error_mask & ~self.sifted_mask))},
            )

    def summary(self, confidence_level: float = 0.95) -> Dict[str, Union[int, float]]:
        if not constants.is_valid_probability(confidence_level) or confidence_level <= 0.0:
            raise exceptions.ParameterValidationError(
                "confidence_level must be in the interval $(0, 1]$.",
                param_name="confidence_level",
                param_value=confidence_level,
            )

        num_sifted = int(np.count_nonzero(self.sifted_mask))
        num_errors = int(np.count_nonzero(self.error_mask))
        if num_sifted == 0:
            return {
                "num_sifted": 0,
                "num_errors": 0,
                "qber": float("nan"),
                "qber_confidence_level": float(confidence_level),
                "qber_ci_low": float("nan"),
                "qber_ci_high": float("nan"),
            }

        qber = num_errors / num_sifted
        alpha = 1.0 - confidence_level

        try:
            interval = qkd_math.clopper_pearson_bounds(
                k=int(num_errors),
                n=int(num_sifted),
                alpha=alpha,
                side="two-sided",
            )
            qber_ci_low = float(interval.lower)
            qber_ci_high = float(interval.upper)
        except (ImportError, ValueError) as e:
            raise exceptions.QKDSimulationError(
                f"Failed to compute QBER confidence interval: {e}"
            ) from e

        return {
            "num_sifted": num_sifted,
            "num_errors": num_errors,
            "qber": float(qber),
            "qber_confidence_level": float(confidence_level),
            "qber_ci_low": qber_ci_low,
            "qber_ci_high": qber_ci_high,
        }


class Protocol(ABC):
    """Abstract Base Class for a QKD Protocol."""

    @property
    @abstractmethod
    def protocol_name(self) -> str:
        pass

    @abstractmethod
    def prepare_states(
        self,
        num_pulses: int,
        rng: RNG,
    ) -> Union[BB84PreparedStates, MDIPreparedStates, B92PreparedStates]:
        pass

    @abstractmethod
    def sift_results(
        self,
        prepared_states: Union[BB84PreparedStates, MDIPreparedStates, B92PreparedStates],
        detection_results: DetectionResults,
        rng: RNG,
    ) -> SiftingResults:
        pass

    @abstractmethod
    def to_config_dict(self) -> Dict[str, Any]:
        pass


class RedundantTransmissionProtocol(Protocol):
    """
    Implementation of the Redundant Encoding scheme.
    Supports general $M$-bit to $N$-qubit encodings in addition to standard repetition codes.
    """

    def __init__(
        self,
        redundancy_M: int,
        source: OpticalSource,
        codeword_mapping: Optional[Dict[str, List[int]]] = None,
    ):
        if not isinstance(redundancy_M, int) or redundancy_M < 1:
            raise exceptions.ParameterValidationError(
                "redundancy_M must be a positive integer.",
                param_name="redundancy_M",
                param_value=redundancy_M,
            )
        if not isinstance(source, OpticalSource):
            raise exceptions.ConfigurationError("source must be an OpticalSource instance.")

        self.redundancy_M = redundancy_M
        self.source = source

        self.inverse_codeword_mapping: Dict[tuple[int, ...], str] = {}

        if codeword_mapping is None:
            self.codeword_mapping = None
            self.msg_length = 1
        else:
            if not codeword_mapping:
                raise exceptions.ConfigurationError("codeword_mapping must not be empty if provided.")

            self.codeword_mapping = codeword_mapping
            first_key = next(iter(codeword_mapping))
            self.msg_length = len(first_key)

            for msg, code in codeword_mapping.items():
                if len(msg) != self.msg_length:
                    raise exceptions.ConfigurationError(
                        "All message keys in codeword_mapping must have equal length."
                    )
                if any(ch not in {"0", "1"} for ch in msg):
                    raise exceptions.ConfigurationError(
                        "All codeword_mapping keys must be binary strings."
                    )
                if len(code) != self.redundancy_M:
                    raise exceptions.ConfigurationError(
                        f"Codeword length {len(code)} does not match redundancy_M={self.redundancy_M}."
                    )
                if any(bit not in (0, 1) for bit in code):
                    raise exceptions.ConfigurationError(
                        "All codeword values must contain only binary entries."
                    )

                code_tuple = tuple(int(bit) for bit in code)
                if code_tuple in self.inverse_codeword_mapping:
                    raise exceptions.ConfigurationError(
                        f"Duplicate codeword {list(code_tuple)} in codeword_mapping: "
                        f"maps both '{self.inverse_codeword_mapping[code_tuple]}' and '{msg}'. "
                        "Codeword mapping must be injective."
                    )
                self.inverse_codeword_mapping[code_tuple] = msg
            expected_num_messages = 2 ** self.msg_length
            if len(self.codeword_mapping) != expected_num_messages:
                raise exceptions.ConfigurationError(
                    f"Codeword mapping has {len(self.codeword_mapping)} entries but "
                    f"msg_length={self.msg_length} implies {expected_num_messages} messages. "
                    "Mapping must cover all possible messages."
                )


    @property
    def protocol_name(self) -> str:
        return "redundant"

    def to_config_dict(self) -> Dict[str, Any]:
        return {
            "redundancy_M": self.redundancy_M,
            "codeword_mapping": self.codeword_mapping,
        }

    def prepare_states(self, num_pulses: int, rng: RNG) -> BB84PreparedStates:
        if not isinstance(num_pulses, int) or num_pulses < 0:
            raise exceptions.ParameterValidationError(
                "num_pulses must be a non-negative integer.",
                param_name="num_pulses",
                param_value=num_pulses,
            )

        if num_pulses == 0:
            return BB84PreparedStates(
                num_pulses=0,
                alice_bits=np.array([], dtype=np.uint8),
                alice_bases=np.array([], dtype=np.uint8),
                alice_pulse_type_indices=np.array([], dtype=np.int64),
                bob_bases=np.array([], dtype=np.uint8),
            )

        if num_pulses % self.redundancy_M != 0:
            logger.warning(
                "Total pulses %s not divisible by M=%s. Last partial block will be ignored during sifting.",
                num_pulses,
                self.redundancy_M,
            )

        num_blocks = num_pulses // self.redundancy_M
        num_messages = 2 ** self.msg_length
        logical_msgs = rng.integers(0, num_messages, size=num_blocks, dtype=np.int64)

        alice_bits = np.zeros(num_pulses, dtype=np.uint8)

        if self.codeword_mapping is not None:
            for i in range(num_blocks):
                msg_val = int(logical_msgs[i])
                msg_str = format(msg_val, f"0{self.msg_length}b")
                codeword = self.codeword_mapping.get(msg_str)
                if codeword is None:
                    raise exceptions.QKDSimulationError(
                        f"Mapping not found for message {msg_str}."
                    )
                start_idx = i * self.redundancy_M
                alice_bits[start_idx:start_idx + self.redundancy_M] = codeword
        else:
            logical_bits = logical_msgs.astype(np.uint8)
            alice_bits[: num_blocks * self.redundancy_M] = np.repeat(logical_bits, self.redundancy_M)

        alice_bases = np.zeros(num_pulses, dtype=np.uint8)
        bob_bases = np.zeros(num_pulses, dtype=np.uint8)


        probs = _normalized_source_probabilities(self.source)
        alice_pulse_indices = rng.choice(len(probs), size=num_pulses, p=probs).astype(np.int64)

        return BB84PreparedStates(
            num_pulses=num_pulses,
            alice_bits=alice_bits,
            alice_bases=alice_bases,
            alice_pulse_type_indices=alice_pulse_indices,
            bob_bases=bob_bases,
        )

    def sift_results(
        self,
        prepared_states: Union[BB84PreparedStates, MDIPreparedStates, B92PreparedStates],
        detection_results: DetectionResults,
        rng: RNG,
    ) -> SiftingResults:
        if not isinstance(prepared_states, BB84PreparedStates):
            raise exceptions.QKDSimulationError(
                "Redundant protocol expects BB84PreparedStates structure."
            )

        num_physical = prepared_states.num_pulses
        if detection_results.num_pulses != num_physical:
            raise exceptions.ParameterValidationError(
                "detection_results.num_pulses must match prepared_states.num_pulses.",
                param_name="detection_results.num_pulses",
                context={
                    "prepared_num_pulses": num_physical,
                    "detection_num_pulses": detection_results.num_pulses,
                },
            )

        M = self.redundancy_M
        num_blocks = num_physical // M

        error_mask = np.zeros(num_physical, dtype=bool)
        sifted_mask = np.zeros(num_physical, dtype=bool)
        sifted_mask[: num_blocks * M] = True

        clicks = (detection_results.click0 | detection_results.click1)[: num_blocks * M].reshape((num_blocks, M))

        if True:
            if self.codeword_mapping is not None:
                received_bits = clicks.astype(int)
                alice_physical = prepared_states.alice_bits[: num_blocks * M].reshape((num_blocks, M))
                block_errors = np.zeros(num_blocks, dtype=bool)

                for i in range(num_blocks):
                    alice_chunk = tuple(int(x) for x in alice_physical[i])
                    original_msg_str = self.inverse_codeword_mapping.get(alice_chunk)
                    if original_msg_str is None:
                        block_errors[i] = True
                        continue

                    rx_chunk = tuple(int(x) for x in received_bits[i])
                    decoded_msg_str = self.inverse_codeword_mapping.get(rx_chunk)

                    if decoded_msg_str is None:
                        best_match_msg = None
                        min_dist = float("inf")
                        candidates = []
                        for code_tuple, msg_str in self.inverse_codeword_mapping.items():
                            dist = int(np.sum(np.abs(np.array(rx_chunk) - np.array(code_tuple))))
                            if dist < min_dist:
                                min_dist = dist
                                candidates = [msg_str]
                            elif dist == min_dist:
                                candidates.append(msg_str)
                        # Deterministic tie-breaking: pick lexicographically smallest message
                        decoded_msg_str = min(candidates) if candidates else best_match_msg

                    block_errors[i] = decoded_msg_str != original_msg_str

                error_mask[: num_blocks * M] = np.repeat(block_errors, M)
            else:
                decoded_bits = np.any(clicks, axis=1).astype(np.uint8)
                logical_alice_bits = prepared_states.alice_bits[: num_blocks * M].reshape((num_blocks, M))[:, 0]
                block_errors = decoded_bits != logical_alice_bits
                error_mask[: num_blocks * M] = np.repeat(block_errors, M)

        return SiftingResults(
            num_pulses=num_physical,
            sifted_mask=sifted_mask,
            error_mask=error_mask,
            sifted_alice_pulse_type_indices=prepared_states.alice_pulse_type_indices[sifted_mask],
        )


class BB84DecoyProtocol(Protocol):
    """Concrete implementation of the BB84 protocol with decoy states."""

    def __init__(
        self,
        alice_z_basis_prob: float,
        bob_z_basis_prob: float,
        source: OpticalSource,
        double_click_policy: Union[DoubleClickPolicy, str] = DoubleClickPolicy.DISCARD,
    ):
        if not constants.is_valid_probability(alice_z_basis_prob):
            raise exceptions.ParameterValidationError(
                "alice_z_basis_prob must be in [0,1].",
                param_name="alice_z_basis_prob",
                param_value=alice_z_basis_prob,
            )
        if not constants.is_valid_probability(bob_z_basis_prob):
            raise exceptions.ParameterValidationError(
                "bob_z_basis_prob must be in [0,1].",
                param_name="bob_z_basis_prob",
                param_value=bob_z_basis_prob,
            )
        if not isinstance(source, OpticalSource):
            raise exceptions.ConfigurationError(
                f"source must be an OpticalSource instance, but got {type(source).__name__}."
            )

        if isinstance(double_click_policy, str):
            try:
                double_click_policy = DoubleClickPolicy[double_click_policy.strip().upper()]
            except KeyError as exc:
                raise exceptions.ParameterValidationError(
                    "Unknown double_click_policy string.",
                    param_name="double_click_policy",
                    param_value=double_click_policy,
                    context={"valid_policies": [p.name for p in DoubleClickPolicy]},
                ) from exc

        self.alice_z_basis_prob = alice_z_basis_prob
        self.bob_z_basis_prob = bob_z_basis_prob
        self.source = source
        self.double_click_policy = double_click_policy

        self.decoy_configs = [pc for pc in source.pulse_configs if getattr(pc, "name", None) != "signal"]
        try:
            self.signal_config = source.get_pulse_config_by_name("signal")
        except (KeyError, AttributeError) as exc:
            raise exceptions.ConfigurationError(
                "Source must contain a pulse config named 'signal' for decoy-state protocol."
            ) from exc

    @property
    def protocol_name(self) -> str:
        return "bb84-decoy"

    def to_config_dict(self) -> Dict[str, Any]:
        return {
            "alice_z_basis_prob": self.alice_z_basis_prob,
            "bob_z_basis_prob": self.bob_z_basis_prob,
            "double_click_policy": self.double_click_policy.name,
        }

    def prepare_states(self, num_pulses: int, rng: RNG) -> BB84PreparedStates:
        if not isinstance(num_pulses, int) or num_pulses < 0:
            raise exceptions.ParameterValidationError(
                "num_pulses must be a non-negative integer.",
                param_name="num_pulses",
                param_value=num_pulses,
            )

        if num_pulses == 0:
            return BB84PreparedStates(
                num_pulses=0,
                alice_bits=np.array([], dtype=np.uint8),
                alice_bases=np.array([], dtype=np.uint8),
                alice_pulse_type_indices=np.array([], dtype=np.int64),
                bob_bases=np.array([], dtype=np.uint8),
            )

        alice_bits = rng.integers(0, 2, size=num_pulses, dtype=np.uint8)
        alice_bases = rng.choice([Z_BASIS, X_BASIS], size=num_pulses,
                                 p=[self.alice_z_basis_prob, 1.0 - self.alice_z_basis_prob]).astype(np.uint8)
        bob_bases = rng.choice([Z_BASIS, X_BASIS], size=num_pulses,
                                p=[self.bob_z_basis_prob, 1.0 - self.bob_z_basis_prob]).astype(np.uint8)

        probs = _normalized_source_probabilities(self.source)
        alice_pulse_indices = rng.choice(len(probs), size=num_pulses, p=probs).astype(np.int64)

        return BB84PreparedStates(
            num_pulses=num_pulses,
            alice_bits=alice_bits,
            alice_bases=alice_bases,
            alice_pulse_type_indices=alice_pulse_indices,
            bob_bases=bob_bases,
        )

    def sift_results(
        self,
        prepared_states: Union[BB84PreparedStates, MDIPreparedStates, B92PreparedStates],
        detection_results: DetectionResults,
        rng: RNG,
    ) -> SiftingResults:
        if not isinstance(prepared_states, BB84PreparedStates):
            raise exceptions.QKDSimulationError(
                "Mismatched prepared_states object passed to BB84 sifting function."
            )
        if detection_results.num_pulses != prepared_states.num_pulses:
            raise exceptions.ParameterValidationError(
                "detection_results.num_pulses must match prepared_states.num_pulses.",
                param_name="detection_results.num_pulses",
            )

        basis_match = prepared_states.alice_bases == prepared_states.bob_bases
        click0, click1 = detection_results.click0, detection_results.click1

        conclusive0 = click0 & ~click1
        conclusive1 = click1 & ~click0
        double_click_mask = click0 & click1

        bob_bits_valid = conclusive0 | conclusive1
        bob_bits = np.zeros(prepared_states.num_pulses, dtype=np.uint8)
        bob_bits[conclusive1] = 1

        if self.double_click_policy == DoubleClickPolicy.RANDOM:
            num_dc = int(np.count_nonzero(double_click_mask))
            if num_dc > 0:
                bob_bits[double_click_mask] = rng.integers(0, 2, size=num_dc, dtype=np.uint8)
                bob_bits_valid[double_click_mask] = True
        elif self.double_click_policy == DoubleClickPolicy.DISCARD:
            bob_bits_valid[double_click_mask] = False

        sifted_mask = basis_match & bob_bits_valid
        error_mask = np.zeros_like(sifted_mask)

        sifted_indices = np.flatnonzero(sifted_mask)
        if sifted_indices.size > 0:
            error_mask[sifted_indices] = (
                prepared_states.alice_bits[sifted_indices] != bob_bits[sifted_indices]
            )

        return SiftingResults(
            num_pulses=prepared_states.num_pulses,
            sifted_mask=sifted_mask,
            error_mask=error_mask,
            sifted_alice_pulse_type_indices=prepared_states.alice_pulse_type_indices[sifted_mask],
        )


class B92Protocol(Protocol):
    """Concrete implementation of the B92 protocol."""

    def __init__(
        self,
        bob_z_basis_prob: float,
        source: OpticalSource,
    ):
        if not constants.is_valid_probability(bob_z_basis_prob):
            raise exceptions.ParameterValidationError(
                "bob_z_basis_prob must be in [0,1].",
                param_name="bob_z_basis_prob",
                param_value=bob_z_basis_prob,
            )
        if not isinstance(source, OpticalSource):
            raise exceptions.ConfigurationError("source must be an OpticalSource instance.")

        self.bob_z_basis_prob = bob_z_basis_prob
        self.source = source

    @property
    def protocol_name(self) -> str:
        return "b92"

    def to_config_dict(self) -> Dict[str, Any]:
        return {
            "bob_z_basis_prob": self.bob_z_basis_prob,
        }

    def prepare_states(self, num_pulses: int, rng: RNG) -> B92PreparedStates:
        if not isinstance(num_pulses, int) or num_pulses < 0:
            raise exceptions.ParameterValidationError(
                "num_pulses must be a non-negative integer.",
                param_name="num_pulses",
                param_value=num_pulses,
            )

        if num_pulses == 0:
            return B92PreparedStates(
                num_pulses=0,
                alice_bits=np.array([], dtype=np.uint8),
                alice_pulse_type_indices=np.array([], dtype=np.int64),
                bob_bases=np.array([], dtype=np.uint8),
            )

        alice_bits = rng.integers(0, 2, size=num_pulses, dtype=np.uint8)
        bob_bases = rng.choice([Z_BASIS, X_BASIS], size=num_pulses,
                                p=[self.bob_z_basis_prob, 1.0 - self.bob_z_basis_prob]).astype(np.uint8)

        probs = _normalized_source_probabilities(self.source)
        alice_pulse_indices = rng.choice(len(probs), size=num_pulses, p=probs).astype(np.int64)

        return B92PreparedStates(
            num_pulses=num_pulses,
            alice_bits=alice_bits,
            alice_pulse_type_indices=alice_pulse_indices,
            bob_bases=bob_bases,
        )

    def sift_results(
        self,
        prepared_states: Union[BB84PreparedStates, MDIPreparedStates, B92PreparedStates],
        detection_results: DetectionResults,
        rng: RNG,
    ) -> SiftingResults:
        if not isinstance(prepared_states, B92PreparedStates):
            raise exceptions.QKDSimulationError(
                "Mismatched prepared_states object passed to B92 sifting function."
            )
        if detection_results.num_pulses != prepared_states.num_pulses:
            raise exceptions.ParameterValidationError(
                "detection_results.num_pulses must match prepared_states.num_pulses.",
                param_name="detection_results.num_pulses",
            )

        click0, click1 = detection_results.click0, detection_results.click1

        bob_basis_z = prepared_states.bob_bases == 0
        bob_basis_x = prepared_states.bob_bases == 1

        valid_z_basis = bob_basis_z & click0 & ~click1
        valid_x_basis = bob_basis_x & click1 & ~click0

        sifted_mask = valid_z_basis | valid_x_basis

        bob_bits = np.zeros(prepared_states.num_pulses, dtype=np.uint8)
        bob_bits[valid_z_basis] = 0
        bob_bits[valid_x_basis] = 1

        error_mask = np.zeros_like(sifted_mask)
        sifted_indices = np.flatnonzero(sifted_mask)
        if sifted_indices.size > 0:
            error_mask[sifted_indices] = (
                prepared_states.alice_bits[sifted_indices] != bob_bits[sifted_indices]
            )

        return SiftingResults(
            num_pulses=prepared_states.num_pulses,
            sifted_mask=sifted_mask,
            error_mask=error_mask,
            sifted_alice_pulse_type_indices=prepared_states.alice_pulse_type_indices[sifted_mask],
        )


class MDIQKDProtocol(Protocol):
    """Concrete implementation of MDI-QKD."""

    def __init__(self, z_basis_prob: float, source: OpticalSource):
        if not constants.is_valid_probability(z_basis_prob):
            raise exceptions.ParameterValidationError(
                "z_basis_prob must be in [0,1].",
                param_name="z_basis_prob",
                param_value=z_basis_prob,
            )
        if not isinstance(source, OpticalSource):
            raise exceptions.ConfigurationError("source must be an OpticalSource instance.")

        self.z_basis_prob = z_basis_prob
        self.source = source

    @property
    def protocol_name(self) -> str:
        return "mdi-qkd"

    def to_config_dict(self) -> Dict[str, Any]:
        return {"z_basis_prob": self.z_basis_prob}

    def prepare_states(self, num_pulses: int, rng: RNG) -> MDIPreparedStates:
        if not isinstance(num_pulses, int) or num_pulses < 0:
            raise exceptions.ParameterValidationError(
                "num_pulses must be a non-negative integer.",
                param_name="num_pulses",
                param_value=num_pulses,
            )

        if num_pulses == 0:
            return MDIPreparedStates(
                num_pulses=0,
                alice_bits=np.array([], dtype=np.uint8),
                alice_bases=np.array([], dtype=np.uint8),
                alice_pulse_type_indices=np.array([], dtype=np.int64),
                bob_bits=np.array([], dtype=np.uint8),
                bob_bases=np.array([], dtype=np.uint8),
                bob_pulse_type_indices=np.array([], dtype=np.int64),
            )

        probs = _normalized_source_probabilities(self.source)

        alice_bits = rng.integers(0, 2, size=num_pulses, dtype=np.uint8)
        alice_bases = rng.choice([Z_BASIS, X_BASIS], size=num_pulses,
                                 p=[self.z_basis_prob, 1.0 - self.z_basis_prob]).astype(np.uint8)
        alice_pulse_indices = rng.choice(len(probs), size=num_pulses, p=probs).astype(np.int64)

        bob_bits = rng.integers(0, 2, size=num_pulses, dtype=np.uint8)
        bob_bases = rng.choice([Z_BASIS, X_BASIS], size=num_pulses,
                                p=[self.z_basis_prob, 1.0 - self.z_basis_prob]).astype(np.uint8)
        bob_pulse_indices = rng.choice(len(probs), size=num_pulses, p=probs).astype(np.int64)

        return MDIPreparedStates(
            num_pulses=num_pulses,
            alice_bits=alice_bits,
            alice_bases=alice_bases,
            alice_pulse_type_indices=alice_pulse_indices,
            bob_bits=bob_bits,
            bob_bases=bob_bases,
            bob_pulse_type_indices=bob_pulse_indices,
        )

    def sift_results(
        self,
        prepared_states: Union[BB84PreparedStates, MDIPreparedStates, B92PreparedStates],
        detection_results: DetectionResults,
        rng: RNG,
    ) -> SiftingResults:
        if not isinstance(prepared_states, MDIPreparedStates):
            raise exceptions.QKDSimulationError(
                "Mismatched prepared_states object passed to MDI-QKD sifting function."
            )
        if detection_results.num_pulses != prepared_states.num_pulses:
            raise exceptions.ParameterValidationError(
                "detection_results.num_pulses must match prepared_states.num_pulses.",
                param_name="detection_results.num_pulses",
            )

        # BSM model: coincident clicks (click0 & click1) are interpreted as a
        # successful projection onto |ψ⁻⟩ (the anti-symmetric Bell state).
        # This model does not distinguish |ψ⁻⟩ from |ψ⁺⟩; the error rules
        # below assume all successful BSM events are |ψ⁻⟩. For |ψ⁻⟩:
        #   Z basis: Alice and Bob bits must differ (anti-correlated).
        #   X basis: Alice and Bob bits must be equal (correlated).
        # Security proofs that require separate e_Z and e_X must use a
        # detector model that resolves individual Bell states.
        successful_bsm = detection_results.click0 & detection_results.click1
        basis_match = prepared_states.alice_bases == prepared_states.bob_bases
        sifted_mask = basis_match & successful_bsm

        error_mask = np.zeros_like(sifted_mask)
        sifted_indices = np.flatnonzero(sifted_mask)

        if sifted_indices.size > 0:
            sifted_alice_bits = prepared_states.alice_bits[sifted_indices]
            sifted_bob_bits = prepared_states.bob_bits[sifted_indices]
            sifted_bases = prepared_states.alice_bases[sifted_indices]

            z_basis_errors = (sifted_bases == Z_BASIS) & (sifted_alice_bits == sifted_bob_bits)
            x_basis_errors = (sifted_bases == X_BASIS) & (sifted_alice_bits != sifted_bob_bits)
            error_mask[sifted_indices] = z_basis_errors | x_basis_errors

        return SiftingResults(
            num_pulses=prepared_states.num_pulses,
            sifted_mask=sifted_mask,
            error_mask=error_mask,
            sifted_alice_pulse_type_indices=prepared_states.alice_pulse_type_indices[sifted_mask],
            sifted_bob_pulse_type_indices=prepared_states.bob_pulse_type_indices[sifted_mask],
        )

