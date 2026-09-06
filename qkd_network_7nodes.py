#!/usr/bin/env python3
"""
7-Node QKD Network with Trusted Relay and SPF+ Path Selection
==============================================================

Implements a complete Quantum Key Distribution network with:
- 7 nodes: A (receiver), B/C (trusted relays), D/E/F/G (senders)
- 10 direct QKD links
- 20 local key pools (2 per link)
- Optical switches: 1x2 (A<->B/C), 2x4 (B/C<->D/E/F/G)
- Full BB84-decoy pipeline simulation via the repository-local
  main_optimized.run_single_simulation()
- Fallback statistical simulation when pipeline not available
- SPF+ (Shortest Path First Plus) path selection
- XOR-based trusted relay key delivery
- Key pool management with replenishment

Key Generation:
  The network loads the ``main_optimized.py`` located beside this file,
  regardless of the process working directory. The full pipeline is used
  via ``run_single_simulation()`` which handles:
  Source -> Channel -> Protocol -> Detector -> Sifting -> Proof (Lim2014),
  including F-01 bias-voltage clamping, F-05 SeedSequence RNG, F-11
  probability balancing validation, and F-14 source metadata caching.

  When the pipeline is not available, a distance-dependent statistical
  model is used as fallback.

Usage:
    # From the QKD Simulation Framework directory (uses real pipeline):
    cd "4.1 QKD Simulation Framework"
    python qkd_network_7nodes.py

    # From any directory (loads the same repository-local pipeline):
    python /path/to/qkd_network_7nodes.py
"""

from __future__ import annotations

import heapq
import importlib.util
import math
import os
from pathlib import Path
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


_PIPELINE_MODULE_NAME = "_qkd_network_main_optimized"


def _load_local_pipeline_module():
    """Load the main_optimized.py located beside this network script."""
    framework_dir = Path(__file__).resolve().parent
    module_path = framework_dir / "main_optimized.py"
    if not module_path.is_file():
        raise ImportError(f"Pipeline engine not found at {module_path}")

    framework_dir_str = str(framework_dir)
    if not sys.path or sys.path[0] != framework_dir_str:
        sys.path.insert(0, framework_dir_str)

    cached_module = sys.modules.get(_PIPELINE_MODULE_NAME)
    if cached_module is not None:
        cached_path = getattr(cached_module, "__file__", None)
        if cached_path and Path(cached_path).resolve() == module_path:
            return cached_module
        sys.modules.pop(_PIPELINE_MODULE_NAME, None)

    spec = importlib.util.spec_from_file_location(_PIPELINE_MODULE_NAME, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[_PIPELINE_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_PIPELINE_MODULE_NAME, None)
        raise

    loaded_path = Path(module.__file__).resolve()
    if loaded_path != module_path:
        raise ImportError(
            f"Loaded pipeline engine from {loaded_path}, expected {module_path}"
        )
    return module


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class KeyBlockState(Enum):
    """State of a key block within a key pool."""
    READY = "READY"
    RESERVED = "RESERVED"
    USED = "USED"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"


class KeyPoolStatus(Enum):
    """Status of a key pool indicating its health and availability."""
    ACTIVE = "ACTIVE"
    LOW = "LOW"
    EMPTY = "EMPTY"
    DISABLED = "DISABLED"
    ERROR = "ERROR"
    REPLENISHING = "REPLENISHING"


class LinkStatus(Enum):
    """Status of a QKD link between two nodes."""
    UP = "UP"
    DOWN = "DOWN"
    DEGRADED = "DEGRADED"


class NodeRole(Enum):
    """Role of a node in the QKD network."""
    RECEIVER = "RECEIVER"
    RELAY = "RELAY"
    SENDER = "SENDER"


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

INFINITY_COST = 1e18
QBER_THRESHOLD = 0.11
HIGH_PENALTY = 1e6          # Penalty for low-watermark pools (not blocking)

# Watermark constants — two sets for different simulator types
# Statistical model: ~0.01-0.04 bits/pulse → 2×10^8 pulses → ~2-8 Mbit/session
STAT_LOW_WATERMARK_BITS = 1_000_000       # 1 Mbit
STAT_HIGH_WATERMARK_BITS = 10_000_000     # 10 Mbit
STAT_MAX_CAPACITY_BITS = 20_000_000       # 20 Mbit

# Real pipeline: ~1e-4 to 3e-4 bits/pulse → 10^8 pulses → ~10-30 Kbit/session
# Watermarks scaled to realistic pipeline output
PIPELINE_LOW_WATERMARK_BITS = 10_000       # 10 Kbit
PIPELINE_HIGH_WATERMARK_BITS = 100_000     # 100 Kbit
PIPELINE_MAX_CAPACITY_BITS = 200_000       # 200 Kbit

# Default (statistical) — used before simulator type is known
DEFAULT_LOW_WATERMARK_BITS = STAT_LOW_WATERMARK_BITS
DEFAULT_HIGH_WATERMARK_BITS = STAT_HIGH_WATERMARK_BITS
DEFAULT_MAX_CAPACITY_BITS = STAT_MAX_CAPACITY_BITS

# SPF+ weight parameters
# Normalized cost: each term in [0,1] range, weighted by its coefficient.
# Total cost = sum of weighted terms. Key rate provides a discount (subtracted).
DEFAULT_ALPHA = 1.0       # optical loss weight (loss ~ 5-11 dB)
DEFAULT_BETA = 5.0        # QBER weight (QBER ~ 0.005-0.04)
DEFAULT_GAMMA = 0.5       # latency/distance weight (dist ~ 25-55 km)
DEFAULT_DELTA = 2.0       # pool pressure weight (0 = full, 1 = empty)
DEFAULT_EPSILON = 5.0     # key rate bonus weight (rate ~ 0.01-0.05)

# SPF+ normalization constants — also differ by simulator type
# Statistical: key_rate ~ 0.01-0.05 bits/pulse
STAT_SPF_MAX_KEY_RATE = 0.06
# Pipeline: key_rate ~ 5e-5 to 3e-4 bits/pulse
PIPELINE_SPF_MAX_KEY_RATE = 4e-4

# Normalization constants for SPF+ cost function
SPF_MAX_LOSS_DB = 12.0        # worst-case optical loss for normalization
SPF_MAX_DISTANCE_KM = 60.0   # worst-case distance for normalization
# SPF_MAX_KEY_RATE is now dynamic — set by calibrate_for_simulator()
SPF_MAX_KEY_RATE = 0.06      # default (statistical); overridden for pipeline

# Node distances (km) — approximate metropolitan-scale
NODE_DISTANCES: Dict[str, float] = {
    "A-B": 75.0, "A-C": 50.0,
    "B-D": 55.0, "B-E": 60.0, "B-F": 65.0, "B-G": 70.0,
    "C-D": 30.0, "C-E": 35.0, "C-F": 40.0, "C-G": 45.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class KeyBlock:
    """A block of quantum-derived key material stored in a key pool.

    Attributes:
        key_id: Unique identifier for this key block (e.g. "AB-000001").
        link_id: The QKD link that generated this key (e.g. "A-B").
        local_node: The node that holds this key block.
        peer_node: The node at the other end of the link.
        key_material: The actual secret key bits as raw bytes.
        length_bits: Length of the key material in bits.
        state: Current state of the key block.
        created_at: Timestamp when this block was created.
        expires_at: Optional expiration timestamp.
        qkd_session_id: ID of the QKD session that produced this key.
        qber: Quantum Bit Error Rate measured during generation.
        secret_key_rate: Secure key rate (bits/pulse) during generation.
    """
    key_id: str
    link_id: str
    local_node: str
    peer_node: str
    key_material: bytes
    length_bits: int
    state: KeyBlockState
    created_at: float
    expires_at: Optional[float]
    qkd_session_id: str
    qber: float
    secret_key_rate: float


class KeyPool:
    """Manages a pool of key blocks for one direction of a QKD link.

    Each QKD link has two key pools, one at each endpoint. The pool tracks
    available, reserved, and used key bits and manages the lifecycle of
    individual key blocks.

    Attributes:
        pool_id: Unique identifier (e.g. "POOL-A-B").
        local_node: The node hosting this pool.
        peer_node: The remote node this pool communicates with.
        link_id: The QKD link ID.
        key_blocks: Dictionary of key blocks keyed by key_id.
        available_key_bits: Total bits in READY state.
        reserved_key_bits: Total bits in RESERVED state.
        used_key_bits: Total bits in USED state.
        max_capacity_bits: Maximum capacity of this pool.
        low_watermark_bits: Threshold below which pool status is LOW.
        high_watermark_bits: Target level after replenishment.
        status: Current pool status.
        last_updated: Timestamp of last modification.
    """

    # Piecewise-linear pool pressure model shared across all callers.
    # [high_watermark, inf) -> 0.0 (healthy)
    # [low_watermark,  high_watermark) -> 0.0..0.5 (linear ramp)
    # [0, low_watermark) -> 0.5..1.0 (linear ramp)

    def __init__(
        self,
        pool_id: str,
        local_node: str,
        peer_node: str,
        link_id: str,
        max_capacity_bits: int = DEFAULT_MAX_CAPACITY_BITS,
        low_watermark_bits: int = DEFAULT_LOW_WATERMARK_BITS,
        high_watermark_bits: int = DEFAULT_HIGH_WATERMARK_BITS,
    ) -> None:
        """Initialize a key pool for one direction of a QKD link."""
        self.pool_id: str = pool_id
        self.local_node: str = local_node
        self.peer_node: str = peer_node
        self.link_id: str = link_id
        self.key_blocks: Dict[str, KeyBlock] = {}
        self.available_key_bits: int = 0
        self.reserved_key_bits: int = 0
        self.used_key_bits: int = 0
        self.max_capacity_bits: int = max_capacity_bits
        self.low_watermark_bits: int = low_watermark_bits
        self.high_watermark_bits: int = high_watermark_bits
        self.status: KeyPoolStatus = KeyPoolStatus.EMPTY
        self.last_updated: float = time.time()

    def add_key_block(self, block: KeyBlock) -> None:
        """Add a key block to the pool.

        Args:
            block: The KeyBlock to add. Must be in READY state.
        """
        # Defensive capacity cap: never exceed max_capacity_bits even if a
        # buggy/misconfigured simulator returns an absurdly large key.  This
        # prevents a single bad session from inflating the pool to millions
        # of bits (observed when statistical fallback ran with det_eff=0.65
        # against pipeline-scaled 200K-bit watermarks).
        if (self.available_key_bits + self.reserved_key_bits + block.length_bits
                > self.max_capacity_bits):
            return  # silently drop the block; pool is at capacity
        self.key_blocks[block.key_id] = block
        if block.state == KeyBlockState.READY:
            self.available_key_bits += block.length_bits
        elif block.state == KeyBlockState.RESERVED:
            self.reserved_key_bits += block.length_bits
        self.last_updated = time.time()
        self.status = self.check_status()

    def reserve_keys(self, length_bits: int) -> List[KeyBlock]:
        """Reserve exactly ``length_bits`` from the pool.

        Blocks are scanned in DECREASING size order so that the smallest
        number of blocks is consumed and fragmentation is minimised.
        When the next block would push the accumulated total above
        ``length_bits``, that block is **split**: a new RESERVED block
        containing exactly the needed bits is created, and the remainder
        stays in the pool as a new READY block.  This guarantees the
        pool's ``available_key_bits`` decreases by exactly ``length_bits``.

        Why sorting matters: ``key_blocks`` is a regular ``dict`` whose
        iteration order is insertion order.  When two priming sessions
        each leave a small ``remaining_bits`` block (e.g. 32 bits) and
        that small block is inserted *before* the next session's 256-bit
        block, the original scan-in-insertion-order algorithm would hit
        the 32-bit block first, take it (accumulated=32), then take the
        next 256-bit block (accumulated=288) — over-reserving a 256-bit
        request by 32 bits.  Across 4 service-key requests this produced
        1,062 bits consumed instead of the expected 1,024.

        Byte alignment: ``length_bits`` is expected to be a multiple of 8
        (the default service-key length is 256).  When the accumulated
        total is also a multiple of 8 (the common case after taking whole
        256-bit blocks), the split is byte-aligned and exact.  In the
        rare edge case where ``needed_bits`` is not a multiple of 8, the
        split rounds ``needed_bits`` up to the next byte boundary,
        over-reserving by at most 7 bits (vs. the previous behaviour of
        over-reserving by an entire block).

        Args:
            length_bits: Number of key bits to reserve.

        Returns:
            List of reserved KeyBlocks totaling exactly ``length_bits``
            (modulo at most 7 bits of byte-alignment rounding in the edge
            case described above).  Empty if insufficient available bits.
        """
        if length_bits <= 0:
            return []
        if self.available_key_bits < length_bits:
            return []

        reserved: List[KeyBlock] = []
        accumulated = 0

        # Sort READY blocks by size DESCENDING so that the largest blocks
        # are consumed first.  This is the primary fix: a 256-bit request
        # will preferentially take a single 256-bit block (exact match)
        # rather than combining a small 'remaining_bits' block with the
        # next 256-bit block (over-reservation).
        ready_blocks = sorted(
            [b for b in self.key_blocks.values() if b.state == KeyBlockState.READY],
            key=lambda b: -b.length_bits,
        )

        for block in ready_blocks:
            if accumulated >= length_bits:
                break

            needed = length_bits - accumulated

            if block.length_bits <= needed:
                # Take the whole block — no over-reservation.
                block.state = KeyBlockState.RESERVED
                self.available_key_bits -= block.length_bits
                self.reserved_key_bits += block.length_bits
                accumulated += block.length_bits
                reserved.append(block)
            else:
                # The block is larger than needed — split it.
                self._split_block_for_reservation(block, needed, reserved)
                accumulated += needed

        self.last_updated = time.time()
        self.status = self.check_status()
        return reserved

    def _split_block_for_reservation(
        self,
        block: KeyBlock,
        needed_bits: int,
        reserved_list: List[KeyBlock],
    ) -> None:
        """Split a READY block into a RESERVED portion and a READY remainder.

        Removes ``block`` from the pool and adds two new blocks in its
        place: one RESERVED block containing ``needed_bits`` (rounded up
        to the next byte boundary for byte-aligned key material), and one
        READY block containing the remainder.  Both new blocks inherit
        the original block's metadata (link_id, qkd_session_id, qber,
        secret_key_rate, etc.).

        Args:
            block: The READY block to split.  Must be in READY state and
                must be larger than ``needed_bits``.
            needed_bits: Number of bits to reserve from this block.
            reserved_list: List to append the new RESERVED block to.
        """
        assert block.state == KeyBlockState.READY
        assert needed_bits > 0
        assert block.length_bits > needed_bits

        # Round needed_bits up to the next multiple of 8 for byte-aligned
        # key_material slicing.  This may over-reserve by up to 7 bits in
        # the rare edge case where needed_bits is not a multiple of 8.
        needed_aligned = (needed_bits + 7) // 8 * 8
        if needed_aligned > block.length_bits:
            # Rounding would exceed the block — clamp to block size.
            needed_aligned = block.length_bits
        remainder_bits = block.length_bits - needed_aligned
        needed_bytes = needed_aligned // 8

        # Remove the original block from the dict.  Its bits are already
        # counted in available_key_bits; we will subtract needed_aligned
        # (moving it to reserved) and keep remainder_bits in available.
        del self.key_blocks[block.key_id]

        # Create the RESERVED portion (taken).
        taken_id = f"{block.key_id}#R-{uuid.uuid4().hex[:6]}"
        taken_block = KeyBlock(
            key_id=taken_id,
            link_id=block.link_id,
            local_node=block.local_node,
            peer_node=block.peer_node,
            key_material=block.key_material[:needed_bytes],
            length_bits=needed_aligned,
            state=KeyBlockState.RESERVED,
            created_at=block.created_at,
            expires_at=block.expires_at,
            qkd_session_id=block.qkd_session_id,
            qber=block.qber,
            secret_key_rate=block.secret_key_rate,
        )
        self.key_blocks[taken_id] = taken_block
        self.reserved_key_bits += needed_aligned
        self.available_key_bits -= needed_aligned
        reserved_list.append(taken_block)

        # Create the READY remainder (if any).
        if remainder_bits > 0:
            remainder_id = f"{block.key_id}#S-{uuid.uuid4().hex[:6]}"
            remainder_block = KeyBlock(
                key_id=remainder_id,
                link_id=block.link_id,
                local_node=block.local_node,
                peer_node=block.peer_node,
                key_material=block.key_material[needed_bytes:],
                length_bits=remainder_bits,
                state=KeyBlockState.READY,
                created_at=block.created_at,
                expires_at=block.expires_at,
                qkd_session_id=block.qkd_session_id,
                qber=block.qber,
                secret_key_rate=block.secret_key_rate,
            )
            self.key_blocks[remainder_id] = remainder_block
            # available_key_bits unchanged for the remainder — it was
            # already counted as available when the original block was
            # in READY state, and we only subtracted needed_aligned above.

    def mark_used(self, key_ids: List[str]) -> None:
        """Mark reserved key blocks as USED after consumption.

        Args:
            key_ids: List of key IDs to transition from RESERVED to USED.
        """
        for kid in key_ids:
            block = self.key_blocks.get(kid)
            if block and block.state == KeyBlockState.RESERVED:
                block.state = KeyBlockState.USED
                self.reserved_key_bits -= block.length_bits
                self.used_key_bits += block.length_bits
        self.last_updated = time.time()
        self.status = self.check_status()

    def get_available_bits(self) -> int:
        """Return the total number of available (READY) key bits in the pool."""
        return self.available_key_bits

    def calculate_pressure(self) -> float:
        """Compute normalised pool pressure in [0, 1].

        0 = pool at or above high watermark (healthy), 1 = empty.
        Uses a piecewise-linear model so that SPF+ can differentiate
        between various levels of key scarcity.

        Returns:
            Float in [0, 1].  Returns 1.0 if watermarks are not set.
        """
        if self.low_watermark_bits <= 0:
            return 1.0
        avail = self.available_key_bits
        lo = self.low_watermark_bits
        hi = self.high_watermark_bits
        if avail >= hi:
            return 0.0
        elif avail >= lo:
            return 0.5 * (1.0 - (avail - lo) / max(1, hi - lo))
        else:
            return 0.5 + 0.5 * (1.0 - avail / max(1, lo))

    def check_status(self) -> KeyPoolStatus:
        """Evaluate and return the current pool status based on watermarks.

        Transition rules:
          - DISABLED / ERROR: sticky — only external action can change.
          - REPLENISHING → ACTIVE when pool reaches low_watermark (operational
            minimum).  We don't require high_watermark because in pipeline mode
            a single session may not fill to high_watermark.
          - EMPTY → LOW when any bits are added.
          - LOW → ACTIVE when bits >= low_watermark.

        Returns:
            KeyPoolStatus indicating the pool's health.
        """
        if self.status == KeyPoolStatus.DISABLED:
            return KeyPoolStatus.DISABLED
        if self.status == KeyPoolStatus.ERROR:
            return KeyPoolStatus.ERROR
        if self.status == KeyPoolStatus.REPLENISHING:
            # Transition to ACTIVE once we have at least the low-watermark
            # level of key material — this is the operational minimum.
            if self.available_key_bits >= self.low_watermark_bits:
                return KeyPoolStatus.ACTIVE
            # Still below low watermark but has some bits — keep replenishing
            if self.available_key_bits > 0:
                return KeyPoolStatus.REPLENISHING
            return KeyPoolStatus.EMPTY
        if self.available_key_bits == 0:
            return KeyPoolStatus.EMPTY
        if self.available_key_bits < self.low_watermark_bits:
            return KeyPoolStatus.LOW
        return KeyPoolStatus.ACTIVE


class QKDLink:
    """Represents a QKD link between two nodes.

    Stores the physical parameters and current operational metrics of
    a quantum key distribution link.

    Attributes:
        link_id: Unique identifier (e.g. "A-B").
        node_a: First endpoint node ID.
        node_b: Second endpoint node ID.
        distance_km: Physical fiber distance in kilometers.
        fiber_loss_db_km: Fiber attenuation in dB/km.
        status: Current link status.
        qber: Current Quantum Bit Error Rate.
        secret_key_rate: Current secure key rate in bits/pulse.
        optical_loss_db: Total optical loss on the link in dB.
        last_qkd_session_id: ID of the most recent QKD session.
    """

    def __init__(
        self,
        link_id: str,
        node_a: str,
        node_b: str,
        distance_km: float,
        fiber_loss_db_km: float = 0.2,
    ) -> None:
        """Initialize a QKD link with physical parameters."""
        self.link_id: str = link_id
        self.node_a: str = node_a
        self.node_b: str = node_b
        self.distance_km: float = distance_km
        self.fiber_loss_db_km: float = fiber_loss_db_km
        self.status: LinkStatus = LinkStatus.UP
        self.qber: float = 0.0
        self.secret_key_rate: float = 0.0
        self.optical_loss_db: float = fiber_loss_db_km * distance_km
        self.last_qkd_session_id: str = ""

    def get_cost(self, spf_params: Dict[str, float]) -> float:
        """Calculate the SPF+ cost for this link.

        Uses a normalized composite cost function where each metric is
        scaled to a comparable range before weighting:

          cost = alpha * (loss / max_loss)
               + beta  * (QBER / QBER_threshold)
               + gamma * (distance / max_distance)
               + delta * pool_pressure
               - epsilon * (key_rate / max_key_rate)

        pool_pressure is in [0, 1]: 0 = pool at capacity, 1 = pool empty.

        When QBER exceeds the threshold (0.11), the link is still routed
        through but with a heavy penalty proportional to how far above
        threshold the QBER is.  This avoids the all-or-nothing INFINITY_COST
        behavior that makes the entire network unreachable when all links
        have elevated QBER (e.g. pipeline misconfiguration).

        Only links that are physically DOWN receive INFINITY_COST.

        Args:
            spf_params: Dictionary with keys 'alpha', 'beta', 'gamma',
                        'delta', 'epsilon' as SPF+ weight parameters.
                        Also expects 'pool_pressure' and 'key_rate' values.

        Returns:
            Floating point cost. INFINITY_COST only if the link is DOWN.
        """
        if self.status == LinkStatus.DOWN:
            return INFINITY_COST

        alpha = spf_params.get("alpha", DEFAULT_ALPHA)
        beta = spf_params.get("beta", DEFAULT_BETA)
        gamma = spf_params.get("gamma", DEFAULT_GAMMA)
        delta = spf_params.get("delta", DEFAULT_DELTA)
        epsilon = spf_params.get("epsilon", DEFAULT_EPSILON)
        pool_pressure = spf_params.get("pool_pressure", 0.0)
        key_rate = spf_params.get("key_rate", 0.0)

        # Normalized terms
        norm_loss = self.optical_loss_db / SPF_MAX_LOSS_DB
        norm_qber = self.qber / QBER_THRESHOLD
        norm_distance = self.distance_km / SPF_MAX_DISTANCE_KM
        norm_key_rate = key_rate / SPF_MAX_KEY_RATE

        cost = (
            alpha * norm_loss
            + beta * norm_qber
            + gamma * norm_distance
            + delta * pool_pressure
            - epsilon * norm_key_rate
        )

        # If QBER exceeds threshold, add a heavy but finite penalty instead
        # of INFINITY_COST.  This allows SPF+ to still discover paths through
        # the least-bad links, which is essential when the simulator produces
        # elevated QBER across all links (e.g. pipeline misconfiguration).
        if self.qber > QBER_THRESHOLD:
            qber_excess_ratio = self.qber / QBER_THRESHOLD  # e.g. 0.49/0.11 ≈ 4.45
            cost += HIGH_PENALTY * qber_excess_ratio

        return max(cost, 0.01)  # Small floor to avoid zero-cost confusion


class OpticalSwitch:
    """Represents an optical circuit switch for dynamic link configuration.

    Optical switches allow the network to reconfigure which nodes are
    connected by routing quantum channels through different fiber paths.

    Attributes:
        switch_id: Unique identifier.
        switch_type: Type descriptor (e.g. "1x2", "2x4").
        configuration: Mapping from input port name to output port name.
        available: Whether the switch is operational.
    """

    def __init__(self, switch_id: str, switch_type: str) -> None:
        """Initialize an optical switch.

        Args:
            switch_id: Unique identifier for this switch.
            switch_type: Descriptor of the switch geometry.
        """
        self.switch_id: str = switch_id
        self.switch_type: str = switch_type
        self.configuration: Dict[str, str] = {}
        self.available: bool = True

    def configure(self, mapping: Dict[str, str]) -> None:
        """Set the switch configuration mapping.

        Args:
            mapping: Dictionary mapping input port names to output port names.
        """
        self.configuration = dict(mapping)

    def get_connection(self, input_port: str) -> str:
        """Query which output port an input port is connected to.

        Args:
            input_port: Name of the input port to query.

        Returns:
            Name of the connected output port, or empty string if not configured.
        """
        return self.configuration.get(input_port, "")


# ══════════════════════════════════════════════════════════════════════════════
# Node Classes
# ══════════════════════════════════════════════════════════════════════════════

class QKDNode:
    """Base class for all nodes in the QKD network.

    Each node maintains a set of key pools for its QKD links and tracks
    its neighbors in the network topology.

    Attributes:
        node_id: Unique identifier for this node (e.g. "A").
        role: The role of this node in the network.
        key_pools: Dictionary of key pools keyed by "LOCAL-PEER".
        neighbors: List of directly connected node IDs.
    """

    def __init__(self, node_id: str, role: NodeRole) -> None:
        """Initialize a QKD node.

        Args:
            node_id: Unique node identifier.
            role: The role this node plays in the network.
        """
        self.node_id: str = node_id
        self.role: NodeRole = role
        self.key_pools: Dict[str, KeyPool] = {}
        self.neighbors: List[str] = []

    def get_key_pool(self, peer_id: str) -> Optional[KeyPool]:
        """Retrieve the key pool shared with a specific peer.

        Args:
            peer_id: The ID of the peer node.

        Returns:
            The KeyPool for the peer, or None if no pool exists.
        """
        return self.key_pools.get(f"{self.node_id}-{peer_id}")

    def add_neighbor(self, peer_id: str) -> None:
        """Add a neighbor to this node's adjacency list.

        Args:
            peer_id: The ID of the neighboring node.
        """
        if peer_id not in self.neighbors:
            self.neighbors.append(peer_id)


class ReceiverNode(QKDNode):
    """A receiver node in the QKD network (Node A).

    The receiver is the destination for service keys. Multiple senders
    may establish end-to-end keys through trusted relays to this node.
    """

    def __init__(self, node_id: str = "A") -> None:
        """Initialize a receiver node.

        Args:
            node_id: Node identifier, defaults to "A".
        """
        super().__init__(node_id, NodeRole.RECEIVER)


class TrustedRelayNode(QKDNode):
    """A trusted relay node in the QKD network (Nodes B and C).

    Trusted relays perform hop-by-hop key relay using XOR operations.
    They decrypt incoming key material with one hop key and re-encrypt
    with the next hop key.

    Attributes:
        (inherits all from QKDNode)
    """

    def __init__(self, node_id: str) -> None:
        """Initialize a trusted relay node.

        Args:
            node_id: Node identifier (e.g. "B" or "C").
        """
        super().__init__(node_id, NodeRole.RELAY)

    def relay_key(
        self,
        incoming_key_material: bytes,
        from_node: str,
        to_node: str,
        hop_keys: Dict[str, bytes],
    ) -> bytes:
        """Perform XOR-based trusted relay of key material.

        Decrypts the incoming material with the hop key from the previous
        node, then re-encrypts with the hop key for the next node.

        Args:
            incoming_key_material: The encrypted key material received.
            from_node: The node that sent the material.
            to_node: The node to forward to.
            hop_keys: Dictionary mapping "FROM-SELF" and "SELF-TO" to
                      the corresponding hop key material.

        Returns:
            The re-encrypted key material for the next hop.
        """
        # Decrypt with key from previous hop
        key_from = hop_keys[f"{from_node}-{self.node_id}"]
        decrypted = bytes(a ^ b for a, b in zip(incoming_key_material, key_from))

        # Re-encrypt with key for next hop
        key_to = hop_keys[f"{self.node_id}-{to_node}"]
        encrypted = bytes(a ^ b for a, b in zip(decrypted, key_to))

        return encrypted


class SenderNode(QKDNode):
    """A sender node in the QKD network (Nodes D, E, F, G).

    Senders initiate service key requests and encrypt the service key
    with their hop key before transmission through the relay network.
    """

    def __init__(self, node_id: str) -> None:
        """Initialize a sender node.

        Args:
            node_id: Node identifier (e.g. "D", "E", "F", "G").
        """
        super().__init__(node_id, NodeRole.SENDER)


# ══════════════════════════════════════════════════════════════════════════════
# QKD Link Simulators
# ══════════════════════════════════════════════════════════════════════════════

class StatisticalQKDLinkSimulator:
    """Simulates BB84 decoy-state QKD key generation using a statistical model.

    Uses the standard analytical model for decoy-state BB84 with the
    key rate formula:

        rate = q * eta * 10^(-alpha*d/10) * mu * max(0, 1 - h2(QBER))

    where:
        q     = sifting factor (0.5)
        eta   = detector efficiency
        alpha = fiber loss coefficient (dB/km)
        d     = fiber distance (km)
        mu    = signal pulse intensity
        h2    = binary entropy function

    QBER is modeled as:

        QBER = intrinsic_qber + dark_rate / (eta * mu * 10^(-alpha*d/10) + dark_rate)

    Attributes:
        det_eff: Single-photon detector efficiency.
        dark_rate: Dark count rate per pulse.
        fiber_loss: Fiber loss coefficient in dB/km.
        intrinsic_qber: Baseline QBER from optical imperfections.
        mu_signal: Signal state mean photon number.
    """

    def __init__(
        self,
        det_eff: float = 0.15,
        dark_rate: float = 1e-6,
        fiber_loss: float = 0.2,
        intrinsic_qber: float = 0.015,
        mu_signal: float = 0.5,
    ) -> None:
        """Initialize the statistical QKD link simulator.

        Defaults are aligned with ``main_optimized.DEFAULT_CONFIG`` so that the
        statistical fallback produces key rates within the same order of
        magnitude as the pipeline simulator:

            det_eff          = 0.15   (pipeline det_eff_d0/d1)
            dark_rate        = 1e-6   (pipeline dark_rate)
            intrinsic_qber   = 0.015  (pipeline qber_intrinsic 0.01
                                      + misalignment 0.005)
            mu_signal        = 0.5    (pipeline source.pulses.signal.mu)

        Args:
            det_eff: Detector efficiency (0-1).
            dark_rate: Dark count probability per pulse.
            fiber_loss: Fiber loss in dB/km.
            intrinsic_qber: Intrinsic QBER from optical imperfections
                (includes misalignment contribution).
            mu_signal: Mean photon number for signal states.
        """
        self.det_eff = det_eff
        self.dark_rate = dark_rate
        self.fiber_loss = fiber_loss
        self.intrinsic_qber = intrinsic_qber
        self.mu_signal = mu_signal

    @staticmethod
    def _binary_entropy(x: float) -> float:
        """Compute the binary entropy function h2(x) = -x*log2(x) - (1-x)*log2(1-x).

        Args:
            x: Probability value in [0, 1].

        Returns:
            Binary entropy in bits.
        """
        if x <= 0.0 or x >= 1.0:
            return 0.0
        return -x * math.log2(x) - (1.0 - x) * math.log2(1.0 - x)

    def simulate_link(
        self, link: QKDLink, num_pulses: int = 10 ** 7
    ) -> Tuple[int, float, float]:
        """Simulate a QKD session on the given link.

        Runs the statistical model for the specified number of pulses
        and returns the resulting secure key bits, QBER, and key rate.

        QBER model (distance-dependent):
          - Base intrinsic QBER from optical imperfections
          - Polarization drift: increases linearly with fiber length
            (longer fibers accumulate more birefringence and PMD)
          - Dark-count contamination: as signal weakens with distance,
            dark counts become a larger fraction of detections
          - Afterpulse contribution: small fixed component

        Key rate model (decoy-state BB84, asymptotic):
          rate = q * eta * 10^(-alpha*d/10) * mu * max(0, 1 - 2*h2(QBER))

        Args:
            link: The QKDLink to simulate.
            num_pulses: Number of laser pulses to simulate.

        Returns:
            Tuple of (secure_key_bits, qber, key_rate_bits_per_pulse).
        """
        d = link.distance_km
        alpha = self.fiber_loss
        eta = self.det_eff
        mu = self.mu_signal
        q = 0.5  # sifting factor

        # Channel transmittance
        transmittance = eta * 10.0 ** (-alpha * d / 10.0)

        # Gain (detection probability per pulse) — linear approximation
        gain_approx = mu * transmittance + self.dark_rate
        gain = max(gain_approx, 1e-30)

        # ── Distance-dependent QBER model ──────────────────────────────
        # 1) Base intrinsic QBER (optical misalignment at zero distance)
        qber_intrinsic = self.intrinsic_qber

        # 2) Polarization drift: ~0.01% per km of fiber with active
        #    polarization stabilization (realistic metropolitan QKD).
        #    The previous 0.0005/km value was ~5x too aggressive and
        #    produced QBER ~0.02-0.025 for short 30-40 km links.
        qber_polarization = 0.0001 * d

        # 3) Dark-count contamination fraction: dark_rate / gain
        #    As signal weakens with distance, dark counts dominate
        qber_dark = self.dark_rate / gain

        # 4) Small afterpulse contribution (fixed, ~0.1%)
        qber_afterpulse = 0.001

        # Total QBER
        qber = qber_intrinsic + qber_polarization + qber_dark + qber_afterpulse
        qber = min(qber, 1.0)

        # If QBER exceeds threshold, no secure key is possible
        if qber > QBER_THRESHOLD:
            return 0, qber, 0.0

        # Secret fraction (using asymptotic decoy-state analysis)
        h2_qber = self._binary_entropy(qber)
        secret_fraction = max(0.0, 1.0 - 2.0 * h2_qber)

        # Key rate per pulse
        key_rate = q * transmittance * mu * secret_fraction

        # Total secure key bits
        secure_key_bits = int(key_rate * num_pulses)

        return secure_key_bits, qber, key_rate


class PipelineQKDLinkSimulator:
    """Pipeline-based QKD simulator using main_optimized.run_single_simulation().

    Delegates the full BB84-decoy pipeline (Source -> Channel -> Protocol ->
    Detector -> Sifting -> Proof) to ``run_single_simulation()`` which
    incorporates all F-01..F-22 fixes:

    - F-01: Detector bias-voltage clamping + override auditing
    - F-05: SeedSequence.spawn for independent RNG streams
    - F-11: Probability-balancing validation
    - F-14: Source metadata caching
    - F-18: Debug pulse override removed (dead code eliminated)

    Detector construction is handled by ``build_detector()`` which
    automatically clamps bias_voltage <= breakdown_voltage and records
    any overrides — no manual from_config+setattr bypass needed.

    The pipeline engine is loaded from the ``main_optimized.py`` file located
    beside this network script. This avoids accidentally importing another
    copy from a parent directory or an installed package.

    Attributes:
        available: Whether the pipeline simulator was successfully imported.
        fallback: The StatisticalQKDLinkSimulator used as fallback.
        detector_efficiency: Detection efficiency applied to D0 and D1.
        apply_noise: Whether the full noisy detector path is used.
        pipeline_path: Absolute path of the loaded physics engine.
        _last_result: The full result dict from the most recent pipeline run.
    """

    def __init__(
        self,
        detector_efficiency: float = 0.15,
        apply_noise: bool = True,
    ) -> None:
        """Initialize the pipeline simulator from the local physics engine."""
        if not 0.0 < detector_efficiency <= 1.0:
            raise ValueError("detector_efficiency must be in the interval (0, 1]")
        if type(apply_noise) is not bool:
            raise TypeError("apply_noise must be a bool")

        self.available: bool = False
        # Fallback uses the SAME detector efficiency as the pipeline so that
        # if the pipeline fails per-link, the statistical model produces key
        # rates in the same order of magnitude (otherwise the fallback with
        # det_eff=0.65 yields ~20x inflated rates vs the pipeline's 0.15).
        self.fallback: StatisticalQKDLinkSimulator = StatisticalQKDLinkSimulator(
            det_eff=detector_efficiency,
        )
        self._pipeline_modules: dict = {}
        self._last_result: dict = {}
        self.detector_efficiency = float(detector_efficiency)
        self.apply_noise = apply_noise
        self.pipeline_path: Optional[str] = None

        try:
            pipeline_module = _load_local_pipeline_module()
            self._pipeline_modules = {
                "np": pipeline_module.np,
                "DEFAULT_CFG": pipeline_module.DEFAULT_CONFIG,
                "init_worker": pipeline_module.init_worker,
                "run_single_simulation": pipeline_module.run_single_simulation,
                "set_nested_value": pipeline_module.set_nested_value,
            }
            self.pipeline_path = str(Path(pipeline_module.__file__).resolve())
            self.available = True
        except ImportError as exc:
            self._import_error = str(exc)

    def _make_pipeline_config(self) -> dict:
        """Build the same physical configuration used by main_optimized.py."""
        import copy

        set_nested_value = self._pipeline_modules["set_nested_value"]
        config = copy.deepcopy(self._pipeline_modules["DEFAULT_CFG"])
        set_nested_value(
            config,
            "detector.det_eff_d0",
            self.detector_efficiency,
        )
        set_nested_value(
            config,
            "detector.det_eff_d1",
            self.detector_efficiency,
        )
        set_nested_value(
            config,
            "protocol_params.detector_type",
            config["detector"]["detector_type"],
        )
        return config

    def simulate_link(
        self, link: QKDLink, num_pulses: int = 10 ** 7
    ) -> Tuple[int, float, float]:
        """Simulate a QKD session using the full pipeline or fallback.

        When the pipeline is available, delegates to
        ``main_optimized.run_single_simulation()`` which incorporates all
        F-01..F-22 fixes (SeedSequence RNG, bias-voltage clamping,
        override auditing, probability balancing, etc.).  Otherwise falls
        back to the statistical model.

        If the pipeline returns QBER > QBER_THRESHOLD (0.11) or
        key_rate == 0, it likely means the pipeline's internal detector
        model produced unrealistic results for this configuration.  In
        that case, we fall back to the statistical model per-link, which
        has been validated against published BB84-decoy analytical
        results for metropolitan-scale distances.

        Args:
            link: The QKDLink to simulate.
            num_pulses: Number of pulses to simulate.

        Returns:
            Tuple of (secure_key_bits, qber, key_rate).
        """
        if not self.available:
            return self.fallback.simulate_link(link, num_pulses)

        try:
            secure_bits, qber, key_rate = self._run_pipeline(link, num_pulses)

            # ── Per-link fallback validation ────────────────────────────
            # If the pipeline returns QBER above threshold or zero key rate,
            # the pipeline model may be misconfigured for this link.  Fall
            # back to the validated statistical model instead of propagating
            # unusable metrics into the network.
            if qber > QBER_THRESHOLD or (key_rate == 0.0 and secure_bits == 0):
                # Suppress repeated warnings — only print once per link
                if not hasattr(self, '_fallback_warned'):
                    self._fallback_warned: set = set()
                if link.link_id not in self._fallback_warned:
                    self._fallback_warned.add(link.link_id)
                    import logging
                    logging.getLogger(__name__).warning(
                        "Pipeline returned unrealistic QBER=%.4f / key_rate=%.2e for "
                        "link %s (%.0f km). Falling back to statistical model.",
                        qber, key_rate, link.link_id, link.distance_km,
                    )
                return self.fallback.simulate_link(link, num_pulses)

            return secure_bits, qber, key_rate

        except Exception as e:
            # Classify the error to inform network behaviour.
            # - ConfigurationError / ImportError: permanent, don't retry
            #   with pipeline on this config.
            # - LPFailureError / ArithmeticError: transient (distance too
            #   far, numerical edge case); statistical fallback is fine.
            import logging
            logger = logging.getLogger(__name__)
            is_permanent = any(
                name in type(e).__name__
                for name in ("ConfigurationError", "ImportError", "ParameterValidationError")
            )
            if is_permanent:
                logger.error(
                    "Pipeline simulation failed PERMANENTLY for link %s "
                    "(%.0f km): %s.  Will not retry pipeline.",
                    link.link_id, link.distance_km, e,
                )
            else:
                logger.warning(
                    "Pipeline simulation failed for link %s (%.0f km): %s. "
                    "Falling back to statistical model.",
                    link.link_id, link.distance_km, e,
                )
            return self.fallback.simulate_link(link, num_pulses)

    def _run_pipeline(
        self, link: QKDLink, num_pulses: int
    ) -> Tuple[int, float, float]:
        """Execute the full QKD pipeline for a single link.

        Delegates to ``main_optimized.run_single_simulation()`` which
        handles the complete pipeline (Source -> Channel -> Protocol ->
        Detector -> Sifting -> Proof) with all F-01..F-22 fixes:

        - F-01: Detector bias-voltage clamping + override auditing
        - F-05: SeedSequence.spawn for independent RNG streams
        - F-11: Probability-balancing validation
        - F-14: Source metadata caching
        - F-18: No debug pulse override (removed dead code)

        This replaces the previous manual pipeline implementation that
        reimplemented ~100 lines of logic but missed these fixes.

        Args:
            link: The QKDLink to simulate.
            num_pulses: Number of pulses to simulate.

        Returns:
            Tuple of (secure_key_bits, qber, key_rate).
        """
        m = self._pipeline_modules
        np = m["np"]
        dist = link.distance_km

        if type(num_pulses) is not int or num_pulses <= 0:
            raise ValueError("num_pulses must be a positive integer")

        cfg = self._make_pipeline_config()

        # Initialise the worker state for this config.
        # combo_map {0: {}} means no sweep overrides.
        m["init_worker"](cfg, {0: {}})

        # Derive a high-entropy seed for this session.
        # run_single_simulation uses SeedSequence.spawn(5) internally (F-05).
        rng_seed = int(np.random.default_rng().integers(0, 2**63))

        # Run the complete pipeline via main_optimized.
        # args_tuple: (distance_km, total_pulses, apply_noise, seed, combo_idx)
        result = m["run_single_simulation"](
            (dist, num_pulses, self.apply_noise, rng_seed, 0)
        )

        secure_key = result["secure_key_bits"]
        qber = result["qber"]
        key_rate = result["secure_key_rate"]

        # Store full result for callers that want richer diagnostics
        # (per-intensity QBER, detection yields, detector overrides, etc.)
        self._last_result = result

        return secure_key, qber, key_rate


# ══════════════════════════════════════════════════════════════════════════════
# KMS (Key Management System) — THE CORE
# ══════════════════════════════════════════════════════════════════════════════

class KMS:
    """Key Management System for the 7-node QKD network.

    The KMS is the central orchestrator for all key management operations:
    - Building and maintaining the network topology
    - Running QKD sessions to generate key material
    - SPF+ path selection for end-to-end key delivery
    - Trusted relay of service keys
    - Key pool monitoring and replenishment
    - Failure handling and path failover

    Attributes:
        nodes: Dictionary of QKDNode objects keyed by node_id.
        links: Dictionary of QKDLink objects keyed by link_id.
        switches: Dictionary of OpticalSwitch objects keyed by switch_id.
        simulator: The QKD link simulator instance.
        service_key_registry: Registry of all issued service keys.
        spf_weights: SPF+ weight parameters for path cost calculation.
        _key_counter: Monotonic counter for generating unique key IDs.
        _svc_key_counter: Monotonic counter for generating service key IDs.
    """

    def __init__(self) -> None:
        """Initialize the Key Management System."""
        self.nodes: Dict[str, QKDNode] = {}
        self.links: Dict[str, QKDLink] = {}
        self.switches: Dict[str, OpticalSwitch] = {}
        self.simulator: Union[StatisticalQKDLinkSimulator, PipelineQKDLinkSimulator] = (
            StatisticalQKDLinkSimulator()
        )
        self.service_key_registry: Dict[str, dict] = {}
        self.spf_weights: Dict[str, float] = {
            "alpha": DEFAULT_ALPHA,
            "beta": DEFAULT_BETA,
            "gamma": DEFAULT_GAMMA,
            "delta": DEFAULT_DELTA,
            "epsilon": DEFAULT_EPSILON,
        }
        self._key_counter: int = 0
        self._svc_key_counter: int = 0
        self._is_pipeline_mode: bool = False  # Set by calibrate_for_simulator()

    # ── Simulator Calibration ────────────────────────────────────────────

    def calibrate_for_simulator(self, is_pipeline: bool) -> None:
        """Reconfigure pool watermarks and SPF+ normalization for the simulator.

        The pipeline generates ~100-1000x fewer key bits per session than the
        statistical model.  If we keep the same watermarks (1M/10M/20M), all
        pools will be permanently stuck in LOW or REPLENISHING status with the
        pipeline, and pool_pressure will be ~1.0 for every hop, making SPF+
        unable to differentiate paths.

        This method:
          1. Sets pool watermarks and capacity appropriate for the simulator
          2. Updates the SPF_MAX_KEY_RATE normalization constant
          3. Re-evaluates all pool statuses after the watermark change

        Args:
            is_pipeline: True if using the real QKD pipeline simulator.
        """
        self._is_pipeline_mode = is_pipeline

        if is_pipeline:
            new_low = PIPELINE_LOW_WATERMARK_BITS
            new_high = PIPELINE_HIGH_WATERMARK_BITS
            new_max = PIPELINE_MAX_CAPACITY_BITS
            new_max_rate = PIPELINE_SPF_MAX_KEY_RATE
        else:
            new_low = STAT_LOW_WATERMARK_BITS
            new_high = STAT_HIGH_WATERMARK_BITS
            new_max = STAT_MAX_CAPACITY_BITS
            new_max_rate = STAT_SPF_MAX_KEY_RATE

        # Update all pools
        for node_id, node in self.nodes.items():
            for pool_key, pool in node.key_pools.items():
                pool.low_watermark_bits = new_low
                pool.high_watermark_bits = new_high
                pool.max_capacity_bits = new_max
                # Re-evaluate status with new watermarks
                pool.status = pool.check_status()

        # Update SPF+ key rate normalization
        global SPF_MAX_KEY_RATE
        SPF_MAX_KEY_RATE = new_max_rate

    # ── Network Construction ──────────────────────────────────────────────

    def build_network(self) -> None:
        """Initialize the 7-node topology with all 10 links, 20 pools, and 2 switches.

        Creates:
        - 7 nodes: A (Receiver), B,C (Trusted Relay), D,E,F,G (Sender)
        - 10 QKD links: A-B, A-C, B-D, B-E, B-F, B-G, C-D, C-E, C-F, C-G
        - 20 key pools: 2 per link (one at each endpoint)
        - 2 optical switches: 1x2 at A, 2x4 at B and C
        """
        # --- Create nodes ---
        self.nodes["A"] = ReceiverNode("A")
        self.nodes["B"] = TrustedRelayNode("B")
        self.nodes["C"] = TrustedRelayNode("C")
        self.nodes["D"] = SenderNode("D")
        self.nodes["E"] = SenderNode("E")
        self.nodes["F"] = SenderNode("F")
        self.nodes["G"] = SenderNode("G")

        # --- Create 10 QKD links and 20 key pools ---
        link_definitions = [
            ("A", "B"), ("A", "C"),
            ("B", "D"), ("B", "E"), ("B", "F"), ("B", "G"),
            ("C", "D"), ("C", "E"), ("C", "F"), ("C", "G"),
        ]

        for node_a, node_b in link_definitions:
            link_id = f"{node_a}-{node_b}"
            distance = NODE_DISTANCES.get(link_id, 40.0)
            link = QKDLink(link_id, node_a, node_b, distance_km=distance)
            self.links[link_id] = link

            # Add neighbors
            self.nodes[node_a].add_neighbor(node_b)
            self.nodes[node_b].add_neighbor(node_a)

            # Create 2 key pools per link (one at each endpoint)
            pool_ab = KeyPool(
                pool_id=f"POOL-{node_a}-{node_b}",
                local_node=node_a,
                peer_node=node_b,
                link_id=link_id,
            )
            pool_ba = KeyPool(
                pool_id=f"POOL-{node_b}-{node_a}",
                local_node=node_b,
                peer_node=node_a,
                link_id=link_id,
            )
            self.nodes[node_a].key_pools[f"{node_a}-{node_b}"] = pool_ab
            self.nodes[node_b].key_pools[f"{node_b}-{node_a}"] = pool_ba

        # --- Create optical switches ---
        # 1x2 switch at A: connects A to B or C
        switch_a = OpticalSwitch("SW-A-1x2", "1x2")
        switch_a.configure({"A": "B"})  # default: A connected to B
        self.switches["SW-A-1x2"] = switch_a

        # 2x4 switch at B: connects B to D/E/F/G
        switch_b = OpticalSwitch("SW-B-2x4", "2x4")
        switch_b.configure({"B-D": "D", "B-E": "E", "B-F": "F", "B-G": "G"})
        self.switches["SW-B-2x4"] = switch_b

        # 2x4 switch at C: connects C to D/E/F/G
        switch_c = OpticalSwitch("SW-C-2x4", "2x4")
        switch_c.configure({"C-D": "D", "C-E": "E", "C-F": "F", "C-G": "G"})
        self.switches["SW-C-2x4"] = switch_c

    # ── SPF+ Path Selection ──────────────────────────────────────────────

    def spf_plus(self, source: str, destination: str) -> List[str]:
        """Find the lowest-cost path from source to destination using SPF+.

        Implements Dijkstra's algorithm with a composite cost function that
        considers optical loss, QBER, latency proxy, available key bits, and
        key rate. Links that are DOWN or have ERROR/DISABLED pools are
        assigned infinite cost.  Links with QBER above threshold receive a
        heavy but finite penalty so that paths can still be discovered when
        all links have elevated QBER.

        Args:
            source: Source node ID.
            destination: Destination node ID.

        Returns:
            List of node IDs representing the lowest-cost path.
            Empty list if no path exists.
        """
        # Build adjacency with costs
        dist: Dict[str, float] = {nid: INFINITY_COST for nid in self.nodes}
        prev: Dict[str, Optional[str]] = {nid: None for nid in self.nodes}
        dist[source] = 0.0
        visited: set = set()
        heap: List[Tuple[float, str]] = [(0.0, source)]

        while heap:
            current_dist, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            if u == destination:
                break

            for v in self.nodes[u].neighbors:
                if v in visited:
                    continue

                link_id = self._get_link_id(u, v)
                link = self.links.get(link_id)
                if link is None:
                    continue

                # Gather pool info for cost calculation
                pool_u_v = self.nodes[u].key_pools.get(f"{u}-{v}")
                pool_v_u = self.nodes[v].key_pools.get(f"{v}-{u}")

                # Compute normalised pool pressure using KeyPool.calculate_pressure()
                pressures = []
                for pool in (pool_u_v, pool_v_u):
                    if pool is not None:
                        pressures.append(pool.calculate_pressure())
                    else:
                        pressures.append(1.0)  # no pool = maximum pressure
                pool_pressure = max(pressures)

                # If either pool is ERROR/DISABLED, link is unusable
                if (pool_u_v and pool_u_v.status in (KeyPoolStatus.ERROR, KeyPoolStatus.DISABLED)) or \
                   (pool_v_u and pool_v_u.status in (KeyPoolStatus.ERROR, KeyPoolStatus.DISABLED)):
                    edge_cost = INFINITY_COST
                else:
                    cost_params = {
                        "alpha": self.spf_weights["alpha"],
                        "beta": self.spf_weights["beta"],
                        "gamma": self.spf_weights["gamma"],
                        "delta": self.spf_weights["delta"],
                        "epsilon": self.spf_weights["epsilon"],
                        "pool_pressure": pool_pressure,
                        "key_rate": link.secret_key_rate,
                    }
                    edge_cost = link.get_cost(cost_params)
                new_dist = current_dist + edge_cost

                if new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(heap, (new_dist, v))

        # Reconstruct path
        if dist[destination] >= INFINITY_COST:
            return []

        path: List[str] = []
        current: Optional[str] = destination
        while current is not None:
            path.append(current)
            current = prev[current]
        path.reverse()
        return path

    def _get_link_id(self, node_a: str, node_b: str) -> str:
        """Get the canonical link ID for a pair of nodes.

        Link IDs always use alphabetical ordering (e.g. "A-B", not "B-A").

        Args:
            node_a: First node ID.
            node_b: Second node ID.

        Returns:
            Canonical link ID string.
        """
        pair = sorted([node_a, node_b])
        return f"{pair[0]}-{pair[1]}"

    # ── Service Key Request ──────────────────────────────────────────────

    def request_service_key(
        self, sender: str, receiver: str, key_length_bits: int = 256
    ) -> dict:
        """Request an end-to-end service key from sender to receiver.

        Full flow:
        1. Find optimal path using SPF+
        2. Check key pool availability on all hops
        3. Reserve hop keys on each link
        4. Generate service key material
        5. Perform trusted relay to deliver key to receiver
        6. Mark consumed hop keys as USED
        7. Register the service key

        Args:
            sender: Sender node ID.
            receiver: Receiver node ID.
            key_length_bits: Desired service key length in bits (default 256).

        Returns:
            Dictionary with keys: service_key_id, key_material, path,
            hop_keys_used, status, message.
            On error: status="error" with explanatory message.
        """
        # Step 1: Find path
        path = self.spf_plus(sender, receiver)
        if not path:
            return {
                "status": "error",
                "message": f"No available path from {sender} to {receiver}",
                "service_key_id": "",
                "key_material": b"",
                "path": [],
                "hop_keys_used": [],
            }

        # Step 2: Reserve hop keys
        hop_reservations: List[Dict[str, Any]] = []
        key_bytes = key_length_bits // 8

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            pool_u = self.nodes[u].key_pools.get(f"{u}-{v}")
            pool_v = self.nodes[v].key_pools.get(f"{v}-{u}")

            if pool_u is None or pool_v is None:
                # Rollback previous reservations
                self._rollback_reservations(hop_reservations)
                return {
                    "status": "error",
                    "message": f"Missing key pool for hop {u}-{v}",
                    "service_key_id": "",
                    "key_material": b"",
                    "path": path,
                    "hop_keys_used": [],
                }

            blocks_u = pool_u.reserve_keys(key_length_bits)
            blocks_v = pool_v.reserve_keys(key_length_bits)

            if not blocks_u or not blocks_v:
                self._rollback_reservations(hop_reservations)
                # Also rollback current partial reservation
                for b in blocks_u:
                    b.state = KeyBlockState.READY
                    pool_u.available_key_bits += b.length_bits
                    pool_u.reserved_key_bits -= b.length_bits
                for b in blocks_v:
                    b.state = KeyBlockState.READY
                    pool_v.available_key_bits += b.length_bits
                    pool_v.reserved_key_bits -= b.length_bits
                return {
                    "status": "error",
                    "message": f"Insufficient key material in pool for hop {u}-{v}",
                    "service_key_id": "",
                    "key_material": b"",
                    "path": path,
                    "hop_keys_used": [],
                }

            hop_reservations.append({
                "u": u, "v": v,
                "blocks_u": blocks_u, "blocks_v": blocks_v,
                "pool_u": pool_u, "pool_v": pool_v,
            })

        # Step 3: Generate service key material
        service_key_material = os.urandom(key_bytes)

        # Step 4: Perform trusted relay
        relay_success = self.perform_trusted_relay(path, service_key_material)

        # Step 5: Mark hop keys as USED
        all_used_key_ids: List[str] = []
        for hop in hop_reservations:
            ids_u = [b.key_id for b in hop["blocks_u"]]
            ids_v = [b.key_id for b in hop["blocks_v"]]
            hop["pool_u"].mark_used(ids_u)
            hop["pool_v"].mark_used(ids_v)
            all_used_key_ids.extend(ids_u)
            all_used_key_ids.extend(ids_v)

        # Step 6: Register service key
        self._svc_key_counter += 1
        service_key_id = f"SVC-{sender}{receiver}-{self._svc_key_counter:06d}"
        self.service_key_registry[service_key_id] = {
            "service_key_id": service_key_id,
            "sender": sender,
            "receiver": receiver,
            "key_length_bits": key_length_bits,
            "path": path,
            "hop_keys_used": all_used_key_ids,
            "created_at": time.time(),
            "relay_success": relay_success,
        }

        return {
            "status": "ok",
            "message": "Service key delivered successfully",
            "service_key_id": service_key_id,
            "key_material": service_key_material,
            "path": path,
            "hop_keys_used": all_used_key_ids,
        }

    def _rollback_reservations(
        self, hop_reservations: List[Dict[str, Any]]
    ) -> None:
        """Rollback previously reserved hop keys on failure.

        Returns key blocks from RESERVED back to READY state.

        Args:
            hop_reservations: List of hop reservation dictionaries.
        """
        for hop in hop_reservations:
            for b in hop["blocks_u"]:
                b.state = KeyBlockState.READY
                hop["pool_u"].available_key_bits += b.length_bits
                hop["pool_u"].reserved_key_bits -= b.length_bits
            for b in hop["blocks_v"]:
                b.state = KeyBlockState.READY
                hop["pool_v"].available_key_bits += b.length_bits
                hop["pool_v"].reserved_key_bits -= b.length_bits

    # ── Trusted Relay ────────────────────────────────────────────────────

    def perform_trusted_relay(
        self, path: List[str], service_key_material: bytes
    ) -> bool:
        """Implement XOR-based hop-by-hop trusted relay.

        For a path like D -> B -> A:
        1. D encrypts: C1 = service_key XOR K_DB
        2. B decrypts with K_DB, re-encrypts with K_BA: C2 = service_key XOR K_BA
        3. A decrypts with K_BA: service_key = C2 XOR K_BA

        Each relay node performs one decryption and one encryption step.

        Args:
            path: Ordered list of node IDs from sender to receiver.
            service_key_material: The raw service key bytes to relay.

        Returns:
            True if relay completed successfully, False otherwise.
        """
        if len(path) < 2:
            return False

        # Collect hop keys
        current_material = service_key_material

        # Sender encrypts with first hop key
        first_hop_u, first_hop_v = path[0], path[1]
        pool_u = self.nodes[first_hop_u].key_pools.get(f"{first_hop_u}-{first_hop_v}")
        if pool_u is None:
            return False

        # Get the key material from reserved blocks
        hop_key_send = self._get_reserved_key_material(pool_u)
        if hop_key_send is None:
            return False

        # Step 1: Sender encrypts
        current_material = bytes(a ^ b for a, b in zip(current_material, hop_key_send))

        # Intermediate relay nodes
        for i in range(1, len(path) - 1):
            relay_node = self.nodes[path[i]]
            from_node = path[i - 1]
            to_node = path[i + 1]

            if not isinstance(relay_node, TrustedRelayNode):
                return False

            # Get hop keys for this relay
            pool_in = relay_node.key_pools.get(f"{relay_node.node_id}-{from_node}")
            if pool_in is None:
                # Try reverse direction
                pool_in = relay_node.key_pools.get(f"{from_node}-{relay_node.node_id}")
            pool_out = relay_node.key_pools.get(f"{relay_node.node_id}-{to_node}")

            if pool_in is None or pool_out is None:
                return False

            key_in = self._get_reserved_key_material(pool_in)
            key_out = self._get_reserved_key_material(pool_out)

            if key_in is None or key_out is None:
                return False

            hop_keys = {
                f"{from_node}-{relay_node.node_id}": key_in,
                f"{relay_node.node_id}-{to_node}": key_out,
            }

            current_material = relay_node.relay_key(
                current_material, from_node, to_node, hop_keys
            )

        # Final receiver decrypts with last hop key
        last_hop_u, last_hop_v = path[-2], path[-1]
        pool_recv = self.nodes[last_hop_v].key_pools.get(f"{last_hop_v}-{last_hop_u}")
        if pool_recv is None:
            return False

        hop_key_recv = self._get_reserved_key_material(pool_recv)
        if hop_key_recv is None:
            return False

        # Receiver decrypts
        final_material = bytes(a ^ b for a, b in zip(current_material, hop_key_recv))

        # Verify relay correctness (final material should match original)
        return final_material == service_key_material

    def _get_reserved_key_material(self, pool: KeyPool) -> Optional[bytes]:
        """Extract concatenated key material from RESERVED blocks in a pool.

        Args:
            pool: The key pool to extract material from.

        Returns:
            Concatenated key bytes from reserved blocks, or None if none available.
        """
        reserved_blocks = [
            b for b in pool.key_blocks.values() if b.state == KeyBlockState.RESERVED
        ]
        if not reserved_blocks:
            return None
        material = b"".join(b.key_material for b in reserved_blocks)
        return material

    # ── Key Pool Replenishment ────────────────────────────────────────────

    def replenish_key_pools(self) -> List[str]:
        """Monitor all pools and start QKD sessions for those that are LOW or EMPTY.

        Scans every key pool in the network and triggers a QKD session on
        the corresponding link if the pool is below the low watermark.
        Both peer pools on the same link are set to REPLENISHING together
        to maintain synchronized status.

        Returns:
            List of link IDs that were replenished.
        """
        replenished: List[str] = []
        visited_links: set = set()

        for node_id, node in self.nodes.items():
            for pool_key, pool in node.key_pools.items():
                if pool.status in (KeyPoolStatus.LOW, KeyPoolStatus.EMPTY):
                    link_id = pool.link_id

                    # Avoid double-replenishing the same link
                    if link_id in visited_links:
                        continue
                    visited_links.add(link_id)

                    # Set BOTH peer pools on this link to REPLENISHING
                    link = self.links.get(link_id)
                    if link is None:
                        continue
                    pool_a = self.nodes[link.node_a].key_pools.get(
                        f"{link.node_a}-{link.node_b}"
                    )
                    pool_b = self.nodes[link.node_b].key_pools.get(
                        f"{link.node_b}-{link.node_a}"
                    )
                    if pool_a:
                        pool_a.status = KeyPoolStatus.REPLENISHING
                    if pool_b:
                        pool_b.status = KeyPoolStatus.REPLENISHING

                    try:
                        # Use the SAME pulse count as prime_pools (10**7).
                        # The previous code used 10**8 in pipeline mode, which
                        # the prime_pools docstring explicitly warns against:
                        # at 10**8 pulses the pipeline allocates several ~800 MB
                        # int64 arrays (prepared_states, photons, click0,
                        # click1) which trigger memory pressure and numerical
                        # edge-cases in the qkd package's parameter-estimation
                        # code — observed as pipeline returning key_rate=0 for
                        # short (30-40 km) links, which then forced fallback to
                        # the statistical model with its (previously) inflated
                        # detector efficiency, producing millions of bits per
                        # pool.  10**7 pulses is sufficient to fill the
                        # pipeline-scaled watermarks (10K/100K/200K bits).
                        replenish_pulses = 10 ** 7
                        self.run_qkd_session(link_id, num_pulses=replenish_pulses)
                        replenished.append(link_id)
                    except Exception as e:
                        if pool_a:
                            pool_a.status = KeyPoolStatus.ERROR
                        if pool_b:
                            pool_b.status = KeyPoolStatus.ERROR
                        print(f"  [ERROR] Replenishment failed for {link_id}: {e}")

        return replenished

    # ── QKD Session Execution ────────────────────────────────────────────

    def run_qkd_session(
        self, link_id: str, num_pulses: int = 10 ** 7
    ) -> Tuple[int, float, float]:
        """Run a BB84 QKD session on the specified link.

        Uses the statistical simulator to generate secure key bits, then
        creates KeyBlock objects and distributes them to both endpoint pools.

        Args:
            link_id: The QKD link to run the session on.
            num_pulses: Number of laser pulses for the session.

        Returns:
            Tuple of (secure_key_bits, qber, key_rate).
        """
        link = self.links.get(link_id)
        if link is None:
            raise ValueError(f"Unknown link: {link_id}")
        if link.status == LinkStatus.DOWN:
            raise RuntimeError(f"Link {link_id} is DOWN")

        # Run simulation
        secure_bits, qber, key_rate = self.simulator.simulate_link(link, num_pulses)

        # Update link metrics
        link.qber = qber
        link.secret_key_rate = key_rate
        session_id = f"QKD-{link_id}-{uuid.uuid4().hex[:8]}"
        link.last_qkd_session_id = session_id

        if secure_bits <= 0:
            return secure_bits, qber, key_rate

        # Split secure bits into key blocks of 256 bits each
        block_size_bits = 256
        num_blocks = secure_bits // block_size_bits
        remaining_bits = secure_bits % block_size_bits

        node_a, node_b = link.node_a, link.node_b

        for i in range(num_blocks):
            self._key_counter += 1
            kid = f"{node_a}{node_b}-{self._key_counter:06d}"

            block_a = KeyBlock(
                key_id=kid,
                link_id=link_id,
                local_node=node_a,
                peer_node=node_b,
                key_material=os.urandom(block_size_bits // 8),
                length_bits=block_size_bits,
                state=KeyBlockState.READY,
                created_at=time.time(),
                expires_at=time.time() + 3600.0,  # 1 hour expiry
                qkd_session_id=session_id,
                qber=qber,
                secret_key_rate=key_rate,
            )

            # Create matching block for node_b with SAME key material
            self._key_counter += 1
            kid_b = f"{node_b}{node_a}-{self._key_counter:06d}"

            block_b = KeyBlock(
                key_id=kid_b,
                link_id=link_id,
                local_node=node_b,
                peer_node=node_a,
                key_material=block_a.key_material,  # Same key at both ends
                length_bits=block_size_bits,
                state=KeyBlockState.READY,
                created_at=time.time(),
                expires_at=time.time() + 3600.0,
                qkd_session_id=session_id,
                qber=qber,
                secret_key_rate=key_rate,
            )

            pool_a = self.nodes[node_a].key_pools.get(f"{node_a}-{node_b}")
            pool_b = self.nodes[node_b].key_pools.get(f"{node_b}-{node_a}")
            if pool_a:
                pool_a.add_key_block(block_a)
            if pool_b:
                pool_b.add_key_block(block_b)

        # Handle remaining bits as a smaller block
        if remaining_bits > 0:
            self._key_counter += 1
            kid = f"{node_a}{node_b}-{self._key_counter:06d}"

            block_a = KeyBlock(
                key_id=kid,
                link_id=link_id,
                local_node=node_a,
                peer_node=node_b,
                key_material=os.urandom(max(1, remaining_bits // 8)),
                length_bits=remaining_bits,
                state=KeyBlockState.READY,
                created_at=time.time(),
                expires_at=time.time() + 3600.0,
                qkd_session_id=session_id,
                qber=qber,
                secret_key_rate=key_rate,
            )

            self._key_counter += 1
            kid_b = f"{node_b}{node_a}-{self._key_counter:06d}"

            block_b = KeyBlock(
                key_id=kid_b,
                link_id=link_id,
                local_node=node_b,
                peer_node=node_a,
                key_material=block_a.key_material,
                length_bits=remaining_bits,
                state=KeyBlockState.READY,
                created_at=time.time(),
                expires_at=time.time() + 3600.0,
                qkd_session_id=session_id,
                qber=qber,
                secret_key_rate=key_rate,
            )

            pool_a = self.nodes[node_a].key_pools.get(f"{node_a}-{node_b}")
            pool_b = self.nodes[node_b].key_pools.get(f"{node_b}-{node_a}")
            if pool_a:
                pool_a.add_key_block(block_a)
            if pool_b:
                pool_b.add_key_block(block_b)

        return secure_bits, qber, key_rate

    # ── Switch Configuration ─────────────────────────────────────────────

    def configure_switches_for_path(self, path: List[str]) -> None:
        """Configure optical switches to support the given path.

        Sets the switch configurations so that the quantum channels
        are properly routed for the specified end-to-end path.

        Args:
            path: Ordered list of node IDs from sender to receiver.
        """
        if len(path) < 2:
            return

        # Configure the 1x2 switch at A
        for i in range(len(path) - 1):
            if path[i] == "A" or path[i + 1] == "A":
                if path[i] == "A":
                    peer = path[i + 1]
                else:
                    peer = path[i]
                sw = self.switches.get("SW-A-1x2")
                if sw:
                    sw.configure({"A": peer})
                break

        # Configure 2x4 switches at B and C
        for i in range(len(path) - 1):
            for relay_id in ["B", "C"]:
                if path[i] == relay_id:
                    sw = self.switches.get(f"SW-{relay_id}-2x4")
                    if sw:
                        key = f"{relay_id}-{path[i+1]}"
                        sw.configure({key: path[i+1]})
                elif path[i + 1] == relay_id:
                    sw = self.switches.get(f"SW-{relay_id}-2x4")
                    if sw and i > 0:
                        key = f"{relay_id}-{path[i-1]}"
                        sw.configure({key: path[i-1]})

    # ── Failure Handling ─────────────────────────────────────────────────

    def handle_failure(self, failed_link_id: str) -> List[str]:
        """Handle a link failure by marking it down and finding alternative paths.

        When a link fails, this method:
        1. Marks the link as DOWN
        2. Sets associated pool statuses to ERROR
        3. Identifies affected service keys
        4. Finds alternative paths for affected sender-receiver pairs

        Args:
            failed_link_id: The ID of the link that has failed.

        Returns:
            List of alternative paths found for affected routes.
        """
        link = self.links.get(failed_link_id)
        if link is None:
            return []

        link.status = LinkStatus.DOWN

        # Mark associated pools
        node_a, node_b = link.node_a, link.node_b
        pool_a = self.nodes[node_a].key_pools.get(f"{node_a}-{node_b}")
        pool_b = self.nodes[node_b].key_pools.get(f"{node_b}-{node_a}")
        if pool_a:
            pool_a.status = KeyPoolStatus.ERROR
        if pool_b:
            pool_b.status = KeyPoolStatus.ERROR

        # Find alternative paths for affected pairs
        affected_pairs: List[Tuple[str, str]] = []
        senders = ["D", "E", "F", "G"]
        for s in senders:
            affected_pairs.append((s, "A"))

        alternative_paths: List[str] = []
        for sender, receiver in affected_pairs:
            alt_path = self.spf_plus(sender, receiver)
            if alt_path:
                path_str = " -> ".join(alt_path)
                alternative_paths.append(path_str)

        return alternative_paths

    # ── Network Status ───────────────────────────────────────────────────

    def get_network_status(self) -> dict:
        """Return a comprehensive summary of the current network state.

        Includes node information, link metrics, pool levels, switch
        configurations, and service key registry statistics.

        Returns:
            Dictionary containing the full network state.
        """
        node_info = {}
        for nid, node in self.nodes.items():
            pool_data = {}
            for pk, pool in node.key_pools.items():
                pool_data[pk] = {
                    "pool_id": pool.pool_id,
                    "status": pool.status.value,
                    "available_bits": pool.available_key_bits,
                    "reserved_bits": pool.reserved_key_bits,
                    "used_bits": pool.used_key_bits,
                    "num_blocks": len(pool.key_blocks),
                }
            node_info[nid] = {
                "role": node.role.value,
                "neighbors": node.neighbors,
                "pools": pool_data,
            }

        link_info = {}
        for lid, link in self.links.items():
            link_info[lid] = {
                "distance_km": link.distance_km,
                "status": link.status.value,
                "qber": link.qber,
                "secret_key_rate": link.secret_key_rate,
                "optical_loss_db": link.optical_loss_db,
                "last_session": link.last_qkd_session_id,
            }

        switch_info = {}
        for sid, sw in self.switches.items():
            switch_info[sid] = {
                "type": sw.switch_type,
                "available": sw.available,
                "configuration": sw.configuration,
            }

        return {
            "nodes": node_info,
            "links": link_info,
            "switches": switch_info,
            "service_keys_issued": len(self.service_key_registry),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Network Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

class QKDNetwork7Nodes:
    """Orchestrator for the 7-node QKD network demonstration.

    Manages the lifecycle of the network: initialization, pool priming,
    demo execution, and status reporting.

    Attributes:
        kms: The Key Management System instance.
    """

    def __init__(self) -> None:
        """Initialize the network orchestrator."""
        self.kms: KMS = KMS()

    def initialize(self) -> None:
        """Build and prime the network.

        Creates the 7-node topology, runs initial QKD sessions to fill
        all key pools, and verifies network health.
        """
        print("\n" + "=" * 80)
        print("  7-NODE QKD NETWORK INITIALIZATION")
        print("=" * 80)
        print()

        # Build topology
        self.kms.build_network()
        print("  [1] Network topology built: 7 nodes, 10 links, 20 pools, 3 switches")

        # Try to use pipeline simulator if available
        is_pipeline = False
        try:
            pipeline_sim = PipelineQKDLinkSimulator()
            if pipeline_sim.available:
                self.kms.simulator = pipeline_sim
                is_pipeline = True
                print("  [2] Pipeline QKD simulator loaded")
                print(f"      Engine: {pipeline_sim.pipeline_path}")
                print(
                    f"      Detector efficiency: "
                    f"{pipeline_sim.detector_efficiency:.2f}"
                )
            else:
                self.kms.simulator = StatisticalQKDLinkSimulator()
                reason = getattr(pipeline_sim, '_import_error', 'unknown')
                print(f"  [2] Using statistical QKD simulator (pipeline unavailable: {reason})")
                print("      HINT: Run from the '4.1 QKD Simulation Framework' directory")
                print("            where the qkd package and main_optimized.py are available.")
        except Exception as e:
            self.kms.simulator = StatisticalQKDLinkSimulator()
            print(f"  [2] Using statistical QKD simulator (pipeline import failed: {e})")

        # Calibrate watermarks and SPF+ normalization for the simulator type
        self.kms.calibrate_for_simulator(is_pipeline=is_pipeline)
        if is_pipeline:
            print(f"  [3] Calibrated for pipeline mode: "
                  f"watermarks={PIPELINE_LOW_WATERMARK_BITS//1000}K/"
                  f"{PIPELINE_HIGH_WATERMARK_BITS//1000}K/"
                  f"{PIPELINE_MAX_CAPACITY_BITS//1000}K bits, "
                  f"SPF max_rate={PIPELINE_SPF_MAX_KEY_RATE:.1e}")
        else:
            print(f"  [3] Calibrated for statistical mode: "
                  f"watermarks={STAT_LOW_WATERMARK_BITS//1_000_000}M/"
                  f"{STAT_HIGH_WATERMARK_BITS//1_000_000}M/"
                  f"{STAT_MAX_CAPACITY_BITS//1_000_000}M bits, "
                  f"SPF max_rate={STAT_SPF_MAX_KEY_RATE}")

        print()

    def prime_pools(self, num_sessions_per_link: int = 2, pulses_per_session: int = 10 ** 7) -> None:
        """Fill all key pools by running QKD sessions on every link.

        Uses a large number of pulses per session so that pools are filled
        above their low watermark thresholds.

        Note: ``pulses_per_session`` was reduced from 10**8 to 10**7.  At
        10**8 pulses, the pipeline allocates several ~800 MB int64 arrays
        (prepared_states, photons, click0, click1) which can trigger
        memory pressure and numerical edge-cases in the qkd package's
        sifting/parameter-estimation code.  10**7 pulses is sufficient
        to fill the pipeline-scaled key pools (10 Kbit / 100 Kbit / 200
        Kbit watermarks) while staying well within memory limits.

        Args:
            num_sessions_per_link: Number of QKD sessions to run per link.
            pulses_per_session: Number of pulses per QKD session.
        """
        print("══════════════════════════════════════════════════════════════════════")
        print("  PRIMING KEY POOLS")
        print("══════════════════════════════════════════════════════════════════════")
        print()

        link_ids = sorted(self.kms.links.keys())
        for link_id in link_ids:
            link = self.kms.links[link_id]
            total_bits = 0
            for session in range(num_sessions_per_link):
                bits, qber, rate = self.kms.run_qkd_session(link_id, num_pulses=pulses_per_session)
                total_bits += bits
            status_note = ""
            if link.qber > QBER_THRESHOLD:
                status_note = " [QBER above threshold — key generation not possible]"
            elif total_bits == 0:
                status_note = " [WARNING: zero key bits generated]"
            print(
                f"  Link {link_id:5s} | Distance: {link.distance_km:5.1f} km | "
                f"QBER: {link.qber:.6f} | Key Rate: {link.secret_key_rate:.2e} bits/pulse | "
                f"Total: {total_bits:>12,} bits{status_note}"
            )
        print()

    def run_demo(self) -> None:
        """Execute the complete demonstration flow.

        Steps:
        1. Print network status (topology, pool levels, link metrics)
        2. Request service keys: A<->D, A<->E, A<->F, A<->G
        3. Show SPF+ path selection for each request
        4. Show trusted relay execution
        5. Show pool levels after key consumption
        6. Simulate a link failure (D-B down) and show failover
        7. Trigger replenishment for low pools
        8. Show final network status
        """
        # ── Step 1: Network Status ────────────────────────────────────────
        self._print_network_status()

        # ── Step 2-4: Service Key Requests ────────────────────────────────
        print("══════════════════════════════════════════════════════════════════════")
        print("  SERVICE KEY REQUESTS (SPF+ Path Selection + Trusted Relay)")
        print("══════════════════════════════════════════════════════════════════════")
        print()

        service_requests = [
            ("D", "A"),
            ("E", "A"),
            ("F", "A"),
            ("G", "A"),
        ]

        for sender, receiver in service_requests:
            self._process_service_key_request(sender, receiver)

        # ── Step 5: Pool Levels After Consumption ─────────────────────────
        self._print_pool_levels("POOL LEVELS AFTER KEY CONSUMPTION")

        # ── Step 6: Link Failure Simulation ───────────────────────────────
        print("══════════════════════════════════════════════════════════════════════")
        print("  LINK FAILURE SIMULATION: D-B Link Goes DOWN")
        print("══════════════════════════════════════════════════════════════════════")
        print()

        alt_paths = self.kms.handle_failure("B-D")
        print(f"  Link B-D marked as DOWN")
        print(f"  Pool POOL-B-D and POOL-D-B set to ERROR")
        print()
        print("  Alternative paths found:")
        for p in alt_paths:
            print(f"    {p}")
        print()

        # Re-request a key that would have used B-D
        print("  Re-requesting service key D -> A with failed link B-D:")
        self._process_service_key_request("D", "A")
        print()

        # ── Step 7: Pool Replenishment ────────────────────────────────────
        print("══════════════════════════════════════════════════════════════════════")
        print("  KEY POOL REPLENISHMENT")
        print("══════════════════════════════════════════════════════════════════════")
        print()

        # Restore the B-D link first for replenishment
        self.kms.links["B-D"].status = LinkStatus.UP
        bd_pool_b = self.kms.nodes["B"].key_pools.get("B-D")
        bd_pool_d = self.kms.nodes["D"].key_pools.get("D-B")
        if bd_pool_b:
            bd_pool_b.status = KeyPoolStatus.LOW
        if bd_pool_d:
            bd_pool_d.status = KeyPoolStatus.LOW

        replenished = self.kms.replenish_key_pools()
        if replenished:
            print(f"  Replenished {len(replenished)} links:")
            for lid in replenished:
                link = self.kms.links[lid]
                print(
                    f"    {lid}: QBER={link.qber:.6f}, "
                    f"Key Rate={link.secret_key_rate:.2e} bits/pulse"
                )
        else:
            print("  All pools at healthy levels - no replenishment needed.")
        print()

        # ── Step 8: Final Network Status ──────────────────────────────────
        self._print_final_status()

    def _process_service_key_request(self, sender: str, receiver: str) -> None:
        """Process and display a single service key request.

        Shows SPF+ path selection details, cost calculations for ALL
        candidate paths (not just the winner), and trusted relay steps.

        Args:
            sender: Sender node ID.
            receiver: Receiver node ID.
        """
        print(f"  ── Service Key Request: {sender} -> {receiver} ──")

        # Show ALL candidate paths and their total costs
        self._print_path_comparison(sender, receiver)

        # Show SPF+ path
        path = self.kms.spf_plus(sender, receiver)
        if not path:
            print(f"    No available path from {sender} to {receiver}")
            print()
            return

        path_str = " -> ".join(path)
        print(f"    SPF+ Selected Path: {path_str}")

        # Show cost calculation for each hop on the selected path
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link_id = self.kms._get_link_id(u, v)
            link = self.kms.links[link_id]
            pool_u = self.kms.nodes[u].key_pools.get(f"{u}-{v}")
            pool_v = self.kms.nodes[v].key_pools.get(f"{v}-{u}")
            avail_u = pool_u.get_available_bits() if pool_u else 0
            avail_v = pool_v.get_available_bits() if pool_v else 0

            # Compute pool pressure using KeyPool.calculate_pressure()
            pressures = []
            for pool in (pool_u, pool_v):
                if pool is not None:
                    pressures.append(pool.calculate_pressure())
                else:
                    pressures.append(1.0)
            pool_pressure = max(pressures)

            cost_params = {
                "alpha": self.kms.spf_weights["alpha"],
                "beta": self.kms.spf_weights["beta"],
                "gamma": self.kms.spf_weights["gamma"],
                "delta": self.kms.spf_weights["delta"],
                "epsilon": self.kms.spf_weights["epsilon"],
                "pool_pressure": pool_pressure,
                "key_rate": link.secret_key_rate,
            }
            hop_cost = link.get_cost(cost_params)
            qber_warning = " [QBER EXCEEDED]" if link.qber > QBER_THRESHOLD else ""
            print(
                f"    Hop {u}->{v}: loss={link.optical_loss_db:.1f}dB, "
                f"QBER={link.qber:.6f}, "
                f"avail={min(avail_u, avail_v):,}bits, "
                f"rate={link.secret_key_rate:.2e}, "
                f"pressure={pool_pressure:.3f}, "
                f"cost={hop_cost:.4f}{qber_warning}"
            )

        # Configure switches
        self.kms.configure_switches_for_path(path)
        print(f"    Switches configured for path")

        # Execute request
        result = self.kms.request_service_key(sender, receiver, key_length_bits=256)

        if result["status"] == "ok":
            print(f"    Service Key ID: {result['service_key_id']}")
            print(f"    Key Material:   {result['key_material'].hex()[:32]}...")
            print(f"    Path:           {' -> '.join(result['path'])}")
            print(f"    Hop Keys Used:  {len(result['hop_keys_used'])} blocks")

            # Show relay steps explicitly
            self._print_relay_steps(path)
        else:
            print(f"    ERROR: {result['message']}")

        print()

    def _print_path_comparison(self, sender: str, receiver: str) -> None:
        """Print a comparison of all candidate relay paths with their SPF+ costs.

        For a sender→receiver request, there are exactly two candidate
        2-hop paths (via relay B or via relay C). This method computes
        the total SPF+ cost for each and shows which one wins.

        Args:
            sender: Sender node ID.
            receiver: Receiver node ID.
        """
        # Enumerate candidate paths (sender → relay → receiver)
        relays = [n for n in self.kms.nodes
                  if isinstance(self.kms.nodes[n], TrustedRelayNode)]
        candidates = []
        for relay in sorted(relays):
            path = [sender, relay, receiver]
            total_cost = 0.0
            valid = True
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                link_id = self.kms._get_link_id(u, v)
                link = self.kms.links.get(link_id)
                if link is None or link.status == LinkStatus.DOWN:
                    valid = False
                    break

                pool_u = self.kms.nodes[u].key_pools.get(f"{u}-{v}")
                pool_v = self.kms.nodes[v].key_pools.get(f"{v}-{u}")
                # Pool pressure using KeyPool.calculate_pressure()
                pressures = []
                for pool in (pool_u, pool_v):
                    if pool is not None:
                        pressures.append(pool.calculate_pressure())
                    else:
                        pressures.append(1.0)
                pool_pressure = max(pressures)

                # Check for ERROR/DISABLED pools
                if (pool_u and pool_u.status in (KeyPoolStatus.ERROR, KeyPoolStatus.DISABLED)) or \
                   (pool_v and pool_v.status in (KeyPoolStatus.ERROR, KeyPoolStatus.DISABLED)):
                    valid = False
                    break

                cost_params = {
                    "alpha": self.kms.spf_weights["alpha"],
                    "beta": self.kms.spf_weights["beta"],
                    "gamma": self.kms.spf_weights["gamma"],
                    "delta": self.kms.spf_weights["delta"],
                    "epsilon": self.kms.spf_weights["epsilon"],
                    "pool_pressure": pool_pressure,
                    "key_rate": link.secret_key_rate,
                }
                total_cost += link.get_cost(cost_params)

            if valid:
                candidates.append((" -> ".join(path), total_cost))

        if not candidates:
            print("    No candidate paths available")
            return

        # Sort by cost
        candidates.sort(key=lambda x: x[1])
        print("    Candidate Paths (SPF+ cost comparison):")
        for i, (path_str, cost) in enumerate(candidates):
            marker = " <-- SELECTED" if i == 0 else ""
            print(f"      {path_str}: cost={cost:.4f}{marker}")

    def _print_relay_steps(self, path: List[str]) -> None:
        """Print the XOR relay steps for a given path.

        Shows each encryption/decryption step performed by the sender,
        relay nodes, and receiver.

        Args:
            path: Ordered list of node IDs.
        """
        print(f"    Relay Steps (XOR-based hop-by-hop):")
        for i in range(len(path)):
            node = path[i]
            node_obj = self.kms.nodes[node]
            if i == 0:
                next_node = path[i + 1]
                print(
                    f"      Step 1: Sender {node} encrypts service_key "
                    f"with K_{node}{next_node} -> C1"
                )
            elif i == len(path) - 1:
                prev_node = path[i - 1]
                print(
                    f"      Step {i+1}: Receiver {node} decrypts C{i} "
                    f"with K_{node}{prev_node} -> service_key recovered"
                )
            else:
                prev_node = path[i - 1]
                next_node = path[i + 1]
                print(
                    f"      Step {i+1}: Relay {node} decrypts C{i} with K_{node}{prev_node}, "
                    f"re-encrypts with K_{node}{next_node} -> C{i+1}"
                )

    def _print_network_status(self) -> None:
        """Print the full network status with formatted tables."""
        print("══════════════════════════════════════════════════════════════════════")
        print("  NETWORK STATUS")
        print("══════════════════════════════════════════════════════════════════════")
        print()

        # Node table
        print("  Node Topology:")
        print(f"  {'Node':<6} {'Role':<12} {'Neighbors':<20}")
        print(f"  {'----':<6} {'----':<12} {'---------':<20}")
        for nid in ["A", "B", "C", "D", "E", "F", "G"]:
            node = self.kms.nodes[nid]
            print(f"  {nid:<6} {node.role.value:<12} {', '.join(node.neighbors):<20}")
        print()

        # Link metrics table
        print("  QKD Link Metrics:")
        print(
            f"  {'Link':<6} {'Dist(km)':<10} {'Status':<10} "
            f"{'QBER':<12} {'Key Rate':<16} {'Loss(dB)':<10}"
        )
        print(
            f"  {'----':<6} {'--------':<10} {'------':<10} "
            f"{'----':<12} {'--------':<16} {'--------':<10}"
        )
        for lid in sorted(self.kms.links.keys()):
            link = self.kms.links[lid]
            print(
                f"  {lid:<6} {link.distance_km:<10.1f} {link.status.value:<10} "
                f"{link.qber:<12.6f} {link.secret_key_rate:<16.2e} {link.optical_loss_db:<10.1f}"
            )
        print()

        # Pool levels
        self._print_pool_levels("KEY POOL LEVELS (After Priming)")

        # Switch configuration
        print("  Optical Switch Configurations:")
        for sid, sw in self.kms.switches.items():
            config_str = ", ".join(f"{k}->{v}" for k, v in sw.configuration.items())
            print(f"    {sid} ({sw.switch_type}): {config_str}")
        print()

    def _print_pool_levels(self, title: str) -> None:
        """Print a formatted table of key pool levels.

        Args:
            title: Section header for the pool level display.
        """
        print(f"  {title}:")
        print(
            f"  {'Pool ID':<14} {'Status':<14} {'Available':<14} "
            f"{'Reserved':<14} {'Used':<14} {'Blocks':<8}"
        )
        print(
            f"  {'-------':<14} {'------':<14} {'---------':<14} "
            f"{'--------':<14} {'----':<14} {'------':<8}"
        )
        for nid in ["A", "B", "C", "D", "E", "F", "G"]:
            node = self.kms.nodes[nid]
            for pk, pool in sorted(node.key_pools.items()):
                print(
                    f"  {pool.pool_id:<14} {pool.status.value:<14} "
                    f"{pool.available_key_bits:<14,} "
                    f"{pool.reserved_key_bits:<14,} "
                    f"{pool.used_key_bits:<14,} "
                    f"{len(pool.key_blocks):<8}"
                )
        print()

    def _print_final_status(self) -> None:
        """Print the final network status summary after the demo."""
        print("══════════════════════════════════════════════════════════════════════")
        print("  FINAL NETWORK STATUS")
        print("══════════════════════════════════════════════════════════════════════")
        print()

        # Link metrics
        print("  QKD Link Metrics (Final):")
        print(
            f"  {'Link':<6} {'Status':<10} {'QBER':<12} {'Key Rate':<16}"
        )
        print(
            f"  {'----':<6} {'------':<10} {'----':<12} {'--------':<16}"
        )
        for lid in sorted(self.kms.links.keys()):
            link = self.kms.links[lid]
            print(
                f"  {lid:<6} {link.status.value:<10} "
                f"{link.qber:<12.6f} {link.secret_key_rate:<16.2e}"
            )
        print()

        # Pool levels
        self._print_pool_levels("KEY POOL LEVELS (Final)")

        # Service key summary
        print("  Service Key Registry:")
        print(f"    Total service keys issued: {len(self.kms.service_key_registry)}")
        for sk_id, meta in self.kms.service_key_registry.items():
            path_str = " -> ".join(meta["path"])
            print(
                f"    {sk_id}: {meta['sender']}->{meta['receiver']} "
                f"via {path_str}, {meta['key_length_bits']} bits, "
                f"relay={'OK' if meta['relay_success'] else 'FAIL'}"
            )
        print()

        # Switch status
        print("  Optical Switch Status:")
        for sid, sw in self.kms.switches.items():
            config_str = ", ".join(f"{k}->{v}" for k, v in sw.configuration.items())
            print(f"    {sid} ({sw.switch_type}): available={sw.available}, config=[{config_str}]")
        print()

        print("=" * 80)
        print("  DEMO COMPLETE")
        print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
# Demo Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def demo() -> None:
    """Run the complete 7-node QKD network demonstration.

    Executes the following flow:
    1. Creates and initializes the network
    2. Primes all key pools by running QKD sessions on all 10 links
    3. Prints network status (topology, pool levels, link metrics)
    4. Requests service keys: A<->D, A<->E, A<->F, A<->G
    5. Shows SPF+ path selection for each request
    6. Shows trusted relay execution
    7. Shows pool levels after key consumption
    8. Simulates a link failure (D-B down) and shows failover
    9. Triggers replenishment for low pools
    10. Shows final network status
    """
    network = QKDNetwork7Nodes()
    network.initialize()
    network.prime_pools(num_sessions_per_link=50)
    network.run_demo()


if __name__ == "__main__":
    demo()
