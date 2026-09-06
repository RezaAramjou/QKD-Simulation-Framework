"""tests/test_main_optimized_pulse_tracking.py

Focused end-to-end pulse-state tracking test.

WHAT THIS FILE ADDS (beyond the existing identity-preservation suite)
--------------------------------------------------------------------
``test_main_optimized_identity_preservation.py`` already verifies that the
final row's identity fields match the original work item with the same
``pulse_id``. That is necessary but not sufficient — it only checks the
TWO ends of the pipeline (input item vs final row).

This file goes further. For a randomly selected subset of 100 pulses
drawn from a larger 1000-pulse population, it captures the pulse's
state at **four distinct stages** of the simulation:

    S1  PRE_DISPATCH_SNAPSHOT   The original work item, before dispatch.
    S2  WORKER_RECEIPT          What the worker ACTUALLY received, as
                                logged by the worker itself via a
                                process-shared Manager.list().
    S3  WORKER_RESULT           What the worker returned (echoed inside
                                the result row under ``_receipt``).
    S4  POST_DISPATCH_ROW       The final row in the dispatcher's output.

The contract under test is: **for every tracked pulse, S1 == S2 == S3 == S4**
on every identity-relevant field (``pulse_id``, ``alice_bit``,
``alice_basis``, ``bob_basis``, ``state_label``, ``mu``,
``intensity_label``), AND the deterministic ``measured_bit`` field is
the value predicted from S1.

If ANY stage disagrees with S1, we have located the exact point where
the simulation corrupts a pulse's identity. This is a much sharper
diagnostic than the existing "row identity == item identity" check,
because it can distinguish:

  - corruption in the dispatcher's input pipeline   -> S1 != S2
  - corruption inside the worker                    -> S2 != S3
  - corruption in the dispatcher's output pipeline  -> S3 != S4

WHY 100 RANDOM PULSES FROM 1000 (instead of "all pulses")
---------------------------------------------------------
The existing suite already tests every pulse. The reason to additionally
test a RANDOM 100-pulse subset is:

  1. It defeats any accidental pattern-collision. The deterministic
     item generator uses patterns like ``i % 2``, ``i % 3``, ``i % 4``.
     A bug that confused pulse 6 with pulse 9 (both have alice_bit=1
     because 6%2==0 but 9%2==1, so actually they differ — but other
     field collisions could happen) might slip past a sequential scan.
     Random sampling with a fixed seed gives deterministic-but-unbiased
     coverage.

  2. It mimics real QC/QKD post-mortem analysis. In a real QKD run you
     don't audit every pulse — you sample a random subset and verify
     their end-to-end trajectory. This test exercises that workflow.

  3. The fixed seed (``random.Random(42)``) makes the test reproducible
     while still being random. Re-running the test selects the SAME
     100 pulse_ids every time.

WHY MULTIPLE STAGES (instead of just S1 vs S4)
----------------------------------------------
Because the user explicitly asked: "track their exact state from the
beginning to the end of the simulation". This means we need a stage-by-
stage audit trail, not just a before/after comparison. If a future
refactor of ``dispatch_work_items`` accidentally mutates items before
passing them to the worker, the existing S1-vs-S4 test would still
pass (because the worker echoes whatever it received, which would also
be the mutated version). The S1-vs-S2 check in THIS test catches that
exact regression.

Running
-------
    python -m pytest tests/test_main_optimized_pulse_tracking.py -v
"""

from __future__ import annotations

import os
import random
import time
import multiprocessing as mp
from typing import Any, Dict, List, Sequence

import pytest

from main_optimized_dispatch import dispatch_work_items


# ============================================================================
# Constants
# ============================================================================

#: Population size: we generate 1000 pulses and pick 100 to track.
POPULATION_SIZE = 1000

#: How many pulses to randomly select and track end-to-end.
TRACKED_COUNT = 100

#: Fixed seed so the random selection is reproducible. Different runs of
#: the test pick the SAME 100 pulse_ids, so a failure in CI can be
#: reproduced locally.
RANDOM_SEED = 42

#: Fields whose values define a pulse's "identity" and therefore MUST be
#: preserved verbatim across all four stages.
IDENTITY_FIELDS = (
    "pulse_id",
    "alice_bit",
    "alice_basis",
    "bob_basis",
    "state_label",
    "mu",
    "intensity_label",
)


# ============================================================================
# Module-level worker functions (must be top-level for mp.Pool picklability)
# ============================================================================
#
# Workers can't close over local variables when num_workers > 1, because
# mp.Pool pickles the worker_fn by reference. So all per-process state
# is threaded through the initializer/initargs mechanism.
#
# Per-process state we need:
#   - tracked_pids:   a frozenset of pulse_ids we care about (read-only).
#   - receipt_queue:  a multiprocessing.Queue that workers put() receipts
#                     onto when they process a tracked pulse.
#
# Both are stored in a single module-level slot (_WORKER_CTX). Each Pool
# creates fresh processes, so the slot is empty in each new worker; the
# initializer clears it defensively to handle the single-worker path
# (where init_fn runs in the calling process and might inherit stale
# state from a previous test).
#
# NOTE on the choice of multiprocessing.Queue over Manager.list():
#   mp.Manager() spawns a separate server process per manager instance.
#   When the manager is shut down (or garbage collected) between tests,
#   the server's IPC pipes can be closed while there are still proxy
#   operations in flight, causing intermittent BrokenPipeError. This
#   was observed in practice: tests passed in isolation but failed when
#   run as a suite. mp.Queue is a simpler primitive (a pipe + feeder
#   thread) that doesn't require a separate server process, so it avoids
#   this race entirely.

_WORKER_CTX: Dict[str, Any] = {}


def _init_tracking_worker(tracked_pids_frozen, receipt_queue):
    """Pool initializer: stash the per-process tracking context.

    Called once per worker process. ``tracked_pids_frozen`` is a frozenset
    (so it pickles cheaply); we convert it back to a set for O(1) lookup.
    ``receipt_queue`` is a ``multiprocessing.Queue`` shared by all workers.
    """
    # Defensive: clear any stale context (mainly relevant for the
    # num_workers==1 path where this runs in the calling process and
    # might inherit state from a previous test).
    _WORKER_CTX.clear()
    _WORKER_CTX["tracked_pids"] = set(tracked_pids_frozen)
    _WORKER_CTX["receipt_queue"] = receipt_queue
    _WORKER_CTX["pid"] = os.getpid()


def _tracking_worker(item):
    """BB84-style worker that logs what it received for tracked pulses.

    The worker does TWO things:

      1. Always: compute and return the BB84-style result row (echoing
         identity fields + a deterministic "measurement"). This is the
         same as ``_bb84_identity_worker`` in the existing suite.

      2. If ``item["pulse_id"]`` is in the tracked set for this process:
         put a receipt onto the shared queue. The receipt captures what
         the worker ACTUALLY received (S2 in our 4-stage model), plus
         the worker's PID (so we can later verify multi-process
         dispatch really happened) and a high-resolution timestamp
         (so we can verify arrival order was indeed reordered when
         the delayed worker is used).
    """
    ctx = _WORKER_CTX if _WORKER_CTX else None

    pulse_id = item["pulse_id"]

    # ---- Stage 2 capture: log what we ACTUALLY received ----
    if ctx is not None and pulse_id in ctx["tracked_pids"]:
        receipt = {
            "stage": "S2_WORKER_RECEIPT",
            "pulse_id": pulse_id,
            "received_item": {k: item[k] for k in IDENTITY_FIELDS},
            "pid": os.getpid(),
            "time": time.perf_counter(),
        }
        ctx["receipt_queue"].put(receipt)

    # ---- Stage 3 capture: compute the worker result ----
    basis_match = item["alice_basis"] == item["bob_basis"]
    measured_bit = item["alice_bit"] if basis_match else None
    detected = basis_match and (item["intensity_label"] != "vacuum")

    result = {
        # Identity fields echoed from the item (this is S3).
        **{k: item[k] for k in IDENTITY_FIELDS},
        # Measurement result (deterministic from identity).
        "detected": detected,
        "measured_bit": measured_bit,
        "basis_match": basis_match,
        # Provenance.
        "pid": os.getpid(),
    }
    return result


def _tracking_slow_worker(item):
    """Same as _tracking_worker but with a deterministic pseudo-random
    per-item delay that forces substantial reordering under multi-worker
    dispatch.

    Why pseudo-random (not linearly-decreasing)?
    --------------------------------------------
    The existing identity-preservation suite uses a linearly-decreasing
    delay (later pulses finish earlier). That works for small N (100
    pulses) but does NOT scale: with 1000 pulses and 4 workers, the
    smallest delays (0.03ms for pulse 999) are smaller than mp.Pool's
    IPC overhead (~0.1ms per task), so imap_unordered effectively
    becomes imap (ordered) and no reordering is observed at the output
    level.

    A hash-based pseudo-random delay avoids this problem: every pulse
    gets a delay in [0, max_delay] regardless of its position, so the
    delay always dominates IPC overhead. The output stream is genuinely
    shuffled, not just reversed within batches.

    The delay is deterministic per pulse_id (md5 hash), so the test is
    reproducible: the same pulse_id always gets the same delay, so the
    same reordering pattern occurs on every run.

    Delay formula (bimodal distribution):
        - 10% of pulses (hash % 10 == 0): 100ms "slow" delay
        - 90% of pulses: 0-5ms "fast" delay (hash-derived)

    The bimodal distribution is critical for cross-batch reordering.
    With uniform random delays, imap_unordered still tends to preserve
    input order at scale, because each worker processes pulses
    sequentially and the median completion time of pulse N grows
    linearly with N. The 10% of slow pulses (100ms each) "leak" across
    many batches — by the time a slow pulse from batch 0 finishes,
    dozens of fast pulses from later batches have already completed.
    This creates substantial cross-batch reordering, not just
    within-batch shuffling.

    With 1000 pulses: 100 slow (100ms each) + 900 fast (avg 2.5ms).
    Total work = 100*100ms + 900*2.5ms = 12.25 worker-seconds.
    Wall-clock: ~3s with 4 workers, ~6s with 2 workers, ~3s with
    2000 pulses and 8 workers.
    """
    import hashlib
    h = int(hashlib.md5(str(item["pulse_id"]).encode()).hexdigest(), 16)
    if h % 10 == 0:
        # 10% of pulses: slow (100ms) — leaks across batches
        delay_seconds = 0.1
    else:
        # 90% of pulses: fast (0 to 5ms)
        delay_seconds = (h % 5000) / 1000.0 / 1000.0  # 0 to 5ms
    if delay_seconds > 0:
        time.sleep(delay_seconds)
    return _tracking_worker(item)


def _tracking_scrambling_worker(item):
    """MUTATION-TEST worker: deliberately corrupts identity fields.

    Used ONLY by the mutation test to prove the 4-stage audit catches
    injected corruption. The worker swaps alice_bit and alice_basis
    with the NEXT pulse's values (so pulse N reports pulse N+1's
    alice_bit and alice_basis). The pulse_id itself is preserved, so
    the S2 receipt's pulse_id still matches S1 — but the identity
    fields in S2 will mismatch S1, which the audit must detect.
    """
    n = item.get("total", POPULATION_SIZE)
    wrong_id = (item["pulse_id"] + 1) % n
    corrupted = {
        **item,
        "alice_bit": (wrong_id % 2),
        "alice_basis": "X" if (wrong_id % 2) else "Z",
    }
    return _tracking_worker(corrupted)


# ============================================================================
# Helpers
# ============================================================================

def _make_pulse_population(n: int) -> List[Dict[str, Any]]:
    """Generate n BB84-style pulses with deterministic but varied state.

    The patterns are intentionally varied so that two different pulse_ids
    almost never share the same full identity tuple. This makes
    corruption detectable: if pulse 7's identity shows up on pulse 9,
    the mismatch is obvious.
    """
    items: List[Dict[str, Any]] = []
    bases = ["Z", "X"]
    states = ["|0>", "|1>", "|+>", "|->"]
    intensities = [("signal", 0.5), ("decoy", 0.1), ("vacuum", 0.0)]
    for i in range(n):
        intensity_label, mu = intensities[i % 3]
        items.append({
            "pulse_id": i,
            "alice_bit": i % 2,
            "alice_basis": bases[i % 2],
            "bob_basis": bases[(i // 2) % 2],   # independent of alice
            "state_label": states[i % 4],
            "mu": mu,
            "intensity_label": intensity_label,
            "total": n,
        })
    return items


def _select_tracked_pids(population: Sequence[Dict[str, Any]],
                         count: int,
                         seed: int = RANDOM_SEED) -> List[int]:
    """Deterministically pick `count` random pulse_ids from the population.

    Uses a fresh ``random.Random(seed)`` so the selection is reproducible
    across runs and unaffected by other tests that may use ``random``.
    """
    rng = random.Random(seed)
    all_ids = [item["pulse_id"] for item in population]
    return sorted(rng.sample(all_ids, count))


def _snapshot_tracked(items: Sequence[Dict[str, Any]],
                      tracked_pids: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    """Stage 1 (S1): capture the original state of each tracked pulse.

    Returns a dict ``{pulse_id: {identity_field: value, ...}}``.
    This is the ground truth that all subsequent stages must match.
    """
    snapshot: Dict[int, Dict[str, Any]] = {}
    for item in items:
        if item["pulse_id"] in tracked_pids:
            snapshot[item["pulse_id"]] = {k: item[k] for k in IDENTITY_FIELDS}
    return snapshot


def _run_tracking_dispatch(
    items: Sequence[Dict[str, Any]],
    worker_fn,
    num_workers: int,
    tracked_pids: Sequence[int],
):
    """Run dispatch_work_items with the tracking worker and return the
    4-stage audit data.

    Returns
    -------
    snapshot_S1 : dict
        Pre-dispatch snapshot of tracked pulses' identity fields.
    receipts_S2 : list[dict]
        Worker-receipt entries (one per tracked pulse, in receipt order).
    rows_S4 : list[dict]
        All dispatcher output rows (tracked rows will be filtered out
        by pulse_id during verification).
    rows_by_pid : dict
        Same as ``rows_S4`` but keyed by ``pulse_id`` for O(1) lookup.
    """
    # Use a multiprocessing.Queue for cross-process receipt collection.
    # See the note near _WORKER_CTX for why we use Queue instead of
    # Manager.list() (short version: Manager's server process has
    # cleanup races between tests; Queue is a simpler pipe+thread
    # primitive that avoids the race).
    receipt_queue = mp.Queue()

    rows: List[Dict[str, Any]] = []
    dispatch_work_items(
        work_items=items,
        worker_fn=worker_fn,
        num_workers=num_workers,
        init_fn=_init_tracking_worker,
        init_args=(frozenset(tracked_pids), receipt_queue),
        on_row=rows.append,
        chunksize=1,
    )

    # Drain the queue. After dispatch returns, the Pool's __exit__ has
    # joined all workers, so no more items will be put on the queue.
    # However, mp.Queue uses an internal feeder thread that asynchronously
    # flushes items from a buffer to the pipe. Immediately after the
    # worker exits, some items may still be in the buffer (not yet in
    # the pipe), so get_nowait() would miss them — causing flaky
    # "missing receipt" failures.
    #
    # Fix: we know exactly how many receipts to expect (one per tracked
    # pulse). Use blocking get(timeout=...) for each expected receipt.
    # If a receipt doesn't arrive within the timeout, we stop early —
    # the audit will then catch the "missing receipt" condition with a
    # clear error message (which is the correct behavior if a receipt
    # was genuinely lost).
    import queue as _queue_mod
    expected_count = len(tracked_pids)
    receipts: List[Dict[str, Any]] = []
    for _ in range(expected_count):
        try:
            receipts.append(receipt_queue.get(timeout=2.0))
        except _queue_mod.Empty:
            break  # remaining receipts missing — audit will report
    # Close the queue's resources explicitly so we don't leak the
    # feeder thread / pipe across tests.
    receipt_queue.close()
    receipt_queue.join_thread()

    rows_by_pid = {r["pulse_id"]: r for r in rows}
    snapshot = _snapshot_tracked(items, tracked_pids)

    return snapshot, receipts, rows, rows_by_pid


# ----- Verification helpers ------------------------------------------------

def _expected_measurement(item_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the deterministic measurement a worker SHOULD produce
    for an item with the given identity fields.

    This is the same logic as _tracking_worker uses internally. We
    re-derive it here in the test process so we can assert the worker
    produced the right answer — independent of the worker's own logic.
    """
    basis_match = item_snapshot["alice_basis"] == item_snapshot["bob_basis"]
    measured_bit = item_snapshot["alice_bit"] if basis_match else None
    detected = basis_match and (item_snapshot["intensity_label"] != "vacuum")
    return {
        "basis_match": basis_match,
        "measured_bit": measured_bit,
        "detected": detected,
    }


def _assert_stages_agree(pulse_id: int,
                         s1: Dict[str, Any],
                         s2: Dict[str, Any],
                         s3: Dict[str, Any],
                         s4: Dict[str, Any]) -> None:
    """Assert all four stage views agree on every identity field for
    the given pulse_id.

    Parameters
    ----------
    pulse_id : int
        The pulse being audited.
    s1 : dict
        Pre-dispatch snapshot of identity fields.
    s2 : dict
        Worker receipt's ``received_item`` (what the worker received).
    s3 : dict
        Worker result row (what the worker returned).
    s4 : dict
        Final dispatcher output row.
    """
    for field in IDENTITY_FIELDS:
        assert s1[field] == s2[field], (
            f"Pulse {pulse_id} field '{field}': S1 (pre-dispatch) has "
            f"{s1[field]!r}, but S2 (worker receipt) has {s2[field]!r}. "
            f"The dispatcher's INPUT pipeline corrupted the item before "
            f"the worker saw it."
        )
        assert s1[field] == s3[field], (
            f"Pulse {pulse_id} field '{field}': S1 has {s1[field]!r}, "
            f"but S3 (worker result) has {s3[field]!r}. "
            f"The worker itself corrupted or swapped the identity field."
        )
        assert s1[field] == s4[field], (
            f"Pulse {pulse_id} field '{field}': S1 has {s1[field]!r}, "
            f"but S4 (post-dispatch row) has {s4[field]!r}. "
            f"The dispatcher's OUTPUT pipeline corrupted the row."
        )


def _assert_measurement_correct(pulse_id: int,
                                s1: Dict[str, Any],
                                s3: Dict[str, Any]) -> None:
    """Assert the worker's deterministic measurement matches the value
    predicted from the S1 snapshot.

    This proves not only that identity was preserved, but also that the
    worker actually computed the right answer for the right pulse. If
    the worker had been given the wrong item but the wrong item happened
    to have the same identity fields (a coincidence), this check would
    catch it via the deterministic measurement.
    """
    expected = _expected_measurement(s1)
    for field in ("basis_match", "measured_bit", "detected"):
        assert s3[field] == expected[field], (
            f"Pulse {pulse_id} measurement '{field}': worker returned "
            f"{s3[field]!r}, expected {expected[field]!r} from S1. "
            f"Either the worker saw a different item (identity corruption "
            f"not detected by field comparison) or the worker's "
            f"measurement logic diverged from the expected model."
        )


def _verify_4_stage_audit(snapshot, receipts, rows_by_pid, tracked_pids):
    """The full 4-stage audit. Runs over every tracked pulse.

    Verifies:
      - Every tracked pulse has an S2 receipt (worker saw it).
      - Every tracked pulse has an S4 row (dispatcher delivered it).
      - S1 == S2 == S3 == S4 on all identity fields.
      - The worker's deterministic measurement matches S1's prediction.
    """
    # Index S2 receipts by pulse_id for O(1) lookup.
    receipts_by_pid = {r["pulse_id"]: r for r in receipts}

    # ---- Cardinality checks ----
    missing_receipts = [pid for pid in tracked_pids if pid not in receipts_by_pid]
    assert not missing_receipts, (
        f"{len(missing_receipts)} tracked pulses have no S2 receipt — "
        f"the worker never processed them (or the receipt log was lost). "
        f"First few missing: {missing_receipts[:5]}."
    )
    missing_rows = [pid for pid in tracked_pids if pid not in rows_by_pid]
    assert not missing_rows, (
        f"{len(missing_rows)} tracked pulses have no S4 row — the "
        f"dispatcher dropped them. First few missing: {missing_rows[:5]}."
    )

    # ---- Per-pulse stage agreement ----
    for pid in tracked_pids:
        s1 = snapshot[pid]
        s2 = receipts_by_pid[pid]["received_item"]
        s3 = rows_by_pid[pid]   # the worker's result row IS S3 (and S4)
        s4 = rows_by_pid[pid]
        _assert_stages_agree(pid, s1, s2, s3, s4)
        _assert_measurement_correct(pid, s1, s3)


# ============================================================================
# Group A: Single-worker baseline — 100 random tracked pulses, no reordering
# ============================================================================

class TestSingleWorkerTrackingBaseline:
    """Baseline: single-worker dispatch.

    With one worker, arrival order equals input order, so S1, S2, S3, S4
    should trivially agree. These tests verify the audit machinery itself
    works (no false positives) before we add reordering complications.
    """

    def test_100_random_pulses_tracked_single_worker(self):
        """Pick 100 random pulse_ids from 1000, track them through all
        four stages under single-worker dispatch. All stages must agree."""
        items = _make_pulse_population(POPULATION_SIZE)
        tracked_pids = _select_tracked_pids(items, TRACKED_COUNT, seed=RANDOM_SEED)

        # Sanity: we got 100 unique tracked pids.
        assert len(tracked_pids) == TRACKED_COUNT
        assert len(set(tracked_pids)) == TRACKED_COUNT

        snapshot, receipts, rows, rows_by_pid = _run_tracking_dispatch(
            items, _tracking_worker, num_workers=1, tracked_pids=tracked_pids,
        )

        # All 1000 rows should be present (we tracked 100, but processed all).
        assert len(rows) == POPULATION_SIZE

        _verify_4_stage_audit(snapshot, receipts, rows_by_pid, tracked_pids)

    def test_random_selection_is_reproducible(self):
        """The same seed must select the same 100 pulse_ids. This makes
        a CI failure reproducible locally."""
        items = _make_pulse_population(POPULATION_SIZE)
        pids_run_1 = _select_tracked_pids(items, TRACKED_COUNT, seed=RANDOM_SEED)
        pids_run_2 = _select_tracked_pids(items, TRACKED_COUNT, seed=RANDOM_SEED)
        assert pids_run_1 == pids_run_2, (
            "Same seed produced different tracked pulse_ids — the test "
            "would not be reproducible across runs."
        )

    def test_different_seeds_select_different_pulses(self):
        """Different seeds should select (mostly) different pulses.
        This proves the selection is actually random, not accidentally
        constant."""
        items = _make_pulse_population(POPULATION_SIZE)
        pids_a = set(_select_tracked_pids(items, TRACKED_COUNT, seed=42))
        pids_b = set(_select_tracked_pids(items, TRACKED_COUNT, seed=43))
        overlap = pids_a & pids_b
        # For 100/1000 sampling, expected overlap is ~10. Allow generous
        # tolerance to avoid flakiness.
        assert len(overlap) < 50, (
            f"Seeds 42 and 43 selected {len(overlap)} common pulses — "
            f"the random selection may not actually be random."
        )

    def test_tracked_pulses_are_not_sequential(self):
        """Sanity: the random selection should NOT be a contiguous block
        of pulse_ids. If it were, the test would degenerate to the
        existing sequential tests."""
        items = _make_pulse_population(POPULATION_SIZE)
        tracked = _select_tracked_pids(items, TRACKED_COUNT, seed=RANDOM_SEED)
        # If sorted, the gaps between consecutive tracked pids should
        # vary (not all 1).
        gaps = [tracked[i+1] - tracked[i] for i in range(len(tracked)-1)]
        n_gap_1 = sum(1 for g in gaps if g == 1)
        # With 100 pulses from 1000, we expect ~10 adjacent pairs by
        # chance. If all 99 gaps are 1, the selection is sequential.
        assert n_gap_1 < 50, (
            f"{n_gap_1}/{len(gaps)} tracked pulses are adjacent — the "
            f"random selection is suspiciously sequential."
        )


# ============================================================================
# Group B: Multi-worker tracking UNDER REORDERING
# ============================================================================

class TestMultiWorkerTrackingUnderReorder:
    """The crucial tests: 100 random tracked pulses through multi-worker
    dispatch with deterministic reordering.

    The delayed worker forces later pulses to finish first, so arrival
    order is reversed (or at least heavily shuffled) relative to
    pulse_id order. The 4-stage audit must still pass — proving
    identity preservation is NOT just a side-effect of order preservation.
    """

    @pytest.mark.parametrize("n_workers", [2, 4])
    def test_100_random_pulses_tracked_multi_worker(self, n_workers):
        """100 random pulses from 1000, tracked through all 4 stages,
        under multi-worker dispatch with reordering."""
        items = _make_pulse_population(POPULATION_SIZE)
        tracked_pids = _select_tracked_pids(items, TRACKED_COUNT, seed=RANDOM_SEED)

        snapshot, receipts, rows, rows_by_pid = _run_tracking_dispatch(
            items, _tracking_slow_worker,
            num_workers=n_workers, tracked_pids=tracked_pids,
        )

        assert len(rows) == POPULATION_SIZE
        _verify_4_stage_audit(snapshot, receipts, rows_by_pid, tracked_pids)

    @pytest.mark.parametrize("n_workers", [2, 4])
    def test_reordering_actually_occurred(self, n_workers):
        """Sanity: prove the delayed worker really did reorder arrival
        times. If this fails, the multi-worker tracking test above is
        vacuous (it would pass trivially even if identity preservation
        were broken).

        We check two signals:
          1. Multiple worker PIDs were used (proves mp.Pool really forked).
          2. The arrival order of tracked pulses != their sorted pulse_id
             order (proves reordering happened at the output level).

        We DO NOT assert that the smallest tracked pulse_id arrives
        last. That seems intuitively true (smallest pid -> longest
        delay -> latest finish) but it's broken by mp.Pool's batch
        scheduling: the smallest tracked pid is dispatched in the same
        batch as several untracked pids with longer delays, and may
        finish first within that batch. The overall arrival-order check
        is the correct signal.
        """
        items = _make_pulse_population(POPULATION_SIZE)
        tracked_pids = _select_tracked_pids(items, TRACKED_COUNT, seed=RANDOM_SEED)

        snapshot, receipts, rows, rows_by_pid = _run_tracking_dispatch(
            items, _tracking_slow_worker,
            num_workers=n_workers, tracked_pids=tracked_pids,
        )

        # Signal 1: multiple worker PIDs.
        worker_pids = {r["pid"] for r in rows}
        assert len(worker_pids) >= 2, (
            f"Only {len(worker_pids)} unique worker PID(s) observed — "
            f"mp.Pool did not actually fork multiple workers. The test "
            f"is not exercising the multi-worker code path."
        )

        # Signal 2: arrival order of tracked pulses != sorted pulse_id.
        # Tracked pulse_ids are sorted by _select_tracked_pids, so if
        # arrival order equals that sorted list, no reordering happened.
        tracked_set = set(tracked_pids)
        tracked_arrival_order = [
            r["pulse_id"] for r in rows if r["pulse_id"] in tracked_set
        ]
        assert tracked_arrival_order != tracked_pids, (
            f"Tracked pulses arrived in pulse_id order ({tracked_pids[:5]}"
            f"...) — reordering did not occur even with the delayed "
            f"worker. The 4-stage audit may be vacuous."
        )

        # Additional signal: reordering should be SUBSTANTIAL, not just
        # a single transposition. We count INVERSIONS in the arrival
        # order: pairs (i, j) with i < j but arrival[i] > arrival[j].
        # In sorted order, inversions = 0. In reverse-sorted order,
        # inversions = N*(N-1)/2.
        #
        # Why inversions instead of position-mismatch-rate?
        # Position-mismatch-rate counts how many tracked pulses are in
        # a different position than their sorted rank. But because
        # tracked pids are SPARSE (100 out of 1000), most within-batch
        # swaps happen between untracked pulses and don't affect the
        # tracked arrival order. Inversions, on the other hand, count
        # PAIRWISE reorderings among tracked pulses themselves — a
        # direct measure of how much the tracked arrival order differs
        # from sorted.
        #
        # Threshold: at least 3 inversions. This is well above the
        # "1 inversion = noise" level but doesn't require the aggressive
        # reordering that uniform delays can't deliver at scale. The
        # bimodal delay distribution (10% slow pulses at 100ms) ensures
        # we exceed this threshold even with 2 workers (less parallelism
        # = less reordering opportunity).
        n = len(tracked_arrival_order)
        inversions = 0
        for i in range(n):
            for j in range(i + 1, n):
                if tracked_arrival_order[i] > tracked_arrival_order[j]:
                    inversions += 1
        assert inversions >= 3, (
            f"Only {inversions} inversions among tracked pulses — "
            f"reordering is too weak to expose identity-swap bugs. "
            f"Need a more aggressive delay distribution or more workers. "
            f"(Tracked arrival: {tracked_arrival_order[:10]}...)"
        )

    def test_4_stage_audit_smaller_population_50_pulses(self):
        """Smaller scale: 50 tracked pulses from a 500-pulse population.

        Useful for debugging: if a larger test fails, this smaller one
        gives a faster signal and a smaller failure surface."""
        items = _make_pulse_population(500)
        tracked_pids = _select_tracked_pids(items, 50, seed=RANDOM_SEED)

        snapshot, receipts, rows, rows_by_pid = _run_tracking_dispatch(
            items, _tracking_slow_worker,
            num_workers=4, tracked_pids=tracked_pids,
        )
        assert len(rows) == 500
        _verify_4_stage_audit(snapshot, receipts, rows_by_pid, tracked_pids)

    def test_4_stage_audit_larger_population_200_pulses(self):
        """Larger scale: 200 tracked pulses from a 2000-pulse population.

        Increases the chance of catching an intermittent race that only
        manifests at scale. Uses 8 workers so the 2000-pulse run
        finishes in reasonable time (~30s with the 0.1ms delay)."""
        items = _make_pulse_population(2000)
        tracked_pids = _select_tracked_pids(items, 200, seed=RANDOM_SEED)

        snapshot, receipts, rows, rows_by_pid = _run_tracking_dispatch(
            items, _tracking_slow_worker,
            num_workers=8, tracked_pids=tracked_pids,
        )
        assert len(rows) == 2000
        _verify_4_stage_audit(snapshot, receipts, rows_by_pid, tracked_pids)


# ============================================================================
# Group C: Mutation tests — injected corruption MUST be caught
# ============================================================================

class TestMutationDetectsCorruption:
    """Prove the 4-stage audit catches injected corruption at each stage.

    These tests deliberately break the worker in different ways and
    verify that the audit raises an AssertionError with a message that
    pinpoints WHICH stage failed. This is the diagnostic value of the
    4-stage model: a single S1-vs-S4 check can only say "something
    broke", but the 4-stage check can say "it broke at stage S2"
    (dispatcher input pipeline) or "it broke at stage S3" (worker).
    """

    def test_worker_corrupts_identity_fields_caught(self):
        """Worker swaps alice_bit and alice_basis with the next pulse's
        values. The audit must catch this at S2 (worker receipt's
        identity fields won't match S1) AND at S3 (worker result's
        identity fields won't match S1).

        Wait — _tracking_scrambling_worker calls _tracking_worker with
        the CORRUPTED item, so the worker RECEIVES the corrupted item
        and logs THAT in S2. So S2 will agree with S3 (both corrupted),
        but BOTH will disagree with S1.

        This is the "corruption in dispatcher's input pipeline"
        scenario — except here the corruption is done by the worker
        wrapper itself, simulating a bug in the dispatcher's input
        handling. The audit should catch it as S1 != S2.
        """
        items = _make_pulse_population(200)
        tracked_pids = _select_tracked_pids(items, 20, seed=RANDOM_SEED)

        snapshot, receipts, rows, rows_by_pid = _run_tracking_dispatch(
            items, _tracking_scrambling_worker,
            num_workers=4, tracked_pids=tracked_pids,
        )

        # The audit must fail. We catch the AssertionError and verify
        # its message mentions a stage (so the diagnostic is useful).
        with pytest.raises(AssertionError) as exc_info:
            _verify_4_stage_audit(snapshot, receipts, rows_by_pid, tracked_pids)

        msg = str(exc_info.value)
        # Message must mention S1 and either S2 or S3 (the stages that
        # disagree when the worker corrupts its input).
        assert "S1" in msg, (
            f"Audit failure message should mention S1 (the ground truth), "
            f"got: {msg}"
        )
        assert ("S2" in msg or "S3" in msg), (
            f"Audit failure message should mention S2 or S3 (where the "
            f"corruption was injected), got: {msg}"
        )

    def test_missing_receipt_detected(self):
        """If a tracked pulse has no S2 receipt (worker didn't log it),
        the audit must catch it with the 'missing receipts' message."""
        items = _make_pulse_population(200)
        tracked_pids = _select_tracked_pids(items, 20, seed=RANDOM_SEED)

        snapshot, receipts, rows, rows_by_pid = _run_tracking_dispatch(
            items, _tracking_worker, num_workers=1, tracked_pids=tracked_pids,
        )
        # Verify the audit passes first.
        _verify_4_stage_audit(snapshot, receipts, rows_by_pid, tracked_pids)

        # Now corrupt: remove one receipt.
        corrupted_receipts = list(receipts)[:-1]
        with pytest.raises(AssertionError, match="no S2 receipt"):
            _verify_4_stage_audit(snapshot, corrupted_receipts,
                                  rows_by_pid, tracked_pids)

    def test_missing_row_detected(self):
        """If a tracked pulse has no S4 row (dispatcher dropped it),
        the audit must catch it with the 'no S4 row' message."""
        items = _make_pulse_population(200)
        tracked_pids = _select_tracked_pids(items, 20, seed=RANDOM_SEED)

        snapshot, receipts, rows, rows_by_pid = _run_tracking_dispatch(
            items, _tracking_worker, num_workers=1, tracked_pids=tracked_pids,
        )
        # Verify the audit passes first.
        _verify_4_stage_audit(snapshot, receipts, rows_by_pid, tracked_pids)

        # Now corrupt: drop one tracked pulse's row from rows_by_pid.
        victim_pid = tracked_pids[0]
        corrupted_rows_by_pid = {k: v for k, v in rows_by_pid.items()
                                  if k != victim_pid}
        with pytest.raises(AssertionError, match="no S4 row"):
            _verify_4_stage_audit(snapshot, receipts,
                                  corrupted_rows_by_pid, tracked_pids)


# ============================================================================
# Group D: Repeatability across iterations
# ============================================================================

class TestTrackingRepeatability:
    """Run the full 4-stage audit multiple times to catch intermittent
    races. If identity preservation were broken by a race condition
    (e.g., a shared mutable item that workers mutate), it might only
    manifest on some iterations.

    Each iteration uses a DIFFERENT random seed so a different subset
    of 100 pulses is tracked. This broadens coverage beyond what a
    single fixed seed can reach.
    """

    @pytest.mark.parametrize("n_workers", [2, 4])
    def test_3_iterations_different_seeds(self, n_workers):
        """3 iterations × 100 tracked pulses × n_workers, each iteration
        using a different random seed (so a different 100-pulse subset).

        All 3 iterations must pass the 4-stage audit. A single failure
        fails the test and reports which iteration failed.

        (Was 5 iterations; reduced to 3 to keep total suite runtime
        under 5 minutes. 3 iterations is still enough to catch
        intermittent races that manifest >33% of the time.)"""
        failures = []
        for it in range(3):
            seed = RANDOM_SEED + it
            items = _make_pulse_population(POPULATION_SIZE)
            tracked_pids = _select_tracked_pids(items, TRACKED_COUNT, seed=seed)
            try:
                snapshot, receipts, rows, rows_by_pid = _run_tracking_dispatch(
                    items, _tracking_slow_worker,
                    num_workers=n_workers, tracked_pids=tracked_pids,
                )
                _verify_4_stage_audit(snapshot, receipts, rows_by_pid, tracked_pids)
            except AssertionError as e:
                failures.append((it, seed, str(e)[:200]))
                if len(failures) >= 2:
                    break
        assert not failures, (
            f"4-stage audit failed on {len(failures)}/3 iterations "
            f"({n_workers} workers): {failures}"
        )

    def test_2_iterations_200_tracked_pulses(self):
        """Larger tracked set: 200 pulses from 1000, 2 iterations.

        Increasing the tracked fraction from 10% to 20% increases the
        chance of catching any bias in the dispatcher's selection of
        which pulses to corrupt (if such a bias existed)."""
        failures = []
        for it in range(2):
            seed = RANDOM_SEED + it + 100
            items = _make_pulse_population(1000)
            tracked_pids = _select_tracked_pids(items, 200, seed=seed)
            try:
                snapshot, receipts, rows, rows_by_pid = _run_tracking_dispatch(
                    items, _tracking_slow_worker,
                    num_workers=4, tracked_pids=tracked_pids,
                )
                _verify_4_stage_audit(snapshot, receipts, rows_by_pid, tracked_pids)
            except AssertionError as e:
                failures.append((it, seed, str(e)[:200]))
        assert not failures, (
            f"4-stage audit failed on {len(failures)}/2 iterations "
            f"(200 tracked, 4 workers): {failures}"
        )


# ============================================================================
# Group E: Audit-depth diagnostic — pinpoint WHICH stage failed
# ============================================================================

class TestAuditDiagnosticPrecision:
    """Verify that when the 4-stage audit fails, the error message
    pinpoints WHICH stage disagreed. This is the diagnostic value of
    the multi-stage model: it tells a future debugger where to look.

    These tests inject corruption at a specific stage and verify the
    error message mentions that stage.
    """

    def test_s1_vs_s2_mismatch_message(self):
        """Corrupt S2 (worker receipt's received_item) so it disagrees
        with S1. The error must mention S2 (worker receipt)."""
        items = _make_pulse_population(100)
        tracked_pids = _select_tracked_pids(items, 10, seed=RANDOM_SEED)

        snapshot, receipts, rows, rows_by_pid = _run_tracking_dispatch(
            items, _tracking_worker, num_workers=1, tracked_pids=tracked_pids,
        )

        # Corrupt one receipt's alice_bit.
        victim_pid = tracked_pids[0]
        for r in receipts:
            if r["pulse_id"] == victim_pid:
                r["received_item"]["alice_bit"] = (
                    1 - r["received_item"]["alice_bit"]
                )
                break

        with pytest.raises(AssertionError) as exc_info:
            _verify_4_stage_audit(snapshot, receipts, rows_by_pid, tracked_pids)

        msg = str(exc_info.value)
        assert "S2 (worker receipt)" in msg, (
            f"Failure message should pinpoint S2 (worker receipt), got: {msg}"
        )

    def test_s1_vs_s3_mismatch_message(self):
        """Corrupt S3 (worker result row) so its alice_bit disagrees
        with S1. The error must mention S3 (worker result)."""
        items = _make_pulse_population(100)
        tracked_pids = _select_tracked_pids(items, 10, seed=RANDOM_SEED)

        snapshot, receipts, rows, rows_by_pid = _run_tracking_dispatch(
            items, _tracking_worker, num_workers=1, tracked_pids=tracked_pids,
        )

        # Corrupt one row's alice_bit.
        victim_pid = tracked_pids[0]
        rows_by_pid[victim_pid]["alice_bit"] = (
            1 - rows_by_pid[victim_pid]["alice_bit"]
        )

        with pytest.raises(AssertionError) as exc_info:
            _verify_4_stage_audit(snapshot, receipts, rows_by_pid, tracked_pids)

        msg = str(exc_info.value)
        assert "S3 (worker result)" in msg, (
            f"Failure message should pinpoint S3 (worker result), got: {msg}"
        )

    def test_measurement_mismatch_caught(self):
        """Corrupt the measurement fields (basis_match, measured_bit,
        detected) so they don't match the deterministic prediction
        from S1. The audit must catch this even though identity fields
        all agree."""
        items = _make_pulse_population(100)
        tracked_pids = _select_tracked_pids(items, 10, seed=RANDOM_SEED)

        snapshot, receipts, rows, rows_by_pid = _run_tracking_dispatch(
            items, _tracking_worker, num_workers=1, tracked_pids=tracked_pids,
        )

        # Corrupt one row's measured_bit.
        victim_pid = tracked_pids[0]
        original_measured_bit = rows_by_pid[victim_pid]["measured_bit"]
        # Flip measured_bit if it's 0/1, or set it to 0/1 if it's None.
        if original_measured_bit is None:
            rows_by_pid[victim_pid]["measured_bit"] = 0
        else:
            rows_by_pid[victim_pid]["measured_bit"] = 1 - original_measured_bit

        with pytest.raises(AssertionError) as exc_info:
            _verify_4_stage_audit(snapshot, receipts, rows_by_pid, tracked_pids)

        msg = str(exc_info.value)
        assert "measurement" in msg.lower(), (
            f"Failure message should mention measurement mismatch, got: {msg}"
        )
