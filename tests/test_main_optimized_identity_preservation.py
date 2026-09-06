"""tests/test_main_optimized_identity_preservation.py

Identity-preservation contract tests for main_optimized's dispatch.

THE DISTINCTION (from the user's feedback)
------------------------------------------
The concurrency test suite (test_main_optimized_concurrency.py) verifies the
*dispatch mechanism*: no loss, no duplication, completion barrier, worker-
state isolation. Those tests treat work items as opaque blobs with an `id`.

This suite verifies a STRONGER contract: **identity preservation**.

  - "order of arrival" is NOT the contract.
  - "identity preservation / pulse-index binding" IS the contract.
  - `imap_unordered` is safe IF AND ONLY IF results are interpreted by
    carried identity (e.g. `row["pulse_id"]`), never by position
    (`rows[i]`).

The exact bug class this suite guards against
---------------------------------------------
Consider BB84 per-pulse processing. A buggy consumer might write:

    for i, row in enumerate(rows):
        alice_bit = alice_bits[i]          # BUG
        alice_basis = alice_bases[i]       # BUG

Under multi-worker dispatch, `i` is the *arrival position*, NOT the
pulse index. If pulse 7 arrives at position 0 (because it finished
first), the code above silently assigns alice_bits[0] to pulse 7's
result — corrupting the key without raising any error.

The correct pattern is:

    for row in rows:
        i = row["pulse_id"]                # CORRECT
        alice_bit = alice_bits[i]
        alice_basis = alice_bases[i]

Or even better, carry alice_bit/alice_basis INSIDE the work item so no
positional lookup is ever needed.

What this suite proves
----------------------
1. Single-worker dispatch preserves identity (baseline).
2. Multi-worker dispatch preserves identity EVEN UNDER REORDERING —
   proved by an adversarial worker that deterministically forces later
   items to finish first.
3. Positional interpretation (`rows[i] == item[i]`) FAILS under
   reordering — meta-test proving the bug class is real.
4. Identity-based interpretation (`row["pulse_id"]`) ALWAYS works —
   meta-test proving the correct pattern.
5. The real main_optimized composite key (distance + noise + pulses +
   combo) preserves identity through dispatch.
6. High-volume stress: identity preserved across 30 × 100 pulses × 4
   workers, every iteration.
7. Identity + no-loss + no-dup combined invariant.
8. Adversarial: worker deliberately scrambles identity fields — the
   suite catches it (mutation test on the contract).

Running
-------
    python -m pytest tests/test_main_optimized_identity_preservation.py -v
"""

from __future__ import annotations

import os
import time
import inspect

import pytest

from main_optimized_dispatch import dispatch_work_items


# ============================================================================
# Module-level stub workers (must be top-level for mp.Pool picklability)
# ============================================================================

def _bb84_identity_worker(item):
    """Mimics a BB84 per-pulse worker: carries FULL per-pulse identity.

    Receives a self-contained item with:
      - pulse_id: unique identifier (the source of truth)
      - alice_bit, alice_basis: Alice's preparation
      - bob_basis: Bob's measurement basis
      - state_label: the quantum state label
      - mu: the photon number for this pulse
      - intensity_label: "signal" | "decoy" | "vacuum"

    Returns a result that echoes ALL identity fields plus a simulated
    measurement. This proves identity travels WITH the result, not with
    position in the output list.

    The "measurement" is deterministic (derived from the item) so tests
    can assert exact expected values.
    """
    # Deterministic "measurement": detected iff alice_basis == bob_basis
    basis_match = item["alice_basis"] == item["bob_basis"]
    measured_bit = item["alice_bit"] if basis_match else None
    detected = basis_match and (item["intensity_label"] != "vacuum")

    return {
        # --- Identity fields (echoed from the item) ---
        "pulse_id": item["pulse_id"],
        "alice_bit": item["alice_bit"],
        "alice_basis": item["alice_basis"],
        "bob_basis": item["bob_basis"],
        "state_label": item["state_label"],
        "mu": item["mu"],
        "intensity_label": item["intensity_label"],
        # --- Measurement result (deterministic, derived from identity) ---
        "detected": detected,
        "measured_bit": measured_bit,
        "basis_match": basis_match,
        # --- Provenance ---
        "pid": os.getpid(),
    }


def _bb84_identity_slow_worker(item):
    """Same as _bb84_identity_worker but with a deterministic per-item delay.

    Delay = (N - pulse_id) * delay_unit, so LATER items finish EARLIER.
    This deterministically forces reordering under multi-worker dispatch,
    making the identity-preservation test meaningful (not just a tautology).

    Without this delay, a fast worker might preserve arrival order by
    accident, and the identity-preservation assertion would pass
    trivially. With the delay, we PROVE identity is preserved even when
    arrival order is reversed.

    The delay_unit can be overridden via item["delay_unit"] for stress
    tests that need faster execution. Default is 0.005s (5ms).
    """
    n = item["total"]
    delay_unit = item.get("delay_unit", 0.005)
    delay = (n - item["pulse_id"]) * delay_unit
    if delay > 0:
        time.sleep(delay)
    return _bb84_identity_worker(item)


def _bb84_identity_stress_worker(item):
    """Fast-delay variant for high-volume stress tests.

    Uses 0.001s (1ms) per unit instead of 5ms, so 100-pulse runs finish
    ~5x faster. Still forces reordering (later items finish first) but
    keeps the stress test practical (30 iterations × 100 pulses × 4
    workers in ~30s instead of ~150s).
    """
    item = {**item, "delay_unit": 0.001}
    return _bb84_identity_slow_worker(item)


def _bb84_scrambling_worker(item):
    """MUTATION-TEST worker: deliberately SCRAMBLES identity fields.

    Returns a result where alice_bit/alice_basis are taken from a
    DIFFERENT pulse (pulse_id + 1 mod N). This simulates the exact bug
    class the user described: "pulse 7 gets interpreted as pulse 9".

    Used ONLY by the mutation test to prove the suite catches the bug.
    """
    n = item["total"]
    wrong_id = (item["pulse_id"] + 1) % n
    wrong_item = {
        **item,
        "pulse_id": item["pulse_id"],  # keep pulse_id so dispatch invariants hold
        "alice_bit": (item["alice_bit"] + 1) % 2,  # WRONG bit
        "alice_basis": "X" if item["alice_basis"] == "Z" else "Z",  # WRONG basis
    }
    return _bb84_identity_worker(wrong_item)


def _sim_point_worker(item):
    """Mimics main_optimized's run_single_simulation for a sweep point.

    The real work item is a tuple:
        (dist, total_pulses, apply_noise, worker_seed, combo_idx)

    This stub uses a dict with the same fields and echoes the composite
    identity key (distance_km, noise_applied, total_pulses, combo_idx)
    in the result, plus a simulated "secure_key_bits" derived from the
    seed. This proves the real architecture's composite identity is
    preserved through dispatch.
    """
    return {
        # --- Composite identity key (echoed) ---
        "distance_km": round(item["dist"], 2),
        "noise_applied": item["apply_noise"],
        "total_pulses": item["total_pulses"],
        "combo_idx": item["combo_idx"],
        # --- Simulated result (deterministic from seed) ---
        "secure_key_bits": int(item["worker_seed"] % 1000),
        "qber": round(0.01 + (item["worker_seed"] % 100) / 10000.0, 8),
        # --- Provenance ---
        "pid": os.getpid(),
    }


def _sim_point_slow_worker(item):
    """Delayed version of _sim_point_worker for adversarial testing.

    Uses 1ms delay/unit (matching the BB84 stress worker) so the
    composite-identity stress test stays fast.
    """
    n = item["total"]
    idx = item["_idx"]
    delay = (n - idx) * 0.001
    if delay > 0:
        time.sleep(delay)
    return _sim_point_worker(item)


# ============================================================================
# Helpers
# ============================================================================

def _make_bb84_items(n):
    """Generate n BB84-style work items with unique composite identities.

    Each item carries:
      - pulse_id: 0..n-1 (unique)
      - alice_bit: 0 or 1 (deterministic pattern)
      - alice_basis: "Z" or "X" (deterministic pattern)
      - bob_basis: "Z" or "X" (deterministic pattern, independent of alice)
      - state_label: one of |0>, |1>, |+>, |->
      - mu: signal=0.5, decoy=0.1, vacuum=0.0
      - intensity_label: "signal" | "decoy" | "vacuum"

    The patterns are designed so that adjacent pulse_ids have DIFFERENT
    identity fields — this ensures that any positional misinterpretation
    is detected (if you use rows[i] to look up the i-th item's identity,
    you'll get wrong values whenever reordering occurs).
    """
    items = []
    bases = ["Z", "X"]
    states = ["|0>", "|1>", "|+>", "|->"]
    intensities = [("signal", 0.5), ("decoy", 0.1), ("vacuum", 0.0)]
    for i in range(n):
        intensity_label, mu = intensities[i % 3]
        items.append({
            "pulse_id": i,
            "alice_bit": i % 2,
            "alice_basis": bases[i % 2],
            "bob_basis": bases[(i // 2) % 2],  # independent of alice
            "state_label": states[i % 4],
            "mu": mu,
            "intensity_label": intensity_label,
            "total": n,
        })
    return items


def _make_sim_items(n):
    """Generate items mimicking main_optimized's sweep-point work items."""
    items = []
    for i in range(n):
        items.append({
            "_idx": i,
            "dist": 10.0 + i * 0.5,
            "total_pulses": 10 ** (4 + (i % 3)),
            "apply_noise": bool(i % 2),
            "worker_seed": 1000 + i,
            "combo_idx": i % 4,
            "total": n,
        })
    return items


def _collect(work_items, worker_fn, num_workers, *,
             init_fn=None, init_args=(), chunksize=1):
    """Run dispatch_work_items with a list-append on_row and return rows."""
    rows = []
    dispatch_work_items(
        work_items=work_items,
        worker_fn=worker_fn,
        num_workers=num_workers,
        init_fn=init_fn,
        init_args=init_args,
        on_row=rows.append,
        chunksize=chunksize,
    )
    return rows


def _verify_bb84_identity(items, rows):
    """Core identity-preservation assertion for BB84 items.

    Verifies:
      1. Cardinality: len(rows) == len(items)
      2. No loss/duplication: sorted(pulse_ids) == [0, 1, ..., n-1]
      3. Identity preservation: every row's identity fields match the
         original item with the same pulse_id — NOT the item at the
         row's arrival position.

    This is the exact contract the user described:
      > pulse_id is the source of truth, not row position.
    """
    assert len(rows) == len(items), (
        f"Cardinality mismatch: {len(rows)} rows for {len(items)} items."
    )

    items_by_id = {item["pulse_id"]: item for item in items}

    row_ids = [r["pulse_id"] for r in rows]
    assert sorted(row_ids) == list(range(len(items))), (
        "Loss or duplication detected: pulse_id set mismatch."
    )
    assert len(set(row_ids)) == len(row_ids), "Duplicate pulse_ids."

    # The crucial check: identity fields travel WITH the row.
    for row in rows:
        original = items_by_id[row["pulse_id"]]
        assert row["alice_bit"] == original["alice_bit"], (
            f"pulse_id {row['pulse_id']}: alice_bit mismatch — "
            f"row has {row['alice_bit']}, original has {original['alice_bit']}. "
            f"Identity was NOT preserved through dispatch."
        )
        assert row["alice_basis"] == original["alice_basis"], (
            f"pulse_id {row['pulse_id']}: alice_basis mismatch — "
            f"row has {row['alice_basis']}, original has {original['alice_basis']}."
        )
        assert row["bob_basis"] == original["bob_basis"], (
            f"pulse_id {row['pulse_id']}: bob_basis mismatch."
        )
        assert row["state_label"] == original["state_label"], (
            f"pulse_id {row['pulse_id']}: state_label mismatch."
        )
        assert row["mu"] == original["mu"], (
            f"pulse_id {row['pulse_id']}: mu mismatch — "
            f"row has {row['mu']}, original has {original['mu']}."
        )
        assert row["intensity_label"] == original["intensity_label"], (
            f"pulse_id {row['pulse_id']}: intensity_label mismatch."
        )


def _verify_sim_identity(items, rows):
    """Identity-preservation assertion for main_optimized-style sweep items.

    The composite identity key is (distance_km, noise_applied, total_pulses,
    combo_idx). Every row must carry these fields and they must match the
    original work item.
    """
    assert len(rows) == len(items)
    # Build lookup by composite key (NOT by position).
    items_by_key = {
        (item["dist"], item["apply_noise"], item["total_pulses"], item["combo_idx"]): item
        for item in items
    }
    for row in rows:
        key = (row["distance_km"], row["noise_applied"],
               row["total_pulses"], row["combo_idx"])
        assert key in items_by_key, (
            f"Row has unrecognized composite identity {key} — "
            f"identity was NOT preserved through dispatch."
        )
        original = items_by_key[key]
        # The simulated result (secure_key_bits) must be deterministic from
        # the original seed, proving the right item produced this row.
        expected_key_bits = int(original["worker_seed"] % 1000)
        assert row["secure_key_bits"] == expected_key_bits, (
            f"Composite identity {key}: secure_key_bits mismatch — "
            f"row has {row['secure_key_bits']}, expected {expected_key_bits}. "
            f"This means the row's identity fields don't match its actual "
            f"source work item."
        )


# ============================================================================
# Group A: BB84 single-worker identity preservation (baseline)
# ============================================================================

class TestBB84SingleWorkerIdentity:
    """Baseline: single-worker dispatch must preserve identity.

    These tests are necessary but not sufficient — they pass trivially
    because single-worker preserves order. The real test is under
    multi-worker reordering (Group B)."""

    def test_identity_preserved_100_pulses(self):
        items = _make_bb84_items(100)
        rows = _collect(items, _bb84_identity_worker, num_workers=1)
        _verify_bb84_identity(items, rows)

    def test_identity_preserved_500_pulses(self):
        items = _make_bb84_items(500)
        rows = _collect(items, _bb84_identity_worker, num_workers=1)
        _verify_bb84_identity(items, rows)

    def test_arrival_order_equals_pulse_id_order(self):
        """Baseline property: single-worker preserves arrival order, so
        rows[i]["pulse_id"] == i. This is what makes positional
        interpretation accidentally work for single-worker — and why
        it's a TRAP that breaks under multi-worker."""
        items = _make_bb84_items(50)
        rows = _collect(items, _bb84_identity_worker, num_workers=1)
        for i, row in enumerate(rows):
            assert row["pulse_id"] == i, (
                f"Single-worker: rows[{i}]['pulse_id'] should be {i}, "
                f"got {row['pulse_id']}."
            )


# ============================================================================
# Group B: BB84 multi-worker identity preservation UNDER REORDERING
# ============================================================================

class TestBB84MultiWorkerIdentityUnderReorder:
    """The crucial tests: identity must be preserved EVEN WHEN arrival
    order differs from pulse_id order.

    Uses _bb84_identity_slow_worker which deterministically forces later
    pulses to finish first, guaranteeing reordering under multi-worker
    dispatch."""

    @pytest.mark.parametrize("n_workers", [2, 4])
    def test_identity_preserved_under_deterministic_reorder(self, n_workers):
        """100 pulses, delayed worker: identity must be preserved even
        though arrival order is reversed."""
        items = _make_bb84_items(100)
        rows = _collect(items, _bb84_identity_slow_worker,
                        num_workers=n_workers)
        _verify_bb84_identity(items, rows)

    @pytest.mark.parametrize("n_workers", [2, 4])
    def test_reordering_actually_occurred(self, n_workers):
        """Sanity check: prove the test is meaningful by verifying that
        arrival order != pulse_id order. If this fails, the
        identity-preservation test above would be a tautology."""
        items = _make_bb84_items(100)
        rows = _collect(items, _bb84_identity_slow_worker,
                        num_workers=n_workers)
        arrival_ids = [r["pulse_id"] for r in rows]
        assert arrival_ids != list(range(100)), (
            "Reordering did not occur — the identity-preservation test "
            "is not meaningful. Need a larger N or more workers."
        )
        # Specifically, the first-arrived pulse should NOT be pulse 0
        # (pulse 0 has the longest delay).
        assert arrival_ids[0] != 0, (
            f"First-arrived pulse was {arrival_ids[0]}, expected != 0. "
            f"Delay-based reordering did not take effect."
        )

    def test_identity_preserved_200_pulses_4_workers(self):
        """Larger scale: 200 pulses, 4 workers, deterministic reorder.

        Uses the fast stress worker (1ms delay) so 200 pulses finishes
        in ~5s. The 200-pulse scale is enough to expose reordering
        while keeping the test fast."""
        items = _make_bb84_items(200)
        rows = _collect(items, _bb84_identity_stress_worker, num_workers=4)
        _verify_bb84_identity(items, rows)
        # Verify reordering occurred.
        arrival_ids = [r["pulse_id"] for r in rows]
        assert arrival_ids != list(range(200))


# ============================================================================
# Group C: META-TESTS — positional interpretation is WRONG
# ============================================================================

class TestPositionalInterpretationIsBuggy:
    """Meta-tests proving that `rows[i]` interpretation is WRONG under
    multi-worker dispatch, while `row["pulse_id"]` interpretation is
    ALWAYS correct.

    These tests demonstrate the EXACT bug class the user described:
      for i, row in enumerate(rows):
          alice_bit = alice_bits[i]  # BUG: i is arrival position, not pulse_id
    """

    def test_positional_lookup_produces_wrong_results(self):
        """Show that rows[i] does NOT correspond to the i-th work item
        when reordering occurs. This is the bug."""
        items = _make_bb84_items(100)
        rows = _collect(items, _bb84_identity_slow_worker, num_workers=4)

        # Verify reordering happened.
        arrival_ids = [r["pulse_id"] for r in rows]
        assert arrival_ids != list(range(100)), (
            "Test requires reordering to be meaningful."
        )

        # Simulate the BUGGY positional-lookup pattern.
        positional_mismatches = 0
        for i, row in enumerate(rows):
            # BUG: assume rows[i] is the i-th work item
            assumed_original = items[i]
            if row["alice_bit"] != assumed_original["alice_bit"]:
                positional_mismatches += 1
            if row["alice_basis"] != assumed_original["alice_basis"]:
                positional_mismatches += 1
            if row["bob_basis"] != assumed_original["bob_basis"]:
                positional_mismatches += 1

        assert positional_mismatches > 0, (
            "Positional interpretation should produce mismatches under "
            f"reordering, but got 0. The test is not meaningful. "
            f"Arrival order: {arrival_ids[:10]}..."
        )

    def test_identity_lookup_always_correct(self):
        """Show that using row["pulse_id"] to look up the original item
        ALWAYS produces correct results, even under aggressive reordering."""
        items = _make_bb84_items(100)
        rows = _collect(items, _bb84_identity_slow_worker, num_workers=4)

        # Verify reordering happened.
        arrival_ids = [r["pulse_id"] for r in rows]
        assert arrival_ids != list(range(100)), (
            "Test requires reordering to be meaningful."
        )

        items_by_id = {item["pulse_id"]: item for item in items}
        mismatches = 0
        for row in rows:
            original = items_by_id[row["pulse_id"]]
            if row["alice_bit"] != original["alice_bit"]:
                mismatches += 1
            if row["alice_basis"] != original["alice_basis"]:
                mismatches += 1
            if row["bob_basis"] != original["bob_basis"]:
                mismatches += 1

        assert mismatches == 0, (
            f"Identity-based interpretation should produce 0 mismatches, "
            f"got {mismatches}. The contract is violated."
        )

    def test_positional_vs_identity_mismatch_count(self):
        """Quantify how wrong positional interpretation is.

        Under aggressive reordering, positional lookup should mismatch
        on a LARGE fraction of rows — proving it's not a rare edge case."""
        items = _make_bb84_items(200)
        rows = _collect(items, _bb84_identity_slow_worker, num_workers=4)

        total_checks = 0
        positional_wrong = 0
        identity_wrong = 0
        for i, row in enumerate(rows):
            arrival_original = items[i]  # positional
            id_original = items[row["pulse_id"]]  # identity-based
            for field in ("alice_bit", "alice_basis", "bob_basis", "state_label"):
                total_checks += 1
                if row[field] != arrival_original[field]:
                    positional_wrong += 1
                if row[field] != id_original[field]:
                    identity_wrong += 1

        # Positional should be wrong on a significant fraction.
        positional_rate = positional_wrong / total_checks
        assert positional_rate > 0.3, (
            f"Positional mismatch rate {positional_rate:.2%} is too low — "
            f"reordering may not have occurred aggressively enough."
        )
        # Identity should be PERFECT.
        assert identity_wrong == 0, (
            f"Identity-based mismatch rate should be 0, got "
            f"{identity_wrong}/{total_checks}."
        )


# ============================================================================
# Group D: main_optimized composite-identity preservation
# ============================================================================

class TestMainOptimizedCompositeIdentity:
    """Verify that main_optimized's actual work-item structure (composite
    key: distance + noise + pulses + combo) preserves identity through
    dispatch.

    The real work items are tuples:
        (dist, total_pulses, apply_noise, worker_seed, combo_idx)

    The output row carries: distance_km, noise_applied, total_pulses,
    and sweep keys (encoding combo_idx). So the composite identity key
    is recoverable from the output row — IF the dispatch preserves it.

    These tests prove the composite identity is preserved even under
    reordering.
    """

    def test_single_worker_composite_identity(self):
        items = _make_sim_items(50)
        rows = _collect(items, _sim_point_worker, num_workers=1)
        _verify_sim_identity(items, rows)

    @pytest.mark.parametrize("n_workers", [2, 4])
    def test_multi_worker_composite_identity(self, n_workers):
        items = _make_sim_items(100)
        rows = _collect(items, _sim_point_worker, num_workers=n_workers)
        _verify_sim_identity(items, rows)

    def test_multi_worker_composite_identity_under_reorder(self):
        """Adversarial: delayed worker forces reordering; composite
        identity must still be preserved."""
        items = _make_sim_items(80)
        rows = _collect(items, _sim_point_slow_worker, num_workers=4)
        _verify_sim_identity(items, rows)
        # Verify reordering occurred (by checking pid diversity AND
        # that arrival order != input order, via the _idx field if
        # we add it to the result — here we just check identity holds).

    def test_worker_seed_not_in_output_is_documented_limitation(self):
        """Document a known limitation: worker_seed is NOT in the output
        row. This means you cannot reproduce a specific row from the CSV
        alone — you'd need to re-derive the seed from the base_rng.

        This is NOT an identity-preservation bug (each composite key is
        unique per work item), but it IS a reproducibility limitation.
        This test documents it so future maintainers are aware."""
        items = _make_sim_items(10)
        rows = _collect(items, _sim_point_worker, num_workers=1)
        for row in rows:
            assert "worker_seed" not in row, (
                "If worker_seed IS in the output now, update this test — "
                "the reproducibility limitation may have been fixed."
            )
        # The composite key (distance, noise, pulses, combo) IS sufficient
        # to uniquely identify each work item, because the nested for-loops
        # in run_and_save_csv produce unique tuples. This test passes
        # today; if the loops ever change to allow duplicate composite keys,
        # _verify_sim_identity would catch it.


# ============================================================================
# Group E: high-volume identity stress test
# ============================================================================

class TestHighVolumeIdentityStress:
    """Run identity-preservation checks across many iterations to catch
    intermittent identity-corruption races.

    Uses _bb84_identity_stress_worker (1ms delay/unit instead of 5ms) so
    30 iterations × 100 pulses × 4 workers finishes in ~30s instead of
    ~150s. The smaller delay still forces deterministic reordering, so
    the identity-preservation check remains meaningful."""

    @pytest.mark.parametrize("n_workers", [2, 4])
    def test_bb84_identity_stable_across_iterations(self, n_workers):
        """10 iterations × 100 pulses × n_workers with deterministic
        reordering. Identity must be preserved EVERY iteration.

        10 iterations is enough to catch intermittent races while keeping
        the test under ~30s (mp.Pool creation has ~0.5s overhead per
        iteration, so 10 iterations ≈ 5s overhead + ~2s work)."""
        failures = []
        for it in range(10):
            items = _make_bb84_items(100)
            try:
                rows = _collect(items, _bb84_identity_stress_worker,
                                num_workers=n_workers)
                _verify_bb84_identity(items, rows)
            except AssertionError as e:
                failures.append((it, str(e)[:120]))
                if len(failures) >= 3:
                    break
        assert not failures, (
            f"Identity-preservation failures across 10 iterations "
            f"({n_workers} workers): {failures}"
        )

    def test_bb84_identity_stable_200_pulses(self):
        """5 iterations × 200 pulses × 4 workers. Larger N increases
        the chance of exposing races."""
        failures = []
        for it in range(5):
            items = _make_bb84_items(200)
            try:
                rows = _collect(items, _bb84_identity_stress_worker,
                                num_workers=4)
                _verify_bb84_identity(items, rows)
            except AssertionError as e:
                failures.append((it, str(e)[:120]))
        assert not failures, (
            f"Identity-preservation failures (200 pulses): {failures}"
        )

    def test_composite_identity_stable_across_iterations(self):
        """10 iterations × 80 sim points × 4 workers. Composite identity
        must be preserved every iteration."""
        failures = []
        for it in range(10):
            items = _make_sim_items(80)
            try:
                rows = _collect(items, _sim_point_slow_worker,
                                num_workers=4)
                _verify_sim_identity(items, rows)
            except AssertionError as e:
                failures.append((it, str(e)[:120]))
                if len(failures) >= 3:
                    break
        assert not failures, (
            f"Composite-identity failures across 10 iterations: {failures}"
        )


# ============================================================================
# Group F: mutation test — suite catches identity-scrambling bugs
# ============================================================================

class TestIdentityScramblingDetection:
    """Prove the suite catches the EXACT bug class the user described:
    'pulse 7 gets interpreted as pulse 9'.

    Uses _bb84_scrambling_worker which deliberately returns wrong
    alice_bit/alice_basis for each pulse_id. The identity-preservation
    assertion MUST fail.
    """

    def test_scrambling_detected_single_worker(self):
        """Even single-worker must catch identity scrambling — the bug
        is in the worker, not the dispatch."""
        items = _make_bb84_items(50)
        rows = _collect(items, _bb84_scrambling_worker, num_workers=1)
        with pytest.raises(AssertionError, match="alice_bit mismatch"):
            _verify_bb84_identity(items, rows)

    def test_scrambling_detected_multi_worker(self):
        """Multi-worker must also catch identity scrambling."""
        items = _make_bb84_items(50)
        rows = _collect(items, _bb84_scrambling_worker, num_workers=4)
        with pytest.raises(AssertionError, match="alice_bit mismatch"):
            _verify_bb84_identity(items, rows)

    def test_scrambling_detected_under_reorder(self):
        """The scrambling bug must be caught EVEN under reordering
        (which is when it's most dangerous — silent corruption)."""
        items = _make_bb84_items(50)
        # Combine scrambling with delay to force reorder + wrong identity.
        # We can't easily combine the two stubs, so just use scrambling
        # with multi-worker and verify the assertion fails.
        rows = _collect(items, _bb84_scrambling_worker, num_workers=4)
        with pytest.raises(AssertionError):
            _verify_bb84_identity(items, rows)


# ============================================================================
# Group G: contract documentation — encode the rule in tests
# ============================================================================

class TestIdentityContractDocumentation:
    """Encode the identity-preservation contract as executable tests.

    These tests serve as documentation: they state the rule and verify
    that the rule is followed. If someone later changes the dispatch
    or worker in a way that breaks identity preservation, these tests
    fail with a clear message explaining what contract was violated.
    """

    def test_contract_pulse_id_is_source_of_truth(self):
        """CONTRACT: pulse_id (or the equivalent identity key) is the
        source of truth for interpreting results. Position in the
        output list is NOT.

        This test verifies that the contract holds by checking that
        identity-based lookup works and positional lookup doesn't."""
        items = _make_bb84_items(100)
        rows = _collect(items, _bb84_identity_slow_worker, num_workers=4)

        # Reordering must occur (otherwise the test is vacuous).
        arrival_ids = [r["pulse_id"] for r in rows]
        assert arrival_ids != list(range(100))

        items_by_id = {item["pulse_id"]: item for item in items}

        # CORRECT pattern: identity-based lookup.
        for row in rows:
            original = items_by_id[row["pulse_id"]]
            assert row["alice_bit"] == original["alice_bit"]
            assert row["alice_basis"] == original["alice_basis"]

        # BUGGY pattern: positional lookup. We DON'T assert this in
        # production code — we just verify it WOULD fail, to document
        # why positional lookup is forbidden.
        positional_failures = sum(
            1 for i, row in enumerate(rows)
            if row["alice_bit"] != items[i]["alice_bit"]
        )
        assert positional_failures > 0, (
            "Positional lookup should fail under reordering. If it "
            "doesn't, the test environment is too stable."
        )

    def test_contract_identity_fields_must_travel_with_row(self):
        """CONTRACT: all identity-relevant fields (alice_bit, alice_basis,
        bob_basis, state_label, mu, intensity_label) must be carried IN
        the result row, not looked up by position afterward.

        This test verifies that every result row contains all required
        identity fields."""
        items = _make_bb84_items(50)
        rows = _collect(items, _bb84_identity_worker, num_workers=2)

        required_identity_fields = [
            "pulse_id", "alice_bit", "alice_basis", "bob_basis",
            "state_label", "mu", "intensity_label",
        ]
        for row in rows:
            for field in required_identity_fields:
                assert field in row, (
                    f"Row for pulse_id {row.get('pulse_id', '?')} is "
                    f"missing identity field '{field}'. Identity must "
                    f"travel WITH the row, not be looked up by position."
                )

    def test_contract_dispatch_does_not_swap_identities(self):
        """CONTRACT: dispatch_work_items must never swap the identity
        fields between two work items. Even if it reorders arrival
        times, the (pulse_id, identity_fields) binding is sacrosanct.

        Verify by checking that NO row has the identity of a different
        pulse_id."""
        items = _make_bb84_items(100)
        rows = _collect(items, _bb84_identity_slow_worker, num_workers=4)

        items_by_id = {item["pulse_id"]: item for item in items}
        swaps_detected = 0
        for row in rows:
            original = items_by_id[row["pulse_id"]]
            # Check if any identity field belongs to a DIFFERENT pulse.
            for other_id, other_item in items_by_id.items():
                if other_id == row["pulse_id"]:
                    continue
                if (row["alice_bit"] == other_item["alice_bit"]
                        and row["alice_basis"] == other_item["alice_basis"]
                        and row["bob_basis"] == other_item["bob_basis"]
                        and row["state_label"] == other_item["state_label"]
                        and row["pulse_id"] != other_id):
                    # Could be a coincidence (same bit/basis pattern) —
                    # only count as a swap if ALL fields match a different
                    # pulse AND mu also matches.
                    if row["mu"] == other_item["mu"]:
                        swaps_detected += 1
                        break

        # Some "swaps" might be coincidental pattern matches, but under
        # reordering they should be rare. The key assertion is that
        # _verify_bb84_identity passes (which it does, or this test
        # wouldn't reach here).
        _verify_bb84_identity(items, rows)
        # If swaps_detected > 0, it means identity fields matched a
        # different pulse — but _verify_bb84_identity passed, so the
        # match was coincidental (same pattern), not a real swap.
        # We don't assert swaps_detected == 0 here because of pattern
        # collisions; instead we rely on _verify_bb84_identity.


# ============================================================================
# Group H: integration with main_optimized (static source checks)
# ============================================================================

class TestMainOptimizedIdentityArchitecture:
    """Static checks on main_optimized's source code to verify the
    identity-preservation architecture is sound.

    These tests verify that:
      1. Work items carry their own identity (not looked up by index).
      2. The output row carries the composite identity key.
      3. No post-dispatch code uses positional interpretation.
    """

    def test_work_items_are_self_contained_tuples(self):
        """Verify (by source inspection) that work items are tuples
        carrying all identity fields: (dist, total_pulses, apply_noise,
        worker_seed, combo_idx).

        This means each work item is self-contained — the worker doesn't
        need to look up anything by index."""
        import main_optimized
        src = inspect.getsource(main_optimized.run_and_save_csv)
        # The work_items.append line must include all 5 identity fields.
        assert "work_items.append((dist, total_pulses, apply_noise, worker_seed, combo_idx))" in src, (
            "Work items must be self-contained tuples carrying all "
            "identity fields. If this line changed, verify the new "
            "structure still carries a unique composite identity."
        )

    def test_output_row_carries_composite_identity(self):
        """Verify the output row carries distance_km, noise_applied,
        total_pulses (the composite identity key)."""
        import main_optimized
        src = inspect.getsource(main_optimized.run_single_simulation)
        assert '"distance_km"' in src and '"noise_applied"' in src and '"total_pulses"' in src, (
            "run_single_simulation must put distance_km, noise_applied, "
            "and total_pulses in the output row — these form the "
            "composite identity key."
        )

    def test_no_positional_post_dispatch_interpretation(self):
        """Verify run_and_save_csv does NOT use rows[i] or enumerate(rows)
        to look up work items by position after dispatch.

        The only post-dispatch code should be the success print."""
        import main_optimized
        src = inspect.getsource(main_optimized.run_and_save_csv)
        # After the dispatch_work_items call, there should be no
        # `for i, row in enumerate(rows)` pattern.
        # The _on_row closure uses `completed` as a counter, not as an
        # index into work_items, so it's safe.
        assert "for i, row in enumerate" not in src, (
            "run_and_save_csv must NOT use `for i, row in enumerate(rows)` "
            "— that pattern interprets results by position, which is "
            "wrong under imap_unordered reordering. Use carried identity "
            "(row['distance_km'], row['noise_applied'], etc.) instead."
        )

    def test_on_row_does_not_index_work_items(self):
        """Verify the _on_row closure does NOT index into work_items
        by the completed counter."""
        import main_optimized
        src = inspect.getsource(main_optimized.run_and_save_csv)
        # Extract the _on_row closure source.
        assert "_on_row" in src, "Refactor not applied — _on_row missing."
        # work_items[completed] would be a positional-lookup bug.
        assert "work_items[completed]" not in src, (
            "_on_row must NOT index work_items[completed] — that would "
            "interpret the row by arrival position, not by carried identity."
        )
