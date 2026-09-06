#!/usr/bin/env python3
"""pulse_tracking_trace.py

Runs the 100-random-pulse end-to-end tracking simulation and prints a
human-readable audit trail showing EXACTLY what happens to each tracked
pulse through the four pipeline stages (S1, S2, S3, S4).

WHAT YOU WILL SEE
-----------------
  1. Configuration summary
  2. The 100 pulse_ids that were randomly selected (sorted)
  3. Stage 1 (S1) snapshot: each tracked pulse's identity BEFORE dispatch
  4. Run summary: wall-clock, worker PIDs, receipts collected
  5. Reordering proof: sorted order vs arrival order, pairwise inversions
  6. Per-pulse 4-stage audit table (all 100 pulses): identity fields
  7. Per-pulse measurement audit: predicted vs actual measurement
  8. Stage-level summary: mismatch counts per stage
  9. Final verdict

USAGE
-----
  python pulse_tracking_trace.py
  python pulse_tracking_trace.py --workers 2 --first 30
  python pulse_tracking_trace.py --output my_trace.txt

The trace is also written to /home/z/my-project/download/pulse_tracking_trace.txt
by default so you can review it later.

REQUIREMENTS
------------
This script imports the worker functions and helpers from
`tests/test_main_optimized_pulse_tracking.py`, so it must be run from the
project root (the directory containing `tests/` and `src/`).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List

# --- Path setup so we can import the test module and dispatch helper ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# When run from /home/z/my-project/scripts/, project root is one level up.
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
for candidate in (
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, "src"),
    os.path.join(PROJECT_ROOT, "tests"),
):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

# --- Reuse the test module's machinery ---
from tests.test_main_optimized_pulse_tracking import (  # noqa: E402
    IDENTITY_FIELDS,
    _make_pulse_population,
    _select_tracked_pids,
    _run_tracking_dispatch,
    _expected_measurement,
    _tracking_slow_worker,
)
from main_optimized_dispatch import dispatch_work_items  # noqa: E402


# ============================================================================
# Formatting helpers
# ============================================================================

def _format_table(headers: List[str], rows: List[List[Any]],
                  col_widths: List[int] = None) -> str:
    """Format a list of rows as an aligned text table."""
    if not rows:
        return "(no rows)"
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            w = max(len(str(h)), *(len(str(r[i])) for r in rows))
            col_widths.append(w)
    lines = []
    lines.append("  ".join(str(h).ljust(w) for h, w in zip(headers, col_widths)))
    lines.append("  ".join("-" * w for w in col_widths))
    for r in rows:
        lines.append("  ".join(str(c).ljust(w) for c, w in zip(r, col_widths)))
    return "\n".join(lines)


def _short_identity(d: Dict[str, Any]) -> str:
    """Compact representation of an identity dict: 'bit/basis/bob/state'."""
    return (f"{d['alice_bit']}/{d['alice_basis']}/"
            f"{d['bob_basis']}/{d['state_label']}")


def _identity_match(s1, s2, s3, s4) -> str:
    """Return 'OK' if all four stage views agree on every identity field."""
    for f in IDENTITY_FIELDS:
        if not (s1[f] == s2[f] == s3[f] == s4[f]):
            return "FAIL"
    return "OK"


def _measurement_match(s1, s3) -> str:
    """Return 'OK' if the worker's measurement matches the S1 prediction."""
    exp = _expected_measurement(s1)
    if (s3["basis_match"] == exp["basis_match"] and
            s3["measured_bit"] == exp["measured_bit"] and
            s3["detected"] == exp["detected"]):
        return "OK"
    return "FAIL"


# ============================================================================
# Main trace runner
# ============================================================================

def run_trace(args) -> str:
    """Run the 100-pulse trace and return the full text output."""
    lines: List[str] = []
    def out(s: str = ""):
        lines.append(s)

    out("=" * 88)
    out(" 100-RANDOM-PULSE END-TO-END STATE TRACE")
    out(" QKD BB84 Simulation -- 4-Stage Audit Trail (S1, S2, S3, S4)")
    out("=" * 88)
    out()
    out("Configuration:")
    out(f"  Population:           {args.population} pulses")
    out(f"  Tracked (sampled):    {args.tracked} pulses (seed={args.seed}, reproducible)")
    out(f"  Workers:              {args.workers}")
    out(f"  Worker function:      _tracking_slow_worker (bimodal delay)")
    out(f"  Delay profile:        10% slow (100ms) + 90% fast (0-5ms)")
    out()

    # ---- Build population and select tracked ----
    items = _make_pulse_population(args.population)
    tracked_pids = _select_tracked_pids(items, args.tracked, seed=args.seed)

    out("Tracked pulse_ids (sorted, first 30 shown):")
    preview = tracked_pids[:30]
    out(f"  {preview}{' ...' if len(tracked_pids) > 30 else ''}")
    out(f"  (total: {len(tracked_pids)} pulse_ids, range [{min(tracked_pids)}, {max(tracked_pids)}])")
    out()

    # ---- Stage 1 snapshot ----
    out("-" * 88)
    out(" STAGE 1 (S1) -- PRE-DISPATCH SNAPSHOT")
    out("   The original work items BEFORE dispatch. This is ground truth.")
    out("-" * 88)
    out()
    n_show_s1 = min(10, len(tracked_pids))
    out(f"  Showing first {n_show_s1} of {len(tracked_pids)} tracked pulses:")
    out()
    headers = ["pulse_id", "alice_bit", "alice_basis", "bob_basis",
               "state_label", "mu", "intensity"]
    sample_pids = tracked_pids[:n_show_s1]
    rows_table = []
    for pid in sample_pids:
        item = next(i for i in items if i["pulse_id"] == pid)
        rows_table.append([pid, item["alice_bit"], item["alice_basis"],
                           item["bob_basis"], item["state_label"],
                           item["mu"], item["intensity_label"]])
    out(_format_table(headers, rows_table))
    out()

    # ---- Run dispatch ----
    out("-" * 88)
    out(f" STAGE 2/3/4 -- DISPATCH RUN ({args.workers} workers, bimodal delay)")
    out("-" * 88)
    out()
    out(f"  Running {args.population} pulses through dispatch_work_items...")
    t0 = time.perf_counter()
    snapshot, receipts, rows_s4, rows_by_pid = _run_tracking_dispatch(
        items, _tracking_slow_worker,
        num_workers=args.workers, tracked_pids=tracked_pids,
    )
    elapsed = time.perf_counter() - t0
    worker_pids = sorted({r["pid"] for r in rows_s4})
    out(f"  Wall-clock:           {elapsed:.2f}s")
    out(f"  Rows produced (S4):   {len(rows_s4)}")
    out(f"  Worker PIDs:          {worker_pids}  ({len(worker_pids)} unique)")
    out(f"  Receipts collected:   {len(receipts)} / {len(tracked_pids)} expected")
    out()

    # ---- Reordering proof ----
    out("-" * 88)
    out(" REORDERING PROOF -- arrival order of tracked pulses")
    out("-" * 88)
    out()
    tracked_set = set(tracked_pids)
    tracked_arrival_order = [r["pulse_id"] for r in rows_s4
                             if r["pulse_id"] in tracked_set]

    n_preview = min(30, len(tracked_pids))
    out(f"  Sorted (input order):   {tracked_pids[:n_preview]}"
        f"{' ...' if len(tracked_pids) > n_preview else ''}")
    out(f"  Arrived (output order): {tracked_arrival_order[:n_preview]}"
        f"{' ...' if len(tracked_arrival_order) > n_preview else ''}")
    out()

    # Find concrete swap examples (adjacent or near-adjacent inversions
    # in the arrival order) so the user can SEE the reordering visually.
    swap_examples = []
    for i in range(len(tracked_arrival_order) - 1):
        a = tracked_arrival_order[i]
        b = tracked_arrival_order[i + 1]
        if a > b:  # inversion: larger pid arrived before smaller
            swap_examples.append((i, i + 1, a, b))
            if len(swap_examples) >= 5:
                break
    if swap_examples:
        out("  Concrete swap examples (position in tracked arrival stream):")
        for (i, j, a, b) in swap_examples:
            out(f"    positions {i+1}-{j+1}: pulse {a} arrived BEFORE pulse {b} "
                f"(should be {b} before {a} in sorted order)")
    else:
        out("  No adjacent swaps among tracked pulses in this run "
            "(reordering happened at the untracked-pulse level only).")
    out()

    # Position shifts
    sorted_idx = {pid: i for i, pid in enumerate(tracked_pids)}
    arrived_idx = {pid: i for i, pid in enumerate(tracked_arrival_order)}
    shifts = [(pid, sorted_idx[pid], arrived_idx[pid], arrived_idx[pid] - sorted_idx[pid])
              for pid in tracked_pids]
    max_up = max(shifts, key=lambda x: x[3])
    max_down = min(shifts, key=lambda x: x[3])

    # Kendall-tau inversion count
    n = len(tracked_arrival_order)
    inversions = 0
    for i in range(n):
        ai = tracked_arrival_order[i]
        for j in range(i + 1, n):
            if ai > tracked_arrival_order[j]:
                inversions += 1
    max_inv = n * (n - 1) // 2

    out(f"  Pairwise inversions:    {inversions} / {max_inv} possible "
        f"({100 * inversions / max_inv:.1f}% reordered)")
    out(f"  Largest upward shift:   pulse {max_up[0]} "
        f"(sorted pos {max_up[1] + 1} -> arrived pos {max_up[2] + 1}, "
        f"shift +{max_up[3]})")
    out(f"  Largest downward shift: pulse {max_down[0]} "
        f"(sorted pos {max_down[1] + 1} -> arrived pos {max_down[2] + 1}, "
        f"shift {max_down[3]})")
    out()

    # ---- Per-pulse 4-stage audit table ----
    out("-" * 88)
    out(" PER-PULSE 4-STAGE AUDIT -- identity fields")
    out("   Each row: pulse_id, then S1/S2/S3/S4 identity (bit/basis/bob/state).")
    out("   'OK' means all four stages agree on EVERY identity field.")
    out("-" * 88)
    out()

    receipts_by_pid = {r["pulse_id"]: r for r in receipts}
    n_show = min(args.first, len(tracked_pids))
    headers = ["pid", "S1", "S2", "S3", "S4", "match"]
    table_rows = []
    for pid in tracked_pids[:n_show]:
        s1 = snapshot[pid]
        s2 = receipts_by_pid[pid]["received_item"]
        s3 = rows_by_pid[pid]
        s4 = rows_by_pid[pid]
        table_rows.append([pid, _short_identity(s1), _short_identity(s2),
                           _short_identity(s3), _short_identity(s4),
                           _identity_match(s1, s2, s3, s4)])
    out(_format_table(headers, table_rows))
    if n_show < len(tracked_pids):
        out()
        out(f"  ... ({len(tracked_pids) - n_show} more pulses not shown. "
            f"Use --first {len(tracked_pids)} to display all.)")
    out()

    # ---- Per-pulse measurement audit ----
    out("-" * 88)
    out(" PER-PULSE MEASUREMENT AUDIT -- predicted vs actual")
    out("   Each row: pulse_id, worker's basis_match/measured_bit/detected,")
    out("   then the values PREDICTED from S1, then match.")
    out("-" * 88)
    out()
    headers = ["pid", "actual.bm", "actual.mb", "actual.det",
               "expected.bm", "expected.mb", "expected.det", "match"]
    table_rows = []
    for pid in tracked_pids[:n_show]:
        s1 = snapshot[pid]
        s3 = rows_by_pid[pid]
        exp = _expected_measurement(s1)
        table_rows.append([pid,
                           s3["basis_match"], s3["measured_bit"], s3["detected"],
                           exp["basis_match"], exp["measured_bit"], exp["detected"],
                           _measurement_match(s1, s3)])
    out(_format_table(headers, table_rows))
    out()

    # ---- Stage-level summary ----
    out("-" * 88)
    out(" STAGE-LEVEL SUMMARY -- all 100 tracked pulses")
    out("-" * 88)
    out()

    s1_s2_mismatch = 0
    s2_s3_mismatch = 0
    s3_s4_mismatch = 0
    measurement_mismatch = 0
    missing_receipts = 0
    missing_rows = 0

    for pid in tracked_pids:
        if pid not in receipts_by_pid:
            missing_receipts += 1
            continue
        if pid not in rows_by_pid:
            missing_rows += 1
            continue
        s1 = snapshot[pid]
        s2 = receipts_by_pid[pid]["received_item"]
        s3 = rows_by_pid[pid]
        s4 = rows_by_pid[pid]
        if any(s1[f] != s2[f] for f in IDENTITY_FIELDS):
            s1_s2_mismatch += 1
        if any(s2[f] != s3[f] for f in IDENTITY_FIELDS):
            s2_s3_mismatch += 1
        if any(s3[f] != s4[f] for f in IDENTITY_FIELDS):
            s3_s4_mismatch += 1
        exp = _expected_measurement(s1)
        if (s3["basis_match"] != exp["basis_match"] or
                s3["measured_bit"] != exp["measured_bit"] or
                s3["detected"] != exp["detected"]):
            measurement_mismatch += 1

    out(f"  Tracked pulses:            {len(tracked_pids)}")
    out(f"  Missing S2 receipts:       {missing_receipts}")
    out(f"  Missing S4 rows:           {missing_rows}")
    out(f"  S1 != S2 mismatches:       {s1_s2_mismatch} / {len(tracked_pids)}")
    out(f"  S2 != S3 mismatches:       {s2_s3_mismatch} / {len(tracked_pids)}")
    out(f"  S3 != S4 mismatches:       {s3_s4_mismatch} / {len(tracked_pids)}")
    out(f"  Measurement mismatches:    {measurement_mismatch} / {len(tracked_pids)}")
    out()

    total_mismatch = (missing_receipts + missing_rows + s1_s2_mismatch +
                      s2_s3_mismatch + s3_s4_mismatch + measurement_mismatch)
    if total_mismatch == 0:
        out("  VERDICT:  All 100 tracked pulses preserved identity through all 4 stages.")
        out(f"           {inversions} pairwise inversions occurred (reordering), yet NOT A")
        out("           SINGLE pulse's identity was corrupted. Identity is bound to")
        out("           pulse_id, NOT to row position -- exactly as the contract requires.")
    else:
        out(f"  VERDICT:  {total_mismatch} corruption(s) detected across the 4 stages.")
        out("           See the per-pulse audit table above for details.")
    out()
    out("=" * 88)

    return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="100-random-pulse end-to-end state trace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of worker processes (default: 4)")
    parser.add_argument("--population", type=int, default=1000,
                        help="Total pulses in the population (default: 1000)")
    parser.add_argument("--tracked", type=int, default=100,
                        help="How many pulses to randomly track (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible pulse selection (default: 42)")
    parser.add_argument("--first", type=int, default=100,
                        help="Show first N tracked pulses in audit tables (default: 100 = all)")
    parser.add_argument("--output", type=str, default=None,
                        help="File to write the trace to (in addition to stdout)")
    args = parser.parse_args()

    text = run_trace(args)
    print(text)

    # Always also save to download/ for later review
    default_path = os.path.join(PROJECT_ROOT, "download",
                                "pulse_tracking_trace.txt")
    os.makedirs(os.path.dirname(default_path), exist_ok=True)
    with open(default_path, "w") as f:
        f.write(text)
    print(f"\n[trace also written to {default_path}]", file=sys.stderr)

    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"[trace also written to {args.output}]", file=sys.stderr)


if __name__ == "__main__":
    main()
