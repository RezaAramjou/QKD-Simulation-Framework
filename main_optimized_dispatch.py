"""main_optimized_dispatch.py

Testability hook extracted from main_optimized.run_and_save_csv.

Why this module exists
----------------------
``main_optimized.run_and_save_csv`` previously contained two inline dispatch
loops (a sequential ``for`` loop for ``num_workers == 1`` and a
``mp.Pool.imap_unordered`` loop for the multi-worker case). Those loops *are*
the synchronization/ordering contract of the program, but they could not be
exercised in isolation because the rest of ``main_optimized`` imports the
heavy ``qkd.*`` stack (sources, detectors, proofs, channel, protocols, ...).

This module deliberately depends only on the Python standard library so the
concurrency contract can be tested with deterministic stub workers, without
needing the ``qkd`` package installed.

The contract encoded here is identical to the original inline code:

* ``num_workers == 1``  -> sequential, init_fn called once in the calling
  process, ``worker_fn(item)`` called inline.  **Order is preserved.**
* ``num_workers  > 1``  -> ``mp.Pool(initializer=init_fn, initargs=...)``
  with ``imap_unordered``.  **Order is NOT preserved** (by design); the
  ``with`` block acts as a completion barrier so no rows are still in
  flight when ``dispatch_work_items`` returns.

Behavior is preserved bit-for-bit; the only change is that the loop body
calls ``on_row(row)`` instead of ``writer.writerow(row)``, so callers can
plug in any sink (CSV writer, list append, assertion, etc.).
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable, Iterable, Optional, Sequence


def dispatch_work_items(
    work_items: Sequence[Any],
    worker_fn: Callable[[Any], Any],
    num_workers: int,
    init_fn: Optional[Callable[..., None]],
    init_args: Sequence[Any],
    on_row: Callable[[Any], None],
    *,
    chunksize: int = 1,
) -> int:
    """Dispatch ``work_items`` to ``worker_fn`` and stream results to ``on_row``.

    Parameters
    ----------
    work_items
        Ordered iterable of items to process.  Each element is passed verbatim
        to ``worker_fn``.
    worker_fn
        Top-level (picklable) callable.  Receives one work item, returns one
        result row.  Must be importable from the worker process (so no
        lambdas / closures when ``num_workers > 1``).
    num_workers
        If ``1``, dispatch is sequential in the calling process.
        If ``> 1``, dispatch goes through ``multiprocessing.Pool``.
    init_fn
        Pool initializer; called once per worker process.  May be ``None``
        for the single-worker path if no per-process setup is needed.
    init_args
        Positional args passed to ``init_fn``.  Must be picklable.
    on_row
        Called exactly once for each completed row, in *delivery* order
        (which equals input order for ``num_workers == 1`` and is unordered
        for ``num_workers > 1``).
    chunksize
        Forwarded to ``Pool.imap_unordered``.  Ignored when ``num_workers == 1``.

    Returns
    -------
    int
        Number of rows delivered to ``on_row``.  Always equals ``len(work_items)``
        when ``worker_fn`` does not raise.
    """
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    if chunksize < 1:
        raise ValueError(f"chunksize must be >= 1, got {chunksize}")

    total = len(work_items)
    completed = 0

    if num_workers == 1:
        # Single-worker path: init_fn runs in the calling process, then a
        # plain sequential for-loop.  Order is preserved.
        if init_fn is not None:
            init_fn(*init_args)
        for item in work_items:
            row = worker_fn(item)
            on_row(row)
            completed += 1
        return completed

    # Multi-worker path: mp.Pool with imap_unordered.
    # - initializer=init_fn runs once per worker process (per-process state)
    # - imap_unordered explicitly does NOT preserve input order
    # - the `with` block is the completion barrier: __exit__ joins all workers
    init_args_tuple = tuple(init_args) if init_args else ()
    with mp.Pool(
        processes=num_workers,
        initializer=init_fn,
        initargs=init_args_tuple,
    ) as pool:
        for row in pool.imap_unordered(worker_fn, work_items, chunksize=chunksize):
            on_row(row)
            completed += 1

    return completed
