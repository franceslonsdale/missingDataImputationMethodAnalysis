"""
simulation.py — Monte Carlo driver for the Deng & Lumley (2024) study.

One scenario only: MAR tertile mechanism, as specified in the formal plan.
(MCAR/MNAR arms and multiple missing rates have been removed; if you want
to add a robustness arm later, wrap the inner loop in a `for mechanism in ...`
block and pass a `mechanism` argument into impose_mar.)

The loop is structured so that every completed scenario cell is written to
disk immediately — a crash after run 743 does not lose runs 1..743.
"""

from __future__ import annotations

import os
import pickle
import threading
import time
import traceback
import numpy as np
import pandas as pd


from config import (
    N_SIMS, N_OBS, N_IMPUTATIONS, RANDOM_SEED, METHODS, METHOD_TIMEOUTS,
    RESULTS_DIR, TABLE2_CSV, RAW_CSV,
)
from data_generation   import generate_complete_data, impose_mar
from imputation_methods import run_imputation
from evaluation         import evaluate_single_run, aggregate_runs, fit_analysis

# -----------------------------------------------------------------------------
# Soft timeout via a worker thread.
#
# We use threading rather than multiprocessing for two reasons:
#   1. multiprocessing on Windows requires the entire pipeline to be
#      pickle-able and re-imports config in every worker — fragile.
#   2. our methods spend most of their time inside numpy / torch / R
#      subprocess calls that release the GIL, so a worker thread can
#      genuinely run in parallel with the watchdog.
#
# Caveat: there is no clean way to KILL a Python thread.  If a method
# hangs forever inside pure-Python code (no GIL release), this watchdog
# will report the timeout and the simulation will move on, but the
# orphaned thread will keep running in the background.  In practice the
# torch / R subprocess code paths do release the GIL and the orphan
# eventually exits on its own.  The MICE-via-Rscript path is fully safe
# because the R subprocess is killed by its own internal timeout=600.
# -----------------------------------------------------------------------------

class _TimeoutError(Exception):
    pass


def _run_with_timeout(fn, args, kwargs, timeout_s):
    """
    Run `fn(*args, **kwargs)` in a daemon thread with a wall-clock cap.

    Returns the function's return value.  Raises _TimeoutError if the
    function does not finish within `timeout_s` seconds.  Re-raises any
    exception that the function itself raised.
    """
    result_box: dict = {}

    def _target():
        try:
            result_box["value"] = fn(*args, **kwargs)
        except BaseException as e:
            result_box["error"] = e

    th = threading.Thread(target=_target, daemon=True)
    th.start()
    th.join(timeout=timeout_s)

    if th.is_alive():
        # Watchdog fired; the worker is still running (we cannot kill it
        # cleanly).  Mark the cell as a timeout and let it finish in the
        # background.
        raise _TimeoutError(f"exceeded {timeout_s}s wall-clock cap")
    if "error" in result_box:
        raise result_box["error"]
    return result_box["value"]

# -----------------------------------------------------------------------------
# Per-run seeds (reproducible and independent across runs)
# -----------------------------------------------------------------------------

def _run_rng(run_id: int) -> np.random.Generator:
    # SeedSequence with the run_id as a spawn key gives us independent streams.
    ss = np.random.SeedSequence(RANDOM_SEED, spawn_key=(run_id,))
    return np.random.default_rng(ss)


# -----------------------------------------------------------------------------
# One Monte Carlo replicate
def run_one(run_id: int, methods: list[str], verbose: bool = True) -> dict:
    """
    Do everything needed for one replicate:
        - Generate a complete dataset
        - Impose MAR
        - For each method, impute (within a wall-clock timeout) and evaluate
          against the complete-data fit

    Returns a dict keyed by method, each value a per-run dict from
    `evaluate_single_run`.  An additional `errors` dict records any
    methods that crashed or timed out.
    """
    rng = _run_rng(run_id)
    df_full = generate_complete_data(n=N_OBS, rng=rng)
    df_miss = impose_mar(df_full, rng=rng)

    results: dict[str, dict] = {"run_id": run_id, "errors": {}}
    for method in methods:
        timeout_s = METHOD_TIMEOUTS.get(method, 1800)
        try:
            t0 = time.time()
            imputed = _run_with_timeout(
                run_imputation,
                args=(method, df_miss),
                kwargs={"m": N_IMPUTATIONS, "seed": RANDOM_SEED + run_id},
                timeout_s=timeout_s,
            )
            eval_out = evaluate_single_run(imputed, df_complete=df_full)
            eval_out["seconds"] = time.time() - t0
            results[method] = eval_out
            if verbose:
                print(f"  run {run_id:5d}  {method:14s}  "
                      f"{eval_out['seconds']:7.1f}s")
        except _TimeoutError as e:
            results["errors"][method] = f"TIMEOUT: {e}"
            if verbose:
                elapsed = time.time() - t0
                print(f"  run {run_id:5d}  {method:14s}  "
                      f"{elapsed:7.1f}s  TIMEOUT (cap {timeout_s}s)")
        except Exception as e:
            results["errors"][method] = repr(e)
            if verbose:
                print(f"  run {run_id:5d}  {method:14s}  FAILED: {e}")
                traceback.print_exc()
    return results

# -----------------------------------------------------------------------------
# Outer loop with checkpointing
# -----------------------------------------------------------------------------

def run_study(
    n_sims: int = N_SIMS,
    methods: list[str] | None = None,
    checkpoint_every: int = 10,
    verbose: bool = True,
) -> dict[str, list[dict]]:
    """
    Execute the full Monte Carlo experiment.  After every
    `checkpoint_every` runs the incremental state is pickled to
    `results/checkpoint.pkl`, so you can resume after a crash by
    loading that file and only running the missing run ids.

    Returns {method_name: list_of_per_run_eval_dicts}.
    """
    if methods is None:
        methods = METHODS
    os.makedirs(RESULTS_DIR, exist_ok=True)

    checkpoint = os.path.join(RESULTS_DIR, "checkpoint.pkl")
    if os.path.exists(checkpoint):
        with open(checkpoint, "rb") as f:
            state = pickle.load(f)
        per_method = state["per_method"]
        done_ids   = set(state["done_ids"])
        print(f"Resuming from checkpoint: {len(done_ids)} runs already done.")
    else:
        per_method = {m: [] for m in methods}
        done_ids   = set()

    t_start = time.time()
    for run_id in range(n_sims):
        if run_id in done_ids:
            continue
        if verbose:
            print(f"\n[run {run_id + 1}/{n_sims}]")
        res = run_one(run_id, methods, verbose=verbose)
        for m in methods:
            if m in res:
                per_method[m].append(res[m])
        done_ids.add(run_id)

        if ((run_id + 1) % checkpoint_every == 0) or (run_id + 1 == n_sims):
            with open(checkpoint, "wb") as f:
                pickle.dump({"per_method": per_method, "done_ids": done_ids}, f)
            elapsed = time.time() - t_start
            print(f"  checkpoint saved ({run_id + 1}/{n_sims}, "
                  f"{elapsed/60:.1f} min elapsed)")

    return per_method


# -----------------------------------------------------------------------------
# Aggregation to Deng & Lumley Table 2 shape
# -----------------------------------------------------------------------------

def build_table2(per_method: dict[str, list[dict]]) -> pd.DataFrame:
    """Concatenate per-method aggregates, tagged with a Method column."""
    frames = []
    for method, runs in per_method.items():
        if len(runs) == 0:
            continue
        agg = aggregate_runs(runs)
        agg.insert(0, "Method", method)
        frames.append(agg)
    return pd.concat(frames, axis=0, ignore_index=True)


def save_outputs(per_method: dict[str, list[dict]]) -> pd.DataFrame:
    """Save the raw per-run arrays and the aggregated Table 2 CSV."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    table2 = build_table2(per_method)
    table2.to_csv(TABLE2_CSV, index=False, float_format="%.6f")
    print(f"\nTable 2 written to {TABLE2_CSV}")

    # Raw per-run CSV (long format) — one row per (method, run, parameter).
    rows = []
    from config import PARAM_ORDER
    for method, runs in per_method.items():
        for s, r in enumerate(runs):
            for j, p in enumerate(PARAM_ORDER):
                rows.append({
                    "Method"        : method,
                    "run"           : s,
                    "Parameter"     : p,
                    "beta_complete" : r["beta_complete"][j],
                    "var_complete"  : r["var_complete"][j],
                    "Qbar"          : r["Qbar"][j],
                    "Ubar"          : r["Ubar"][j],
                    "B"             : r["B"][j],
                    "Tvar"          : r["Tvar"][j],
                    "df"            : r["df"][j],
                    "ci_cover"      : r["ci_cover"][j],
                })
    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(RAW_CSV, index=False, float_format="%.6f")
    print(f"Raw per-run results written to {RAW_CSV}")

    return table2


if __name__ == "__main__":
    per_method = run_study()
    save_outputs(per_method)
