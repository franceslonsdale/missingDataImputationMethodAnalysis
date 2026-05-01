#!/usr/bin/env python3
"""
run_study.py — Entry point for the simulation study.

Usage
-----
    python run_study.py                 # run with config.TEST_MODE or full
    python run_study.py --n-sims 20     # override S
    python run_study.py --methods MICE MGAIN
    python run_study.py --format-only   # just regenerate tables/figures
                                        # from existing CSVs
"""

from __future__ import annotations

import argparse
import sys
import time

import config
from simulation         import run_study, save_outputs
from results_formatting import generate_all as generate_tables
from plotting           import generate_all as generate_figures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sims",  type=int, default=None,
                    help="override number of Monte Carlo runs")
    ap.add_argument("--methods", nargs="+", default=None,
                    choices=["MICE", "GAIN_SI", "MGAIN", "MIDAS"])
    ap.add_argument("--format-only", action="store_true",
                    help="skip the Monte Carlo; just re-render tables & figures")
    args = ap.parse_args()

    if args.format_only:
        generate_tables()
        generate_figures()
        return

    n_sims  = args.n_sims  or config.N_SIMS
    methods = args.methods or config.METHODS

    print("=" * 72)
    print("  Simulation study — Deng & Lumley (2024) MAR tertile scenario")
    print(f"  S  = {n_sims}")
    print(f"  n  = {config.N_OBS}")
    print(f"  m  = {config.N_IMPUTATIONS}")
    print(f"  methods = {', '.join(methods)}")
    print(f"  TEST_MODE = {config.TEST_MODE}")
    print("=" * 72)

    t0 = time.time()
    per_method = run_study(n_sims=n_sims, methods=methods)
    table2 = save_outputs(per_method)
    generate_tables(table2)
    generate_figures(table2_df=table2)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
