"""
run_mcar_deng.py — D&L Table 2 style evaluation under MCAR at 10%, 25%, 50%.

Produces the full variance decomposition (Bias, Var_W, Var_B, Var_T,
Var targets, CI coverage) for each MCAR rate, matching the format of
the MAR and MNAR tables already in the dissertation.

Intended for the appendix: one Table 2 per missingness rate.

Output:
    results/mcar_deng_10pct.csv / .txt
    results/mcar_deng_25pct.csv / .txt
    results/mcar_deng_50pct.csv / .txt
    results/mcar_deng_all.tex          — all three as LaTeX tables

Usage:
    python run_mcar_deng.py                 # default S=100
    python run_mcar_deng.py --n-sims 50     # quicker
    python run_mcar_deng.py --rates 25 50   # only specific rates
"""

from __future__ import annotations

import argparse
import os
import time
import traceback
import numpy as np
import pandas as pd
from scipy import stats

from config import (
    N_OBS, N_IMPUTATIONS, RANDOM_SEED, METHODS, PARAM_ORDER,
    TRUE_BETA, CONFIDENCE_LEVEL, RESULTS_DIR, FIGURES_DIR,
)
from data_generation import generate_complete_data
from imputation_methods import run_imputation
from evaluation import fit_analysis, rubin_pool, aggregate_runs

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

METHOD_LABELS = {
    "MICE": "MICE", "GAIN_SI": "GAIN(SI)", "MGAIN": "MGAIN",
    "MGAIN_RESAMPLE": "MGAIN-R", "MIDAS": "MIDAS",
}

PARAM_SHORT = {
    "Intercept": "Intercept", "C(bin1)[T.1]": "bin1",
    "C(ord1)[T.1]": "ord1=1", "C(ord1)[T.2]": "ord1=2",
    "norm1": "norm1", "norm2": "norm2", "norm3": "norm3",
    "norm5": "norm5", "norm7": "norm7",
    "I(norm1 ** 2)": r"norm1$^2$",
    "norm2:norm3": r"norm2$\times$norm3",
    "norm5:C(bin1)[T.1]": r"norm5$\times$bin1",
    "norm7:C(ord1)[T.1]": r"norm7$\times$ord1=1",
    "norm7:C(ord1)[T.2]": r"norm7$\times$ord1=2",
}

TRUE_BETA_VEC = np.array([TRUE_BETA[p] for p in PARAM_ORDER], dtype=float)

ALL_MISS_VARS = ["norm1", "norm2", "norm3", "norm5", "norm7", "bin1", "ord1"]


# ─────────────────────────────────────────────────────────────
# MCAR missingness
# ─────────────────────────────────────────────────────────────

def impose_mcar(df: pd.DataFrame, miss_rate: float,
                rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    for v in ALL_MISS_VARS:
        mask = rng.random(n) < miss_rate
        out.loc[mask, v] = np.nan
    return out


# ─────────────────────────────────────────────────────────────
# Per-run RNG
# ─────────────────────────────────────────────────────────────

def _run_rng(run_id: int, rate_idx: int) -> np.random.Generator:
    ss = np.random.SeedSequence(RANDOM_SEED + 888888,
                                spawn_key=(run_id, rate_idx))
    return np.random.default_rng(ss)


# ─────────────────────────────────────────────────────────────
# Main loop — one rate at a time
# ─────────────────────────────────────────────────────────────

def run_one_rate(miss_rate: float, n_sims: int, methods, verbose=True):
    """Run the D&L evaluation for a single MCAR rate. Returns per_method dict."""
    per_method: dict[str, list[dict]] = {m: [] for m in methods}
    rate_idx = int(miss_rate * 100)

    t0 = time.time()
    for s in range(n_sims):
        rng = _run_rng(s, rate_idx)
        df_full = generate_complete_data(n=N_OBS, rng=rng)
        df_miss = impose_mcar(df_full, miss_rate=miss_rate, rng=rng)

        beta_complete, var_complete = fit_analysis(df_full)

        for method in methods:
            try:
                t_m = time.time()
                imputed = run_imputation(
                    method, df_miss, m=N_IMPUTATIONS,
                    seed=RANDOM_SEED + 888888 + s * 100 + rate_idx,
                )

                ests  = np.stack([fit_analysis(d)[0] for d in imputed])
                varis = np.stack([fit_analysis(d)[1] for d in imputed])
                pooled = rubin_pool(ests, varis)

                alpha = 1 - CONFIDENCE_LEVEL
                tcrit = stats.t.ppf(1 - alpha / 2, pooled["df"])
                se = np.sqrt(np.maximum(pooled["Tvar"], 0))
                lo = pooled["Qbar"] - tcrit * se
                hi = pooled["Qbar"] + tcrit * se
                cover = ((TRUE_BETA_VEC >= lo) & (TRUE_BETA_VEC <= hi)).astype(float)

                eval_out = {
                    "beta_complete": beta_complete,
                    "var_complete":  var_complete,
                    "Qbar":          pooled["Qbar"],
                    "Ubar":          pooled["Ubar"],
                    "B":             pooled["B"],
                    "Tvar":          pooled["Tvar"],
                    "df":            pooled["df"],
                    "ci_cover":      cover,
                    "m":             len(imputed),
                    "seconds":       time.time() - t_m,
                }
                per_method[method].append(eval_out)

                if verbose and (s + 1) % max(1, n_sims // 10) == 0:
                    elapsed = (time.time() - t0) / 60
                    print(f"  MCAR {rate_idx}%  sim {s+1}/{n_sims}  "
                          f"{method:14s}  ({elapsed:.0f} min)")

            except Exception as e:
                if verbose:
                    print(f"  FAILED: {method} MCAR {rate_idx}% sim {s}: {e}")
                    traceback.print_exc()

    return per_method


# ─────────────────────────────────────────────────────────────
# Build Table 2
# ─────────────────────────────────────────────────────────────

def build_table2(per_method):
    frames = []
    for method, runs in per_method.items():
        if len(runs) == 0:
            continue
        agg = aggregate_runs(runs)
        agg.insert(0, "Method", method)
        frames.append(agg)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


# ─────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────

def _pretty_table(table2: pd.DataFrame, rate_pct: int) -> str:
    scaled = table2.copy()
    for c in ("Var_T", "Var_T_target", "Var_W", "Var_W_target",
              "Var_B", "Var_B_target"):
        scaled[c] = scaled[c] * 1000
    scaled["CI_cov"] = scaled["CI_cov"] * 100

    hdr = (f"{'Parameter':22s}  {'Method':14s}  {'Bias':>8s}  "
           f"{'VarT':>8s}  {'VarT*':>8s}  {'VarW':>8s}  {'VarW*':>8s}  "
           f"{'VarB':>8s}  {'VarB*':>8s}  {'CR%':>6s}  {'Nval':>5s}")
    lines = [f"\nMCAR {rate_pct}% — Table 2", "=" * len(hdr), hdr, "-" * len(hdr)]

    for p in PARAM_ORDER:
        block = scaled[scaled["Parameter"] == p]
        first = True
        for method in METHODS:
            row = block[block["Method"] == method]
            if row.empty:
                continue
            r = row.iloc[0]
            label = p if first else ""
            first = False

            def _f(x, spec=".2f"):
                return f"{x:{spec}}" if pd.notna(x) else "—"

            lines.append(
                f"{label:22s}  {METHOD_LABELS.get(method, method):14s}  "
                f"{_f(r['Bias'], '+.3f'):>8s}  "
                f"{_f(r['Var_T']):>8s}  {_f(r['Var_T_target']):>8s}  "
                f"{_f(r['Var_W']):>8s}  {_f(r['Var_W_target']):>8s}  "
                f"{_f(r['Var_B']):>8s}  {_f(r['Var_B_target']):>8s}  "
                f"{_f(r['CI_cov'], '.1f'):>6s}  "
                f"{int(r['n_valid']):>5d}"
            )
        lines.append("")
    return "\n".join(lines)


def _latex_table(table2: pd.DataFrame, rate_pct: int) -> str:
    scaled = table2.copy()
    for c in ("Var_T", "Var_T_target", "Var_W", "Var_W_target",
              "Var_B", "Var_B_target"):
        scaled[c] = scaled[c] * 1000
    scaled["CI_cov"] = scaled["CI_cov"] * 100

    def _esc(s):
        return s.replace("_", r"\_").replace("**", r"\text{**}")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering\scriptsize",
        f"\\caption{{Simulation results under MCAR at {rate_pct}\\% missingness. "
        r"All variance columns reported $\times 1000$.}}",
        f"\\label{{tab:mcar_deng_{rate_pct}}}",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Parameter & Method & Bias & $\mathrm{Var}_T$ & $\mathrm{Var}_T^{*}$ "
        r"& $\mathrm{Var}_W$ & $\mathrm{Var}_W^{*}$ "
        r"& $\mathrm{Var}_B$ & $\mathrm{Var}_B^{*}$ & CR\% \\",
        r"\midrule",
    ]

    for p in PARAM_ORDER:
        block = scaled[scaled["Parameter"] == p]
        first = True
        for method in METHODS:
            row = block[block["Method"] == method]
            if row.empty:
                continue
            r = row.iloc[0]
            label = _esc(p) if first else ""
            first = False

            def _f(x, spec=".2f"):
                return f"{x:{spec}}" if pd.notna(x) else "---"

            lines.append(
                f"{label} & {_esc(method)} "
                f"& {_f(r['Bias'], '+.3f')} "
                f"& {_f(r['Var_T'])} & {_f(r['Var_T_target'])} "
                f"& {_f(r['Var_W'])} & {_f(r['Var_W_target'])} "
                f"& {_f(r['Var_B'])} & {_f(r['Var_B_target'])} "
                f"& {_f(r['CI_cov'], '.1f')} \\\\"
            )
        lines.append(r"\addlinespace[2pt]")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def save_outputs(results_by_rate: dict[int, pd.DataFrame]):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_latex = []

    for rate_pct, table2 in sorted(results_by_rate.items()):
        if table2.empty:
            continue

        # CSV
        csv_path = os.path.join(RESULTS_DIR, f"mcar_deng_{rate_pct}pct.csv")
        table2.to_csv(csv_path, index=False, float_format="%.6f")

        # Pretty text
        pretty = _pretty_table(table2, rate_pct)
        print(pretty)
        txt_path = os.path.join(RESULTS_DIR, f"mcar_deng_{rate_pct}pct.txt")
        with open(txt_path, "w") as f:
            f.write(pretty + "\n")

        # LaTeX
        all_latex.append(_latex_table(table2, rate_pct))

        print(f"  Wrote results/mcar_deng_{rate_pct}pct.csv and .txt")

    # Combined LaTeX
    tex_path = os.path.join(RESULTS_DIR, "mcar_deng_all.tex")
    with open(tex_path, "w") as f:
        f.write("\n\n\\clearpage\n\n".join(all_latex) + "\n")
    print(f"  Wrote results/mcar_deng_all.tex")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="MCAR D&L Table 2 evaluation at 10/25/50%"
    )
    ap.add_argument("--n-sims", type=int, default=100)
    ap.add_argument("--rates", nargs="+", type=int, default=[10, 25, 50],
                    help="missingness rates as integers (default: 10 25 50)")
    ap.add_argument("--methods", nargs="+", default=None)
    args = ap.parse_args()

    methods = args.methods or METHODS
    rates = [r / 100.0 for r in args.rates]

    print("=" * 70)
    print("  MCAR D&L Table 2 — Appendix tables")
    print(f"  S = {args.n_sims},  n = {N_OBS},  m = {N_IMPUTATIONS}")
    print(f"  Rates: {[f'{int(r*100)}%' for r in rates]}")
    print(f"  Methods: {', '.join(methods)}")
    print("=" * 70)

    t0 = time.time()
    results_by_rate = {}

    for rate in rates:
        rate_pct = int(rate * 100)
        print(f"\n{'='*60}")
        print(f"  Running MCAR {rate_pct}%")
        print(f"{'='*60}")

        per_method = run_one_rate(rate, n_sims=args.n_sims,
                                  methods=methods, verbose=True)
        table2 = build_table2(per_method)
        results_by_rate[rate_pct] = table2

    save_outputs(results_by_rate)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} minutes.")
