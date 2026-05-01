"""
per run compute:
    RMSE_continuous = sqrt( mean_over_missing_cells( (x_imp - x_true)^2 ) )
        averaged across the m imputations, separately for each incomplete
        continuous variable (norm1, norm2, norm3, norm5, norm7).
    Accuracy_categorical = fraction of correctly imputed categories
        averaged across the m imputations, for bin1 and ord1.

Output:
    results/imp_rmse.csv       — per-variable, per-method RMSE/accuracy
    results/imp_rmse.tex       — LaTeX table
    figures/imp_rmse.pdf/.png  — grouped bar chart

to use (+ add number of simulations) = python compImpRmse.py (--n-sims 50)  
"""

from __future__ import annotations

import argparse
import os
import time
import traceback
import numpy as np
import pandas as pd

from config import (
    N_OBS, N_IMPUTATIONS, RANDOM_SEED, METHODS, RESULTS_DIR, FIGURES_DIR,
)
from data_generation   import genCompleteData, imposeMar
from imputation_methods import runImp

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
METHOD_COLOURS = {
    "MICE": "#2166ac", "GAIN_SI": "#b2182b", "MGAIN": "#1b7837",
    "MGAIN_RESAMPLE": "#e08214", "MIDAS": "#7b3294",
}

CONT_VARS = ["norm1", "norm2", "norm3", "norm5", "norm7"]
CAT_VARS  = ["bin1", "ord1"]


def _run_rng(run_id: int) -> np.random.Generator:
    ss = np.random.SeedSequence(RANDOM_SEED + 999999, spawn_key=(run_id,))
    return np.random.default_rng(ss)


def compImpRmse(n_sims: int = 50, methods=None, verbose=True):
    """
    Run `n_sims` replicates and compute per-variable imputation metrics.
    """
    if methods is None:
        methods = METHODS

    # Accumulators: {method: {var: list_of_per_run_metrics}}
    cont_mse  = {m: {v: [] for v in CONT_VARS} for m in methods}
    cat_acc   = {m: {v: [] for v in CAT_VARS}  for m in methods}

    t0 = time.time()
    for s in range(n_sims):
        rng = _run_rng(s)
        df_full = genCompleteData(n=N_OBS, rng=rng)
        df_miss = imposeMar(df_full, rng=rng)

        for method in methods:
            try:
                imputed = runImp(
                    method, df_miss, m=N_IMPUTATIONS,
                    seed=RANDOM_SEED + 999999 + s,
                )

                # Continuous: RMSE averaged across m imputations
                for v in CONT_VARS:
                    na_mask = df_miss[v].isna().to_numpy()
                    if not na_mask.any():
                        continue
                    true_vals = df_full[v].to_numpy()[na_mask]
                    # Average imputed value across m draws
                    imp_avg = np.mean(
                        [d[v].to_numpy()[na_mask] for d in imputed], axis=0
                    )
                    mse = np.mean((imp_avg - true_vals) ** 2)
                    cont_mse[method][v].append(mse)

                # Categorical: accuracy averaged across m imputations
                for v in CAT_VARS:
                    na_mask = df_miss[v].isna().to_numpy()
                    if not na_mask.any():
                        continue
                    true_vals = df_full[v].astype(int).to_numpy()[na_mask]
                    # Majority vote across m draws
                    imp_vals = np.stack(
                        [d[v].astype(int).to_numpy()[na_mask] for d in imputed]
                    )
                    # Per-cell mode across m imputations
                    from scipy.stats import mode as _mode
                    voted, _ = _mode(imp_vals, axis=0, keepdims=False)
                    acc = np.mean(voted == true_vals)
                    cat_acc[method][v].append(acc)

                if verbose:
                    print(f"  sim {s+1:3d}/{n_sims}  {method:14s}  OK")

            except Exception as e:
                if verbose:
                    print(f"  sim {s+1:3d}/{n_sims}  {method:14s}  FAILED: {e}")
                    traceback.print_exc()

    elapsed = time.time() - t0
    if verbose:
        print(f"\nImputation RMSE study complete in {elapsed/60:.1f} min.")

    # ── Aggregate ────────────────────────────────────────────
    rows = []
    for method in methods:
        for v in CONT_VARS:
            vals = cont_mse[method][v]
            if len(vals) == 0:
                continue
            rmse = np.sqrt(np.mean(vals))
            rows.append({
                "Method": method, "Variable": v, "Type": "continuous",
                "RMSE": rmse, "Accuracy": np.nan, "n_valid": len(vals),
            })
        for v in CAT_VARS:
            vals = cat_acc[method][v]
            if len(vals) == 0:
                continue
            rows.append({
                "Method": method, "Variable": v, "Type": "categorical",
                "RMSE": np.nan, "Accuracy": np.mean(vals),
                "n_valid": len(vals),
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────

def saveRmseOutputs(df: pd.DataFrame) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "imp_rmse.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Wrote {csv_path}")

    # ── Console table ────────────────────────────────────────
    print("\nImputation-level metrics:")
    print(f"{'Variable':12s}  {'Method':14s}  {'RMSE':>8s}  {'Accuracy':>8s}")
    print("-" * 50)
    for v in CONT_VARS + CAT_VARS:
        sub = df[df["Variable"] == v]
        for method in METHODS:
            row = sub[sub["Method"] == method]
            if row.empty:
                continue
            r = row.iloc[0]
            rmse_s = f"{r['RMSE']:.4f}" if pd.notna(r["RMSE"]) else "—"
            acc_s  = f"{r['Accuracy']:.3f}" if pd.notna(r["Accuracy"]) else "—"
            print(f"{v:12s}  {METHOD_LABELS[method]:14s}  {rmse_s:>8s}  {acc_s:>8s}")
        print()

    # ── LaTeX ────────────────────────────────────────────────
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Imputation-level error: RMSE for continuous variables "
        r"(averaged over $m$ imputations) and classification accuracy for "
        r"categorical variables (majority vote across $m$ draws).}",
        r"\label{tab:imp_rmse}",
        r"\begin{tabular}{ll" + "c" * len(METHODS) + "}",
        r"\toprule",
        "Type & Variable & " + " & ".join(
            METHOD_LABELS[m].replace("_", r"\_") for m in METHODS
        ) + r" \\",
        r"\midrule",
    ]
    for vtype, vlist, metric in [
        ("Continuous", CONT_VARS, "RMSE"),
        ("Categorical", CAT_VARS, "Accuracy"),
    ]:
        for i, v in enumerate(vlist):
            sub = df[df["Variable"] == v]
            label = vtype if i == 0 else ""
            cells = []
            for method in METHODS:
                row = sub[sub["Method"] == method]
                if row.empty:
                    cells.append("—")
                else:
                    val = row.iloc[0][metric]
                    cells.append(f"{val:.4f}" if pd.notna(val) else "—")
            lines.append(
                f"{label} & {v} & " + " & ".join(cells) + r" \\"
            )
        lines.append(r"\addlinespace")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex_path = os.path.join(RESULTS_DIR, "imp_rmse.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {tex_path}")

    # Create figure — grouped bar chart 
    fig, axes = plt.subplots(1, 2, figsize=(11, 4),
                             gridspec_kw={"width_ratios": [5, 2]})

    # RMSE for continuous variables
    ax = axes[0]
    cont_df = df[df["Type"] == "continuous"]
    n_v = len(CONT_VARS)
    n_m = len(METHODS)
    width = 0.75 / n_m
    xbase = np.arange(n_v)
    for j, method in enumerate(METHODS):
        sub = cont_df[cont_df["Method"] == method].set_index("Variable").reindex(CONT_VARS)
        vals = sub["RMSE"].fillna(0).values
        ax.bar(
            xbase + (j - (n_m - 1) / 2) * width,
            vals, width=width,
            color=METHOD_COLOURS[method],
            label=METHOD_LABELS[method],
            edgecolor="white", linewidth=0.3,
        )
    ax.set_xticks(xbase)
    ax.set_xticklabels(CONT_VARS, fontsize=8)
    ax.set_ylabel("RMSE")
    ax.set_title("(a) Continuous variables — imputation RMSE",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, ncol=n_m, frameon=False, loc="upper right")
    ax.grid(axis="y", lw=0.2, alpha=0.4)

    # (b) Accuracy for categorical variables
    ax = axes[1]
    cat_df = df[df["Type"] == "categorical"]
    n_v2 = len(CAT_VARS)
    xbase2 = np.arange(n_v2)
    for j, method in enumerate(METHODS):
        sub = cat_df[cat_df["Method"] == method].set_index("Variable").reindex(CAT_VARS)
        vals = sub["Accuracy"].fillna(0).values
        ax.bar(
            xbase2 + (j - (n_m - 1) / 2) * width,
            vals, width=width,
            color=METHOD_COLOURS[method],
            edgecolor="white", linewidth=0.3,
        )
    ax.set_xticks(xbase2)
    ax.set_xticklabels(CAT_VARS, fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("(b) Categorical — classification accuracy",
                 fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", lw=0.2, alpha=0.4)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES_DIR, f"imp_rmse.{ext}"))
    plt.close(fig)
    print(f"Wrote figures/imp_rmse.pdf and .png")


# cli

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sims", type=int, default=50,
                    help="number of replicates for imputation RMSE (default 50)")
    ap.add_argument("--methods", nargs="+", default=None)
    args = ap.parse_args()

    df = compImpRmse(
        n_sims=args.n_sims,
        methods=args.methods or METHODS,
    )
    saveRmseOutputs(df)
