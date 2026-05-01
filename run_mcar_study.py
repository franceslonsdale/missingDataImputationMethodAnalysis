"""
run_mcar_study.py — MCAR experiment at 10%, 25%, 50% missingness.

This is the second arm of the simulation study, satisfying the project
spec requirements:
    - MCAR at 10%, 25%, 50% on selected variables
    - Direct imputation error: RMSE (continuous), accuracy/F1 (categorical)
    - Predictive utility: classifier trained on imputed data, test-set metrics

Uses the same DGM (Deng & Lumley eq. 3.1), same 5 methods, same analysis
model, but with MCAR missingness at controlled rates instead of the
tertile MAR mechanism.

Output:
    results/mcar_imp_error.csv       — per-variable imputation error
    results/mcar_predictive.csv      — predictive utility metrics
    results/mcar_imp_error.tex       — LaTeX table
    results/mcar_predictive.tex      — LaTeX table
    figures/mcar_rmse_by_rate.pdf    — RMSE vs miss rate, per method
    figures/mcar_accuracy_by_rate.pdf — categorical accuracy vs miss rate
    figures/mcar_predictive.pdf      — predictive utility comparison

Usage:
    python run_mcar_study.py                    # default S=100
    python run_mcar_study.py --n-sims 50        # quicker
    python run_mcar_study.py --n-sims 200       # more precision
"""

from __future__ import annotations

import argparse
import os
import time
import traceback
import warnings
import numpy as np
import pandas as pd
from scipy.stats import mode as _mode

import matplotlib.pyplot as plt
import matplotlib as mpl

from config import (
    N_OBS, N_IMPUTATIONS, RANDOM_SEED, METHODS,
    RESULTS_DIR, FIGURES_DIR,
)
from data_generation import generate_complete_data
from imputation_methods import run_imputation

# ── Styling ──────────────────────────────────────────────────
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

MISS_RATES = [0.10,0.25,0.50]

# Variables that get missingness (same as the MAR study, minus the ancillaries)
CONT_MISS_VARS = ["norm1", "norm2", "norm3", "norm5", "norm7"]
CAT_MISS_VARS  = ["bin1", "ord1"]
ALL_MISS_VARS  = CONT_MISS_VARS + CAT_MISS_VARS

# All columns in the dataset
ALL_COLS = ["Y", "norm1", "norm2", "norm3", "norm4",
            "norm5", "norm6", "norm7", "norm8", "bin1", "ord1"]


# ─────────────────────────────────────────────────────────────
# MCAR missingness
# ─────────────────────────────────────────────────────────────

def impose_mcar(df: pd.DataFrame, miss_rate: float,
                rng: np.random.Generator) -> pd.DataFrame:
    """
    Impose MCAR missingness at a fixed rate on the 7 incomplete variables.
    Y, norm4, norm6, norm8 remain fully observed (same convention as the
    MAR study, so the imputation models have the same auxiliary info).
    """
    out = df.copy()
    n = len(out)
    for v in ALL_MISS_VARS:
        mask = rng.random(n) < miss_rate
        out.loc[mask, v] = np.nan
    return out


# ─────────────────────────────────────────────────────────────
# Predictive utility — fit a classifier on imputed data
# ─────────────────────────────────────────────────────────────

def _predictive_utility(imputed_datasets: list[pd.DataFrame],
                        df_test: pd.DataFrame) -> dict:
    """
    Fit a Random Forest classifier on each imputed training set to predict
    Y_binary = 1{Y > median(Y)}, then evaluate on a held-out test set.

    Returns averaged (across m imputations) test-set accuracy and F1.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score

    # Create binary outcome from Y
    y_threshold = df_test["Y"].median()

    feature_cols = [c for c in ALL_COLS if c != "Y"]
    accs, f1s = [], []

    for df_imp in imputed_datasets:
        # Prepare features — convert categoricals to int
        X_train = df_imp[feature_cols].copy()
        X_train["bin1"] = X_train["bin1"].astype(int)
        X_train["ord1"] = X_train["ord1"].astype(int)
        y_train = (df_imp["Y"] > y_threshold).astype(int)

        X_test = df_test[feature_cols].copy()
        X_test["bin1"] = X_test["bin1"].astype(int)
        X_test["ord1"] = X_test["ord1"].astype(int)
        y_test = (df_test["Y"] > y_threshold).astype(int)

        clf = RandomForestClassifier(n_estimators=100, random_state=42,
                                     n_jobs=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average="binary"))

    return {"accuracy": np.mean(accs), "f1": np.mean(f1s)}


# ─────────────────────────────────────────────────────────────
# Per-run RNG
# ─────────────────────────────────────────────────────────────

def _run_rng(run_id: int, rate_idx: int) -> np.random.Generator:
    ss = np.random.SeedSequence(RANDOM_SEED + 777777,
                                spawn_key=(run_id, rate_idx))
    return np.random.default_rng(ss)


# ─────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────

def run_mcar_study(n_sims: int = 100, methods=None, verbose=True):
    if methods is None:
        methods = METHODS

    # Accumulators
    imp_rows = []      # imputation-level error
    pred_rows = []     # predictive utility

    total = n_sims * len(MISS_RATES) * len(methods)
    done = 0
    t0 = time.time()

    for s in range(n_sims):
        for ri, rate in enumerate(MISS_RATES):
            rng = _run_rng(s, ri)

            # Generate complete data — use n_train + n_test
            n_train = N_OBS
            n_test  = 2000
            df_all = generate_complete_data(n=n_train + n_test, rng=rng)
            df_train_full = df_all.iloc[:n_train].reset_index(drop=True)
            df_test       = df_all.iloc[n_train:].reset_index(drop=True)

            # Impose MCAR on training data only
            df_miss = impose_mcar(df_train_full, miss_rate=rate, rng=rng)

            for method in methods:
                done += 1
                try:
                    imputed = run_imputation(
                        method, df_miss, m=N_IMPUTATIONS,
                        seed=RANDOM_SEED + 777777 + s * 100 + ri,
                    )

                    # ── Imputation error ─────────────────────────
                    # Continuous: RMSE averaged across m imputations
                    for v in CONT_MISS_VARS:
                        na_mask = df_miss[v].isna().to_numpy()
                        if not na_mask.any():
                            continue
                        true_vals = df_train_full[v].to_numpy()[na_mask]
                        imp_avg = np.mean(
                            [d[v].to_numpy()[na_mask] for d in imputed],
                            axis=0,
                        )
                        rmse = np.sqrt(np.mean((imp_avg - true_vals) ** 2))
                        imp_rows.append({
                            "run": s, "miss_rate": rate, "method": method,
                            "variable": v, "type": "continuous",
                            "rmse": rmse, "accuracy": np.nan,
                        })

                    # Categorical: accuracy (majority vote across m)
                    for v in CAT_MISS_VARS:
                        na_mask = df_miss[v].isna().to_numpy()
                        if not na_mask.any():
                            continue
                        true_vals = df_train_full[v].astype(int).to_numpy()[na_mask]
                        imp_vals = np.stack(
                            [d[v].astype(int).to_numpy()[na_mask] for d in imputed]
                        )
                        voted, _ = _mode(imp_vals, axis=0, keepdims=False)
                        acc = np.mean(voted == true_vals)
                        imp_rows.append({
                            "run": s, "miss_rate": rate, "method": method,
                            "variable": v, "type": "categorical",
                            "rmse": np.nan, "accuracy": acc,
                        })

                    # ── Predictive utility ───────────────────────
                    pu = _predictive_utility(imputed, df_test)
                    pred_rows.append({
                        "run": s, "miss_rate": rate, "method": method,
                        "accuracy": pu["accuracy"], "f1": pu["f1"],
                    })

                    if verbose and done % max(1, total // 20) == 0:
                        pct = 100 * done / total
                        elapsed = (time.time() - t0) / 60
                        print(f"  [{done}/{total}] {pct:.0f}%  "
                              f"{method:14s}  MCAR {int(rate*100)}%  "
                              f"sim {s+1}  ({elapsed:.0f} min)")

                except Exception as e:
                    if verbose:
                        print(f"  FAILED: {method} MCAR {int(rate*100)}% "
                              f"sim {s}: {e}")
                        traceback.print_exc()

    imp_df  = pd.DataFrame(imp_rows)
    pred_df = pd.DataFrame(pred_rows)
    return imp_df, pred_df


# ─────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────

def aggregate_mcar(imp_df: pd.DataFrame, pred_df: pd.DataFrame):
    """Aggregate across runs → one row per (miss_rate, method, variable)."""
    # Imputation error
    imp_agg = (
        imp_df.groupby(["miss_rate", "method", "variable", "type"])
        .agg(rmse=("rmse", "mean"), accuracy=("accuracy", "mean"),
             n_runs=("run", "count"))
        .reset_index()
    )

    # Predictive utility
    pred_agg = (
        pred_df.groupby(["miss_rate", "method"])
        .agg(accuracy=("accuracy", "mean"), f1=("f1", "mean"),
             n_runs=("run", "count"))
        .reset_index()
    )

    return imp_agg, pred_agg


# ─────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────

def save_mcar_outputs(imp_agg, pred_agg):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── CSVs ─────────────────────────────────────────────────
    imp_agg.to_csv(os.path.join(RESULTS_DIR, "mcar_imp_error.csv"),
                   index=False, float_format="%.4f")
    pred_agg.to_csv(os.path.join(RESULTS_DIR, "mcar_predictive.csv"),
                    index=False, float_format="%.4f")
    print(f"Wrote results/mcar_imp_error.csv and mcar_predictive.csv")

    # ── Console tables ───────────────────────────────────────
    print("\n=== Imputation RMSE (continuous) by miss rate & method ===")
    cont = imp_agg[imp_agg["type"] == "continuous"].copy()
    for rate in MISS_RATES:
        print(f"\n  MCAR {int(rate*100)}%:")
        sub = cont[cont["miss_rate"] == rate]
        pivot = sub.pivot(index="variable", columns="method", values="rmse")
        pivot = pivot.reindex(columns=METHODS)
        pivot.columns = [METHOD_LABELS.get(m, m) for m in METHODS]
        print(pivot.round(4).to_string())

    print("\n=== Categorical Accuracy by miss rate & method ===")
    cat = imp_agg[imp_agg["type"] == "categorical"].copy()
    for rate in MISS_RATES:
        print(f"\n  MCAR {int(rate*100)}%:")
        sub = cat[cat["miss_rate"] == rate]
        pivot = sub.pivot(index="variable", columns="method", values="accuracy")
        pivot = pivot.reindex(columns=METHODS)
        pivot.columns = [METHOD_LABELS.get(m, m) for m in METHODS]
        print(pivot.round(4).to_string())

    print("\n=== Predictive Utility (Random Forest) ===")
    for rate in MISS_RATES:
        print(f"\n  MCAR {int(rate*100)}%:")
        sub = pred_agg[pred_agg["miss_rate"] == rate]
        for _, r in sub.iterrows():
            print(f"    {METHOD_LABELS.get(r['method'], r['method']):14s}  "
                  f"Acc={r['accuracy']:.4f}  F1={r['f1']:.4f}")

    # ── LaTeX: Imputation error ──────────────────────────────
    _write_imp_latex(imp_agg)

    # ── LaTeX: Predictive utility ────────────────────────────
    _write_pred_latex(pred_agg)

    # ── Figures ──────────────────────────────────────────────
    _plot_rmse_by_rate(imp_agg)
    _plot_cat_accuracy(imp_agg)
    _plot_predictive(pred_agg)


def _write_imp_latex(imp_agg):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\caption{Imputation error under MCAR at 10\%, 25\%, and 50\% "
        r"missingness. RMSE for continuous variables; classification "
        r"accuracy for categorical variables (majority vote across $m=5$ "
        r"imputations).}",
        r"\label{tab:mcar_imp}",
        r"\begin{tabular}{llc" + "c" * len(METHODS) + "}",
        r"\toprule",
        r"Rate & Variable & Metric & " + " & ".join(
            METHOD_LABELS[m] for m in METHODS) + r" \\",
        r"\midrule",
    ]
    for rate in MISS_RATES:
        rate_label = f"{int(rate*100)}\\%"
        first_rate = True
        # Continuous
        cont = imp_agg[(imp_agg["miss_rate"] == rate) &
                       (imp_agg["type"] == "continuous")]
        for v in CONT_MISS_VARS:
            sub = cont[cont["variable"] == v]
            cells = []
            for m in METHODS:
                row = sub[sub["method"] == m]
                cells.append(f"{row['rmse'].values[0]:.4f}" if len(row) > 0 else "—")
            rl = rate_label if first_rate else ""
            first_rate = False
            lines.append(f"{rl} & {v} & RMSE & " + " & ".join(cells) + r" \\")
        # Categorical
        cat = imp_agg[(imp_agg["miss_rate"] == rate) &
                      (imp_agg["type"] == "categorical")]
        for v in CAT_MISS_VARS:
            sub = cat[cat["variable"] == v]
            cells = []
            for m in METHODS:
                row = sub[sub["method"] == m]
                cells.append(f"{row['accuracy'].values[0]:.3f}" if len(row) > 0 else "—")
            lines.append(f" & {v} & Acc & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path = os.path.join(RESULTS_DIR, "mcar_imp_error.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def _write_pred_latex(pred_agg):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Predictive utility: Random Forest classification "
        r"(predicting $Y > \mathrm{median}(Y)$) trained on imputed data "
        r"and evaluated on a held-out test set ($n_{\mathrm{test}} = 2000$).}",
        r"\label{tab:mcar_pred}",
        r"\begin{tabular}{lcc" + "c" * len(METHODS) + "}",
        r"\toprule",
        r"Rate & Metric & " + " & ".join(
            METHOD_LABELS[m] for m in METHODS) + r" \\",
        r"\midrule",
    ]
    for rate in MISS_RATES:
        rate_label = f"{int(rate*100)}\\%"
        sub = pred_agg[pred_agg["miss_rate"] == rate]
        for metric, mname in [("accuracy", "Acc"), ("f1", "F1")]:
            cells = []
            for m in METHODS:
                row = sub[sub["method"] == m]
                cells.append(f"{row[metric].values[0]:.4f}" if len(row) > 0 else "—")
            rl = rate_label if metric == "accuracy" else ""
            lines.append(f"{rl} & {mname} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path = os.path.join(RESULTS_DIR, "mcar_predictive.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")


# ─────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────

def _plot_rmse_by_rate(imp_agg):
    """RMSE vs miss rate, one panel per continuous variable."""
    cont = imp_agg[imp_agg["type"] == "continuous"]
    n_v = len(CONT_MISS_VARS)
    fig, axes = plt.subplots(1, n_v, figsize=(3.0 * n_v, 3.2), squeeze=False)

    for i, v in enumerate(CONT_MISS_VARS):
        ax = axes[0, i]
        sub = cont[cont["variable"] == v]
        for method in METHODS:
            ms = sub[sub["method"] == method].sort_values("miss_rate")
            if ms.empty:
                continue
            ax.plot(ms["miss_rate"] * 100, ms["rmse"],
                    "o-", color=METHOD_COLOURS[method],
                    label=METHOD_LABELS[method],
                    markersize=5, linewidth=1.5)
        ax.set_xlabel("Missing rate (%)")
        ax.set_ylabel("RMSE")
        ax.set_title(v, fontsize=9, fontweight="bold")
        ax.set_xticks([10, 25, 50])
        ax.grid(lw=0.2, alpha=0.4)

    axes[0, -1].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Imputation RMSE by missingness rate (MCAR)",
                 fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES_DIR, f"mcar_rmse_by_rate.{ext}"))
    plt.close(fig)
    print("Wrote figures/mcar_rmse_by_rate.pdf/.png")


def _plot_cat_accuracy(imp_agg):
    """Categorical accuracy vs miss rate."""
    cat = imp_agg[imp_agg["type"] == "categorical"]
    n_v = len(CAT_MISS_VARS)
    fig, axes = plt.subplots(1, n_v, figsize=(3.5 * n_v, 3.2), squeeze=False)

    for i, v in enumerate(CAT_MISS_VARS):
        ax = axes[0, i]
        sub = cat[cat["variable"] == v]
        for method in METHODS:
            ms = sub[sub["method"] == method].sort_values("miss_rate")
            if ms.empty:
                continue
            ax.plot(ms["miss_rate"] * 100, ms["accuracy"],
                    "o-", color=METHOD_COLOURS[method],
                    label=METHOD_LABELS[method],
                    markersize=5, linewidth=1.5)
        ax.set_xlabel("Missing rate (%)")
        ax.set_ylabel("Accuracy")
        ax.set_title(v, fontsize=9, fontweight="bold")
        ax.set_xticks([10, 25, 50])
        ax.set_ylim(0, 1.05)
        ax.grid(lw=0.2, alpha=0.4)

    axes[0, -1].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("Categorical imputation accuracy by missingness rate (MCAR)",
                 fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES_DIR, f"mcar_accuracy_by_rate.{ext}"))
    plt.close(fig)
    print("Wrote figures/mcar_accuracy_by_rate.pdf/.png")


def _plot_predictive(pred_agg):
    """Predictive utility: grouped bar chart, accuracy + F1."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    n_m = len(METHODS)
    width = 0.75 / n_m

    for ax, metric, title in zip(
        axes, ("accuracy", "f1"),
        ("Test-set accuracy", "Test-set F1"),
    ):
        xbase = np.arange(len(MISS_RATES))
        for j, method in enumerate(METHODS):
            sub = pred_agg[pred_agg["method"] == method].sort_values("miss_rate")
            vals = sub[metric].values
            ax.bar(
                xbase + (j - (n_m - 1) / 2) * width,
                vals, width=width,
                color=METHOD_COLOURS[method],
                label=METHOD_LABELS[method],
                edgecolor="white", linewidth=0.3,
            )
        ax.set_xticks(xbase)
        ax.set_xticklabels([f"{int(r*100)}%" for r in MISS_RATES])
        ax.set_xlabel("MCAR missing rate")
        ax.set_ylabel(title)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.grid(axis="y", lw=0.2, alpha=0.4)

    axes[0].legend(fontsize=7, ncol=n_m, frameon=False, loc="lower left")
    fig.suptitle(
        "Predictive utility: Random Forest on imputed data (MCAR)",
        fontsize=10, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES_DIR, f"mcar_predictive.{ext}"))
    plt.close(fig)
    print("Wrote figures/mcar_predictive.pdf/.png")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="MCAR experiment: imputation error + predictive utility "
                    "at 10%, 25%, 50% missingness."
    )
    ap.add_argument("--n-sims", type=int, default=100,
                    help="Monte Carlo replicates (default 100)")
    ap.add_argument("--methods", nargs="+", default=None)
    args = ap.parse_args()

    methods = args.methods or METHODS
    n_sims  = args.n_sims

    print("=" * 70)
    print("  MCAR Experiment")
    print(f"  S = {n_sims},  n = {N_OBS},  m = {N_IMPUTATIONS}")
    print(f"  Miss rates: {[f'{int(r*100)}%' for r in MISS_RATES]}")
    print(f"  Methods: {', '.join(methods)}")
    print("=" * 70)

    t0 = time.time()
    imp_df, pred_df = run_mcar_study(n_sims=n_sims, methods=methods)
    imp_agg, pred_agg = aggregate_mcar(imp_df, pred_df)
    save_mcar_outputs(imp_agg, pred_agg)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} minutes.")
