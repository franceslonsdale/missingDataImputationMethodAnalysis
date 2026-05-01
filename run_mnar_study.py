"""
run_mnar_study.py — MNAR experiment at 50% missingness.

MNAR (Missing Not At Random): missingness in each variable depends on
the variable's own value.  This is the hardest mechanism — no imputation
method can fully correct for it without modelling the missingness process
itself.  We include it as a sensitivity/robustness arm per the project spec.

Mechanism
---------
For each incomplete variable V, we compute a logistic probability of
missingness:
    P(V missing) = expit(a + b * V)
where b = 1 and a is calibrated via bisection so that the marginal
missingness rate equals the target (50%).  This means extreme values
of V are more likely to be missing — classic self-censoring.

Output structure matches the MAR study (Deng & Lumley Table 2):
    - Bias, Var_T, Var_T_target, Var_W, Var_W_target, Var_B, Var_B_target,
      CI coverage, per coefficient per method.
    - Imputation-level RMSE and categorical accuracy.

Output files:
    results/mnar_table2.csv          — Table 2 style
    results/mnar_table2_pretty.txt   — console version
    results/mnar_table2.tex          — LaTeX version
    results/mnar_imp_error.csv       — imputation-level error
    figures/mnar_coverage_heatmap.pdf — coverage comparison
    figures/mnar_bias_forest.pdf     — bias forest plot
    figures/mnar_imp_rmse.pdf        — imputation RMSE bars

Usage:
    python run_mnar_study.py                # default S=100
    python run_mnar_study.py --n-sims 50    # quicker
    python run_mnar_study.py --n-sims 250   # match the MAR study
"""

from __future__ import annotations

import argparse
import os
import time
import traceback
import warnings
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import mode as _mode

from config import (
    N_OBS, N_IMPUTATIONS, RANDOM_SEED, METHODS, PARAM_ORDER,
    TRUE_BETA, CONFIDENCE_LEVEL, RESULTS_DIR, FIGURES_DIR,
    ANALYSIS_FORMULA,
)
from data_generation import generate_complete_data
from imputation_methods import run_imputation
from evaluation import fit_analysis, rubin_pool

import matplotlib.pyplot as plt
import matplotlib as mpl

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

MISS_RATE = 0.50
CONT_MISS_VARS = ["norm1", "norm2", "norm3", "norm5", "norm7"]
CAT_MISS_VARS  = ["bin1", "ord1"]
ALL_MISS_VARS  = CONT_MISS_VARS + CAT_MISS_VARS


# ─────────────────────────────────────────────────────────────
# MNAR missingness mechanism
# ─────────────────────────────────────────────────────────────

def _calibrate_intercept(values: np.ndarray, target_rate: float,
                         b: float = 1.0) -> float:
    """Bisection to find intercept a such that mean(expit(a + b*v)) ≈ target."""
    a_lo, a_hi = -10.0, 10.0
    for _ in range(100):
        a_mid = (a_lo + a_hi) / 2
        if np.mean(expit(a_mid + b * values)) < target_rate:
            a_lo = a_mid
        else:
            a_hi = a_mid
    return a_lo


def impose_mnar(df: pd.DataFrame, miss_rate: float,
                rng: np.random.Generator) -> pd.DataFrame:
    """
    MNAR: P(V missing) = expit(a + V) where a is calibrated so that
    the marginal rate equals `miss_rate`.  Missingness depends on V itself.
    """
    out = df.copy()
    n = len(out)

    for v in CONT_MISS_VARS:
        vals = out[v].to_numpy(dtype=float)
        a = _calibrate_intercept(vals, miss_rate)
        probs = expit(a + vals)
        mask = rng.random(n) < probs
        out.loc[mask, v] = np.nan

    # For categorical variables, use the latent numeric representation
    # to drive missingness (the variable itself is discrete, so logistic
    # on 0/1 or 0/1/2 is degenerate).  We use the variable's numeric
    # code as the driver.
    for v in CAT_MISS_VARS:
        vals = out[v].astype(float).to_numpy()
        a = _calibrate_intercept(vals, miss_rate)
        probs = expit(a + vals)
        mask = rng.random(n) < probs
        out.loc[mask, v] = np.nan

    return out


# ─────────────────────────────────────────────────────────────
# Per-run RNG
# ─────────────────────────────────────────────────────────────

def _run_rng(run_id: int) -> np.random.Generator:
    ss = np.random.SeedSequence(RANDOM_SEED + 555555, spawn_key=(run_id,))
    return np.random.default_rng(ss)


# ─────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────

def run_mnar_study(n_sims: int = 100, methods=None, verbose=True):
    if methods is None:
        methods = METHODS

    # Accumulators for Table 2
    per_method: dict[str, list[dict]] = {m: [] for m in methods}

    # Accumulators for imputation error
    imp_rows = []

    total = n_sims * len(methods)
    done = 0
    t0 = time.time()

    for s in range(n_sims):
        rng = _run_rng(s)
        df_full = generate_complete_data(n=N_OBS, rng=rng)
        df_miss = impose_mnar(df_full, miss_rate=MISS_RATE, rng=rng)

        # Complete-data fit (for bias & variance targets)
        beta_complete, var_complete = fit_analysis(df_full)

        for method in methods:
            done += 1
            try:
                t_m = time.time()
                imputed = run_imputation(
                    method, df_miss, m=N_IMPUTATIONS,
                    seed=RANDOM_SEED + 555555 + s,
                )

                # ── Rubin pooling for Table 2 ────────────────
                ests  = np.stack([fit_analysis(d)[0] for d in imputed])
                varis = np.stack([fit_analysis(d)[1] for d in imputed])
                pooled = rubin_pool(ests, varis)

                alpha = 1 - CONFIDENCE_LEVEL
                from scipy import stats
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

                # ── Imputation error ─────────────────────────
                for v in CONT_MISS_VARS:
                    na_mask = df_miss[v].isna().to_numpy()
                    if not na_mask.any():
                        continue
                    true_vals = df_full[v].to_numpy()[na_mask]
                    imp_avg = np.mean(
                        [d[v].to_numpy()[na_mask] for d in imputed], axis=0
                    )
                    rmse = np.sqrt(np.mean((imp_avg - true_vals) ** 2))
                    imp_rows.append({
                        "run": s, "method": method, "variable": v,
                        "type": "continuous", "rmse": rmse, "accuracy": np.nan,
                    })

                for v in CAT_MISS_VARS:
                    na_mask = df_miss[v].isna().to_numpy()
                    if not na_mask.any():
                        continue
                    true_vals = df_full[v].astype(int).to_numpy()[na_mask]
                    imp_vals = np.stack(
                        [d[v].astype(int).to_numpy()[na_mask] for d in imputed]
                    )
                    voted, _ = _mode(imp_vals, axis=0, keepdims=False)
                    acc = np.mean(voted == true_vals)
                    imp_rows.append({
                        "run": s, "method": method, "variable": v,
                        "type": "categorical", "rmse": np.nan, "accuracy": acc,
                    })

                if verbose and done % max(1, total // 20) == 0:
                    pct = 100 * done / total
                    elapsed = (time.time() - t0) / 60
                    print(f"  [{done}/{total}] {pct:.0f}%  "
                          f"{method:14s}  sim {s+1}  ({elapsed:.0f} min)")

            except Exception as e:
                if verbose:
                    print(f"  FAILED: {method} sim {s}: {e}")
                    traceback.print_exc()

    imp_df = pd.DataFrame(imp_rows)
    return per_method, imp_df


# ─────────────────────────────────────────────────────────────
# Aggregation (reuses evaluation.aggregate_runs logic)
# ─────────────────────────────────────────────────────────────

def build_mnar_table2(per_method: dict[str, list[dict]]) -> pd.DataFrame:
    from evaluation import aggregate_runs
    frames = []
    for method, runs in per_method.items():
        if len(runs) == 0:
            continue
        agg = aggregate_runs(runs)
        agg.insert(0, "Method", method)
        frames.append(agg)
    return pd.concat(frames, axis=0, ignore_index=True)


# ─────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────

def save_mnar_outputs(table2: pd.DataFrame, imp_df: pd.DataFrame):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Table 2 CSV ──────────────────────────────────────────
    table2.to_csv(os.path.join(RESULTS_DIR, "mnar_table2.csv"),
                  index=False, float_format="%.6f")
    print(f"Wrote results/mnar_table2.csv")

    # ── Pretty console table ─────────────────────────────────
    scaled = table2.copy()
    for c in ("Var_T", "Var_T_target", "Var_W", "Var_W_target",
              "Var_B", "Var_B_target"):
        scaled[c] = scaled[c] * 1000
    scaled["CI_cov"] = scaled["CI_cov"] * 100

    hdr = (f"{'Parameter':22s}  {'Method':14s}  {'Bias':>8s}  "
           f"{'VarT':>8s}  {'VarT*':>8s}  {'VarW':>8s}  {'VarW*':>8s}  "
           f"{'VarB':>8s}  {'VarB*':>8s}  {'CR%':>6s}  {'Nval':>5s}")
    lines = [f"\nMNAR 50% — Table 2", "=" * len(hdr), hdr, "-" * len(hdr)]

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

    pretty = "\n".join(lines)
    print(pretty)
    with open(os.path.join(RESULTS_DIR, "mnar_table2_pretty.txt"), "w") as f:
        f.write(pretty + "\n")

    # ── Coverage heatmap ─────────────────────────────────────
    mat = (
        table2.pivot(index="Parameter", columns="Method", values="CI_cov")
        .reindex(index=PARAM_ORDER, columns=METHODS) * 100
    )
    short  = [PARAM_SHORT.get(p, p) for p in PARAM_ORDER]
    labels = [METHOD_LABELS.get(m, m) for m in METHODS]

    fig, ax = plt.subplots(
        figsize=(1.3 * len(METHODS) + 1.8, 0.42 * len(PARAM_ORDER) + 1.0)
    )
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "cov", ["#b2182b", "#f4a582", "#fddbc7", "#d1e5f0", "#4393c3", "#2166ac"])
    im = ax.imshow(mat.values, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(len(PARAM_ORDER)))
    ax.set_yticklabels(short, fontsize=7.5)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i, j]
            if np.isfinite(val):
                c = "white" if val < 25 or val > 85 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        color=c, fontsize=7.5, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("95% CI coverage (%)", fontsize=8)
    cbar.ax.axhline(95, color="k", lw=1.0, ls="--")
    ax.set_title("MNAR 50% — CI coverage by parameter and method",
                 fontsize=10, fontweight="bold", pad=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES_DIR, f"mnar_coverage_heatmap.{ext}"))
    plt.close(fig)
    print("Wrote figures/mnar_coverage_heatmap.pdf/.png")

    # ── Imputation RMSE bars ─────────────────────────────────
    if len(imp_df) > 0:
        imp_agg = (
            imp_df.groupby(["method", "variable", "type"])
            .agg(rmse=("rmse", "mean"), accuracy=("accuracy", "mean"))
            .reset_index()
        )
        imp_agg.to_csv(os.path.join(RESULTS_DIR, "mnar_imp_error.csv"),
                       index=False, float_format="%.4f")

        cont = imp_agg[imp_agg["type"] == "continuous"]
        n_v = len(CONT_MISS_VARS)
        n_m = len(METHODS)
        width = 0.75 / n_m
        xbase = np.arange(n_v)

        fig, ax = plt.subplots(figsize=(max(8, 1.5 * n_v), 3.5))
        for j, method in enumerate(METHODS):
            sub = cont[cont["method"] == method].set_index("variable").reindex(CONT_MISS_VARS)
            vals = sub["rmse"].fillna(0).values
            ax.bar(xbase + (j - (n_m - 1) / 2) * width, vals, width=width,
                   color=METHOD_COLOURS[method], label=METHOD_LABELS[method],
                   edgecolor="white", linewidth=0.3)
        ax.set_xticks(xbase)
        ax.set_xticklabels(CONT_MISS_VARS, fontsize=8)
        ax.set_ylabel("RMSE")
        ax.set_title("MNAR 50% — Imputation RMSE (continuous variables)",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, ncol=n_m, frameon=False, loc="upper right")
        ax.grid(axis="y", lw=0.2, alpha=0.4)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIGURES_DIR, f"mnar_imp_rmse.{ext}"))
        plt.close(fig)
        print("Wrote figures/mnar_imp_rmse.pdf/.png")

        # Console print
        print("\nMNAR 50% — Imputation error:")
        print(f"{'Variable':12s}  {'Method':14s}  {'RMSE':>8s}  {'Accuracy':>8s}")
        print("-" * 50)
        for v in CONT_MISS_VARS + CAT_MISS_VARS:
            sub = imp_agg[imp_agg["variable"] == v]
            for method in METHODS:
                row = sub[sub["method"] == method]
                if row.empty:
                    continue
                r = row.iloc[0]
                rs = f"{r['rmse']:.4f}" if pd.notna(r["rmse"]) else "—"
                ac = f"{r['accuracy']:.3f}" if pd.notna(r["accuracy"]) else "—"
                print(f"{v:12s}  {METHOD_LABELS[method]:14s}  {rs:>8s}  {ac:>8s}")
            print()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="MNAR 50% experiment — D&L Table 2 style output"
    )
    ap.add_argument("--n-sims", type=int, default=100,
                    help="Monte Carlo replicates (default 100)")
    ap.add_argument("--methods", nargs="+", default=None)
    args = ap.parse_args()

    methods = args.methods or METHODS
    n_sims  = args.n_sims

    print("=" * 70)
    print("  MNAR 50% Experiment — Deng & Lumley Table 2 style")
    print(f"  S = {n_sims},  n = {N_OBS},  m = {N_IMPUTATIONS}")
    print(f"  Methods: {', '.join(methods)}")
    print("=" * 70)

    t0 = time.time()
    per_method, imp_df = run_mnar_study(n_sims=n_sims, methods=methods)
    table2 = build_mnar_table2(per_method)
    save_mnar_outputs(table2, imp_df)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} minutes.")
