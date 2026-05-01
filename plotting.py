"""
plotting.py — Publication-quality figures for §6 of the formal plan.

1. Bias forest plot           (figures/bias_forest.pdf/.png)     — D&L Fig. 3 style
2. Relative variance ratios   (figures/var_ratio.pdf/.png)       — D&L Fig. 4 style
3. Coverage heatmap           (figures/coverage_heatmap.pdf/.png)
4. Summary bar chart          (figures/summary_bars.pdf/.png)

All figures read the table2 CSV.  Saves PDF (vector, for thesis) and
PNG (raster, for quick viewing).
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

from config import PARAM_ORDER, METHODS, FIGURES_DIR, TABLE2_CSV, RAW_CSV, TRUE_BETA

# ── Thesis styling ───────────────────────────────────────────
mpl.rcParams.update({
    "font.family"      : "serif",
    "font.serif"       : ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset" : "cm",
    "font.size"        : 9,
    "axes.titlesize"   : 10,
    "axes.labelsize"   : 9,
    "xtick.labelsize"  : 8,
    "ytick.labelsize"  : 8,
    "legend.fontsize"  : 8,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"       : 150,
    "savefig.dpi"      : 300,
    "savefig.bbox"     : "tight",
})

METHOD_LABELS = {
    "MICE"           : "MICE",
    "GAIN_SI"        : "GAIN(SI)",
    "MGAIN"          : "MGAIN",
    "MGAIN_RESAMPLE" : "MGAIN-R",
    "MIDAS"          : "MIDAS",
}

METHOD_COLOURS = {
    "MICE"           : "#2166ac",
    "GAIN_SI"        : "#b2182b",
    "MGAIN"          : "#1b7837",
    "MGAIN_RESAMPLE" : "#e08214",
    "MIDAS"          : "#7b3294",
}

PARAM_SHORT = {
    "Intercept"              : "Intercept",
    "C(bin1)[T.1]"           : "bin1",
    "C(ord1)[T.1]"           : "ord1=1",
    "C(ord1)[T.2]"           : "ord1=2",
    "norm1"                  : "norm1",
    "norm2"                  : "norm2",
    "norm3"                  : "norm3",
    "norm5"                  : "norm5",
    "norm7"                  : "norm7",
    "I(norm1 ** 2)"          : r"norm1$^2$",
    "norm2:norm3"            : r"norm2$\times$norm3",
    "norm5:C(bin1)[T.1]"     : r"norm5$\times$bin1",
    "norm7:C(ord1)[T.1]"     : r"norm7$\times$ord1=1",
    "norm7:C(ord1)[T.2]"     : r"norm7$\times$ord1=2",
}


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES_DIR, f"{name}.{ext}"))
    print(f"  Wrote figures/{name}.pdf and .png")


# ─────────────────────────────────────────────────────────────
# 1. Bias forest plot
# ─────────────────────────────────────────────────────────────

def plot_bias_forest(raw_df: pd.DataFrame) -> None:
    params = PARAM_ORDER
    ncol = 4
    nrow = int(np.ceil(len(params) / ncol))
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(3.2 * ncol, 2.0 * nrow),
                             squeeze=False)
    n_methods = len(METHODS)
    for i, p in enumerate(params):
        ax = axes[i // ncol, i % ncol]
        sub = raw_df[raw_df["Parameter"] == p]
        for j, method in enumerate(METHODS):
            s = sub[sub["Method"] == method]
            if s.empty:
                continue
            mean = s["Qbar"].mean()
            se   = s["Qbar"].std(ddof=1) / np.sqrt(len(s))
            ax.errorbar(
                mean, j, xerr=1.96 * se,
                fmt="o", color=METHOD_COLOURS[method],
                capsize=2.5, capthick=1.0,
                markersize=4, markeredgewidth=0.5, markeredgecolor="white",
                linewidth=1.2,
            )
        ax.axvline(TRUE_BETA[p], color="grey", lw=0.7, ls="--", zorder=0)
        ax.set_yticks(range(n_methods))
        ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=7)
        ax.set_title(PARAM_SHORT[p], fontsize=9, fontweight="bold")
        ax.grid(axis="x", lw=0.2, alpha=0.4)
        ax.invert_yaxis()
        ax.tick_params(axis="x", labelsize=7)

    for k in range(len(params), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")

    fig.suptitle(
        r"Pooled coefficient estimates $\bar{\beta}_M$ with $\pm 1.96$ MCSE"
        "\n(dashed line = DGM true value)",
        y=1.01, fontsize=11,
    )
    fig.tight_layout()
    _save(fig, "bias_forest")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# 2. Relative variance ratio plot
# ─────────────────────────────────────────────────────────────

def plot_var_ratio(table2_df: pd.DataFrame) -> None:
    df = table2_df.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["ratio_W"] = np.where(
            df["Var_W_target"] > 0,
            df["Var_W"] / df["Var_W_target"] - 1.0, np.nan)
        df["ratio_B"] = np.where(
            df["Var_B_target"] > 0,
            df["Var_B"] / df["Var_B_target"] - 1.0, np.nan)

    params = PARAM_ORDER
    short  = [PARAM_SHORT[p] for p in params]
    n_m    = len(METHODS)
    width  = 0.75 / n_m
    xbase  = np.arange(len(params))

    fig, axes = plt.subplots(2, 1,
                             figsize=(max(11, 0.8 * len(params)), 7),
                             sharex=True)
    for ax, col, ylabel in zip(
        axes, ("ratio_W", "ratio_B"),
        (r"$\mathrm{Var}_W\, /\, \mathrm{Var}_W^{\mathrm{target}} - 1$",
         r"$\mathrm{Var}_B\, /\, \mathrm{Var}_B^{\mathrm{target}} - 1$"),
    ):
        for j, method in enumerate(METHODS):
            s = df[df["Method"] == method].set_index("Parameter").reindex(params)
            vals = s[col].fillna(0).values
            ax.bar(
                xbase + (j - (n_m - 1) / 2) * width,
                vals, width=width,
                color=METHOD_COLOURS[method],
                label=METHOD_LABELS[method],
                edgecolor="white", linewidth=0.3,
            )
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", lw=0.2, alpha=0.4)

    axes[1].set_xticks(xbase)
    axes[1].set_xticklabels(short, rotation=40, ha="right", fontsize=7.5)
    axes[0].legend(
        loc="upper right", ncol=n_m, fontsize=7.5, frameon=False,
        handlelength=1.2, columnspacing=0.8,
    )
    axes[0].set_title(
        "Relative ratio of imputation variance to target variance",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "var_ratio")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# 3. Coverage heatmap
# ─────────────────────────────────────────────────────────────

def plot_coverage_heatmap(table2_df: pd.DataFrame) -> None:
    mat = (
        table2_df
        .pivot(index="Parameter", columns="Method", values="CI_cov")
        .reindex(index=PARAM_ORDER, columns=METHODS)
        * 100
    )
    short  = [PARAM_SHORT[p] for p in PARAM_ORDER]
    labels = [METHOD_LABELS[m] for m in METHODS]

    fig, ax = plt.subplots(
        figsize=(1.3 * len(METHODS) + 1.8, 0.42 * len(PARAM_ORDER) + 1.0)
    )
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "cov", ["#b2182b", "#f4a582", "#fddbc7", "#d1e5f0", "#4393c3", "#2166ac"],
    )
    im = ax.imshow(mat.values, aspect="auto", cmap=cmap, vmin=0, vmax=100)

    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(len(PARAM_ORDER)))
    ax.set_yticklabels(short, fontsize=7.5)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i, j]
            if np.isfinite(val):
                txt_col = "white" if val < 25 or val > 85 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        color=txt_col, fontsize=7.5, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("95% CI coverage (%)", fontsize=8)
    cbar.ax.axhline(95, color="k", lw=1.0, ls="--")

    ax.set_title("95% CI coverage by parameter and method",
                 fontsize=10, fontweight="bold", pad=8)
    fig.tight_layout()
    _save(fig, "coverage_heatmap")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# 4. Summary bar chart
# ─────────────────────────────────────────────────────────────

def plot_summary_bars(table2_df: pd.DataFrame) -> None:
    df = table2_df.copy()
    rows = []
    for method in METHODS:
        sub = df[df["Method"] == method]
        mean_abs_bias = sub["Bias"].abs().mean()
        mean_cov      = sub["CI_cov"].mean() * 100
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_B = np.where(
                sub["Var_B_target"] > 0,
                sub["Var_B"] / sub["Var_B_target"], np.nan)
        mean_ratio_B = np.nanmean(ratio_B)
        rows.append(dict(Method=method, MAB=mean_abs_bias,
                         Mean_Cov=mean_cov, Mean_RatB=mean_ratio_B))
    sdf = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    x = np.arange(len(METHODS))
    colours = [METHOD_COLOURS[m] for m in METHODS]
    labels  = [METHOD_LABELS[m] for m in METHODS]

    ax = axes[0]
    ax.bar(x, sdf["MAB"], color=colours, edgecolor="white", linewidth=0.4)
    ax.set_ylabel(r"Mean $|\mathrm{Bias}|$")
    ax.set_title("(a) Absolute bias", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)

    ax = axes[1]
    ax.bar(x, sdf["Mean_Cov"], color=colours, edgecolor="white", linewidth=0.4)
    ax.axhline(95, color="k", lw=0.7, ls="--")
    ax.set_ylabel("Mean 95% CI coverage (%)")
    ax.set_title("(b) CI coverage", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylim(0, 105)

    ax = axes[2]
    ax.bar(x, sdf["Mean_RatB"], color=colours, edgecolor="white", linewidth=0.4)
    ax.axhline(1.0, color="k", lw=0.7, ls="--")
    ax.set_ylabel(r"Mean $\mathrm{Var}_B\, /\, \mathrm{Var}_B^{\mathrm{target}}$")
    ax.set_title("(c) Between-var calibration", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)

    fig.suptitle(
        "Summary metrics averaged across all 14 analysis-model parameters",
        fontsize=10, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "summary_bars")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def generate_all(table2_df: pd.DataFrame | None = None,
                 raw_df:    pd.DataFrame | None = None) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    if table2_df is None:
        table2_df = pd.read_csv(TABLE2_CSV)
    if raw_df is None and os.path.exists(RAW_CSV):
        raw_df = pd.read_csv(RAW_CSV)

    if raw_df is not None:
        plot_bias_forest(raw_df)
    else:
        print("  Skipping bias_forest (no raw CSV available)")

    plot_var_ratio(table2_df)
    plot_coverage_heatmap(table2_df)
    plot_summary_bars(table2_df)


if __name__ == "__main__":
    generate_all()
