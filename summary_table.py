"""
summary_table.py — Executive summary table: one row per method, with
metrics averaged across all 14 analysis-model parameters.

Columns:
    Method | Mean|Bias| | Mean VarW/VarW* | Mean VarB/VarB* | Mean CR%

Writes:
    results/summary_table.txt    (fixed-width for terminal)
    results/summary_table.tex    (LaTeX booktabs for thesis)
    results/summary_table.csv    (machine-readable)

Can be run standalone:  python summary_table.py
Or called from run_study.py / results_formatting.py.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from config import PARAM_ORDER, METHODS, RESULTS_DIR, TABLE2_CSV

METHOD_LABELS = {
    "MICE"           : "MICE",
    "GAIN_SI"        : "GAIN(SI)",
    "MGAIN"          : "MGAIN",
    "MGAIN_RESAMPLE" : "MGAIN-R",
    "MIDAS"          : "MIDAS",
}


def build_summary(table2_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-method summary metrics averaged across all parameters.
    """
    rows = []
    for method in METHODS:
        sub = table2_df[table2_df["Method"] == method]
        if sub.empty:
            continue

        mean_abs_bias = sub["Bias"].abs().mean()

        # Var_W / Var_W_target ratio
        with np.errstate(divide="ignore", invalid="ignore"):
            r_W = np.where(sub["Var_W_target"] > 0,
                           sub["Var_W"] / sub["Var_W_target"], np.nan)
            r_B = np.where(sub["Var_B_target"] > 0,
                           sub["Var_B"] / sub["Var_B_target"], np.nan)

        mean_cov   = sub["CI_cov"].mean() * 100
        mean_n     = sub["n_valid"].mean()

        rows.append({
            "Method"          : method,
            "Label"           : METHOD_LABELS[method],
            "Mean_Abs_Bias"   : mean_abs_bias,
            "Mean_VarW_ratio" : np.nanmean(r_W),
            "Mean_VarB_ratio" : np.nanmean(r_B),
            "Mean_CR"         : mean_cov,
            "Mean_Nvalid"     : mean_n,
        })
    return pd.DataFrame(rows)


def pretty_summary(sdf: pd.DataFrame) -> str:
    hdr = (f"{'Method':12s}  {'|Bias|':>8s}  {'VarW/VarW*':>10s}  "
           f"{'VarB/VarB*':>10s}  {'CR%':>6s}  {'Nval':>6s}")
    lines = [hdr, "-" * len(hdr)]
    for _, r in sdf.iterrows():
        lines.append(
            f"{r['Label']:12s}  {r['Mean_Abs_Bias']:>8.3f}  "
            f"{r['Mean_VarW_ratio']:>10.2f}  "
            f"{r['Mean_VarB_ratio']:>10.2f}  "
            f"{r['Mean_CR']:>6.1f}  "
            f"{r['Mean_Nvalid']:>6.0f}"
        )
    return "\n".join(lines)


def latex_summary(sdf: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Summary of simulation results averaged across all 14 "
        r"analysis-model parameters ($S = 250$, $n = 10\,000$, $m = 5$). "
        r"Variance ratios equal to 1 indicate perfect calibration; "
        r"values below 1 indicate under-dispersion, above 1 over-dispersion.}",
        r"\label{tab:summary}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Mean $|\mathrm{Bias}|$ "
        r"& $\mathrm{Var}_W / \mathrm{Var}_W^{*}$ "
        r"& $\mathrm{Var}_B / \mathrm{Var}_B^{*}$ "
        r"& CR (\%) \\",
        r"\midrule",
    ]
    for _, r in sdf.iterrows():
        label = r["Label"].replace("_", r"\_")
        lines.append(
            f"{label} & {r['Mean_Abs_Bias']:.3f} "
            f"& {r['Mean_VarW_ratio']:.2f} "
            f"& {r['Mean_VarB_ratio']:.2f} "
            f"& {r['Mean_CR']:.1f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_summary(table2_df: pd.DataFrame | None = None) -> pd.DataFrame:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if table2_df is None:
        table2_df = pd.read_csv(TABLE2_CSV)

    sdf = build_summary(table2_df)

    # Console
    pretty = pretty_summary(sdf)
    print("\nSummary table:")
    print(pretty)
    with open(os.path.join(RESULTS_DIR, "summary_table.txt"), "w") as f:
        f.write(pretty + "\n")

    # LaTeX
    with open(os.path.join(RESULTS_DIR, "summary_table.tex"), "w") as f:
        f.write(latex_summary(sdf) + "\n")

    # CSV
    sdf.to_csv(os.path.join(RESULTS_DIR, "summary_table.csv"),
               index=False, float_format="%.4f")

    print(f"  Wrote results/summary_table.txt, .tex, .csv")
    return sdf


if __name__ == "__main__":
    generate_summary()
