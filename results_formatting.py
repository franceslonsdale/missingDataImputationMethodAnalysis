"""
results_formatting.py — Produce Deng & Lumley (2024) Table 2 from the
aggregated results.

Layout: one block of rows per analysis-model coefficient; within each
block, one row per imputation method, with columns

    Bias | VarT | VarT* | VarW | VarW* | VarB | VarB* | CR(%)

where * denotes the "target" quantity (see evaluation.py) and all
variances are shown ×1000 to match the paper.

Writes:
    results/table2_pretty.txt    fixed-width, for a terminal
    results/table2.tex           LaTeX (booktabs), for the thesis
"""

from __future__ import annotations

import os
import pandas as pd

from config import PARAM_ORDER, METHODS, RESULTS_DIR, TABLE2_CSV


def _fmt(x, spec=".2f"):
    if pd.isna(x):
        return "—"
    return format(x, spec)


def _scale_vars(df: pd.DataFrame) -> pd.DataFrame:
    """Multiply the variance columns by 1000 in-place on a copy."""
    out = df.copy()
    for c in ("Var_T", "Var_T_target", "Var_W", "Var_W_target",
              "Var_B", "Var_B_target"):
        out[c] = out[c] * 1000
    out["CI_cov"] = out["CI_cov"] * 100
    return out


# -----------------------------------------------------------------------------
# Pretty console table
# -----------------------------------------------------------------------------

def pretty_table(df_scaled: pd.DataFrame) -> str:
    hdr = (
        f"{'Parameter':22s}  {'Method':8s}  "
        f"{'Bias':>8s}  {'VarT':>8s}  {'VarT*':>8s}  "
        f"{'VarW':>8s}  {'VarW*':>8s}  "
        f"{'VarB':>8s}  {'VarB*':>8s}  {'CR%':>6s}  {'Nval':>5s}"
    )
    out = [hdr, "-" * len(hdr)]

    for p in PARAM_ORDER:
        block = df_scaled[df_scaled["Parameter"] == p]
        if block.empty:
            continue
        first = True
        for method in METHODS:
            row = block[block["Method"] == method]
            if row.empty:
                continue
            r = row.iloc[0]
            label = p if first else ""
            first = False
            out.append(
                f"{label:22s}  {method:8s}  "
                f"{_fmt(r['Bias'], '+.3f'):>8s}  "
                f"{_fmt(r['Var_T']):>8s}  {_fmt(r['Var_T_target']):>8s}  "
                f"{_fmt(r['Var_W']):>8s}  {_fmt(r['Var_W_target']):>8s}  "
                f"{_fmt(r['Var_B']):>8s}  {_fmt(r['Var_B_target']):>8s}  "
                f"{_fmt(r['CI_cov'], '.1f'):>6s}  "
                f"{int(r['n_valid']):>5d}"
            )
        out.append("")
    return "\n".join(out)


# -----------------------------------------------------------------------------
# LaTeX (booktabs) — same structure
# -----------------------------------------------------------------------------

def _latex_escape(s: str) -> str:
    """Minimal LaTeX escape for the characters that actually appear in our
    parameter and method labels: _, ^, **, and the bracket/paren glyphs which
    are fine in math or text mode but we wrap carefully."""
    return (
        s.replace("\\", r"\textbackslash{}")
         .replace("_", r"\_")
         .replace("**", r"\text{**}")
         .replace("^", r"\^{}")
    )


def latex_table(df_scaled: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering\scriptsize",
        r"\caption{Simulation results: bias, variance decomposition and 95\% CI coverage for "
        r"MICE, GAIN(SI), MGAIN and MIDAS under the Deng \& Lumley (2024) MAR tertile mechanism. "
        r"All variance columns are reported $\times 1000$. Stars denote target quantities "
        r"(see text).}",
        r"\label{tab:simstudy_table2}",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Parameter & Method & Bias & $\mathrm{Var}_T$ & $\mathrm{Var}_T^{*}$ "
        r"& $\mathrm{Var}_W$ & $\mathrm{Var}_W^{*}$ "
        r"& $\mathrm{Var}_B$ & $\mathrm{Var}_B^{*}$ & CR\% \\",
        r"\midrule",
    ]
    for p in PARAM_ORDER:
        block = df_scaled[df_scaled["Parameter"] == p]
        if block.empty:
            continue
        first = True
        for method in METHODS:
            row = block[block["Method"] == method]
            if row.empty:
                continue
            r = row.iloc[0]
            label = _latex_escape(p) if first else ""
            first = False
            lines.append(
                f"{label} & {_latex_escape(method)} "
                f"& {_fmt(r['Bias'], '+.3f')} "
                f"& {_fmt(r['Var_T'])} & {_fmt(r['Var_T_target'])} "
                f"& {_fmt(r['Var_W'])} & {_fmt(r['Var_W_target'])} "
                f"& {_fmt(r['Var_B'])} & {_fmt(r['Var_B_target'])} "
                f"& {_fmt(r['CI_cov'], '.1f')} \\\\"
            )
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def generate_all(table2_df: pd.DataFrame | None = None) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if table2_df is None:
        table2_df = pd.read_csv(TABLE2_CSV)

    scaled = _scale_vars(table2_df)

    pretty = pretty_table(scaled)
    print(pretty)
    with open(os.path.join(RESULTS_DIR, "table2_pretty.txt"), "w") as f:
        f.write(pretty + "\n")

    with open(os.path.join(RESULTS_DIR, "table2.tex"), "w") as f:
        f.write(latex_table(scaled) + "\n")

    print(f"\nWrote {os.path.join(RESULTS_DIR, 'table2_pretty.txt')}")
    print(f"Wrote {os.path.join(RESULTS_DIR, 'table2.tex')}")


if __name__ == "__main__":
    generate_all()
