"""
evaluation.py — Per-run analysis, Rubin pooling, and cross-run aggregation.

Implements the evaluation criteria of Deng & Lumley (2024) §3.2 exactly.

Per-run quantities
------------------
For a single simulation run s and a single imputation method:
    - Fit the analysis model on the complete (pre-masking) data once,
      obtaining beta_hat_s and U_s (we keep only the diagonal of the
      variance-covariance).
    - Fit the analysis model on each of the m imputed datasets,
      obtaining beta*_{s,l} and U*_{s,l} for l = 1, ..., m.
    - Combine via Rubin (1987):
          Q_bar_s = mean_l  beta*_{s,l}
          U_bar_s = mean_l  U*_{s,l}              (within-imp variance)
          B_s     = var_l   beta*_{s,l}  (ddof=1) (between-imp variance)
          T_s     = U_bar_s + (1 + 1/m) B_s

Cross-run targets (Deng & Lumley §3.2)
--------------------------------------
Averaging over simulation runs (h = 1, ..., S):
    Var_W        := E_h[U_bar_s]
    Var_B        := (1 + 1/m) E_h[B_s]
    Var_T        := Var_W + Var_B
    Var_W^target := E_h[U_s]                     (avg complete-data OLS var)
    Var_B^target := Var_h(Q_bar_s)               (MC var of pooled estimates)
    Var_T^target := Var_W^target + Var_B^target

Bias  := E_h[Q_bar_s - beta_hat_s]               (relative to per-run β̂)
Cov   := Pr_h(beta_true in [Q_bar_s ± t_{ν,.975} sqrt(T_s)])

All variance quantities are reported ×1000 downstream, as in D&L Table 2.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from config import (
    ANALYSIS_FORMULA, PARAM_ORDER, TRUE_BETA, CONFIDENCE_LEVEL,
)

TRUE_BETA_VEC = np.array([TRUE_BETA[p] for p in PARAM_ORDER], dtype=float)


# -----------------------------------------------------------------------------
# One OLS fit aligned to PARAM_ORDER
# -----------------------------------------------------------------------------

def fit_analysis(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit the analysis model on a (complete) data frame and return
    `(beta, var_diag)` aligned to PARAM_ORDER.

    A coefficient that is absent from the fit (should be vanishingly
    rare at n = 10,000 but possible in principle if a factor level
    drops) is returned as NaN; the aggregator skips those cells.
    """
    df_fit = df.copy()
    # statsmodels' C() handles Categorical fine, but we cast to plain int
    # for a stable set of coefficient names.
    if hasattr(df_fit["bin1"], "cat"):
        df_fit["bin1"] = df_fit["bin1"].astype(int)
    if hasattr(df_fit["ord1"], "cat"):
        df_fit["ord1"] = df_fit["ord1"].astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.ols(ANALYSIS_FORMULA, data=df_fit).fit()

    params = fit.params
    vardia = pd.Series(np.diag(fit.cov_params()), index=params.index)

    beta = np.array([params.get(p, np.nan) for p in PARAM_ORDER])
    varv = np.array([vardia.get(p, np.nan) for p in PARAM_ORDER])
    return beta, varv


# -----------------------------------------------------------------------------
# Rubin's rules (univariate, per coefficient)
# -----------------------------------------------------------------------------

def rubin_pool(ests: np.ndarray, varis: np.ndarray) -> dict:
    """
    Pool m imputed estimates using Rubin (1987).

    Parameters
    ----------
    ests  : (m, k) array of per-imputation point estimates
    varis : (m, k) array of per-imputation within-imputation variances

    Returns a dict with keys
        Qbar  (k,) pooled estimates
        Ubar  (k,) within-imputation variance
        B     (k,) between-imputation variance (ddof=1); NaN if m == 1
        Tvar  (k,) total variance U_bar + (1 + 1/m) B
        df    (k,) Rubin (Barnard-Rubin) degrees of freedom
    """
    m = ests.shape[0]
    Qbar = np.nanmean(ests, axis=0)
    Ubar = np.nanmean(varis, axis=0)

    if m == 1:
        # Single imputation: no between-imputation variance.  This arm
        # (GAIN_SI) is included precisely to illustrate what happens if you
        # ignore B — we treat B = 0 and set df very large.
        B    = np.zeros_like(Qbar)
        Tvar = Ubar.copy()
        df   = np.full_like(Qbar, 1e6)
        return dict(Qbar=Qbar, Ubar=Ubar, B=B, Tvar=Tvar, df=df)

    B    = np.nanvar(ests, axis=0, ddof=1)
    Tvar = Ubar + (1.0 + 1.0 / m) * B

    # Barnard-Rubin (1999) degrees of freedom; fall back to large df
    # when B is tiny (to avoid division-by-zero).
    with np.errstate(divide="ignore", invalid="ignore"):
        r  = (1.0 + 1.0 / m) * B / np.where(Ubar > 0, Ubar, np.nan)
        df = (m - 1) * (1.0 + 1.0 / r) ** 2
    df = np.where(np.isfinite(df), df, 1e6)
    df = np.minimum(df, 1e6)
    return dict(Qbar=Qbar, Ubar=Ubar, B=B, Tvar=Tvar, df=df)


# -----------------------------------------------------------------------------
# One simulation run (one method)
# -----------------------------------------------------------------------------

def evaluate_single_run(
    imputed_datasets: list[pd.DataFrame],
    df_complete: pd.DataFrame,
) -> dict:
    """
    Evaluate one (method, run) cell.

    Returns a dict with the per-run quantities needed to feed the
    cross-run aggregator:

        beta_complete : (k,) complete-data OLS estimates (β̂_s)
        var_complete  : (k,) complete-data OLS diagonal variance (U_s)
        Qbar          : (k,) Rubin-pooled estimates
        Ubar, B, Tvar : (k,) Rubin-pooled variance components
        df            : (k,) Rubin degrees of freedom
        ci_cover      : (k,) 0/1 indicator that beta_true ∈ CI
        m             : number of imputations (informational)
    """
    # Fit on the complete (pre-masking) data — supplies β̂_s and U_s.
    beta_complete, var_complete = fit_analysis(df_complete)

    ests  = np.stack([fit_analysis(d)[0] for d in imputed_datasets])
    varis = np.stack([fit_analysis(d)[1] for d in imputed_datasets])

    pooled = rubin_pool(ests, varis)

    # Coverage CIs — Rubin's rules with Barnard-Rubin df.
    alpha = 1 - CONFIDENCE_LEVEL
    tcrit = stats.t.ppf(1 - alpha / 2, pooled["df"])
    se    = np.sqrt(np.maximum(pooled["Tvar"], 0))
    lo    = pooled["Qbar"] - tcrit * se
    hi    = pooled["Qbar"] + tcrit * se
    cover = ((TRUE_BETA_VEC >= lo) & (TRUE_BETA_VEC <= hi)).astype(float)

    return {
        "beta_complete": beta_complete,
        "var_complete" : var_complete,
        "Qbar"         : pooled["Qbar"],
        "Ubar"         : pooled["Ubar"],
        "B"            : pooled["B"],
        "Tvar"         : pooled["Tvar"],
        "df"           : pooled["df"],
        "ci_cover"     : cover,
        "m"            : len(imputed_datasets),
    }


# -----------------------------------------------------------------------------
# Cross-run aggregation (Deng & Lumley §3.2)
# -----------------------------------------------------------------------------

def aggregate_runs(run_results: list[dict]) -> pd.DataFrame:
    """
    Collapse a list of per-run dicts (from evaluate_single_run) into a
    DataFrame in the shape of D&L Table 2, for ONE imputation method.

    One row per analysis-model coefficient, with columns:
        Parameter, true_beta,
        Bias,
        Var_T, Var_T_target,
        Var_W, Var_W_target,
        Var_B, Var_B_target,
        CI_cov,
        n_valid
    """
    if len(run_results) == 0:
        raise ValueError("No runs to aggregate.")

    m = run_results[0]["m"]
    stack = lambda key: np.stack([r[key] for r in run_results])   # (S, k)

    beta_c = stack("beta_complete")
    var_c  = stack("var_complete")
    Qbar   = stack("Qbar")
    Ubar   = stack("Ubar")
    B      = stack("B")
    cover  = stack("ci_cover")

    # Per-coefficient count of non-NaN runs
    n_valid = np.sum(np.isfinite(Qbar), axis=0)

    # Empirical bias — relative to the per-run complete-data estimate.
    bias = np.nanmean(Qbar - beta_c, axis=0)

    # Within and between (estimated):
    var_W = np.nanmean(Ubar, axis=0)
    var_B = (1.0 + 1.0 / m) * np.nanmean(B, axis=0)
    var_T = var_W + var_B

    # Targets (D&L §3.2):
    var_W_tgt = np.nanmean(var_c, axis=0)         # = E_h[U_s]
    var_B_tgt = np.nanvar(Qbar, axis=0, ddof=1)   # = Var_h(Q_bar_s)
    var_T_tgt = var_W_tgt + var_B_tgt

    ci_cov = np.nanmean(cover, axis=0)

    out = pd.DataFrame({
        "Parameter"    : PARAM_ORDER,
        "true_beta"    : TRUE_BETA_VEC,
        "Bias"         : bias,
        "Var_T"        : var_T,
        "Var_T_target" : var_T_tgt,
        "Var_W"        : var_W,
        "Var_W_target" : var_W_tgt,
        "Var_B"        : var_B,
        "Var_B_target" : var_B_tgt,
        "CI_cov"       : ci_cov,
        "n_valid"      : n_valid,
    })
    return out
