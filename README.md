# Simulation study: MICE vs GAIN(SI) vs MGAIN vs MIDAS

Implementation of the formal plan: a Deng & Lumley (2024) §3-style
comparison of four imputation methods under the tertile MAR mechanism
with n = 10,000, S = 1,000, m = 5.

## Layout

    config.py                 — all tunable parameters; flip TEST_MODE for smoke tests
    data_generation.py        — DGM (eq. 3.1 of D&L) + tertile MAR
    evaluation.py             — OLS analysis + Rubin pooling + D&L variance targets
    gain.py                   — GAIN (Yoon et al. 2018), PyTorch, iteration-based
    midas_imputer.py          — MIDAS (Lall & Robinson 2022), PyTorch, MC-dropout
    imputation_methods.py     — unified dispatch, factor encode/decode, rpy2 MICE
    simulation.py             — Monte Carlo driver + checkpointing
    results_formatting.py     — Table 2 (console + LaTeX)
    plotting.py               — bias forest, variance ratios, coverage heatmap
    run_study.py              — CLI entry point

    results/                  — raw + aggregated CSVs, Table 2, checkpoint.pkl
    figures/                  — PNGs for the thesis

## Quick start

    # smoke test: S=10, ~2-5 minutes end-to-end
    python run_study.py

    # full run (edit config.TEST_MODE = False first, then):
    python run_study.py

    # just rebuild tables/plots from the existing CSVs
    python run_study.py --format-only

Checkpointing is automatic (every 10 runs) to
`results/checkpoint.pkl`; if you kill a long run and restart,
it picks up where it left off.

## Dependencies

See `requirements.txt`.  For the **MICE** arm to match the plan
exactly you need R and the R `mice` package installed, plus `rpy2`:

    R -e 'install.packages("mice")'
    pip install rpy2

Without these, MICE silently falls back to
`sklearn.IterativeImputer` with a runtime warning.  The fallback
is fine for debugging but it is **not** the PMM method described
in the plan and should not be used for the final run.

## Parameters to verify before the full run

In `config.py`:

- `TEST_MODE = False`  (sets `S = 1000`)
- `N_OBS = 10_000`
- `N_IMPUTATIONS = 5`
- `GAIN_CONFIG["iterations"] = 5000`, `alpha = 10`
- `MIDAS_CONFIG["train_epochs"] = 300`, `layer_structure = [256, 256]`

And double-check the sanity output from

    python data_generation.py

which prints (i) the empirical correlations `cor(norm5,bin1) ≈ 0.55`,
`cor(norm7,ord1) ≈ 0.65`, etc. and (ii) marginal missing rates ≈ 0.43
per incomplete variable.  If those are off, the DGM is wrong and
nothing downstream is trustworthy.

## What changed vs the old codebase

- Missingness: logistic calibration → tertile MAR (D&L §3.1).
- DGM: 4-variable → 11-variable mixed-type (eq. 3.1 of D&L).
- Factors: `int` → `pd.Categorical` throughout.
- Evaluation targets: `Var_B^target := Var_h(Q̄_M)` (was wrong before).
- Bias: now relative to per-run complete-data β̂ (D&L §3.2).
- MICE: sklearn BayesianRidge → R `mice(pmm)` via rpy2.
- GAIN: epoch-based → iteration-based (5000 updates); `α = 10`.
- MIDAS: bug fix — MC-dropout now actually fires at inference time.
- `GAIN_MI` renamed to `MGAIN` (five independent trainings).
- MCAR/MNAR arms removed (not in the formal plan; easy to re-add).
