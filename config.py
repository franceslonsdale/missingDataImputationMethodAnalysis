"""
config.py — Central configuration for the simulation study.

Reference: Deng & Lumley (2024) "Multiple Imputation through XGBoost",
           JCGS 33(2), 352–363.
           The data generation model, missingness mechanism, analysis
           model and evaluation criteria in this study are designed to
           match their Section 3 exactly, so that our four methods
           (MICE, GAIN_SI, MGAIN, MIDAS) can be dropped into their
           Table 2 framework.

Compared to Deng & Lumley:
    - They run 1,000 replicates at n=10,000.  We replicate that.
    - They use m=5 imputations.  We do the same.
    - They analyse mice-default / mice-cart / mice-ranger / mixgb / mixgb-sub.
      We substitute MICE (PMM via R's `mice`), GAIN(SI), MGAIN, and MIDAS.

Set TEST_MODE=True for a fast end-to-end smoke test (S=10, ~2 minutes).
"""

import os

# =============================================================================
# Simulation scale
# =============================================================================#TEST_MODE = True              # False for the full 1000-run study
TEST_MODE = False             # full run

if TEST_MODE:
    N_SIMS = 10
else:
    N_SIMS = 250              # SE on coverage ~1.4 percentage points;
                              # sufficient precision for a pp-scale
                              # comparison between methods, fits overnight.

N_OBS         = 10_000        # Deng & Lumley §3.1
N_IMPUTATIONS = 5             # M; Deng & Lumley §3.1
RANDOM_SEED   = 20260409      # master seed for the L'Ecuyer-style stream

# Number of worker processes for the outer simulation loop.
# Set to 1 while GAIN/MIDAS are using a GPU (can't share CUDA context).
N_WORKERS = 1

# =============================================================================
# Missingness
# =============================================================================
# Deng & Lumley §3.1: tertile MAR on the score Z.
#   Variables with missingness: norm1, norm2, norm3, norm5, bin1, norm7, ord1
#   Drivers:   norm1/2/3 ~ Y + norm4
#              norm5, bin1 ~ Y + norm6
#              norm7, ord1 ~ Y + norm8
#   Probabilities: top third -> 0.6, middle third -> 0.1, bottom third -> 0.6
# Expected marginal missing rate per variable ≈ (0.6 + 0.1 + 0.6) / 3 ≈ 0.433.
MECHANISM = "MAR_TERTILE"
P_MISS_TAIL = 0.6
P_MISS_MID  = 0.1

# =============================================================================
# Data-generation correlation structure (§3.1)
# =============================================================================
# - norm1..norm4:  pairwise correlation 0.5
# - norm5, norm6:  correlation 0.7, each correlated (~0.55) with bin1
# - norm7, norm8:  correlation 0.7, each correlated (~0.65) with ord1
#
# The bin1/ord1 correlations are constructed through a Gaussian copula
# (latent normal + threshold); the latent correlations below are calibrated
# so the observed point-biserial / polyserial correlations land at the
# targets above.  See data_generation.generate_complete_data() for detail.
RHO_NORM14    = 0.5
RHO_56        = 0.7
RHO_56_LATENT = 0.69     # -> observed cor(norm5, bin1) ≈ 0.55
RHO_78        = 0.7
RHO_78_LATENT = 0.72     # -> observed cor(norm7, ord1) ≈ 0.65

# =============================================================================
# Analysis model (formal plan §4)
# =============================================================================
# Y ~ norm1 + norm2 + norm3 + norm5 + norm7 + C(bin1) + C(ord1)
#     + I(norm1**2) + norm2:norm3 + norm5:C(bin1) + norm7:C(ord1)
#
# We use statsmodels formula syntax; C(...) is a categorical contrast
# with the first level as reference, so bin1 contributes C(bin1)[T.1]
# and ord1 contributes C(ord1)[T.1] and C(ord1)[T.2].
ANALYSIS_FORMULA = (
    "Y ~ norm1 + norm2 + norm3 + norm5 + norm7 "
    "+ C(bin1) + C(ord1) "
    "+ I(norm1 ** 2) + norm2:norm3 "
    "+ norm5:C(bin1) + norm7:C(ord1)"
)

# Canonical ordering of analysis-model parameters, with the DGM true values
# attached.  The "true" values come directly from equation (3.1) of D&L; the
# OLS fit to a complete dataset will not recover these exactly in a single
# run, but will be unbiased for them in expectation.
TRUE_BETA = {
    "Intercept"              :  0.0,
    "C(bin1)[T.1]"           :  1.0,
    "C(ord1)[T.1]"           : -1.0,
    "C(ord1)[T.2]"           : -2.0,
    "norm1"                  :  1.0,
    "norm2"                  :  1.0,
    "norm3"                  :  1.0,
    "norm5"                  :  1.0,
    "norm7"                  :  1.0,
    "I(norm1 ** 2)"          :  1.0,
    "norm2:norm3"            :  1.0,
    "norm5:C(bin1)[T.1]"     : -3.0,
    "norm7:C(ord1)[T.1]"     : -2.0,
    "norm7:C(ord1)[T.2]"     :  1.0,
}
PARAM_ORDER = list(TRUE_BETA.keys())
N_PARAMS    = len(PARAM_ORDER)

CONFIDENCE_LEVEL = 0.95

# =============================================================================
# Methods
# =============================================================================
METHODS = ["MICE", "GAIN_SI", "MGAIN", "MGAIN_RESAMPLE", "MIDAS"]

# Per-method wall-clock timeout in seconds.  If a single (run, method) call
# exceeds its timeout it is killed and recorded as a failure.  The
# aggregator handles partial-failure cells via n_valid.  Values are ~10x
# the median observed wall time on the smoke test, generous enough to
# absorb noise but tight enough that one rogue run cannot trash the study.
METHOD_TIMEOUTS = {
    "MICE"          :  600,    # observed median ~40s
    "GAIN_SI"       :  500,    # observed median ~26s
    "MGAIN"         : 1800,    # observed median ~125s
    "MGAIN_RESAMPLE":  500,    # similar to GAIN_SI -- one training
    "MIDAS"         :  900,    # observed median ~55s
}

# Use R's mice (method="pmm") via rpy2 if available.  When False, MICE falls
# back to sklearn.IterativeImputer with a warning — fine for smoke tests but
# NOT what the formal plan specifies.
USE_RPY2_MICE = True

MICE_CONFIG = {
    "m"     : N_IMPUTATIONS,
    "maxit" : 5,                  # Deng & Lumley §3.1
    "pmm_k" : 5,                  # default in mice
}

# GAIN — Yoon et al. (2018) defaults, matching the formal plan §3.
# NOTE: 'iterations' is the number of minibatch updates, NOT epochs.
GAIN_CONFIG = {
    "batch_size"  : 128,
    "hint_rate"   : 0.9,
    "alpha"       : 10.0,         # Yoon et al. reference implementation
    "iterations"  : 5000,
    "learning_rate": 1e-3,
    "hidden_dim"  : None,         # None -> set to ncol(X) in the Python module
}

# MIDAS — Lall & Robinson (2022); formal plan §3.
MIDAS_CONFIG = {
    "layer_structure" : [256, 256],
    "learn_rate"      : 1e-4,
    "input_drop"      : 0.5,
    "train_epochs"    : 300,
    "batch_size"      : 256,
    "early_stop_patience": 20,
}

# =============================================================================
# I/O
# =============================================================================
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
RAW_CSV     = os.path.join(RESULTS_DIR, "simulation_results_raw.csv")
TABLE2_CSV  = os.path.join(RESULTS_DIR, "table2.csv")
