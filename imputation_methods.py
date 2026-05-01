"""
imputation_methods.py — Unified imputation dispatch.

Each method exposes a function `impute_<name>(df_miss, m, seed) -> list[pd.DataFrame]`
returning `m` completed copies of `df_miss` with the *same dtypes* (numeric
numerics, categorical bin1/ord1 with the original level sets).

Methods
-------
- MICE     : predictive mean matching via R's `mice` package, called through
             rpy2.  Falls back to sklearn.IterativeImputer if rpy2 or R are
             not available — this fallback is NOT the method in the formal
             plan and should only be used for smoke tests.
- GAIN_SI  : train GAIN once, produce m IDENTICAL completed datasets.
             Var_B ≡ 0 by construction (this is the point of the baseline).
- MGAIN    : train GAIN m times independently (fresh weight init + fresh
             seeds), producing m genuinely different completions.
- MIDAS    : fit a single denoising autoencoder, generate m imputations via
             MC dropout.

Factor handling for the NN methods
----------------------------------
GAIN and MIDAS operate on a purely numeric matrix.  We encode:
    - Y, norm1..norm8 as-is (standardised internally by the NN module)
    - bin1 and ord1 as one-hot blocks over their fixed level sets.
Observed cells are handed to the NN as 0/1 flags; missing cells become NaN
across the whole one-hot block for that row.  On decode we argmax over the
block to recover a level label.  Numeric columns are un-standardised.

Observed entries are restored exactly from the input after decoding, so
round-trip drift cannot contaminate the analysis model.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import warnings
from typing import Callable
import numpy as np
import pandas as pd

from config import GAIN_CONFIG, MIDAS_CONFIG, MICE_CONFIG
from gain import GAIN
from midas_imputer import MIDAS

# Path to the R script that does the actual MICE work.
_THIS_DIR        = os.path.dirname(os.path.abspath(__file__))
_MICE_R_SCRIPT   = os.path.join(_THIS_DIR, "mice_impute.R")
_RSCRIPT_EXE     = shutil.which("Rscript") or "Rscript"


NUMERIC_COLS = ["Y", "norm1", "norm2", "norm3", "norm4",
                "norm5", "norm6", "norm7", "norm8"]
BIN_COL      = "bin1"
ORD_COL      = "ord1"
BIN_LEVELS   = [0, 1]
ORD_LEVELS   = [0, 1, 2]


# =============================================================================
# Encode / decode helpers for GAIN and MIDAS
# =============================================================================

def _encode(df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """
    Encode a (possibly-incomplete) data frame to a numeric matrix for the
    neural imputers.  Returns (X, meta) where X has NaN in missing cells
    and meta carries the information needed by `_decode`.
    """
    num = df[NUMERIC_COLS].to_numpy(dtype=float)  # NaN preserved

    def _one_hot(col: pd.Series, levels) -> np.ndarray:
        oh = np.full((len(col), len(levels)), np.nan, dtype=float)
        idx = col.astype("object").map({lv: i for i, lv in enumerate(levels)})
        for i, lv in enumerate(levels):
            oh[:, i] = 0.0
        obs = idx.notna().to_numpy()
        # Zero the observed rows then set the chosen column to 1.
        oh[~obs, :] = np.nan
        oh[obs, :] = 0.0
        rows = np.arange(len(col))[obs]
        cols = idx.dropna().astype(int).to_numpy()
        oh[rows, cols] = 1.0
        return oh

    bin_oh = _one_hot(df[BIN_COL], BIN_LEVELS)
    ord_oh = _one_hot(df[ORD_COL], ORD_LEVELS)
    X = np.concatenate([num, bin_oh, ord_oh], axis=1)

    meta = dict(
        n_num   = num.shape[1],
        bin_slc = slice(num.shape[1],               num.shape[1] + len(BIN_LEVELS)),
        ord_slc = slice(num.shape[1] + len(BIN_LEVELS),
                        num.shape[1] + len(BIN_LEVELS) + len(ORD_LEVELS)),
    )
    return X, meta


def _decode(X_imp: np.ndarray, meta: dict, template: pd.DataFrame) -> pd.DataFrame:
    """
    Invert `_encode` on a completed numeric matrix.  Only missing cells in
    `template` are overwritten; observed cells are kept exactly.
    """
    out = template.copy()

    # Numeric
    num_block = X_imp[:, : meta["n_num"]]
    for j, col in enumerate(NUMERIC_COLS):
        na = out[col].isna().to_numpy()
        if na.any():
            filled = out[col].to_numpy(dtype=float)
            filled[na] = num_block[na, j]
            out[col] = filled

    # Binary
    bin_block = X_imp[:, meta["bin_slc"]]
    bin_pred  = np.array(BIN_LEVELS)[np.argmax(bin_block, axis=1)]
    na_bin = out[BIN_COL].isna().to_numpy()
    if na_bin.any():
        current = out[BIN_COL].astype("object")
        current[na_bin] = bin_pred[na_bin]
        out[BIN_COL] = pd.Categorical(current.astype(int),
                                      categories=BIN_LEVELS, ordered=False)

    # Ordinal
    ord_block = X_imp[:, meta["ord_slc"]]
    ord_pred  = np.array(ORD_LEVELS)[np.argmax(ord_block, axis=1)]
    na_ord = out[ORD_COL].isna().to_numpy()
    if na_ord.any():
        current = out[ORD_COL].astype("object")
        current[na_ord] = ord_pred[na_ord]
        out[ORD_COL] = pd.Categorical(current.astype(int),
                                      categories=ORD_LEVELS, ordered=True)

    return out


# =============================================================================
# MICE via rpy2 (primary) or sklearn (fallback)
# =============================================================================

# =============================================================================
# MICE via subprocess to R's mice package
# =============================================================================
# We shell out to Rscript rather than using rpy2 because rpy2 has fragile
# version-coupling problems on Windows.  The R script is mice_impute.R in
# the same directory as this file.

def _check_rscript_available() -> None:
    """Raise a clear error if Rscript or the R script cannot be found."""
    if not os.path.exists(_MICE_R_SCRIPT):
        raise FileNotFoundError(
            f"Cannot find {_MICE_R_SCRIPT}.  The R script must live next "
            f"to imputation_methods.py."
        )
    if shutil.which("Rscript") is None:
        raise FileNotFoundError(
            "Rscript is not on PATH.  Install R and ensure "
            "<R install>/bin/x64 is on PATH so that `Rscript --version` works."
        )


def impute_mice(df_miss: pd.DataFrame, m: int, seed: int) -> list[pd.DataFrame]:
    """
    MICE via R's mice package, called as a subprocess.

    Writes the incomplete dataset to a temp CSV, calls Rscript on
    mice_impute.R, reads the m completed CSVs back as pandas frames
    with the original dtypes restored.
    """
    _check_rscript_available()

    # Use a per-call temp directory so concurrent runs (if we ever add
    # multiprocessing) don't stomp on each other.
    with tempfile.TemporaryDirectory(prefix="mice_") as tmpdir:
        in_csv  = os.path.join(tmpdir, "in.csv")
        out_pre = os.path.join(tmpdir, "out")

        # Write to CSV with bin1/ord1 as integers (mice_impute.R re-casts
        # them to factors with the right levels).
        df_to_write = df_miss.copy()
        df_to_write["bin1"] = df_to_write["bin1"].astype("Int64")
        df_to_write["ord1"] = df_to_write["ord1"].astype("Int64")
        df_to_write.to_csv(in_csv, index=False)

        cmd = [
            _RSCRIPT_EXE, _MICE_R_SCRIPT,
            in_csv, out_pre,
            str(int(m)),
            str(int(MICE_CONFIG["maxit"])),
            str(int(seed)),
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=600,            # 10 min hard cap on a single MICE call
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "mice_impute.R failed (returncode "
                f"{proc.returncode}):\nSTDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )

        out: list[pd.DataFrame] = []
        for l in range(1, m + 1):
            csv_path = f"{out_pre}_{l}.csv"
            d = pd.read_csv(csv_path)
            d["bin1"] = pd.Categorical(d["bin1"].astype(int),
                                       categories=BIN_LEVELS, ordered=False)
            d["ord1"] = pd.Categorical(d["ord1"].astype(int),
                                       categories=ORD_LEVELS, ordered=True)
            out.append(d)
    return out


# =============================================================================
# GAIN (single imputation) — train once, return m identical copies
# =============================================================================

def impute_gain_si(df_miss: pd.DataFrame, m: int, seed: int) -> list[pd.DataFrame]:
    X, meta = _encode(df_miss)
    gain = GAIN(
        batch_size    = GAIN_CONFIG["batch_size"],
        hint_rate     = GAIN_CONFIG["hint_rate"],
        alpha         = GAIN_CONFIG["alpha"],
        iterations    = GAIN_CONFIG["iterations"],
        learning_rate = GAIN_CONFIG["learning_rate"],
        hidden_dim    = GAIN_CONFIG["hidden_dim"],
        seed          = int(seed),
    )
    X_imp = gain.fit_transform(X, n_imputations=1)
    one = _decode(X_imp, meta, df_miss)
    return [one.copy() for _ in range(m)]


# =============================================================================
# MGAIN — m independent GAIN trainings
# =============================================================================

def impute_mgain(df_miss: pd.DataFrame, m: int, seed: int) -> list[pd.DataFrame]:
    X, meta = _encode(df_miss)
    out: list[pd.DataFrame] = []
    for l in range(m):
        gain = GAIN(
            batch_size    = GAIN_CONFIG["batch_size"],
            hint_rate     = GAIN_CONFIG["hint_rate"],
            alpha         = GAIN_CONFIG["alpha"],
            iterations    = GAIN_CONFIG["iterations"],
            learning_rate = GAIN_CONFIG["learning_rate"],
            hidden_dim    = GAIN_CONFIG["hidden_dim"],
            seed          = int(seed) + 1000 * (l + 1),
        )
        X_imp = gain.fit_transform(X, n_imputations=1)
        out.append(_decode(X_imp, meta, df_miss))
    return out

# =============================================================================
# MGAIN_RESAMPLE — train one GAIN, sample m completions with fresh noise
# =============================================================================
# This is the "right" multiple imputation strategy for GAIN, per Yoon et al.
# (2018): train the generator once, then call it m times with independently
# sampled noise vectors Z.  Var_B comes from the stochastic noise injection
# at inference time, not from training-noise differences across networks.
# Compare with:
#   GAIN_SI -- one training, one forward pass, m identical copies (Var_B = 0)
#   MGAIN   -- m independent trainings (Var_B inflated by training variance)

def impute_mgain_resample(df_miss: pd.DataFrame, m: int, seed: int) -> list[pd.DataFrame]:
    X, meta = _encode(df_miss)
    gain = GAIN(
        batch_size    = GAIN_CONFIG["batch_size"],
        hint_rate     = GAIN_CONFIG["hint_rate"],
        alpha         = GAIN_CONFIG["alpha"],
        iterations    = GAIN_CONFIG["iterations"],
        learning_rate = GAIN_CONFIG["learning_rate"],
        hidden_dim    = GAIN_CONFIG["hidden_dim"],
        seed          = int(seed),
    )
    X_imps = gain.fit_transform(X, n_imputations=m)
    if not isinstance(X_imps, list):
        X_imps = [X_imps]
    return [_decode(Xi, meta, df_miss) for Xi in X_imps]

# =============================================================================
# MIDAS — single fit, m MC-dropout draws
# =============================================================================

def impute_midas(df_miss: pd.DataFrame, m: int, seed: int) -> list[pd.DataFrame]:
    X, meta = _encode(df_miss)
    midas = MIDAS(
        layer_structure     = MIDAS_CONFIG["layer_structure"],
        learn_rate          = MIDAS_CONFIG["learn_rate"],
        input_drop          = MIDAS_CONFIG["input_drop"],
        train_epochs        = MIDAS_CONFIG["train_epochs"],
        batch_size          = MIDAS_CONFIG["batch_size"],
        early_stop_patience = MIDAS_CONFIG["early_stop_patience"],
        seed                = int(seed),
    )
    X_imps = midas.fit_transform(X, n_imputations=m)
    return [_decode(Xi, meta, df_miss) for Xi in X_imps]


# =============================================================================
# Dispatcher
# =============================================================================

IMPUTERS: dict[str, Callable] = {
    "MICE"           : impute_mice,
    "GAIN_SI"        : impute_gain_si,
    "MGAIN"          : impute_mgain,
    "MGAIN_RESAMPLE" : impute_mgain_resample,
    "MIDAS"          : impute_midas,
}


def runImp(method: str, df_miss: pd.DataFrame, m: int, seed: int) -> list[pd.DataFrame]:
    if method not in IMPUTERS:
        raise ValueError(f"Unknown method {method!r}; choose from {list(IMPUTERS)}")
    return IMPUTERS[method](df_miss, m, seed)
