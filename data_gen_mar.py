"""
data_generation.py — Data generation and MAR missingness for the
                     Deng & Lumley (2024) simulation design.

The generator reproduces equation (3.1) of Deng & Lumley:

    Y_i = norm1 + norm2 + norm3 + norm5 + norm7
          + (bin1 == 1) - (ord1 == 1) - 2*(ord1 == 2)
          + norm1^2 + norm2*norm3
          - 3*norm5*(bin1 == 1)
          - 2*norm7*(ord1 == 1) + norm7*(ord1 == 2)
          + epsilon_i,      epsilon_i ~ N(0, 1)

with the correlation structure specified in §3.1:
    - norm1..norm4 pairwise 0.5
    - norm5, norm6  corr 0.7, each cor(·, bin1) ≈ 0.55
    - norm7, norm8  corr 0.7, each cor(·, ord1) ≈ 0.65
    - bin1 ~ Bernoulli(0.5);   ord1 ~ Binomial(2, 0.5)

The continuous/discrete correlations are realised through a Gaussian
copula: each factor has an underlying latent normal, jointly normal
with its two associated norms, then thresholded.  The latent
correlations are set in config.py so that the observed point-biserial
and polyserial correlations match the D&L targets.

The missingness mechanism is the *tertile* MAR of D&L §3.1 (not a
logistic calibration):

    P(R_i = 0 | Z_i) =  0.6   if Z_i is in the top third of Z
                        0.1   if Z_i is in the middle third of Z
                        0.6   if Z_i is in the bottom third of Z

with driver Z = Y + norm4 for (norm1, norm2, norm3);
                Y + norm6 for (norm5, bin1);
                Y + norm8 for (norm7, ord1).

Y, norm4, norm6 and norm8 are always fully observed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

from config import (
    N_OBS,
    RHO_NORM14, RHO_56, RHO_56_LATENT, RHO_78, RHO_78_LATENT,
    P_MISS_TAIL, P_MISS_MID,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _equicorr(d: int, rho: float) -> np.ndarray:
    """Equicorrelation matrix of size d x d."""
    M = np.full((d, d), rho, dtype=float)
    np.fill_diagonal(M, 1.0)
    return M


def _assert_psd(M: np.ndarray, name: str) -> None:
    eig = np.linalg.eigvalsh(M)
    if eig.min() < -1e-10:
        raise ValueError(
            f"{name} is not positive semi-definite (min eig = {eig.min():.3e})"
        )


# -----------------------------------------------------------------------------
# Data generator
# -----------------------------------------------------------------------------

# Ordinal thresholds realise Bin(2, 0.5) via a single latent N(0,1):
#   ord1 = 0 if z < -0.6745;  1 if -0.6745 <= z <= 0.6745;  2 if z > 0.6745
# because P(N(0,1) < -0.6745) = P(N(0,1) > 0.6745) = 0.25 = P(Bin(2,0.5)=0).
_ORD_TAU = _norm.ppf([0.25, 0.75])   # approx [-0.6745, +0.6745]


def generate_complete_data(n: int = N_OBS, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """
    Generate a single complete dataset of size `n`.

    Returns a DataFrame with columns
        Y, norm1, norm2, ..., norm8, bin1, ord1
    where bin1 and ord1 are pandas Categorical (ordered=False for bin1;
    ordered=True for ord1) with explicit levels so that downstream
    imputation and analysis produce a consistent design matrix even if
    one level happens to be absent in a subset of rows.
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- Block 1: norm1..norm4, pairwise correlation 0.5 --------------------
    S1 = _equicorr(4, RHO_NORM14)
    _assert_psd(S1, "Sigma_norm14")
    N14 = rng.multivariate_normal(np.zeros(4), S1, size=n)
    norm1, norm2, norm3, norm4 = N14.T

    # --- Block 2: (norm5, norm6, z_bin) --------------------------------------
    r56, rl56 = RHO_56, RHO_56_LATENT
    S2 = np.array([
        [1.0,  r56,  rl56],
        [r56,  1.0,  rl56],
        [rl56, rl56, 1.0 ],
    ])
    _assert_psd(S2, "Sigma_56bin")
    B2 = rng.multivariate_normal(np.zeros(3), S2, size=n)
    norm5, norm6, z_bin = B2.T
    bin1 = (z_bin > 0).astype(np.int8)           # 0 or 1

    # --- Block 3: (norm7, norm8, z_ord) --------------------------------------
    r78, rl78 = RHO_78, RHO_78_LATENT
    S3 = np.array([
        [1.0,  r78,  rl78],
        [r78,  1.0,  rl78],
        [rl78, rl78, 1.0 ],
    ])
    _assert_psd(S3, "Sigma_78ord")
    B3 = rng.multivariate_normal(np.zeros(3), S3, size=n)
    norm7, norm8, z_ord = B3.T
    ord1 = (z_ord > _ORD_TAU[0]).astype(np.int8) + (z_ord > _ORD_TAU[1]).astype(np.int8)

    # --- Outcome Y (equation 3.1) -------------------------------------------
    eps = rng.standard_normal(n)
    Y = (
        norm1 + norm2 + norm3 + norm5 + norm7
        + (bin1 == 1).astype(float)
        - (ord1 == 1).astype(float)
        - 2.0 * (ord1 == 2).astype(float)
        + norm1 ** 2
        + norm2 * norm3
        - 3.0 * norm5 * (bin1 == 1).astype(float)
        - 2.0 * norm7 * (ord1 == 1).astype(float)
        + norm7 * (ord1 == 2).astype(float)
        + eps
    )

    df = pd.DataFrame({
        "Y"    : Y,
        "norm1": norm1, "norm2": norm2, "norm3": norm3, "norm4": norm4,
        "norm5": norm5, "norm6": norm6, "norm7": norm7, "norm8": norm8,
        "bin1" : pd.Categorical(bin1, categories=[0, 1],    ordered=False),
        "ord1" : pd.Categorical(ord1, categories=[0, 1, 2], ordered=True),
    })
    return df


# -----------------------------------------------------------------------------
# Missingness mechanism
# -----------------------------------------------------------------------------

def _tertile_mask(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Return a boolean mask (True = to be set missing) for the tertile
    mechanism of Deng & Lumley §3.1.
    """
    q1, q2 = np.quantile(z, [1.0 / 3.0, 2.0 / 3.0])
    in_mid = (z > q1) & (z < q2)
    prob = np.where(in_mid, P_MISS_MID, P_MISS_TAIL)
    return rng.random(len(z)) < prob


def impose_mar(df: pd.DataFrame, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """
    Impose the D&L tertile MAR on a complete dataset.  Returns a copy; the
    input is not modified.  Missing entries are NaN for numerics and NaN
    for categoricals (pandas preserves the dtype).
    """
    if rng is None:
        rng = np.random.default_rng()

    out = df.copy()
    z_a = (out["Y"] + out["norm4"]).to_numpy()   # drives norm1, norm2, norm3
    z_b = (out["Y"] + out["norm6"]).to_numpy()   # drives norm5, bin1
    z_c = (out["Y"] + out["norm8"]).to_numpy()   # drives norm7, ord1

    # Independent draws per variable, conditional on the shared driver Z.
    for v in ("norm1", "norm2", "norm3"):
        mask = _tertile_mask(z_a, rng)
        out.loc[mask, v] = np.nan

    for v in ("norm5", "bin1"):
        mask = _tertile_mask(z_b, rng)
        out.loc[mask, v] = np.nan

    for v in ("norm7", "ord1"):
        mask = _tertile_mask(z_c, rng)
        out.loc[mask, v] = np.nan

    return out


def missing_rates(df: pd.DataFrame) -> pd.Series:
    """Fraction missing per column — diagnostic."""
    return df.isna().mean()


# -----------------------------------------------------------------------------
# Sanity check — run directly to verify correlation construction
# -----------------------------------------------------------------------------

def check_correlations(n: int = 200_000, seed: int = 0) -> pd.Series:
    """Verify that the Gaussian-copula latent correlations produce
    the target observed correlations.  Should be run once at setup."""
    rng = np.random.default_rng(seed)
    df = generate_complete_data(n=n, rng=rng)
    bin_num = df["bin1"].astype(int).to_numpy()
    ord_num = df["ord1"].astype(int).to_numpy()
    return pd.Series({
        "cor(norm1,norm2)"  : np.corrcoef(df["norm1"], df["norm2"])[0, 1],
        "cor(norm5,norm6)"  : np.corrcoef(df["norm5"], df["norm6"])[0, 1],
        "cor(norm7,norm8)"  : np.corrcoef(df["norm7"], df["norm8"])[0, 1],
        "cor(norm5,bin1)"   : np.corrcoef(df["norm5"], bin_num)[0, 1],
        "cor(norm6,bin1)"   : np.corrcoef(df["norm6"], bin_num)[0, 1],
        "cor(norm7,ord1)"   : np.corrcoef(df["norm7"], ord_num)[0, 1],
        "cor(norm8,ord1)"   : np.corrcoef(df["norm8"], ord_num)[0, 1],
        "P(bin1=1)"         : bin_num.mean(),
        "E[ord1]"           : ord_num.mean(),
        "Var[ord1]"         : ord_num.var(),
    })


if __name__ == "__main__":
    print("Correlation sanity check (n = 200,000):")
    print(check_correlations().round(3).to_string())
    print()
    rng = np.random.default_rng(0)
    df = generate_complete_data(n=10_000, rng=rng)
    dm = impose_mar(df, rng=rng)
    print("Marginal missing rates after MAR tertile mechanism:")
    print(missing_rates(dm).round(3).to_string())
