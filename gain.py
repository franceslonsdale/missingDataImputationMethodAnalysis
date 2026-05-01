"""
GAIN: Generative Adversarial Imputation Nets (Yoon, Jordon, van der Schaar, ICML 2018).

PyTorch implementation.

Changes from the original uploaded version:
  - Training loop is *iteration-based* (a fixed number of minibatch updates),
    matching Yoon et al.'s reference TF implementation and the formal plan.
    Previously the loop ran `epochs` full passes over the dataloader, which
    is a different amount of work and a different optimiser trajectory.
  - Default alpha = 10  (was 100; Yoon et al. use 10 for tabular data).
  - `fit_transform` can return `n_imputations` draws by re-sampling the
    noise and re-running the generator forward pass.  This is what MGAIN
    uses when it calls GAIN `m` times with distinct seeds (fresh weights).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class _Generator(nn.Module):
    def __init__(self, dim, h_dim, dropout_p: float = 0.5):
        super().__init__()
        # Dropout after each hidden ReLU provides the stochasticity used
        # for MC-dropout inference (see fit_transform's generation loop).
        self.net = nn.Sequential(
            nn.Linear(dim * 2, h_dim), nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(h_dim,   h_dim), nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(h_dim,   dim  ), nn.Sigmoid(),
        )

    def forward(self, x, m):
        return self.net(torch.cat([x, m], dim=1))


class _Discriminator(nn.Module):
    def __init__(self, dim, h_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, h_dim), nn.ReLU(),
            nn.Linear(h_dim,   h_dim), nn.ReLU(),
            nn.Linear(h_dim,   dim  ), nn.Sigmoid(),
        )

    def forward(self, x, h):
        return self.net(torch.cat([x, h], dim=1))


class GAIN:
    """
    GAIN imputer.  See Yoon, Jordon & van der Schaar (ICML 2018).

    Parameters
    ----------
    batch_size    : minibatch size
    hint_rate     : probability of revealing mask bit to discriminator
    alpha         : weight on the reconstruction loss (Yoon et al. use 10)
    iterations    : total number of generator/discriminator updates
    learning_rate : Adam learning rate for both G and D
    hidden_dim    : size of hidden layers; None means `dim`
    seed          : RNG seed (controls weight init, noise, batch sampling)
    """

    def __init__(
        self,
        batch_size: int   = 128,
        hint_rate:  float = 0.9,
        alpha:      float = 10.0,
        iterations: int   = 5000,
        learning_rate: float = 1e-3,
        hidden_dim: int | None = None,
        seed:       int | None = None,
    ):
        self.batch_size = int(batch_size)
        self.hint_rate  = float(hint_rate)
        self.alpha      = float(alpha)
        self.iterations = int(iterations)
        self.lr         = float(learning_rate)
        self.h_dim_arg  = hidden_dim
        self.seed       = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------

    def fit_transform(
        self,
        X_miss: np.ndarray,
        n_imputations: int = 1,
    ) -> np.ndarray | list[np.ndarray]:
        """
        Train GAIN on `X_miss` (with NaN = missing) and return imputation(s).

        If n_imputations == 1 returns a single ndarray; otherwise returns
        a list of `n_imputations` ndarrays that differ only in the
        resampled noise at the final forward pass (for MGAIN-style use,
        call the class `m` times with different `seed`s instead).
        """
        if self.seed is not None:
            np.random.seed(int(self.seed))
            torch.manual_seed(int(self.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.seed))

        X_miss = np.asarray(X_miss, dtype=np.float64)
        n, dim = X_miss.shape
        h_dim  = int(self.h_dim_arg) if self.h_dim_arg is not None else dim

        # Mask: 1 = observed, 0 = missing
        M = (~np.isnan(X_miss)).astype(np.float32)

        # Min-max normalise to [0, 1] so the sigmoid output is appropriate.
        mins  = np.nanmin(X_miss, axis=0)
        maxs  = np.nanmax(X_miss, axis=0)
        rng_  = np.where((maxs - mins) < 1e-6, 1.0, maxs - mins)
        X_std = (X_miss - mins) / rng_
        X_std = np.nan_to_num(X_std, nan=0.0).astype(np.float32)

        X_t = torch.tensor(X_std, device=self.device)
        M_t = torch.tensor(M,     device=self.device)

        G = _Generator(dim, h_dim).to(self.device)
        D = _Discriminator(dim, h_dim).to(self.device)
        opt_G = optim.Adam(G.parameters(), lr=self.lr)
        opt_D = optim.Adam(D.parameters(), lr=self.lr)

        # --- iteration-based training loop --------------------------------
        G.train()
        D.train()
        bs = min(self.batch_size, n)
        for it in range(self.iterations):
            idx = np.random.choice(n, size=bs, replace=False)
            X_mb = X_t[idx]
            M_mb = M_t[idx]

            # Hint: H = B*M + 0.5*(1 - B), B ~ Bernoulli(hint_rate)
            B_mb = torch.bernoulli(torch.full((bs, dim), self.hint_rate,
                                              device=self.device))
            H_mb = B_mb * M_mb + 0.5 * (1.0 - B_mb)

            # Noise injection on missing entries -- Yoon et al.'s Z ~ U(0, 0.01).
            # Stochasticity at inference comes from MC dropout in the
            # generator, not from Z.
            Z_mb = torch.rand(bs, dim, device=self.device) * 0.01
            X_input = M_mb * X_mb + (1.0 - M_mb) * Z_mb

            # ---- Discriminator update ----
            G_sample = G(X_input, M_mb)
            X_hat    = M_mb * X_mb + (1.0 - M_mb) * G_sample
            D_prob   = D(X_hat.detach(), H_mb)
            D_loss   = -torch.mean(
                M_mb * torch.log(D_prob + 1e-8) +
                (1.0 - M_mb) * torch.log(1.0 - D_prob + 1e-8)
            )
            opt_D.zero_grad(); D_loss.backward(); opt_D.step()

            # ---- Generator update ----
            G_sample = G(X_input, M_mb)
            X_hat    = M_mb * X_mb + (1.0 - M_mb) * G_sample
            D_prob   = D(X_hat, H_mb)
            G_loss_adv = -torch.mean((1.0 - M_mb) * torch.log(D_prob + 1e-8))
            G_loss_mse = (
                torch.mean(M_mb * (X_mb - G_sample) ** 2)
                / (torch.mean(M_mb) + 1e-8)
            )
            G_loss = G_loss_adv + self.alpha * G_loss_mse
            opt_G.zero_grad(); G_loss.backward(); opt_G.step()

        # --- generate imputation(s) via MC dropout -----------------------
        # Keep the generator in train() mode so Dropout layers fire at
        # inference, giving genuinely different draws across calls.  This
        # is the same trick MIDAS uses and is the standard Bayesian-NN
        # approximation (Gal & Ghahramani 2016).  The Z noise is back to
        # Yoon et al.'s 0.01 scale, so variance between draws comes
        # entirely from dropout masks.
        G.train()
        outs: list[np.ndarray] = []
        with torch.no_grad():
            for _ in range(max(1, int(n_imputations))):
                Z_full = torch.rand(n, dim, device=self.device) * 0.01
                X_in   = M_t * X_t + (1.0 - M_t) * Z_full
                G_full = G(X_in, M_t).cpu().numpy()
                imp = X_std.copy()
                miss_mask = (M == 0.0)
                imp[miss_mask] = G_full[miss_mask]
                imp = imp * rng_ + mins
                imp[~miss_mask] = X_miss[~miss_mask]
                outs.append(imp.astype(np.float64))

        return outs[0] if n_imputations == 1 else outs
