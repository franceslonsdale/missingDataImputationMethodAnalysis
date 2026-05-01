"""
MIDAS: Multiple Imputation with Denoising Autoencoders.
Based on Lall & Robinson (2022), "The MIDAS Touch".

Changes from the original uploaded version:
  - **Bug fix:** MC-dropout at inference time.  The previous version called
    `model(X_t, apply_input_drop=False)` inside a dead `with torch.no_grad()`
    block and produced `n_imputations` IDENTICAL draws.  This one keeps the
    model in train() mode at generation time AND enables input dropout, so
    each of the m imputations is a genuinely different stochastic draw.
  - Uses the Lall & Robinson hyperparameters from the formal plan
    (layers [256, 256], lr = 1e-4, input_drop = 0.5, up to 300 epochs).
  - Early stopping on a held-out 10% mask of observed entries.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class _DAE(nn.Module):
    """Denoising autoencoder with input-level dropout corruption."""

    def __init__(self, input_dim: int, layer_structure, input_drop: float):
        super().__init__()
        self.input_drop = nn.Dropout(p=input_drop)

        enc, prev = [], input_dim
        for h in layer_structure:
            enc += [nn.Linear(prev, h), nn.Tanh(), nn.Dropout(0.5)]
            prev = h
        self.encoder = nn.Sequential(*enc)

        dec = []
        for h in list(reversed(layer_structure))[1:]:
            dec += [nn.Linear(prev, h), nn.Tanh(), nn.Dropout(0.5)]
            prev = h
        dec += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor, apply_input_drop: bool) -> torch.Tensor:
        if apply_input_drop:
            x = self.input_drop(x)
        return self.decoder(self.encoder(x))


class MIDAS:
    """
    MIDAS imputer — overcomplete denoising autoencoder with MC dropout.

    Parameters
    ----------
    layer_structure : widths of the encoder hidden layers (decoder mirrors)
    learn_rate      : Adam learning rate
    input_drop      : input-corruption dropout probability
    train_epochs    : maximum training epochs
    batch_size      : minibatch size
    early_stop_patience : stop if val loss does not improve for this many epochs
    seed            : RNG seed
    """

    def __init__(
        self,
        layer_structure=(256, 256),
        learn_rate: float = 1e-4,
        input_drop: float = 0.5,
        train_epochs: int = 300,
        batch_size: int = 256,
        early_stop_patience: int = 20,
        seed: int | None = None,
    ):
        self.layer_structure = list(layer_structure)
        self.lr              = float(learn_rate)
        self.input_drop      = float(input_drop)
        self.train_epochs    = int(train_epochs)
        self.batch_size      = int(batch_size)
        self.patience        = int(early_stop_patience)
        self.seed            = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------

    def fit_transform(self, X_miss: np.ndarray, n_imputations: int = 5) -> list[np.ndarray]:
        if self.seed is not None:
            np.random.seed(int(self.seed))
            torch.manual_seed(int(self.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.seed))

        X_miss = np.asarray(X_miss, dtype=np.float64)
        n, dim = X_miss.shape
        M = (~np.isnan(X_miss)).astype(np.float32)

        # Standardise using observed-only moments.
        mu  = np.nanmean(X_miss, axis=0)
        sd  = np.nanstd(X_miss,  axis=0)
        sd  = np.where(sd < 1e-8, 1.0, sd)
        X0  = np.where(np.isnan(X_miss), mu, X_miss)
        X_s = ((X0 - mu) / sd).astype(np.float32)

        # Held-out validation mask: 10% of observed entries (M==1) are
        # hidden from the training loss, used only to trigger early stopping.
        val_mask = np.zeros_like(M)
        obs_idx = np.argwhere(M == 1.0)
        n_val = int(0.10 * len(obs_idx))
        sel = np.random.choice(len(obs_idx), size=n_val, replace=False)
        val_mask[obs_idx[sel, 0], obs_idx[sel, 1]] = 1.0
        train_M = M * (1.0 - val_mask)   # 1 for observed-and-used-for-training

        X_t   = torch.tensor(X_s,      device=self.device)
        Mtr_t = torch.tensor(train_M,  device=self.device)
        Mvl_t = torch.tensor(val_mask, device=self.device)

        model = _DAE(dim, self.layer_structure, self.input_drop).to(self.device)
        opt   = optim.Adam(model.parameters(), lr=self.lr)

        ds = TensorDataset(X_t, Mtr_t, Mvl_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        best_val = float("inf")
        since_improve = 0
        best_state = None

        for epoch in range(self.train_epochs):
            model.train()
            for Xb, Mb_tr, _ in dl:
                opt.zero_grad()
                Xr = model(Xb, apply_input_drop=True)
                loss = torch.sum(Mb_tr * (Xb - Xr) ** 2) / (torch.sum(Mb_tr) + 1e-8)
                loss.backward()
                opt.step()

            # Validation — use eval-mode (no dropout) for a clean signal.
            model.eval()
            with torch.no_grad():
                Xr_full = model(X_t, apply_input_drop=False)
                val_loss = (
                    torch.sum(Mvl_t * (X_t - Xr_full) ** 2) /
                    (torch.sum(Mvl_t) + 1e-8)
                ).item()

            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                since_improve = 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                since_improve += 1
                if since_improve >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # -------------------------------------------------------------------
        # MC-dropout inference: train() mode + apply_input_drop=True +
        # independent forward passes => genuinely different imputations.
        # -------------------------------------------------------------------
        model.train()
        out: list[np.ndarray] = []
        M_t = torch.tensor(M, device=self.device)
        for _ in range(int(n_imputations)):
            with torch.no_grad():
                Xr = model(X_t, apply_input_drop=True)
            Xi = (M_t * X_t + (1.0 - M_t) * Xr).cpu().numpy()
            Xi = Xi * sd + mu
            # Restore observed entries exactly.
            Xi[M == 1.0] = X_miss[M == 1.0]
            out.append(Xi.astype(np.float64))
        return out
