"""
Stage 3: Writer Enrollment

WriterEnrollment encodes R genuine reference signatures for a writer and
computes three cached statistics (no gradient updates):

  F_refs      : (R, d)      — GAP patch features per reference
  mu          : (d,)        — writer prototype (mean of F_refs)
  sigma_tilde : (d, d) or (d,) — regularized covariance

Regularization strategy (per spec):
  R <= 10          : isotropic    Sigma_tilde = Sigma_hat + eps * I
  10 < R <= 30     : Ledoit-Wolf  Sigma_tilde = (1-rho)*Sigma_hat + rho*mu_LW*I
  R > 30           : diagonal     Sigma_tilde = diag(sigma_j^2 + eps)
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class WriterEnrollment:
    """
    Manages a registry of enrolled writers.

    Each entry in self.registry[writer_id] contains:
        'F_refs'      : np.ndarray (R, d)          individual reference GAP features
        'mu'          : np.ndarray (d,)             writer prototype
        'sigma_tilde' : np.ndarray (d, d) or (d,)  regularized covariance
        'sigma_type'  : str  'full' | 'diag'
        'R'           : int  number of references
    """

    def __init__(self, encoder: nn.Module, embed_dim: int = 768, epsilon: float = 0.05):
        """
        Args:
            encoder   : fine-tuned ViTEncoder, already in eval mode on the target device.
            embed_dim : encoder hidden dimension d.
            epsilon   : regularisation constant for isotropic and diagonal branches.
                        Should be in [0.01, 0.1] as per spec.
        """
        assert 0.01 <= epsilon <= 0.1, f"epsilon must be in [0.01, 0.1], got {epsilon}"
        self.encoder   = encoder
        self.embed_dim = embed_dim
        self.epsilon   = epsilon
        self.registry: Dict[str, dict] = {}

    # ── Encoding ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _extract_gap_features(
        self,
        images: torch.Tensor,
        device: torch.device,
    ) -> np.ndarray:
        """
        Run the encoder and return GAP patch features.

        Args:
            images : (R, C, H, W) float32 tensor, already normalised to [-1, 1].
            device : device the encoder lives on.

        Returns:
            F_refs : (R, d) float32 numpy array.
                     F_{r_i} = (1/L) * sum_l  Z_{r_i}^{(l)}
        """
        self.encoder.eval()
        images = images.to(device)

        tokens   = self.encoder(images, return_all_tokens=True)  # (R, L+1, d)
        Z_patch  = tokens[:, 1:, :]                              # (R, L, d)  — exclude CLS (index 0)
        F_refs   = Z_patch.mean(dim=1)                           # (R, d)     — GAP over L

        return F_refs.cpu().float().numpy()

    # ── Covariance ───────────────────────────────────────────────────────────

    def _regularized_covariance(
        self,
        F_refs: np.ndarray,
    ) -> Tuple[np.ndarray, str]:
        """
        Compute the regularised covariance Sigma_tilde from (R, d) features.

        Returns:
            sigma_tilde : (d, d) float32 for 'full';  (d,) float32 for 'diag'.
            sigma_type  : 'full' or 'diag'.
        """
        R, d = F_refs.shape
        mu      = F_refs.mean(axis=0)           # (d,)
        centered = F_refs - mu[None, :]          # (R, d)

        # ── Sample covariance Sigma_hat (needed for isotropic & LW branches) ─
        if R > 1:
            sigma_hat = (centered.T @ centered) / (R - 1)   # (d, d)
        else:
            sigma_hat = np.zeros((d, d), dtype=np.float64)

        # ── Branch selection ──────────────────────────────────────────────────
        if R <= 10:
            # Isotropic regularization: Sigma_tilde = Sigma_hat + eps * I
            sigma_tilde = sigma_hat + self.epsilon * np.eye(d)
            return sigma_tilde.astype(np.float32), 'full'

        elif R <= 30:
            # Ledoit-Wolf analytical shrinkage (sklearn handles rho and mu_LW)
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf(assume_centered=False, store_precision=False)
            lw.fit(F_refs)                                   # fits on (R, d)
            return lw.covariance_.astype(np.float32), 'full'

        else:
            # Diagonal approximation: diag(sigma_j^2 + eps)
            # ddof=1 for unbiased sample variance
            var         = F_refs.var(axis=0, ddof=1) if R > 1 else np.zeros(d)
            sigma_tilde = (var + self.epsilon).astype(np.float32)  # (d,)
            return sigma_tilde, 'diag'

    # ── Public API ────────────────────────────────────────────────────────────

    def enroll_writer(
        self,
        writer_id: str,
        reference_images: torch.Tensor,
        device: torch.device,
    ) -> None:
        """
        Enroll a single writer (no gradient updates).

        Args:
            writer_id        : unique string identifier (e.g. 'u0001').
            reference_images : (R, C, H, W) float32 tensor, normalised to [-1, 1].
            device           : device the encoder runs on.
        """
        R = reference_images.shape[0]

        # 1. Extract GAP features: F_{r_i} = (1/L) sum_l Z_{r_i}^{(l)}
        F_refs = self._extract_gap_features(reference_images, device)  # (R, d)

        # 2. Writer prototype: mu = (1/R) sum_i F_{r_i}
        mu = F_refs.mean(axis=0).astype(np.float32)                    # (d,)

        # 3. Regularised covariance
        sigma_tilde, sigma_type = self._regularized_covariance(F_refs)

        self.registry[writer_id] = {
            'F_refs'      : F_refs,          # (R, d)
            'mu'          : mu,              # (d,)
            'sigma_tilde' : sigma_tilde,     # (d, d) or (d,)
            'sigma_type'  : sigma_type,      # 'full' | 'diag'
            'R'           : R,
        }

    def get(self, writer_id: str) -> Optional[dict]:
        """Return the enrollment record for a writer, or None if not enrolled."""
        return self.registry.get(writer_id)

    def __len__(self) -> int:
        return len(self.registry)

    def __contains__(self, writer_id: str) -> bool:
        return writer_id in self.registry

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Save the registry to a compressed .npz file.

        Uses numpy compression — effective because sigma_tilde matrices
        are low-rank and compress well.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Flatten registry into named arrays for npz storage
        payload = {}
        writer_ids = list(self.registry.keys())
        payload['__writer_ids__'] = np.array(writer_ids)

        for wid, rec in self.registry.items():
            key = wid  # use writer_id directly as key prefix
            payload[f'{key}__F_refs']       = rec['F_refs']
            payload[f'{key}__mu']           = rec['mu']
            payload[f'{key}__sigma_tilde']  = rec['sigma_tilde']
            payload[f'{key}__sigma_type']   = np.array(rec['sigma_type'])
            payload[f'{key}__R']            = np.array(rec['R'])

        np.savez_compressed(path, **payload)
        size_mb = os.path.getsize(path + '.npz' if not path.endswith('.npz') else path) / 1e6
        print(f"Saved {len(self.registry)} writers to '{path}' ({size_mb:.1f} MB)")

    def load(self, path: str) -> None:
        """Load a registry previously saved with save()."""
        if not path.endswith('.npz'):
            path = path + '.npz'
        data = np.load(path, allow_pickle=False)
        writer_ids = data['__writer_ids__'].tolist()

        self.registry = {}
        for wid in writer_ids:
            self.registry[wid] = {
                'F_refs'      : data[f'{wid}__F_refs'],
                'mu'          : data[f'{wid}__mu'],
                'sigma_tilde' : data[f'{wid}__sigma_tilde'],
                'sigma_type'  : str(data[f'{wid}__sigma_type']),
                'R'           : int(data[f'{wid}__R']),
            }
        print(f"Loaded {len(self.registry)} writers from '{path}'")
