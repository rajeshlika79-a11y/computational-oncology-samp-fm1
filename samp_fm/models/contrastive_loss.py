"""
Cross-Modal Spatial Contrastive Loss (L_CMSC).

The manuscript states the one-directional form only. This implementation
uses the correct symmetrized form (mean of morphology->transcriptomic and
transcriptomic->morphology directions), which is what should actually be
optimized -- see math supplement Sec. 4.1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

TAU_MIN, TAU_MAX = 0.01, 0.20


class CrossModalSpatialContrastiveLoss(nn.Module):
    def __init__(self, init_tau: float = 0.07, learnable: bool = True):
        super().__init__()
        tau = torch.tensor(float(init_tau))
        if learnable:
            self.log_tau = nn.Parameter(torch.log(tau))
        else:
            self.register_buffer("log_tau", torch.log(tau))
        self._learnable = learnable

    @property
    def tau(self) -> torch.Tensor:
        # Clamp is applied on the *value* used in the forward pass, not the
        # raw parameter, matching "tau is strictly constrained to [0.01,0.20]
        # during backpropagation" in the manuscript.
        return self.log_tau.exp().clamp(TAU_MIN, TAU_MAX)

    def forward(self, z_m: torch.Tensor, z_t: torch.Tensor, extra_negatives: torch.Tensor | None = None):
        """
        z_m, z_t: (B, d) matched morphology / transcriptomic embeddings,
            row i of z_m paired with row i of z_t.
        extra_negatives: optional (Q, d) queued negative transcriptomic
            embeddings (the manuscript's 65,536-slot negative buffer).

        Returns: (loss, diagnostics dict)
        """
        z_m = F.normalize(z_m, dim=-1)
        z_t = F.normalize(z_t, dim=-1)
        tau = self.tau

        t_bank = z_t if extra_negatives is None else torch.cat([z_t, F.normalize(extra_negatives, dim=-1)], dim=0)

        sim_m2t = z_m @ t_bank.t() / tau            # (B, B[+Q])
        sim_t2m = z_t @ z_m.t() / tau                # (B, B)  (queue not symmetric-usable for t->m)

        targets = torch.arange(z_m.size(0), device=z_m.device)
        loss_m2t = F.cross_entropy(sim_m2t, targets)
        loss_t2m = F.cross_entropy(sim_t2m, targets)
        loss = 0.5 * (loss_m2t + loss_t2m)

        with torch.no_grad():
            pos_sim = (z_m * z_t).sum(-1).mean().item()
            diagnostics = {
                "tau": tau.item(),
                "mean_positive_cosine_sim": pos_sim,
                "loss_m2t": loss_m2t.item(),
                "loss_t2m": loss_t2m.item(),
            }
        return loss, diagnostics
