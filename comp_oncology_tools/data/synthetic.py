"""
Synthetic data generators.

These exist ONLY to verify the code runs and behaves correctly (loss
decreases, AUROC computation is self-consistent, DeLong/CI/survival code
matches known references). They do NOT stand in for real HEST-1k /
GEO (GSE115978, GSE203612, GSE123813) data, and nothing computed from
them should be reported as a biological or clinical finding.
"""
from __future__ import annotations

import numpy as np
import torch


def synthetic_paired_batch(batch_size: int = 16, embed_dim: int = 1024, seed: int = 0,
                            true_correlation: float = 0.9):
    """Generate a batch of correlated (z_m, z_t) pairs: z_t = corr * z_m +
    sqrt(1-corr^2) * noise, so a well-implemented contrastive loss should
    be able to drive alignment loss down as correlation -> 1."""
    g = torch.Generator().manual_seed(seed)
    z_m = torch.randn(batch_size, embed_dim, generator=g)
    noise = torch.randn(batch_size, embed_dim, generator=g)
    z_t = true_correlation * z_m + (1 - true_correlation**2) ** 0.5 * noise
    return z_m, z_t


def synthetic_st_graph(n_nodes: int = 200, gene_dim: int = 480, coord_range: float = 500.0,
                        seed: int = 0):
    """Random spatial transcriptomics graph: node coordinates in a
    coord_range x coord_range um field, random transcript vectors."""
    g = torch.Generator().manual_seed(seed)
    coords = torch.rand(n_nodes, 2, generator=g) * coord_range
    x = torch.rand(n_nodes, gene_dim, generator=g).clamp(min=0)  # nonneg "counts"
    return x, coords


def synthetic_cohort(n_patients: int = 300, auc_target: float = 0.80, seed: int = 0):
    """
    Generate a synthetic (label, score) cohort with an approximately known
    AUROC, for validating bootstrap CI coverage and DeLong's test -- NOT a
    substitute for GSE115978 / GSE203612 / GSE123813.

    Construction: draw scores from two shifted Gaussians whose separation
    is calibrated (via the probit relationship AUC = Phi(delta/sqrt(2)))
    to hit the target AUROC in expectation.
    """
    rng = np.random.default_rng(seed)
    delta = np.sqrt(2) * _norm_ppf(auc_target)
    labels = rng.integers(0, 2, size=n_patients)
    scores = rng.normal(loc=labels * delta, scale=1.0)
    return labels, scores


def _norm_ppf(p: float) -> float:
    from scipy.stats import norm
    return norm.ppf(p)


def synthetic_survival_cohort(n_patients: int = 300, hr: float = 2.4, seed: int = 0):
    """Synthetic exponential-hazard survival data with a known ground-truth
    hazard ratio between high- and low-risk groups, for validating the
    Cox/KM/log-rank plumbing -- NOT real PFS data."""
    rng = np.random.default_rng(seed)
    high_risk = rng.integers(0, 2, size=n_patients).astype(bool)
    baseline_hazard = 0.05
    hazard = baseline_hazard * np.where(high_risk, hr, 1.0)
    true_time = rng.exponential(scale=1.0 / hazard)
    censor_time = rng.uniform(0, 24, size=n_patients)  # 24-month admin censoring
    durations = np.minimum(true_time, censor_time)
    events = (true_time <= censor_time).astype(int)
    age = rng.normal(60, 10, size=n_patients)
    return {
        "duration": durations, "event": events, "high_risk": high_risk.astype(int), "age": age,
    }
