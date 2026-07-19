import numpy as np
import torch
import pytest

from samp_fm.models.contrastive_loss import CrossModalSpatialContrastiveLoss, TAU_MIN, TAU_MAX
from samp_fm.models.spatial_gnn import SpatialTranscriptomicEncoder, build_radius_graph
from samp_fm.models.morphology_encoder import MorphologyEncoder, PatchEmbedViT
from samp_fm.metrics.delong import delong_test
from samp_fm.metrics.survival_and_ci import bootstrap_auroc_ci, km_and_logrank, cox_ph
from samp_fm.data.synthetic import (
    synthetic_paired_batch, synthetic_st_graph, synthetic_cohort, synthetic_survival_cohort,
)


# ---------- Contrastive loss ----------

def test_contrastive_loss_decreases_with_correlation():
    loss_fn = CrossModalSpatialContrastiveLoss(learnable=False)
    losses = []
    for corr in [0.0, 0.5, 0.99]:
        z_m, z_t = synthetic_paired_batch(batch_size=64, embed_dim=128, true_correlation=corr, seed=1)
        loss, _ = loss_fn(z_m, z_t)
        losses.append(loss.item())
    # higher true pairing correlation -> easier alignment -> lower contrastive loss
    assert losses[0] > losses[1] > losses[2]


def test_contrastive_loss_collapse_is_penalized():
    """Proposition (math supplement Sec 4.2): constant collapsed embeddings
    give loss = log(B), the MAXIMUM over similarity configurations, not a
    minimum -- i.e. collapse should score strictly worse than a random but
    non-degenerate batch."""
    loss_fn = CrossModalSpatialContrastiveLoss(learnable=False)
    B, d = 32, 64
    const = torch.ones(1, d).repeat(B, 1)
    loss_collapse, _ = loss_fn(const, const.clone())
    assert abs(loss_collapse.item() - np.log(B)) < 1e-4

    z_m, z_t = synthetic_paired_batch(batch_size=B, embed_dim=d, true_correlation=0.9, seed=2)
    loss_normal, _ = loss_fn(z_m, z_t)
    assert loss_normal.item() < loss_collapse.item()


def test_temperature_is_clamped():
    loss_fn = CrossModalSpatialContrastiveLoss(init_tau=0.07, learnable=True)
    with torch.no_grad():
        loss_fn.log_tau.fill_(torch.log(torch.tensor(100.0)))  # try to blow past TAU_MAX
    assert loss_fn.tau.item() <= TAU_MAX + 1e-6
    with torch.no_grad():
        loss_fn.log_tau.fill_(torch.log(torch.tensor(1e-6)))   # try to go below TAU_MIN
    assert loss_fn.tau.item() >= TAU_MIN - 1e-6


# ---------- Spatial GNN ----------

def test_radius_graph_respects_radius():
    coords = torch.tensor([[0., 0.], [10., 0.], [100., 0.]])
    edges = build_radius_graph(coords, radius=25.0)
    edge_set = set(map(tuple, edges.t().tolist()))
    assert (0, 1) in edge_set and (1, 0) in edge_set
    assert (0, 2) not in edge_set and (2, 0) not in edge_set


def test_spatial_gnn_output_shape_and_finite():
    x, coords = synthetic_st_graph(n_nodes=50, gene_dim=480, seed=3)
    enc = SpatialTranscriptomicEncoder(gene_dim=480, hidden_dim=64, n_layers=2, heads=4, radius=100.0)
    out = enc(x, coords)
    assert out.shape == (50, 64)
    assert torch.isfinite(out).all()


def test_neighborhood_pooling():
    x, coords = synthetic_st_graph(n_nodes=50, gene_dim=480, seed=4)
    enc = SpatialTranscriptomicEncoder(gene_dim=480, hidden_dim=32, n_layers=1, heads=4, radius=100.0)
    node_embeds = enc(x, coords)
    mask = torch.zeros(50, dtype=torch.bool)
    mask[:10] = True
    pooled = enc.pool_neighborhood(node_embeds, mask)
    assert pooled.shape == (32,)
    with pytest.raises(ValueError):
        enc.pool_neighborhood(node_embeds, torch.zeros(50, dtype=torch.bool))


# ---------- Morphology encoder ----------

def test_multiscale_attention_sums_to_one():
    backbone = PatchEmbedViT(embed_dim=64, depth=1, heads=2, img_size=28, patch_size=14)
    enc = MorphologyEncoder(embed_dim=64, backbone=backbone)
    p40 = torch.randn(4, 3, 28, 28)
    p20 = torch.randn(4, 3, 28, 28)
    p5 = torch.randn(4, 3, 28, 28)
    z_m, alpha = enc(p40, p20, p5)
    assert z_m.shape == (4, 64)
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(4), atol=1e-5)


# ---------- DeLong's test ----------

def test_delong_identical_models_gives_p_one():
    labels, scores = synthetic_cohort(n_patients=200, auc_target=0.8, seed=5)
    result = delong_test(labels, scores, scores.copy())
    assert result["p_value"] == pytest.approx(1.0, abs=1e-8)
    assert result["z"] == pytest.approx(0.0, abs=1e-8)


def test_delong_matches_sklearn_auc():
    from sklearn.metrics import roc_auc_score
    labels, scores_a = synthetic_cohort(n_patients=200, auc_target=0.85, seed=6)
    _, scores_b = synthetic_cohort(n_patients=200, auc_target=0.65, seed=7)
    result = delong_test(labels, scores_a, scores_b)
    assert result["auc_a"] == pytest.approx(roc_auc_score(labels, scores_a), abs=1e-8)
    assert result["auc_b"] == pytest.approx(roc_auc_score(labels, scores_b), abs=1e-8)


def test_delong_detects_large_gap_as_significant():
    labels, scores_a = synthetic_cohort(n_patients=400, auc_target=0.90, seed=8)
    _, scores_b = synthetic_cohort(n_patients=400, auc_target=0.55, seed=9)
    result = delong_test(labels, scores_a, scores_b)
    assert result["p_value"] < 0.05
    assert result["auc_a"] > result["auc_b"]


# ---------- Bootstrap CI ----------

def test_bootstrap_ci_contains_point_estimate_and_has_sensible_width():
    labels, scores = synthetic_cohort(n_patients=300, auc_target=0.8, seed=10)
    result = bootstrap_auroc_ci(labels, scores, n_boot=500, seed=11)
    assert result["ci_low"] <= result["auc"] <= result["ci_high"]
    assert 0 < (result["ci_high"] - result["ci_low"]) < 0.3


def test_bootstrap_ci_narrows_with_more_patients():
    labels_small, scores_small = synthetic_cohort(n_patients=50, auc_target=0.8, seed=12)
    labels_large, scores_large = synthetic_cohort(n_patients=1000, auc_target=0.8, seed=13)
    small = bootstrap_auroc_ci(labels_small, scores_small, n_boot=500, seed=14)
    large = bootstrap_auroc_ci(labels_large, scores_large, n_boot=500, seed=15)
    assert (large["ci_high"] - large["ci_low"]) < (small["ci_high"] - small["ci_low"])


# ---------- Survival ----------

def test_km_and_logrank_detects_known_separation():
    data = synthetic_survival_cohort(n_patients=400, hr=3.0, seed=16)
    kmfs, lr = km_and_logrank(data["duration"], data["event"], data["high_risk"])
    assert lr["p_value"] < 0.05
    # high-risk group's median survival should be shorter (or KM curve lower)
    assert kmfs["High-Risk"].median_survival_time_ <= kmfs["Low-Risk"].median_survival_time_


def test_cox_ph_recovers_approximate_hr():
    import pandas as pd
    data = synthetic_survival_cohort(n_patients=800, hr=2.5, seed=17)
    df = pd.DataFrame(data)
    cph = cox_ph(df, duration_col="duration", event_col="event", covariates=["high_risk", "age"])
    hr_est = np.exp(cph.params_["high_risk"])
    # Loose tolerance: this is a stochastic recovery check, not an exact match.
    assert 1.5 < hr_est < 4.5
