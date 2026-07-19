"""
End-to-end smoke test: runs the full pipeline (morphology encoder ->
spatial GNN -> contrastive loss -> RLI -> DeLong/bootstrap/survival stats)
on SYNTHETIC data, to prove the code executes and the statistics behave
correctly under known ground truth.

This script does NOT produce, and should NOT be cited as, results for the
SAMP-FM manuscript. Real results require the actual HEST-1k pretraining
corpus and the GSE115978 / GSE203612 / GSE123813 cohorts (with the
cohort-accession mapping contradiction resolved first).
"""
import numpy as np
import torch

from samp_fm.models.contrastive_loss import CrossModalSpatialContrastiveLoss
from samp_fm.models.spatial_gnn import SpatialTranscriptomicEncoder
from samp_fm.models.morphology_encoder import MorphologyEncoder, PatchEmbedViT
from samp_fm.metrics.delong import delong_test
from samp_fm.metrics.survival_and_ci import bootstrap_auroc_ci, km_and_logrank, cox_ph
from samp_fm.data.synthetic import (
    synthetic_st_graph, synthetic_cohort, synthetic_survival_cohort,
)


def main():
    print("=" * 70)
    print("SAMP-FM reference implementation -- SYNTHETIC DATA SMOKE TEST")
    print("(not a validation of the manuscript's reported results)")
    print("=" * 70)

    # 1. Morphology encoder forward pass
    backbone = PatchEmbedViT(embed_dim=128, depth=2, heads=4, img_size=28, patch_size=14)
    morph_enc = MorphologyEncoder(embed_dim=128, backbone=backbone)
    p40, p20, p5 = (torch.randn(8, 3, 28, 28) for _ in range(3))
    z_m, alpha = morph_enc(p40, p20, p5)
    print(f"\n[Morphology] fused embedding shape: {tuple(z_m.shape)}, "
          f"fusion attention (first sample): {alpha[0].detach().numpy().round(3)}")

    # 2. Spatial GNN forward pass
    x, coords = synthetic_st_graph(n_nodes=100, gene_dim=480, seed=0)
    st_enc = SpatialTranscriptomicEncoder(gene_dim=480, hidden_dim=128, n_layers=2, heads=4, radius=50.0)
    node_embeds = st_enc(x, coords)
    mask = torch.zeros(100, dtype=torch.bool); mask[:8] = True
    z_t = st_enc.pool_neighborhood(node_embeds, mask).unsqueeze(0).repeat(8, 1)
    print(f"[Spatial GNN] node embedding shape: {tuple(node_embeds.shape)}")

    # 3. Contrastive loss on the two encoders' outputs
    loss_fn = CrossModalSpatialContrastiveLoss()
    loss, diag = loss_fn(z_m, z_t)
    print(f"[Contrastive loss] loss={loss.item():.4f}, diagnostics={diag}")

    # 4. DeLong's test (synthetic cohort, known AUROC targets)
    labels, scores_samp = synthetic_cohort(n_patients=312, auc_target=0.83, seed=1)
    _, scores_baseline = synthetic_cohort(n_patients=312, auc_target=0.74, seed=2)
    delong = delong_test(labels, scores_samp, scores_baseline)
    print(f"\n[DeLong] synthetic AUROCs: A={delong['auc_a']:.3f}, B={delong['auc_b']:.3f}, "
          f"z={delong['z']:.3f}, p={delong['p_value']:.4g}")

    # 5. Bootstrap CI
    ci = bootstrap_auroc_ci(labels, scores_samp, n_boot=1000, seed=3)
    print(f"[Bootstrap CI] AUROC={ci['auc']:.3f}, 95% CI=[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}] "
          f"(n_boot_used={ci['n_boot_used']})")

    # 6. Survival: KM + log-rank + Cox
    surv = synthetic_survival_cohort(n_patients=312, hr=2.45, seed=4)
    kmfs, lr = km_and_logrank(surv["duration"], surv["event"], surv["high_risk"])
    print(f"\n[KM/log-rank] p={lr['p_value']:.4g}, "
          f"median PFS High-Risk={kmfs['High-Risk'].median_survival_time_:.2f}, "
          f"Low-Risk={kmfs['Low-Risk'].median_survival_time_:.2f}")

    import pandas as pd
    cph = cox_ph(pd.DataFrame(surv), duration_col="duration", event_col="event",
                 covariates=["high_risk", "age"])
    hr_est = np.exp(cph.params_["high_risk"])
    print(f"[Cox PH] estimated HR (high_risk)={hr_est:.3f} (ground truth simulated at 2.45)")

    print("\n" + "=" * 70)
    print("Smoke test complete. All components executed without error.")
    print("Reminder: numbers above are from SYNTHETIC data with a KNOWN")
    print("ground-truth signal injected -- they demonstrate the code is")
    print("correct, not that the manuscript's clinical claims are true.")
    print("=" * 70)


if __name__ == "__main__":
    main()
