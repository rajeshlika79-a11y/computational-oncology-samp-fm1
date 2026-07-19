# SAMP-FM: Reference Implementation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![CI](https://github.com/<USERNAME>/<REPOSITORY>/actions/workflows/ci.yml/badge.svg)


This repository implements, as runnable and tested code, the architecture and
statistical framework described in the SAMP-FM manuscript:

- `samp_fm/models/morphology_encoder.py` — multi-scale ViT patch encoder +
  cross-scale attention fusion (`E_m`)
- `samp_fm/models/spatial_gnn.py` — radius-graph construction + 4-layer,
  8-head GATv2 spatial transcriptomic encoder (`E_t`)
- `samp_fm/models/contrastive_loss.py` — Cross-Modal Spatial Contrastive Loss,
  symmetrized, with the `[0.01, 0.20]` temperature clamp
- `samp_fm/metrics/delong.py` — DeLong's test for correlated ROC curves
- `samp_fm/metrics/survival_and_ci.py` — patient-level bootstrap AUROC CIs,
  Kaplan-Meier, log-rank, and Cox PH wrappers
- `samp_fm/data/synthetic.py` — synthetic data generators used **only** to
  smoke-test the pipeline
- `tests/test_pipeline.py` — 14 unit tests covering correctness properties
  (see below)
- `scripts/run_smoke_test.py` — end-to-end run on synthetic data

## What this repository demonstrates

Running `pytest` and `scripts/run_smoke_test.py` shows that:

1. The contrastive loss decreases as true morphology-transcriptomic
   correlation increases, and is *maximized* (not minimized) by
   representation collapse, matching the collapse-prevention argument in the
   math supplement.
2. The temperature parameter is correctly clamped to `[0.01, 0.20]`.
3. The GATv2 spatial encoder builds a correct radius graph and produces
   finite, correctly-shaped outputs.
4. The multi-scale fusion attention weights are a valid probability
   distribution over the three magnification scales.
5. DeLong's test recovers `AUROC_A == AUROC_B` (p=1) for identical scores,
   matches `sklearn.roc_auc_score` on point estimates, and detects a large
   simulated AUROC gap as significant.
6. Bootstrap CIs contain the point estimate and narrow with more patients.
7. The KM/log-rank/Cox pipeline recovers a statistically significant
   separation and an approximately correct hazard ratio from synthetic
   survival data with a known injected effect.

**This is a correctness check on the code, not a validation of the
manuscript's clinical claims.**

## What this repository does NOT demonstrate

- It does not reproduce the manuscript's reported AUROC/CI/HR/p-values.
  Those would require training on the real HEST-1k pretraining corpus and
  evaluating on the real GSE115978 / GSE203612 / GSE123813 cohorts — none of
  which this environment has network access to.
- The morphology encoder here uses a small randomly-initialized ViT for
  testing the fusion math; the manuscript specifies initialization from
  pretrained UNI weights, which must be loaded separately.
- **The manuscript's cohort–accession mapping is currently self-contradictory**
  (see below) and should be corrected before any real run, since every
  downstream table is indexed by cohort identity.

## Known issue to fix before running on real data

The manuscript's Abstract and Section 2.3 disagree on which GEO accession
corresponds to which cancer type and sample size:

| | Abstract | Section 2.3 |
|---|---|---|
| GSE115978 | melanoma, N=312 | melanoma, N=280 |
| GSE123813 | lung, N=280 | lung, N=312 |

Resolve this against the actual GEO records before treating any
cohort-indexed result as meaningful.

## Setup

```bash
pip install -e .
# or: pip install -r requirements.txt

pytest tests/ -v
python scripts/run_smoke_test.py
```


---

## Repository Structure

```text
samp_fm/
├── models/
├── metrics/
├── data/
scripts/
tests/
```

## Quick Start

```bash
pip install -r requirements.txt
pytest -v
python scripts/run_smoke_test.py
```

## Citation

If you use this software in academic work, please cite the accompanying manuscript and this repository. A `CITATION.cff` file is included for GitHub citation support.

## Zenodo

After creating a GitHub release, connect the repository to Zenodo to obtain a DOI for archival and citation.
