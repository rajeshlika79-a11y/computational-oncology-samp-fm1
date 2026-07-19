"""
DeLong's test for comparing two correlated AUROCs (same patients, two
models/biomarkers), via the structural components / placement values
formulation. See math supplement Sec. 6.2.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _midrank_placements(pos_scores: np.ndarray, neg_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute V10 (per-positive placement values) and V01 (per-negative
    placement values) and the point AUROC, using the Mann-Whitney U
    equivalence (ties count as 0.5), exactly matching psi(x,y) in the
    math supplement.
    """
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    # psi(d,k) matrix, vectorized: 1 if pos>neg, 0.5 if equal, 0 if pos<neg
    diff = pos_scores[:, None] - neg_scores[None, :]
    psi = (diff > 0).astype(float) + 0.5 * (diff == 0).astype(float)

    v10 = psi.mean(axis=1)   # (n_pos,) -- average over negatives, per positive
    v01 = psi.mean(axis=0)   # (n_neg,) -- average over positives, per negative
    auc = psi.mean()
    return v10, v01, auc


def delong_test(labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray):
    """
    Two-sided DeLong test for H0: AUROC_A == AUROC_B on the SAME patients
    (paired design -- both score vectors must be aligned to `labels`).

    Args:
        labels: (n,) binary array, 1 = event/diseased/resistant, 0 = other.
        scores_a, scores_b: (n,) model/biomarker scores.

    Returns:
        dict with auc_a, auc_b, z, p_value, var_a, var_b, cov_ab
    """
    labels = np.asarray(labels)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    assert labels.shape == scores_a.shape == scores_b.shape

    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos, n_neg = pos_mask.sum(), neg_mask.sum()
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need at least one positive and one negative case.")

    v10_a, v01_a, auc_a = _midrank_placements(scores_a[pos_mask], scores_a[neg_mask])
    v10_b, v01_b, auc_b = _midrank_placements(scores_b[pos_mask], scores_b[neg_mask])

    # Sample covariance of the structural components (DeLong et al. 1988)
    s10 = np.cov(np.vstack([v10_a, v10_b])) / n_pos     # 2x2
    s01 = np.cov(np.vstack([v01_a, v01_b])) / n_neg     # 2x2
    cov_matrix = s10 + s01

    var_a, var_b, cov_ab = cov_matrix[0, 0], cov_matrix[1, 1], cov_matrix[0, 1]
    var_diff = var_a + var_b - 2 * cov_ab
    if var_diff <= 0:
        # identical scores -> zero variance of the difference
        z = 0.0
        p = 1.0
    else:
        z = (auc_a - auc_b) / np.sqrt(var_diff)
        p = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "auc_a": auc_a, "auc_b": auc_b,
        "var_a": var_a, "var_b": var_b, "cov_ab": cov_ab,
        "z": z, "p_value": p,
    }
