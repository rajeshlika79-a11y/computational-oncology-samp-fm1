"""
Bootstrap AUROC CIs (patient-level resampling) and survival analysis
(Kaplan-Meier + Cox PH) wrappers. See math supplement Secs. 6.1, 6.3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test


def bootstrap_auroc_ci(labels: np.ndarray, scores: np.ndarray, n_boot: int = 1000,
                        alpha: float = 0.05, seed: int | None = 0):
    """
    Patient-level nonparametric bootstrap CI for AUROC.

    IMPORTANT: resamples (label, score) pairs jointly at the patient level
    (not spot/tile level) -- see math supplement Sec 6.1 on why this
    distinction changes what the CI means.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    n = len(labels)
    point_estimate = roc_auc_score(labels, scores)

    boot_aucs = []
    tries = 0
    while len(boot_aucs) < n_boot and tries < n_boot * 20:
        tries += 1
        idx = rng.integers(0, n, size=n)
        yb, sb = labels[idx], scores[idx]
        if len(np.unique(yb)) < 2:
            continue  # skip degenerate resamples (all one class)
        boot_aucs.append(roc_auc_score(yb, sb))

    boot_aucs = np.array(boot_aucs)
    lo, hi = np.percentile(boot_aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "auc": point_estimate,
        "ci_low": lo, "ci_high": hi,
        "n_boot_used": len(boot_aucs),
        "boot_aucs": boot_aucs,
    }


def km_and_logrank(durations: np.ndarray, events: np.ndarray, groups: np.ndarray,
                    group_labels=("Low-Risk", "High-Risk")):
    """
    Kaplan-Meier curves for two risk groups + log-rank test.

    durations: (n,) time-to-event or censoring
    events: (n,) 1 = event observed, 0 = censored
    groups: (n,) boolean/0-1, True/1 = high-risk group
    """
    groups = np.asarray(groups).astype(bool)
    kmfs = {}
    for label, mask in zip(group_labels, [~groups, groups]):
        kmf = KaplanMeierFitter()
        kmf.fit(durations[mask], event_observed=events[mask], label=label)
        kmfs[label] = kmf

    lr = logrank_test(
        durations[~groups], durations[groups],
        event_observed_A=events[~groups], event_observed_B=events[groups],
    )
    return kmfs, {"test_statistic": lr.test_statistic, "p_value": lr.p_value}


def cox_ph(df: pd.DataFrame, duration_col: str, event_col: str, covariates: list[str]):
    """
    Fit a multivariable Cox PH model.

    df must contain duration_col, event_col, and all covariates (including
    the high-risk indicator as one of the covariates). Returns the fitted
    CoxPHFitter (summary() gives HR = exp(coef) with 95% CI and p-values).
    """
    cph = CoxPHFitter()
    cph.fit(df[[duration_col, event_col] + covariates], duration_col=duration_col, event_col=event_col)
    return cph
