"""Independent planning data and risk-model estimation (no main-sample outcomes used)."""
from __future__ import annotations

import warnings

import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from scipy.stats import spearmanr

from . import dgp

MODELS = {
    "COR": ["zA", "Z", "G1", "G2", "ZG2"],
    "MIS-I": ["zA", "Z", "G1", "G2"],
    "MIS-G": ["zA", "Z"],
    "KS": ["zA", "Z", "G1", "G2", "ZG2", "W"],
    "STRAT": ["S1", "S2", "S3", "S4"],
}
INFO_LEVELS = {  # label -> (model, planning m)
    "COR-10000": ("COR", 10000), "COR-2000": ("COR", 2000), "COR-500": ("COR", 500),
    "MIS-I-2000": ("MIS-I", 2000), "MIS-G-2000": ("MIS-G", 2000),
    "MIS-G-10000": ("MIS-G", 10000), "KS-2000": ("KS", 2000),
}


def design_matrix(frame, cols, bounds=None):
    feats = dict(zA=frame.zA, Z=frame.Z, G1=(frame.G == 1) * 1.0, G2=(frame.G == 2) * 1.0,
                 ZG2=frame.Z * (frame.G == 2), W=frame.W)
    if any(c.startswith("S") for c in cols):
        s = dgp.assign_strata(frame.A, bounds)
        for k in range(1, dgp.N_STRATA):
            feats[f"S{k}"] = (s == k) * 1.0
    return np.column_stack([np.ones(frame.N)] + [feats[c] for c in cols])


def draw_planning(m, dgp_name, sigma, alpha, bounds, rng):
    fr = dgp.draw_frame(m, rng)
    p = dgp.risk(fr, dgp_name, sigma, alpha, bounds)
    E = (rng.random(m) < p).astype(float)
    return fr, E


def fit_predict(fr_plan, E, model, target, bounds):
    """Return (phat on target, failed flag, n_pos). Fallback: intercept-only."""
    cols = MODELS[model]
    n_pos = int(E.sum())
    if n_pos == 0 or n_pos == E.size:
        return np.full(target.N, np.clip(E.mean(), 1e-6, 1 - 1e-6)), True, n_pos
    X = design_matrix(fr_plan, cols, bounds)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = sm.GLM(E, X, family=sm.families.Binomial()).fit(maxiter=100)
        if not res.converged or not np.all(np.isfinite(res.params)) or np.abs(res.params).max() > 25:
            raise RuntimeError("nonconvergence/separation")
        phat = expit(design_matrix(target, cols, bounds) @ res.params)
        return np.clip(phat, 1e-8, 1 - 1e-8), False, n_pos
    except Exception:
        return np.full(target.N, E.mean()), True, n_pos


def quality(phat, p, E_real):
    """Planning-model quality on the target frame (true p known)."""
    lp, lph = np.log(p / (1 - p)), np.log(phat / (1 - phat))
    brier = np.mean(phat**2 - 2 * phat * p + p)                 # E[(phat - E)^2]
    logloss = -np.mean(p * np.log(phat) + (1 - p) * np.log(1 - phat))
    if np.std(lph) > 1e-9 and np.std(lp) > 1e-9:
        slope = np.polyfit(lph, lp, 1)[0]
    else:
        slope = np.nan
    order = np.argsort(phat)
    r = np.empty_like(order, dtype=float); r[order] = np.arange(phat.size)
    npos = E_real.sum(); nneg = E_real.size - npos
    auc = (r[E_real == 1].sum() - npos * (npos - 1) / 2) / (npos * nneg) if npos and nneg else np.nan
    return dict(brier=brier, logloss=logloss, calib_in_large=phat.mean() - p.mean(),
                calib_slope=slope, auc=auc)


def score_corr(s_hat, s_true):
    if np.std(s_hat) < 1e-12 or np.std(s_true) < 1e-12:
        return np.nan, np.nan
    return float(np.corrcoef(s_hat, s_true)[0, 1]), float(spearmanr(s_hat, s_true).statistic)
