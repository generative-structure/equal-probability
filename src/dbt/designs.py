"""Design construction and exact anticipated variances.

Moments per unit under the planning model (true p for evaluation):
    rate:    y = E,    mu = p,     v = p(1-p),       E y^2 = p
    dollars: y = A E,  mu = A p,   v = A^2 p(1-p),   E y^2 = A^2 p
DIFF with working mean w: residual r = y - w, E r^2 = v + (mu - w)^2.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


# ------------------------------------------------------------------ moments -- #
def moments(estimand, p, A):
    if estimand == "rate":
        return p, p * (1 - p)                      # mu, v
    if estimand == "dollars":
        return A * p, A**2 * p * (1 - p)
    raise ValueError(estimand)


def working_mean(estimand, phat, A):
    return phat if estimand == "rate" else A * phat


def second_moment(estimand, estimator, p, A, phat=None):
    """Allocation-relevant E[(Y - w)^2 | X] under the model with risk p.
    estimator 'HT': w = 0;  'DIFF': w = working mean from phat (phat=None -> oracle w=mu)."""
    mu, v = moments(estimand, p, A)
    if estimator == "HT":
        return v + mu**2
    if estimator == "DIFF":
        w = mu if phat is None else working_mean(estimand, phat, A)
        return v + (mu - w) ** 2
    raise ValueError(estimator)


# ---------------------------------------------------------------- allocation -- #
def waterfill(score, n, floor=0.0):
    """pi = clip(lam*score, floor, 1), sum pi = n (exact KKT for score = sqrt(m))."""
    s = np.asarray(score, float)
    s = s / s.max()
    if n >= s.size:
        return np.ones_like(s)
    pi0 = n * s / s.sum()
    if pi0.max() <= 1 and pi0.min() >= floor:
        return pi0
    f = lambda L: np.clip(np.exp(L) * s, floor, 1.0).sum() - n
    lo, hi = np.log(n / s.sum()) - 1, np.log(n / s.sum()) + 1
    while f(lo) > 0:
        lo -= 5
    while f(hi) < 0:
        hi += 5
    L = brentq(f, lo, hi, xtol=1e-13, rtol=1e-14, maxiter=500)
    return np.clip(np.exp(L) * s, floor, 1.0)


def defensive(pi_ra, n, N, lam):
    return (1 - lam) * n / N + lam * pi_ra


# ----------------------------------------------------- anticipated variances -- #
def av_poisson(m_true, pi):
    return float(np.sum(m_true * (1 / pi - 1)))


def av_hajek_fixed(mu, v, pi):
    """SENSITIVITY: Hajek (1964) approximation to the variance of the HT estimator under a
    fixed-size high-entropy (conditional Poisson) design with first-order pi, taken in model
    expectation:  Var ~ sum c y^2/pi^2 - (sum c y/pi)^2 / sum c,  c = pi(1-pi).
    Certainty units (pi=1) contribute nothing.  For pi = n/N it equals the SRSWOR variance
    times (N-1)/N."""
    c = pi * (1 - pi)
    a = c / pi
    Ey2 = v + mu**2
    return float(np.sum(c * Ey2 / pi**2) - (np.sum(a**2 * v) + np.sum(a * mu) ** 2) / c.sum())


def expected_S2(mu, v):
    """Model expectation of the finite-population variance S^2 of y (independent units)."""
    Nh = mu.size
    Ey2 = np.sum(v + mu**2)
    Esum2 = np.sum(v) + mu.sum() ** 2
    return (Ey2 - Esum2 / Nh) / (Nh - 1)


def av_srswor(mu, v, n):
    N = mu.size
    return N**2 * (1 - n / N) * expected_S2(mu, v) / n


def av_stratified(mu, v, strata, nh):
    tot = 0.0
    for h, n_h in enumerate(nh):
        idx = strata == h
        Nh = idx.sum()
        if n_h >= Nh:
            continue
        tot += Nh**2 * (1 - n_h / Nh) * expected_S2(mu[idx], v[idx]) / n_h
    return float(tot)


def alloc_prop(strata, n, K, integer=True):
    Nh = np.bincount(strata, minlength=K).astype(float)
    raw = n * Nh / Nh.sum()
    return _finish(raw, Nh, n, integer)


def alloc_equal(strata, n, K, integer=True):
    Nh = np.bincount(strata, minlength=K).astype(float)
    return _finish(np.full(K, n / K), Nh, n, integer)


def alloc_neyman(strata, n, K, S_hat, integer=True):
    Nh = np.bincount(strata, minlength=K).astype(float)
    w = Nh * np.maximum(S_hat, 1e-12)
    return _finish(n * w / w.sum(), Nh, n, integer)


def _finish(raw, Nh, n, integer):
    """Enforce 2 <= n_h <= N_h and sum n_h = n (largest-remainder rounding)."""
    raw = np.clip(raw, 2, Nh)
    for _ in range(50):                       # re-spread excess after caps
        exc = n - raw.sum()
        free = (raw < Nh) & (raw > 2 - 1e-12) if exc > 0 else raw > 2
        if abs(exc) < 1e-9 or not free.any():
            break
        raw[free] += exc * raw[free] / raw[free].sum()
        raw = np.clip(raw, 2, Nh)
    if not integer:
        return raw
    fl = np.floor(raw); rem = int(round(n - fl.sum()))
    order = np.argsort(-(raw - fl))
    for j in order[:max(rem, 0)]:
        fl[j] += 1
    return fl.astype(int)


# ------------------------------------------------------------- stability ----- #
def stability(pi):
    w = 1 / pi
    n = pi.sum()
    mw = np.sum(pi * w) / n
    cvw = np.sqrt(np.sum(pi * (w - mw) ** 2) / n) / mw
    return dict(min_pi=pi.min(), max_pi=pi.max(), max_weight=w.max(), cv_weights=cvw,
                n_certainty=int(np.sum(pi >= 1 - 1e-12)), sd_size=float(np.sqrt(np.sum(pi * (1 - pi)))))


def eq_precision_n(av_of_n, target, n_lo=2.0, n_hi=19_000.0):
    g = lambda n: av_of_n(n) - target
    if g(n_hi) > 0:
        return np.inf
    if g(n_lo) < 0:
        return n_lo
    return brentq(g, n_lo, n_hi, xtol=1e-6, rtol=1e-9)
