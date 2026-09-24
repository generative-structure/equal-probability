"""Finite-population data-generating process (synthetic public improper-payment frame).

Frame variables known before audit: amount A, continuous risk score Z, category G,
and an outcome-irrelevant covariate W (correlated with A and G, not with E given
A, Z, G).  Audit outcome E ~ Bernoulli(p), all-or-nothing improper payment A*E.

    logit p = alpha + sigma_eta * eta0_std
    eta0    = 0.6 zA + 0.8 Z + bG[G] + 1.0 Z 1{G=2},  bG = (0, 0.8, 1.6)

eta0_std is eta0 standardized with fixed superpopulation constants (ETA0_MEAN,
ETA0_SD), so sigma_eta is the SD of the logit linear predictor.  alpha is solved
on the target population so that mean(p) equals the prevalence exactly.

STRATCAP variant (Control 6): eta0 = amount-stratum index (5 equal-dollar strata of
the target frame), standardized the same way, so conventional amount strata carry
all predictable heterogeneity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

A_MEANLOG, A_SDLOG = np.log(250.0), 1.2
G_PROBS = (0.6, 0.3, 0.1)
B_G = np.array([0.0, 0.8, 1.6])
B_A, B_Z, B_INT = 0.6, 0.8, 1.0
N_STRATA = 5


def eta0_raw(zA, Z, G):
    return B_A * zA + B_Z * Z + B_G[G] + B_INT * Z * (G == 2)


def _superpop_constants(n=2_000_000, seed=1):
    rng = np.random.default_rng(seed)
    zA = rng.standard_normal(n); Z = rng.standard_normal(n)
    G = rng.choice(3, size=n, p=G_PROBS)
    e = eta0_raw(zA, Z, G)
    return float(e.mean()), float(e.std())


ETA0_MEAN, ETA0_SD = _superpop_constants()


@dataclass
class Frame:
    A: np.ndarray
    zA: np.ndarray
    Z: np.ndarray
    G: np.ndarray
    W: np.ndarray
    stratum: np.ndarray | None = None

    @property
    def N(self):
        return self.A.size


def draw_frame(N, rng):
    zA = rng.standard_normal(N)
    A = np.exp(A_MEANLOG + A_SDLOG * zA)
    Z = rng.standard_normal(N)
    G = rng.choice(3, size=N, p=G_PROBS)
    W = 0.8 * zA + 0.6 * (G - np.dot(G_PROBS, [0, 1, 2])) + 0.5 * rng.standard_normal(N)
    return Frame(A, zA, Z, G, W)


def equal_dollar_boundaries(A, k=N_STRATA):
    """Boundaries splitting total dollars into k equal parts (PERM-style)."""
    a = np.sort(A); c = np.cumsum(a) / a.sum()
    idx = [np.searchsorted(c, j / k) for j in range(1, k)]
    return a[idx]


def assign_strata(A, bounds):
    return np.searchsorted(bounds, A, side="right")


def eta0_std(frame, dgp, bounds=None):
    if dgp == "MAIN":
        return (eta0_raw(frame.zA, frame.Z, frame.G) - ETA0_MEAN) / ETA0_SD
    if dgp == "STRATCAP":
        s = assign_strata(frame.A, bounds).astype(float)
        return (s - STRATCAP_MEAN) / STRATCAP_SD
    raise ValueError(dgp)


def _stratcap_constants():
    # superpopulation moments of the stratum index under the target-frame boundaries
    # (computed from a large draw with its own equal-dollar boundaries; shape only)
    rng = np.random.default_rng(2)
    A = np.exp(A_MEANLOG + A_SDLOG * rng.standard_normal(2_000_000))
    s = assign_strata(A, equal_dollar_boundaries(A)).astype(float)
    return float(s.mean()), float(s.std())


STRATCAP_MEAN, STRATCAP_SD = _stratcap_constants()


def solve_alpha(eta_s, sigma, prevalence):
    f = lambda a: expit(a + sigma * eta_s).mean() - prevalence
    return brentq(f, -30, 30, xtol=1e-14)


def risk(frame, dgp, sigma, alpha, bounds=None):
    return expit(alpha + sigma * eta0_std(frame, dgp, bounds))


def diagnostics(p, A):
    sp = np.sqrt(p)
    return dict(mean_p=p.mean(), p10=np.quantile(p, .1), p50=np.quantile(p, .5),
                p90=np.quantile(p, .9), cv_sqrt_p=sp.std() / sp.mean(),
                corr_p_A=np.corrcoef(p, A)[0, 1],
                corr_p_logA=np.corrcoef(p, np.log(A))[0, 1])
