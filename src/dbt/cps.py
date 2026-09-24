"""Conditional Poisson (maximum-entropy, rejective) sampling with prescribed first-order pi.

Working probabilities are found by the Chen-Dempster-Liu (1994) fixed point; first-order
inclusion probabilities of CPS with working odds w are computed by the recursion
    pi_k(n) = n w_k (1 - pi_k(n-1)) / sum_l w_l (1 - pi_l(n-1)),   pi(0) = 0
(Tille 2006, Sampling Algorithms, Sec. 5.6).  Validated against R `sampling`
(UPMEpikfrompiktilde / UPMEpiktildefrompik) in tests/test_cps_vs_R.py and by empirical
inclusion frequencies.  Certainty units (pi = 1) are removed before CPS.
"""
from __future__ import annotations

import numpy as np


def cps_pi_from_w(w, n):
    pi = np.zeros_like(w)
    for k in range(1, n + 1):
        t = w * (1 - pi)
        pi = k * t / t.sum()
    return pi


def working_odds(pi_target, n, tol=1e-10, maxit=2000):
    """Fixed point w <- w * (pi_target / pi(w)) on the odds scale (Chen et al. 1994 form)."""
    w = pi_target / (1 - pi_target)
    for _ in range(maxit):
        pi = cps_pi_from_w(w, n)
        err = np.max(np.abs(pi - pi_target))
        if err < tol:
            return w, err
        w = w * (pi_target / pi) * (1 - pi) / (1 - pi_target)
    return w, err


class CPS:
    def __init__(self, pi):
        self.pi = np.asarray(pi, float)
        self.cert = self.pi >= 1 - 1e-12
        self.idx = np.flatnonzero(~self.cert)
        self.n_r = int(round(self.pi[~self.cert].sum()))
        pr = self.pi[~self.cert]
        self.w, self.err = working_odds(pr, self.n_r)
        self.q = self.w / (1 + self.w)                 # Poisson working probabilities

    def sample(self, rng, reps, batch=64):
        """Return boolean inclusion matrix (reps x N) by rejection: Poisson(q) | size = n_r."""
        N = self.pi.size
        out = np.zeros((reps, N), bool)
        out[:, self.cert] = True
        got = 0
        while got < reps:
            U = rng.random((batch, self.idx.size))
            I = U < self.q
            ok = np.flatnonzero(I.sum(1) == self.n_r)
            take = ok[: reps - got]
            if take.size:
                rows = np.zeros((take.size, N), bool)
                rows[:, self.idx] = I[take]
                rows[:, self.cert] = True
                out[got:got + take.size] = rows
                got += take.size
        return out


def deville_var(y, pi, I):
    """Deville (1999) approximate variance estimator for max-entropy designs (per row of I).
    Certainty units contribute zero ((1-pi)=0)."""
    yk = y / pi
    c = (1 - pi)
    nr = (I & (pi < 1 - 1e-12)).sum(1)
    Sc = (I * c).sum(1)
    Ahat = (I * c * yk).sum(1) / np.where(Sc > 0, Sc, 1)
    ss = (I * c * (yk[None, :] - Ahat[:, None]) ** 2).sum(1)
    return nr / np.maximum(nr - 1, 1) * ss
