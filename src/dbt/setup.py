"""Shared scenario setup: fixed target frame, strata, realized outcomes, seeds."""
from __future__ import annotations

import zlib

import numpy as np

from . import dgp

N_TARGET = 20_000
SEED_FRAME, SEED_E = 20260923, 20260924
SIGMAS = (0.0, 0.75, 1.5)
PREVS = (0.05, 0.25, 0.70)
NS = (100, 400)
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
R_PLAN = 200
ESTIMANDS = ("rate", "dollars")


def seed_for(*parts):
    return [zlib.crc32(str(p).encode()) for p in parts]


class Target:
    def __init__(self):
        self.frame = dgp.draw_frame(N_TARGET, np.random.default_rng(SEED_FRAME))
        self.bounds = dgp.equal_dollar_boundaries(self.frame.A)
        self.strata = dgp.assign_strata(self.frame.A, self.bounds)
        self.frame.stratum = self.strata
        self.U_E = np.random.default_rng(SEED_E).random(N_TARGET)   # CRN for outcomes

    def scenario(self, dgp_name, sigma, prev):
        eta = dgp.eta0_std(self.frame, dgp_name, self.bounds)
        alpha = dgp.solve_alpha(eta, sigma, prev)
        p = dgp.risk(self.frame, dgp_name, sigma, alpha, self.bounds)
        E = (self.U_E < p).astype(float)
        return alpha, p, E
