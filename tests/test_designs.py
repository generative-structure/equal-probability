"""Property tests for designs, variances and stop rules 1-2.  Run: .venv/bin/python -m pytest -q tests"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dbt import designs as D  # noqa: E402
from dbt.setup import Target  # noqa: E402


def test_waterfill_budget_caps_and_equal_case():
    rng = np.random.default_rng(1)
    s = rng.lognormal(0, 2, 5000)
    for n in (50, 400, 1500):
        pi = D.waterfill(s, n)
        assert abs(pi.sum() - n) < 1e-6 and pi.max() <= 1 + 1e-12 and pi.min() > 0
    assert np.allclose(D.waterfill(np.full(100, 3.0), 10), 0.1)


def test_oracle_dominates_equal_poisson_rule1():
    rng = np.random.default_rng(2)
    for _ in range(200):
        m = rng.lognormal(0, rng.uniform(0, 3), 2000); n = rng.uniform(10, 1500)
        assert D.av_poisson(m, D.waterfill(np.sqrt(m), n)) <= D.av_poisson(m, np.full(2000, n / 2000)) * (1 + 1e-9)


def test_homogeneous_rate_oracle_equals_equal_rule2():
    T = Target()
    for prev in (0.05, 0.25, 0.70):
        _, p, _ = T.scenario("MAIN", 0.0, prev)
        m = p  # rate, HT
        po = D.waterfill(np.sqrt(m), 100)
        assert abs(D.av_poisson(m, po) / D.av_poisson(m, np.full(p.size, 100 / p.size)) - 1) < 1e-9


def test_hajek_equals_srswor_for_equal_pi():
    rng = np.random.default_rng(3); N = 800
    mu, v = rng.random(N), rng.random(N) * 0.2
    assert np.isclose(D.av_hajek_fixed(mu, v, np.full(N, 40 / N)) / D.av_srswor(mu, v, 40), (N - 1) / N)


def test_expected_S2_matches_simulation():
    rng = np.random.default_rng(4); N = 50
    p = rng.random(N); mu, v = p, p * (1 - p)
    sims = [np.var((rng.random(N) < p).astype(float), ddof=1) for _ in range(40000)]
    assert abs(np.mean(sims) / D.expected_S2(mu, v) - 1) < 0.01


def test_stratified_allocations():
    T = Target()
    for n in (100, 400):
        for nh in (D.alloc_prop(T.strata, n, 5), D.alloc_equal(T.strata, n, 5)):
            assert nh.sum() == n and nh.min() >= 2


def test_defensive_mixture_budget_and_floor():
    rng = np.random.default_rng(5); N, n = 3000, 120
    pr = D.waterfill(rng.lognormal(0, 2, N), n)
    for lam in (0.25, 0.5, 0.75):
        pl = D.defensive(pr, n, N, lam)
        assert abs(pl.sum() - n) < 1e-6 and pl.min() >= (1 - lam) * n / N - 1e-15 and pl.max() <= 1
