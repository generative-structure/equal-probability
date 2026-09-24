"""Independent verification of the clean Poisson/HT benchmark (core_benchmark_check.md).

Symbolic (sympy) checks + exact enumeration on a tiny population + numeric
dominance/equality checks.  Does not use the design code in src/dbt.

    .venv/bin/python scripts/core_benchmark_check.py
"""
import itertools
import json
from pathlib import Path

import numpy as np
import sympy as sp

OUT = Path(__file__).resolve().parents[1] / "results" / "core_benchmark"
OUT.mkdir(parents=True, exist_ok=True)
res = {}

# 1. Poisson design variance of HT total by exact enumeration over all 2^N samples
y = np.array([3.0, 0.0, 7.5, 1.2, 4.4]); pi = np.array([0.3, 0.5, 0.8, 0.2, 0.6])
T = y.sum(); ev = 0.0; e1 = 0.0
for I in itertools.product([0, 1], repeat=y.size):
    I = np.array(I); pr = np.prod(np.where(I == 1, pi, 1 - pi))
    th = np.sum(I * y / pi); e1 += pr * th; ev += pr * (th - T) ** 2
res["HT_unbiased_enum"] = bool(np.isclose(e1, T))
res["HT_var_formula_enum"] = bool(np.isclose(ev, np.sum(y**2 * (1 / pi - 1))))
# unbiasedness of v = sum_s y^2 (1-pi)/pi^2
ev_v = sum(np.prod(np.where(np.array(I) == 1, pi, 1 - pi)) * np.sum(np.array(I) * y**2 * (1 - pi) / pi**2)
           for I in itertools.product([0, 1], repeat=y.size))
res["HT_var_estimator_unbiased_enum"] = bool(np.isclose(ev_v, ev))

# 2. Anticipated variance, KKT optimum, Cauchy-Schwarz dominance (symbolic, 3 units)
m = sp.symbols("m1:4", positive=True); n, N = sp.symbols("n N", positive=True)
sq = sum(sp.sqrt(mi) for mi in m)
pis = [n * sp.sqrt(mi) / sq for mi in m]
AV_opt = sp.simplify(sum(m[i] * (1 / pis[i] - 1) for i in range(3)))
res["AV_opt_equals_(sum sqrt m)^2/n - sum m"] = sp.simplify(AV_opt - (sq**2 / n - sum(m))) == 0
AV_eq = sum(m[i] * (3 / n - 1) for i in range(3))            # pi = n/N, N = 3
gap = sp.factor(sp.simplify((AV_eq - AV_opt) * n))
res["AV_eq_minus_AV_opt_times_n"] = str(gap)
# (N sum m - (sum sqrt m)^2) = sum_{i<j} (sqrt m_i - sqrt m_j)^2 (Lagrange identity), N=3
s = [sp.sqrt(mi) for mi in m]
lag = sum((s[i] - s[j])**2 for i in range(3) for j in range(i + 1, 3))
res["Lagrange_identity_N3"] = sp.simplify(3 * sum(m) - sq**2 - lag) == 0

# 3. rate / all-or-nothing dollar second moments and model-assisted residual moments
p, A, mu = sp.symbols("p A mu", positive=True)
E2 = p * 1**2 + (1 - p) * 0                         # E[E^2|X]
res["rate_m"] = str(sp.simplify(E2))
res["dollar_m"] = str(sp.simplify(p * A**2))
res["diff_rate_residual"] = str(sp.factor(sp.expand(p * (1 - mu)**2 + (1 - p) * mu**2)))
res["diff_rate_residual_at_mu_eq_p"] = str(sp.factor(sp.expand((p * (1 - mu)**2 + (1 - p) * mu**2).subs(mu, p))))
res["diff_dollar_residual_at_mu_eq_Ap"] = str(sp.factor(sp.expand(p * (A - A * p)**2 + (1 - p) * (A * p)**2)))

# 4. numeric: capped optimum (water-filling) dominates equal pi; equality iff m constant
def waterfill(s, n):
    lo, hi = 0.0, n / s.min() * 10
    for _ in range(200):
        lam = (lo + hi) / 2
        if np.minimum(1, lam * s).sum() > n: hi = lam
        else: lo = lam
    return np.minimum(1, lam * s)

rng = np.random.default_rng(3); ok = True; worst = 0.0
for _ in range(500):
    Nn = rng.integers(20, 200); nn = rng.uniform(1, Nn * 0.8)
    mm = rng.lognormal(0, rng.uniform(0, 3), Nn)
    po = waterfill(np.sqrt(mm), nn)
    AVo = np.sum(mm * (1 / po - 1)); AVe = np.sum(mm * (Nn / nn - 1))
    ok &= AVo <= AVe * (1 + 1e-9); worst = max(worst, AVo / AVe)
res["capped_optimum_never_worse_500_random"] = bool(ok)
mm = np.full(50, 2.3); po = waterfill(np.sqrt(mm), 10.0)
res["constant_m_optimum_equals_equal_pi"] = bool(np.allclose(po, 10 / 50))

(OUT / "core_benchmark_checks.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
assert all(v is not False for v in res.values())
