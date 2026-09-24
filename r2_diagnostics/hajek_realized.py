"""R2 post-freeze diagnostic: like-with-like check of the Hajek (1964) approximation.
For each conditional-Poisson (CPS) Monte Carlo cell, compares the empirical CPS variance
of the HT estimator on the realized outcome vector Y with the Hajek quadratic form H(Y)
evaluated on that same Y. Uses the per-draw estimates saved by coverage_by_fit.py (the
archived Monte Carlo rerun with archived seeds). Writes only to r2_diagnostics/out/."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import mc_complete_procedure as M
from dbt import designs as D
from dbt.setup import Target

OUT = Path(__file__).parent / "out"
Z = np.load(OUT / "mc_replicates.npz")

def H(y, pi):
    c = pi * (1 - pi)
    return float(np.sum(c * y**2 / pi**2) - np.sum(c * y / pi) ** 2 / c.sum())

T = Target(); A = T.frame.A; rows = []
for sid in sorted(M.CPS_SCEN):
    dg, sigma, prev, n = M.SCEN[sid]
    alpha, p, E = T.scenario(dg, sigma, prev)
    ph_c = M.load_phat(dg, sigma, prev, "COR-2000")
    for est in ("rate", "dollars"):
        y = E if est == "rate" else A * E
        mu, v = D.moments(est, p, A); mHT = v + mu**2
        po = D.waterfill(np.sqrt(mHT), n)
        pis = {"CPS-RA-OR/HT": [po],
               "CPS-RA-EST(COR-2000)/HT": [D.waterfill(np.sqrt(D.second_moment(est, "HT", ph, A)), n)
                                           for ph in ph_c[:M.CPS_PLAN_REPS]]}
        for k, plist in pis.items():
            i = np.flatnonzero((Z["scenario"] == sid) & (Z["estimand"] == est) & (Z["design"] == k))[0]
            e = Z["est"][i]; Tt = float(Z["T"][i]); R = e.size
            per = int(np.ceil(R / len(plist)))
            # per-vector empirical variance about the true total (HT is design-unbiased)
            mse = np.array([np.mean((e[j*per:(j+1)*per] - Tt) ** 2) for j in range(len(plist))])
            hy = np.array([H(y, piv) for piv in plist])
            av = np.array([D.av_hajek_fixed(mu, v, piv) for piv in plist])
            sq = (e - Tt) ** 2
            ratio_h = mse.mean() / hy.mean()
            se_ratio = sq.std(ddof=1) / np.sqrt(R) / hy.mean()
            rows.append(dict(scenario=sid, estimand=est, design=k, draws=R,
                             emp_over_HY=ratio_h, emp_over_HY_mcse=se_ratio,
                             emp_over_AV=mse.mean() / av.mean(), HY_over_AV=hy.mean() / av.mean()))
res = pd.DataFrame(rows)
res.to_csv(OUT / "hajek_realized_check.csv", index=False)
print(res.to_string())
