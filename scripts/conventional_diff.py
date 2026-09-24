"""SENSITIVITY (added after the primary grid; see CHANGELOG): conventional designs under the
SAME model-assisted difference estimator, so that DIFF-family comparisons hold the estimator
fixed.  Planning fits are re-derived from the identical seeds used in analytic_grid.py.

Designs (all with working mean w from the planning model; oracle rows use w = mu):
  SRSWOR/DIFF (exact), ST-PROP/DIFF and ST-EQD/DIFF (exact stratified), PPS-A/DIFF (fixed-size,
  Hajek approximation).  RA-EST/DIFF fixed-size values come from the primary grid.

    .venv/bin/python scripts/conventional_diff.py
Output: results/analytic/conventional_diff_replicates.parquet, conventional_diff_oracle.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dbt import designs as D, dgp, planning as P  # noqa: E402
from dbt.setup import ESTIMANDS, NS, PREVS, R_PLAN, SIGMAS, Target, seed_for  # noqa: E402

K = dgp.N_STRATA


def conv(mu_r, v, T, n):
    A, st = T.frame.A, T.strata
    return {"SRSWOR/DIFF": D.av_srswor(mu_r, v, n),
            "ST-PROP/DIFF": D.av_stratified(mu_r, v, st, D.alloc_prop(st, n, K)),
            "ST-EQD/DIFF": D.av_stratified(mu_r, v, st, D.alloc_equal(st, n, K)),
            "PPS-A/DIFF(fixed)": D.av_hajek_fixed(mu_r, v, D.waterfill(A, n))}


def main():
    T = Target(); A = T.frame.A
    rows, orows = [], []
    for dgp_name in ("MAIN", "STRATCAP"):
        infos = P.INFO_LEVELS if dgp_name == "MAIN" else {"STRAT-2000": ("STRAT", 2000)}
        for sigma in SIGMAS:
            for prev in PREVS:
                alpha, p, E = T.scenario(dgp_name, sigma, prev)
                key = dict(dgp=dgp_name, sigma=sigma, prevalence=prev)
                for n in NS:
                    for est in ESTIMANDS:
                        mu, v = D.moments(est, p, A)
                        for k, av in conv(np.zeros_like(mu), v, T, n).items():
                            orows.append(dict(**key, n=n, estimand=est, design=k, AV=av))
                for info, (model, m) in infos.items():
                    for r in range(R_PLAN):
                        rng = np.random.default_rng(seed_for("plan", dgp_name, sigma, prev, info, r))
                        frp, Ep = P.draw_planning(m, dgp_name, sigma, alpha, T.bounds, rng)
                        phat, failed, npos = P.fit_predict(frp, Ep, model, T.frame, T.bounds)
                        for est in ESTIMANDS:
                            mu, v = D.moments(est, p, A)
                            w = D.working_mean(est, phat, A)
                            for n in NS:
                                for k, av in conv(mu - w, v, T, n).items():
                                    rows.append(dict(**key, n=n, estimand=est, info=info, rep=r, design=k, AV=av))
                print(dgp_name, sigma, prev, flush=True)
    pd.DataFrame(rows).to_parquet(ROOT / "results" / "analytic" / "conventional_diff_replicates.parquet", index=False)
    pd.DataFrame(orows).to_csv(ROOT / "results" / "analytic" / "conventional_diff_oracle.csv", index=False)


if __name__ == "__main__":
    main()
