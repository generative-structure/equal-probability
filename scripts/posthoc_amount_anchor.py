"""POST HOC diagnostic (added after seeing Fig. 3; see CHANGELOG): for the improper-dollar
estimand the prespecified defensive anchor (equal probability) is itself far from the
objective-appropriate conventional design (PPS on amount).  This diagnostic evaluates the
alternative anchor  pi = (1-lam) pi_PPS-A + lam pi_RA-EST  (HT, dollars), with planning fits
re-derived from the identical seeds.  The mixture keeps pi >= (1-lam) pi_PPS-A.

    .venv/bin/python scripts/posthoc_amount_anchor.py
Output: results/posthoc/amount_anchor_frontier.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dbt import designs as D, planning as P  # noqa: E402
from dbt.setup import LAMBDAS, NS, PREVS, R_PLAN, SIGMAS, Target, seed_for  # noqa: E402

OUT = ROOT / "results" / "posthoc"; OUT.mkdir(parents=True, exist_ok=True)


def main():
    T = Target(); A = T.frame.A
    cb = pd.read_csv(ROOT / "results" / "summary" / "conventional_best.csv")
    rows = []
    infos = {k: P.INFO_LEVELS[k] for k in ("COR-10000", "COR-2000", "COR-500", "MIS-G-2000")}
    for sigma in SIGMAS:
        for prev in PREVS:
            alpha, p, E = T.scenario("MAIN", sigma, prev)
            mu, v = D.moments("dollars", p, A); m = v + mu**2
            pa = {n: D.waterfill(A, n) for n in NS}
            for info, (model, mm) in infos.items():
                for r in range(R_PLAN):
                    rng = np.random.default_rng(seed_for("plan", "MAIN", sigma, prev, info, r))
                    frp, Ep = P.draw_planning(mm, "MAIN", sigma, alpha, T.bounds, rng)
                    phat, failed, _ = P.fit_predict(frp, Ep, model, T.frame, T.bounds)
                    s = np.sqrt(D.second_moment("dollars", "HT", phat, A))
                    for n in NS:
                        pr = D.waterfill(s, n)
                        base = cb[(cb.dgp == "MAIN") & (cb.sigma == sigma) & (cb.prevalence == prev) &
                                  (cb.n == n) & (cb.estimand == "dollars")].iloc[0]
                        for lam in LAMBDAS:
                            pl = (1 - lam) * pa[n] + lam * pr
                            rows.append(dict(sigma=sigma, prevalence=prev, n=n, info=info, rep=r, anchor="PPS-A",
                                             lam=lam, rel_bestconv_fixed=D.av_hajek_fixed(mu, v, pl) / base.AV_best_conv,
                                             rel_bestconv_poisson=D.av_poisson(m, pl) / base.AV_best_conv,
                                             max_weight=float((1 / pl).max())))
            print(sigma, prev, flush=True)
    df = pd.DataFrame(rows)
    q = lambda x: (lambda s: s.quantile(x))
    agg = df.groupby(["sigma", "prevalence", "n", "info", "anchor", "lam"]).agg(
        rel_mean=("rel_bestconv_fixed", "mean"), rel_median=("rel_bestconv_fixed", "median"),
        rel_p90=("rel_bestconv_fixed", q(.9)), share_worse=("rel_bestconv_fixed", lambda s: (s > 1).mean()),
        max_weight_p90=("max_weight", q(.9))).reset_index()
    agg.to_csv(OUT / "amount_anchor_frontier.csv", index=False)
    print(agg[(agg.n == 100)].round(3).to_string())


if __name__ == "__main__":
    main()
