"""R2 labeled correction/sensitivity: separation-aware planning fallback.

The archived fit_predict() documents fallback 'after separation' but detects separation only
indirectly (statsmodels warnings, non-convergence, |beta|>25). The fallback audit
(audit_fallback.py) shows that statsmodels 0.15 issued no separation warning and that 218
LP-detected separated fits were retained with |beta| in (20, 25). This script re-evaluates
every planning condition containing such fits with the documented rule implemented
faithfully: intercept-only fallback when the fit is separated (LP check) OR any archived
trigger fires. All other planning fits are identical to the archive. Writes only to
r2_diagnostics/out/."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
import analytic_grid as G, conventional_diff as CD, summarize as S
from dbt import designs as D, planning as P
from dbt.setup import ESTIMANDS, NS, R_PLAN, Target, seed_for
from audit_fallback import separated

OUT = Path(__file__).parent / "out"
aud = pd.read_csv(OUT / "fallback_audit_replicates.csv")
chg = aud[aud.separated & ~aud.fallback]
CELLS = sorted({(r.dgp, r.sigma, r.prevalence, r.info) for r in chg.itertuples()})
print("affected planning conditions:", CELLS)

T = Target(); A = T.frame.A
rows, crow = [], []
for dg, sigma, prev, info in CELLS:
    model, m = (P.INFO_LEVELS[info] if dg == "MAIN" else ("STRAT", 2000))
    alpha, p, E = T.scenario(dg, sigma, prev)
    key = dict(dgp=dg, sigma=sigma, prevalence=prev)
    for r in range(R_PLAN):
        rng = np.random.default_rng(seed_for("plan", dg, sigma, prev, info, r))
        frp, Ep = P.draw_planning(m, dg, sigma, alpha, T.bounds, rng)
        phat_a, failed, _ = P.fit_predict(frp, Ep, model, T.frame, T.bounds)
        X = P.design_matrix(frp, P.MODELS[model], T.bounds)
        sep = separated(X, Ep) if 0 < Ep.sum() < m else True
        for rule, phat in (("archived", phat_a),
                           ("separation-aware", np.full(T.frame.N, np.clip(Ep.mean(), 1e-6, 1 - 1e-6)) if (sep and not failed) else phat_a)):
            for est in ESTIMANDS:
                mu, v = D.moments(est, p, A); w = D.working_mean(est, phat, A)
                for n in NS:
                    rows += [dict(**key, n=n, estimand=est, rule=rule, changed=bool(sep and not failed), **x)
                             for x in G.estimated_rows(T, p, E, phat, n, est, info, r)]
                    for k, av in CD.conv(mu - w, v, T, n).items():
                        crow.append(dict(**key, n=n, estimand=est, rule=rule, info=info, rep=r, design=k, AV=av))
    print(dg, sigma, prev, info, flush=True)
e = pd.DataFrame(rows); cd = pd.DataFrame(crow)
f = pd.read_csv(S.AN / "fixed_designs.csv"); S.OUT = OUT
cb = S.conventional_best(S.oracle_value(f)); KEY = S.KEY
e = e.merge(cb[KEY + ["AV_best_conv"]], on=KEY)
cdb = cd.groupby(KEY + ["rule", "info", "rep"]).AV.min().rename("AV_bcd").reset_index()
e = e.merge(cdb, on=KEY + ["rule", "info", "rep"], how="left")
e["base"] = np.where(e.estimator == "DIFF", np.minimum(e.AV_best_conv, e.AV_bcd), e.AV_best_conv)
e["rel"] = np.where(e.AV_fixed_hajek.notna(), e.AV_fixed_hajek, e.AV) / e.base
q9 = lambda s: s.quantile(.9)
agg = e.groupby(KEY + ["info", "estimator", "design", "rule"]).agg(
    rel_mean=("rel", "mean"), rel_median=("rel", "median"), rel_p90=("rel", q9),
    maxw_p90=("max_weight", q9), n_changed=("changed", "sum")).reset_index()
agg.to_csv(OUT / "separation_sensitivity_summary.csv", index=False)
e.to_parquet(OUT / "separation_sensitivity_replicates.parquet", index=False)
# check: archived-rule rows reproduce the frozen summary
fr = pd.read_csv(ROOT / "results" / "summary" / "estimated_summary.csv")
chk = agg[agg.rule == "archived"].merge(fr, on=KEY + ["info", "estimator", "design"])
print("max |archived-rule mean - frozen mean|:", float(np.max(np.abs(chk.rel_mean - chk.rel_bestconv_mean))))
pd.set_option("display.width", 250)
w = agg[agg.design.isin(["RA-EST", "DEF-0.5"])].pivot_table(index=KEY + ["info", "estimator", "design"], columns="rule",
                                                           values=["rel_mean", "rel_median", "rel_p90"])
print(w.to_string())
