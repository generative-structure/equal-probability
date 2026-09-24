"""Summarize fallback_audit_replicates.csv (R2 post-freeze diagnostic) and record the fitted
probability range of separated fits retained by the archived rule."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from dbt import planning as P
from dbt.setup import Target, seed_for
OUT = Path(__file__).parent / "out"
d = pd.read_csv(OUT / "fallback_audit_replicates.csv")
d["warn"] = d.sm_warning.fillna("").str.len() > 0
d["nonconv"] = d.converged == False
d["beta25"] = d.max_abs_beta > 25
d["retained_sep"] = d.separated & ~d.fallback
g = d.groupby(["dgp", "sigma", "prevalence", "info"]).agg(
    fits=("rep", "size"), separated=("separated", "sum"), degenerate=("degenerate", "sum"),
    sm_warning=("warn", "sum"), nonconverged=("nonconv", "sum"), beta_gt_25=("beta25", "sum"),
    archived_fallback=("fallback", "sum"), separated_retained=("retained_sep", "sum")).reset_index()
g["separation_aware_fallback"] = g.archived_fallback + g.separated_retained
g.to_csv(OUT / "fallback_audit_summary.csv", index=False)
print(g[(g.separated > 0) | (g.archived_fallback > 0)].to_string())
print("totals:", d.shape[0], "fits;", int(d.separated.sum()), "separated;", int(d.fallback.sum()), "archived fallbacks;",
      int(d.retained_sep.sum()), "separated retained; warnings:", d.sm_warning.value_counts().to_dict())
r = d[d.retained_sep]
print("max|beta| of retained separated fits: %.2f-%.2f" % (r.max_abs_beta.min(), r.max_abs_beta.max()))
T = Target(); lo, hi = [], []
for x in r.itertuples():
    model, m = (P.INFO_LEVELS[x.info] if x.dgp == "MAIN" else ("STRAT", 2000))
    alpha, p, E = T.scenario(x.dgp, x.sigma, x.prevalence)
    rng = np.random.default_rng(seed_for("plan", x.dgp, x.sigma, x.prevalence, x.info, x.rep))
    frp, Ep = P.draw_planning(m, x.dgp, x.sigma, alpha, T.bounds, rng)
    ph, _, _ = P.fit_predict(frp, Ep, model, T.frame, T.bounds)
    lo.append(ph.min()); hi.append(ph.max())
print("retained separated fits: min phat range %.2e-%.2e; max phat range 1-%.2e..1-%.2e" %
      (min(lo), max(lo), 1 - max(hi), 1 - min(hi)))
print("share with min phat < 1e-7:", np.mean(np.array(lo) < 1e-7), " max phat > 1-1e-7:", np.mean(1 - np.array(hi) < 1e-7))
