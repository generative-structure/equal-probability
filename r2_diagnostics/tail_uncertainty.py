"""R2 post-freeze diagnostic: planning-level uncertainty for tail-heavy estimated-design
means (Online Appendix Table OA7 / main Section 7.6). Uses archived per-replicate
anticipated variances only (results/analytic/estimated_designs_replicates.parquet);
no new simulation. Writes only to r2_diagnostics/out/."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import summarize as S
OUT = Path(__file__).parent / "out"; OUT.mkdir(exist_ok=True)
S.OUT = OUT                                   # never write into results/summary
f, e, q = S.load()
ov = S.oracle_value(f); cb = S.conventional_best(ov)
KEY = S.KEY
e = e.merge(cb[KEY + ["AV_best_conv"]], on=KEY)
cd = pd.read_parquet(S.AN / "conventional_diff_replicates.parquet")
cdb = cd.groupby(KEY + ["info", "rep"]).AV.min().rename("AV_best_conv_diff").reset_index()
e = e.merge(cdb, on=KEY + ["info", "rep"], how="left")
e["base"] = np.where(e.estimator == "DIFF", np.minimum(e.AV_best_conv, e.AV_best_conv_diff), e.AV_best_conv)
e["rel"] = np.where(e.AV_fixed_hajek.notna(), e.AV_fixed_hajek, e.AV) / e.base
rng = np.random.default_rng(20260923)
rows = []
sel = e[(e.dgp == "MAIN") & (e.estimator == "HT") & (e["info"] == "COR-500") & (e.sigma == 0) & (e.prevalence == 0.05)]
for (est, n, design), g in sel.groupby(["estimand", "n", "design"]):
    x = np.sort(g.rel.to_numpy())[::-1]; R = x.size; m = x.mean()
    boot = rng.choice(x, (20000, R)).mean(1)
    rows.append(dict(estimand=est, n=n, design=design, R=R, mean=m, median=np.median(x),
                     se=x.std(ddof=1) / np.sqrt(R), boot_lo=np.quantile(boot, .025), boot_hi=np.quantile(boot, .975),
                     max=x[0], top1_share=x[0] / x.sum(), top5_share=x[:5].sum() / x.sum(),
                     top10_share=x[:10].sum() / x.sum(), mean_excl_top5=x[5:].mean(),
                     n_above_10=(x > 10).sum()))
out = pd.DataFrame(rows)
out.to_csv(OUT / "tail_planning_uncertainty.csv", index=False)
pd.set_option("display.width", 250)
print(out[out.design.isin(["RA-EST", "DEF-0.5"])].to_string())

# same summaries under the separation-aware fallback (separation_sensitivity.py output)
sp = OUT / "separation_sensitivity_replicates.parquet"
if sp.exists():
    s2 = pd.read_parquet(sp)
    s2 = s2[(s2.dgp == "MAIN") & (s2.estimator == "HT") & (s2["info"] == "COR-500") & (s2.sigma == 0) & (s2.prevalence == 0.05)]
    rows = []
    for (rule, est, n, design), g in s2.groupby(["rule", "estimand", "n", "design"]):
        x = np.sort(g.rel.to_numpy())[::-1]; R = x.size
        boot = rng.choice(x, (20000, R)).mean(1)
        rows.append(dict(rule=rule, estimand=est, n=n, design=design, R=R, mean=x.mean(), median=np.median(x),
                         se=x.std(ddof=1) / np.sqrt(R), boot_lo=np.quantile(boot, .025), boot_hi=np.quantile(boot, .975),
                         max=x[0], top1_share=x[0] / x.sum(), top5_share=x[:5].sum() / x.sum(),
                         top10_share=x[:10].sum() / x.sum(), mean_excl_top5=x[5:].mean(), n_above_10=(x > 10).sum(),
                         n_changed=int(g.changed.sum())))
    o2 = pd.DataFrame(rows); o2.to_csv(OUT / "tail_planning_uncertainty_by_rule.csv", index=False)
    print(o2[o2.design.isin(["RA-EST", "DEF-0.5"])].to_string())
