"""R2 post-freeze diagnostic: rerun the archived complete-procedure Monte Carlo with the
archived seeds and code (scripts/mc_complete_procedure.py, unmodified), keeping the
per-replicate estimates so that coverage can be summarized at the planning-fit level.
Writes only to r2_diagnostics/out/. Verifies the rerun reproduces the frozen summary."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import mc_complete_procedure as M

OUT = Path(__file__).parent / "out"; OUT.mkdir(exist_ok=True)
M.OUT = OUT
CAPT = []
_orig = M.summarize
def summarize(est, var, Ttrue, extra):
    CAPT.append((est.copy(), var.copy(), Ttrue))
    return _orig(est, var, Ttrue, extra)
M.summarize = summarize

df = M.run(10000, "r2_rerun")
frozen = pd.read_csv(ROOT / "results" / "monte_carlo" / "complete_procedure_final.csv")
num = frozen.select_dtypes("number").columns
diff = np.nanmax(np.abs(df[num].to_numpy(float) - frozen[num].to_numpy(float)))
print("max abs difference vs frozen summary:", diff)
assert list(df.design) == list(frozen.design)
np.savez_compressed(OUT / "mc_replicates.npz",
                    est=np.array([c[0] for c in CAPT]), var=np.array([c[1] for c in CAPT]),
                    T=np.array([c[2] for c in CAPT]), design=df.design.to_numpy(str),
                    scenario=df.scenario.to_numpy(str), estimand=df.estimand.to_numpy(str))

Z90 = norm.ppf(0.90)
out = []
for i, row in df.iterrows():
    est, var, Tt = CAPT[i]
    cov = (est - Z90 * np.sqrt(np.maximum(var, 0))) <= Tt
    learned = ("EST" in row.design) or ("DIFF(" in row.design) or ("DEF" in row.design)
    rec = dict(scenario=row.scenario, estimand=row.estimand, design=row.design,
               cov_lb90=cov.mean(), wilson_lo=row.cov_lb90_lo, wilson_hi=row.cov_lb90_hi,
               learned=learned)
    if learned:
        R = cov.size
        if row.design.startswith("CPS-"):
            g = np.arange(R) // int(np.ceil(R / M.CPS_PLAN_REPS))
        else:
            g = np.arange(R) % 200
        cj = pd.Series(cov).groupby(g).mean().to_numpy()
        J = cj.size
        se = cj.std(ddof=1) / np.sqrt(J)
        rec.update(n_fits=J, cluster_se=se, cluster_lo=cov.mean() - 1.96 * se,
                   cluster_hi=cov.mean() + 1.96 * se, fit_cov_min=cj.min(),
                   fit_cov_p10=np.quantile(cj, .1), fit_cov_p50=np.median(cj),
                   design_effect=(se**2) / (cov.mean() * (1 - cov.mean()) / R))
    out.append(rec)
res = pd.DataFrame(out)
res.to_csv(OUT / "coverage_planning_fit_level.csv", index=False)
print(res[res.learned].to_string())
