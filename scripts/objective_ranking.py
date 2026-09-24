"""Objective-ranking diagnostic (protocol amendment §7).

For each complete-procedure scenario x estimand, rank the designs under: anticipated variance
(analytic, model expectation), empirical RMSE, mean 95% interval width, mean 90% lower-bound
shortfall T-LB (decision metric; designs with material undercoverage per constitution §11 are
not ranked on it and are flagged), and maximum survey weight.

    .venv/bin/python scripts/objective_ranking.py
Outputs: results/objective_ranking_comparison.csv, results/objective_ranking_stability.csv
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "results" / "analytic"
KEY = ["dgp", "sigma", "prevalence", "n", "estimand"]
import sys
sys.path.insert(0, str(ROOT / "src"))
from dbt.setup import Target  # noqa: E402
NH = np.bincount(Target().strata, minlength=5)


def analytic_lookup():
    f = pd.read_csv(AN / "fixed_designs.csv")
    e = pd.read_parquet(AN / "estimated_designs_replicates.parquet")
    em = e.groupby(KEY + ["info", "design", "estimator"]).agg(
        AV=("AV", "mean"), AVfix=("AV_fixed_hajek", "mean"), maxw=("max_weight", "median")).reset_index()
    em20 = e[e.rep < 20].groupby(KEY + ["info", "design", "estimator"]).AV_fixed_hajek.mean().rename("AVfix20").reset_index()
    em = em.merge(em20, on=KEY + ["info", "design", "estimator"], how="left")
    return f, em


def analytic_for(row, f, em):
    k = {c: row[c] for c in KEY}
    sel = lambda df: df[(df.dgp == k["dgp"]) & (df.sigma == k["sigma"]) & (df.prevalence == k["prevalence"]) &
                        (df.n == k["n"]) & (df.estimand == k["estimand"])]
    F, E = sel(f), sel(em)
    d = row["design"]; N, n = 20000, k["n"]
    def fx(des, est, col="AV"):
        r = F[(F.design == des) & (F.estimator == est)].iloc[0]
        return r[col], r.get("max_weight", np.nan)
    def ex(des, est, info, col="AV"):
        r = E[(E.design == des) & (E.estimator == est) & (E["info"] == info)].iloc[0]
        return r[col], r["maxw"]
    info = re.search(r"\(([^)]*)\)", d).group(1) if "(" in d else None
    if d == "EQ-P/HT": return fx("EQ-P", "HT")
    if d == "PPS-A/HT": return fx("PPS-A", "HT")
    if d == "RA-OR/HT": return fx("RA-OR", "HT")
    if d == "SRSWOR/EXP": return fx("SRSWOR", "EXP")[0], N / n
    if d in ("ST-PROP/EXP", "ST-EQD/EXP"):
        des = d.split("/")[0]; r = F[(F.design == des) & (F.estimator == "EXP")].iloc[0]
        nh = np.array(eval(r["nh"]))
        return r["AV"], float(np.max(NH / nh))
    if d == "CPS-RA-OR/HT": return fx("RA-OR", "HT", "AV_fixed_hajek")
    if d.startswith("CPS-RA-EST"): return ex("RA-EST", "HT", info, "AVfix20")
    if d.startswith("RA-EST(") and d.endswith("/HT"): return ex("RA-EST", "HT", info)
    if d.startswith("DEF-0.5(") and d.endswith("/HT"): return ex("DEF-0.5", "HT", info)
    if d.startswith("EQ-P/DIFF"): return ex("DEF-0.0", "DIFF", info)
    if d.startswith("RA-EST/DIFF"): return ex("RA-EST", "DIFF", info)
    if d.startswith("DEF-0.5/DIFF"): return ex("DEF-0.5", "DIFF", info)
    if d.startswith("SRSWOR/DIFF"): return ex("SRSWOR", "DIFF", info)[0], N / n
    raise KeyError(d)


def main():
    mc = pd.read_csv(ROOT / "results" / "monte_carlo" / "complete_procedure_final.csv")
    f, em = analytic_lookup()
    rows = []
    for _, r in mc.iterrows():
        av, mw = analytic_for(r, f, em)
        est = "HT" if r.design.endswith("/HT") else ("EXP" if r.design.endswith("/EXP") else "DIFF")
        design = r.design.split("/")[0]
        inf = "Wald LB90/2s95; " + {"HT": "Poisson HT var est" if not design.startswith("CPS") else "Deville approx var est",
                                    "EXP": "SRSWOR/stratified var est", "DIFF": "residual var est"}[est]
        under = r.cov_lb90_hi < 0.88
        rows.append(dict(scenario_id=r.scenario, estimand=r.estimand, estimator=est, sampling_design=design,
                         design_label=r.design, inference_method=inf, anticipated_variance=av,
                         empirical_rmse=r.rmse, mean_width95=r.mean_width95, lb_shortfall=r.mean_lb_shortfall,
                         max_weight=mw, coverage_lb90=r.cov_lb90, coverage_lb90_ci=f"[{r.cov_lb90_lo:.3f},{r.cov_lb90_hi:.3f}]",
                         coverage_2s95=r.cov_2s95, material_undercoverage=under, rel_bias=r.rel_bias))
    df = pd.DataFrame(rows)
    out, stab = [], []
    for (s, e), g in df.groupby(["scenario_id", "estimand"]):
        g = g.copy()
        g["anticipated_variance_rank"] = g.anticipated_variance.rank(method="min")
        g["empirical_rmse_rank"] = g.empirical_rmse.rank(method="min")
        g["interval_width_rank"] = g.mean_width95.rank(method="min")
        ok = ~g.material_undercoverage
        g["decision_metric_rank"] = np.nan
        g.loc[ok, "decision_metric_rank"] = g.loc[ok, "lb_shortfall"].rank(method="min")
        g["weight_instability_rank"] = g.max_weight.rank(method="min")
        g["notes"] = np.where(g.material_undercoverage, "material undercoverage of 90% LB: not ranked on decision metric", "")
        out.append(g)
        top = lambda col: g.loc[g[col].idxmin(), "design_label"] if g[col].notna().any() else None
        rho = lambda a, b: spearmanr(g[a], g[b], nan_policy="omit").statistic
        stab.append(dict(scenario_id=s, estimand=e, n_designs=len(g),
                         best_by_AV=top("anticipated_variance"), best_by_RMSE=top("empirical_rmse"),
                         best_by_width=top("mean_width95"), best_by_decision_metric=top("decision_metric_rank"),
                         rho_AV_RMSE=rho("anticipated_variance_rank", "empirical_rmse_rank"),
                         rho_AV_width=rho("anticipated_variance_rank", "interval_width_rank"),
                         rho_AV_decision=rho("anticipated_variance_rank", "decision_metric_rank"),
                         n_material_undercoverage=int(g.material_undercoverage.sum()),
                         undercovering=";".join(g.loc[g.material_undercoverage, "design_label"])))
    out = pd.concat(out)
    cols = ["scenario_id", "estimand", "estimator", "sampling_design", "design_label", "inference_method",
            "anticipated_variance_rank", "empirical_rmse_rank", "interval_width_rank", "coverage_lb90",
            "coverage_lb90_ci", "coverage_2s95", "decision_metric_rank", "weight_instability_rank",
            "anticipated_variance", "empirical_rmse", "mean_width95", "lb_shortfall", "max_weight",
            "rel_bias", "material_undercoverage", "notes"]
    out[cols].to_csv(ROOT / "results" / "objective_ranking_comparison.csv", index=False)
    pd.DataFrame(stab).to_csv(ROOT / "results" / "objective_ranking_stability.csv", index=False)
    print(pd.DataFrame(stab).round(3).to_string())


if __name__ == "__main__":
    main()
