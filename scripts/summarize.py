"""Summaries of the analytic grid: oracle value, estimated-design distributions, lambda
frontier, decision regions (constitution §11 conventions), pilot-cost economics.

    .venv/bin/python scripts/summarize.py
Outputs: results/summary/*.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dbt import designs as D  # noqa: E402
from dbt.setup import N_TARGET, NS, Target  # noqa: E402

AN = ROOT / "results" / "analytic"; OUT = ROOT / "results" / "summary"; OUT.mkdir(parents=True, exist_ok=True)
KEY = ["dgp", "sigma", "prevalence", "n", "estimand"]


def load():
    f = pd.read_csv(AN / "fixed_designs.csv")
    e = pd.read_parquet(AN / "estimated_designs_replicates.parquet")
    q = pd.read_parquet(AN / "planning_quality_replicates.parquet")
    return f, e, q


def oracle_value(f):
    f = f.copy(); f["de"] = f.design + "/" + f.estimator
    srs = f[f.de == "SRSWOR/EXP"][KEY + ["AV"]].rename(columns={"AV": "AV_srs"})
    eqp = f[f.de == "EQ-P/HT"][KEY + ["AV"]].rename(columns={"AV": "AV_eqp"})
    out = f.merge(srs, on=KEY).merge(eqp, on=KEY)
    out["rel_SRSWOR"] = out.AV / out.AV_srs
    out["rel_EQP"] = out.AV / out.AV_eqp
    out["relfix_SRSWOR"] = out.AV_fixed_hajek / out.AV_srs
    out.to_csv(OUT / "oracle_value.csv", index=False)
    return out


def conventional_best(ov):
    """Best conventional design per scenario in the fixed-size class: SRSWOR, ST-PROP, ST-EQD
    (exact) and PPS-A under fixed-size CPS (Hajek approx)."""
    rows = []
    for k, g in ov.groupby(KEY):
        cand = {"SRSWOR/EXP": g.loc[g.de == "SRSWOR/EXP", "AV"].iloc[0],
                "ST-PROP/EXP": g.loc[g.de == "ST-PROP/EXP", "AV"].iloc[0],
                "ST-EQD/EXP": g.loc[g.de == "ST-EQD/EXP", "AV"].iloc[0],
                "PPS-A/HT(fixed)": g.loc[g.de == "PPS-A/HT", "AV_fixed_hajek"].iloc[0]}
        b = min(cand, key=cand.get)
        rows.append(dict(zip(KEY, k), best_conv=b, AV_best_conv=cand[b], AV_srswor=cand["SRSWOR/EXP"],
                         AV_raor_fixed=g.loc[g.de == "RA-OR/HT", "AV_fixed_hajek"].iloc[0],
                         AV_raor_diff_fixed=g.loc[g.de == "RA-OR/DIFF-OR", "AV_fixed_hajek"].iloc[0],
                         AV_raor_poisson=g.loc[g.de == "RA-OR/HT", "AV"].iloc[0],
                         AV_eqp_poisson=g.loc[g.de == "EQ-P/HT", "AV"].iloc[0]))
    out = pd.DataFrame(rows)
    co = pd.read_csv(AN / "conventional_diff_oracle.csv")
    cob = co.groupby(KEY).AV.min().rename("AV_best_conv_diff_or").reset_index()
    out = out.merge(cob, on=KEY)
    out["AV_best_conv_diff_family"] = np.minimum(out.AV_best_conv, out.AV_best_conv_diff_or)
    return out


def estimated_summary(e, f, cb):
    ref_srs = f[(f.design == "SRSWOR") & (f.estimator == "EXP")][KEY + ["AV"]].rename(columns={"AV": "AV_srs"})
    ref_eqp = f[(f.design == "EQ-P") & (f.estimator == "HT")][KEY + ["AV"]].rename(columns={"AV": "AV_eqp"})
    e = e.merge(ref_srs, on=KEY).merge(ref_eqp, on=KEY).merge(cb[KEY + ["AV_best_conv", "best_conv"]], on=KEY)
    # same-rep equal-probability references for the DIFF regime
    d0 = e[(e.design == "DEF-0.0") & (e.estimator == "DIFF")][KEY + ["info", "rep", "AV"]].rename(columns={"AV": "AV_eqp_diff"})
    sd = e[(e.design == "SRSWOR") & (e.estimator == "DIFF")][KEY + ["info", "rep", "AV"]].rename(columns={"AV": "AV_srs_diff"})
    e = e.merge(d0, on=KEY + ["info", "rep"], how="left").merge(sd, on=KEY + ["info", "rep"], how="left")
    # SENSITIVITY: conventional designs under the same DIFF estimator (per replicate)
    cd = pd.read_parquet(AN / "conventional_diff_replicates.parquet")
    cdb = cd.groupby(KEY + ["info", "rep"]).AV.min().rename("AV_best_conv_diff").reset_index()
    e = e.merge(cdb, on=KEY + ["info", "rep"], how="left")
    e["AV_best_conv_family"] = np.where(e.estimator == "DIFF",
                                        np.minimum(e.AV_best_conv, e.AV_best_conv_diff), e.AV_best_conv)
    ht = e.estimator == "HT"
    e["rel_poisson_equal"] = np.where(ht, e.AV / e.AV_eqp, e.AV / e.AV_eqp_diff)
    e["rel_fixed_equal"] = np.where(ht, e.AV_fixed_hajek / e.AV_srs, e.AV_fixed_hajek / e.AV_srs_diff)
    e["rel_fixed_srswor_exp"] = np.where(e.AV_fixed_hajek.notna(), e.AV_fixed_hajek, e.AV) / e.AV_srs
    e["rel_fixed_bestconv"] = np.where(e.AV_fixed_hajek.notna(), e.AV_fixed_hajek, e.AV) / e.AV_best_conv_family
    q = lambda p: (lambda s: s.quantile(p))
    agg = e.groupby(KEY + ["estimator", "info", "design"]).agg(
        rel_fixed_equal_mean=("rel_fixed_equal", "mean"), rel_fixed_equal_median=("rel_fixed_equal", "median"),
        rel_fixed_equal_p90=("rel_fixed_equal", q(.9)), share_worse_than_equal=("rel_fixed_equal", lambda s: (s > 1).mean()),
        rel_bestconv_mean=("rel_fixed_bestconv", "mean"), rel_bestconv_median=("rel_fixed_bestconv", "median"),
        rel_bestconv_p90=("rel_fixed_bestconv", q(.9)), share_worse_than_bestconv=("rel_fixed_bestconv", lambda s: (s > 1).mean()),
        rel_srswor_exp_mean=("rel_fixed_srswor_exp", "mean"), rel_srswor_exp_p90=("rel_fixed_srswor_exp", q(.9)),
        rel_poisson_equal_mean=("rel_poisson_equal", "mean"), rel_poisson_equal_p90=("rel_poisson_equal", q(.9)),
        max_weight_median=("max_weight", "median"), max_weight_p90=("max_weight", q(.9)),
        cv_weights_median=("cv_weights", "median"), min_pi_median=("min_pi", "median")).reset_index()
    agg.to_csv(OUT / "estimated_summary.csv", index=False)
    return agg


def quality_summary(q):
    s = q.groupby(["dgp", "sigma", "prevalence", "info", "estimand"]).agg(
        fail_rate=("failed", "mean"), n_pos_plan_mean=("n_pos_plan", "mean"), n_pos_plan_min=("n_pos_plan", "min"),
        brier=("brier", "mean"), logloss=("logloss", "mean"), calib_in_large=("calib_in_large", "mean"),
        calib_slope=("calib_slope", "mean"), auc=("auc", "mean"),
        score_pearson_HT=("score_pearson_HT", "mean"), score_spearman_HT=("score_spearman_HT", "mean"),
        score_pearson_DIFF=("score_pearson_DIFF", "mean"), score_spearman_DIFF=("score_spearman_DIFF", "mean")).reset_index()
    s.to_csv(OUT / "planning_quality_summary.csv", index=False)
    return s


def decision_regions(cb, agg):
    """Mechanical application of constitution §11 conventions, fixed-size class.
    HT family: baseline = best conventional practice design (SRSWOR, ST-PROP, ST-EQD with the
    expansion estimator; PPS-A/HT fixed).  DIFF family: baseline = best of those and the same
    conventional designs under the difference estimator (oracle rows: oracle working mean)."""
    rows = []
    for _, r in cb.iterrows():
        k = {c: r[c] for c in KEY}
        for fam, av_or, base in (("HT", r.AV_raor_fixed, r.AV_best_conv),
                                 ("DIFF", r.AV_raor_diff_fixed, r.AV_best_conv_diff_family)):
            og_equal = 1 - av_or / r.AV_srswor
            og_conv = 1 - av_or / base
            sub = agg[(agg.dgp == k["dgp"]) & (agg.sigma == k["sigma"]) & (agg.prevalence == k["prevalence"]) &
                      (agg.n == k["n"]) & (agg.estimand == k["estimand"]) & (agg.estimator == fam)]
            for info, g in sub.groupby("info"):
                ra = g[g.design == "RA-EST"].iloc[0]
                defs = g[g.design.isin(["DEF-0.25", "DEF-0.5", "DEF-0.75"])]
                rel = lambda x: (x.rel_bestconv_mean <= 0.90) and (x.rel_bestconv_p90 <= 1.0)
                def_ok = [d.design for _, d in defs.iterrows() if rel(d)]
                if og_conv < 0.05 and og_equal < 0.05:
                    region = "A"      # equal probability effectively optimal (structural)
                elif og_conv < 0.05:
                    region = "A-conv" # conventional design captures the structure
                elif og_conv < 0.10:
                    region = "A-modest"  # oracle gain 5-10%: below the reliability convention
                elif rel(ra):
                    region = "B"      # estimated objective-aligned design reliably better
                elif def_ok:
                    region = "C"      # only a defensive mixture reliably better
                else:
                    region = "A-info" # oracle gain >=10% not reliably recoverable with this information
                rows.append(dict(**k, estimator_family=fam, info=info,
                                 baseline=r.best_conv if fam == "HT" else "best conventional (EXP or DIFF)",
                                 oracle_gain_vs_srswor=og_equal, oracle_gain_vs_bestconv=og_conv,
                                 raest_rel_bestconv_mean=ra.rel_bestconv_mean, raest_rel_bestconv_p90=ra.rel_bestconv_p90,
                                 raest_share_worse=ra.share_worse_than_bestconv,
                                 def_reliable=";".join(def_ok), region=region))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "decision_regions.csv", index=False)
    return out


def lambda_frontier(agg):
    cols = KEY + ["estimator", "info", "design", "rel_bestconv_mean", "rel_bestconv_median", "rel_bestconv_p90",
                  "share_worse_than_bestconv", "rel_fixed_equal_mean", "rel_fixed_equal_p90",
                  "max_weight_median", "max_weight_p90", "cv_weights_median"]
    fr = agg[agg.design.str.startswith("DEF") | (agg.design == "RA-EST")][cols].copy()
    fr["lambda"] = fr.design.map(lambda d: 1.0 if d == "RA-EST" else float(d.split("-")[1]))
    fr.sort_values(KEY + ["estimator", "info", "lambda"]).to_csv(OUT / "lambda_frontier.csv", index=False)
    return fr


def economics(cb):
    """Oracle maximum main-sample saving (fixed-size class, Hajek) vs SRSWOR and vs best
    conventional at n0; compared with dedicated-pilot sizes (Setting B)."""
    T = Target(); A = T.frame.A
    rows = []
    for _, r in cb[(cb.dgp == "MAIN")].iterrows():
        alpha, p, E = T.scenario("MAIN", r.sigma, r.prevalence)
        mu, v = D.moments(r.estimand, p, A); mHT = v + mu**2
        f_or = lambda n: D.av_hajek_fixed(mu, v, D.waterfill(np.sqrt(mHT), n))
        n_srs = D.eq_precision_n(f_or, r.AV_srswor, n_lo=5.0)
        n_conv = D.eq_precision_n(f_or, r.AV_best_conv, n_lo=5.0)
        rows.append(dict(sigma=r.sigma, prevalence=r.prevalence, n0=r.n, estimand=r.estimand,
                         best_conv=r.best_conv, n_oracle_match_srswor=n_srs, n_oracle_match_bestconv=n_conv,
                         max_saving_vs_srswor=r.n - n_srs, max_saving_vs_bestconv=r.n - n_conv,
                         pilot_500_exceeds_saving=500 > r.n - n_conv,
                         pilot_2000_exceeds_saving=2000 > r.n - n_conv))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "economics.csv", index=False)
    return out


def main():
    f, e, q = load()
    ov = oracle_value(f)
    cb = conventional_best(ov); cb.to_csv(OUT / "conventional_best.csv", index=False)
    agg = estimated_summary(e, f, cb)
    quality_summary(q)
    decision_regions(cb, agg)
    lambda_frontier(agg)
    economics(cb)
    print("summaries written to", OUT)


if __name__ == "__main__":
    main()
