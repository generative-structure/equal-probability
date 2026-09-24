"""Primary analytic grid (test_constitution.md §5-7): exact anticipated variances on the
fixed target frame; estimated designs averaged over R_PLAN independent planning replicates.

    .venv/bin/python scripts/analytic_grid.py
Outputs: results/analytic/*.parquet|csv, results/planning_phat/*.npz (for Monte Carlo)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dbt import designs as D, dgp, planning as P  # noqa: E402
from dbt.setup import (ESTIMANDS, LAMBDAS, NS, PREVS, R_PLAN, SIGMAS, N_TARGET,  # noqa: E402
                       Target, seed_for)
import os
R_PLAN = int(os.environ.get("DBT_R_PLAN", R_PLAN))   # smoke tests only; primary run uses 200
SUFFIX = os.environ.get("DBT_SUFFIX", "")

OUT = ROOT / "results" / ("analytic" + os.environ.get("DBT_SUFFIX", "")); OUT.mkdir(parents=True, exist_ok=True)
PH = ROOT / "results" / "planning_phat"; PH.mkdir(parents=True, exist_ok=True)
K = dgp.N_STRATA
MC_KEEP = {("MAIN", 0.0, 0.25), ("MAIN", 0.75, 0.25), ("MAIN", 1.5, 0.05), ("MAIN", 1.5, 0.25),
           ("MAIN", 1.5, 0.70), ("STRATCAP", 1.5, 0.25)}
MC_INFOS = {"COR-2000", "MIS-G-2000", "STRAT-2000"}


def fixed_designs(T, p, n, est):
    """Designs not depending on planning estimates. Returns list of dict rows."""
    A, N, st = T.frame.A, N_TARGET, T.strata
    mu, v = D.moments(est, p, A); mHT = v + mu**2
    rows = []
    def add(design, estimator, av, pi=None, **kw):
        r = dict(design=design, estimator=estimator, AV=av, **kw)
        if pi is not None:
            r.update(D.stability(pi))
            mu_r = mu if estimator == "HT" else np.zeros(N)
            r["AV_fixed_hajek"] = D.av_hajek_fixed(mu_r, v, pi)   # SENSITIVITY
        rows.append(r)
    pe = np.full(N, n / N)
    add("EQ-P", "HT", D.av_poisson(mHT, pe), pe)
    add("SRSWOR", "EXP", D.av_srswor(mu, v, n))
    nh_p = D.alloc_prop(st, n, K); nh_e = D.alloc_equal(st, n, K)
    add("ST-PROP", "EXP", D.av_stratified(mu, v, st, nh_p), nh=str(list(nh_p)))
    add("ST-EQD", "EXP", D.av_stratified(mu, v, st, nh_e), nh=str(list(nh_e)))
    pa = D.waterfill(A, n); add("PPS-A", "HT", D.av_poisson(mHT, pa), pa)
    po = D.waterfill(np.sqrt(mHT), n); add("RA-OR", "HT", D.av_poisson(mHT, po), po)
    for lam in LAMBDAS[1:-1]:
        pl = D.defensive(po, n, N, lam); add(f"DEF-{lam}-OR", "HT", D.av_poisson(mHT, pl), pl)
    # oracle model-assisted regime (working mean = true mu)
    add("EQ-P", "DIFF-OR", D.av_poisson(v, pe), pe)
    add("SRSWOR", "DIFF-OR", D.av_srswor(np.zeros(N), v, n))
    pdo = D.waterfill(np.sqrt(v), n); add("RA-OR", "DIFF-OR", D.av_poisson(v, pdo), pdo)
    return rows, dict(mu=mu, v=v, mHT=mHT, po=po, pdo=pdo, pa=pa)


def eq_precision_rows(T, p, n0, est):
    """Expected reviews to match SRSWOR/EXP at n0 (oracle & conventional designs)."""
    A, N, st = T.frame.A, N_TARGET, T.strata
    mu, v = D.moments(est, p, A); mHT = v + mu**2
    target = D.av_srswor(mu, v, n0)
    f = {
        "EQ-P/HT": lambda n: D.av_poisson(mHT, np.full(N, n / N)),
        "SRSWOR/EXP": lambda n: D.av_srswor(mu, v, n),
        "ST-PROP/EXP": lambda n: D.av_stratified(mu, v, st, D.alloc_prop(st, n, K, integer=False)),
        "ST-EQD/EXP": lambda n: D.av_stratified(mu, v, st, D.alloc_equal(st, n, K, integer=False)),
        "PPS-A/HT": lambda n: D.av_poisson(mHT, D.waterfill(A, n)),
        "RA-OR/HT": lambda n: D.av_poisson(mHT, D.waterfill(np.sqrt(mHT), n)),
        "EQ-P/DIFF-OR": lambda n: D.av_poisson(v, np.full(N, n / N)),
        "SRSWOR/DIFF-OR": lambda n: D.av_srswor(np.zeros(N), v, n),
        "RA-OR/DIFF-OR": lambda n: D.av_poisson(v, D.waterfill(np.sqrt(v), n)),
    }
    return [dict(design_estimator=k, n_equal_precision=D.eq_precision_n(g, target, n_lo=10.0))
            for k, g in f.items()]


def estimated_rows(T, p, E, phat, n, est, info, rep, S_hat=None):
    A, N, st = T.frame.A, N_TARGET, T.strata
    mu, v = D.moments(est, p, A); mHT = v + mu**2
    w = D.working_mean(est, phat, A)
    m_diff_true = v + (mu - w) ** 2
    rows = []
    def add(design, estimator, av, pi=None):
        r = dict(info=info, rep=rep, design=design, estimator=estimator, AV=av)
        if pi is not None:
            s = D.stability(pi); r.update(max_weight=s["max_weight"], cv_weights=s["cv_weights"],
                                          min_pi=s["min_pi"], n_certainty=s["n_certainty"])
            mu_r = mu if estimator == "HT" else (mu - w)
            r["AV_fixed_hajek"] = D.av_hajek_fixed(mu_r, v, pi)   # SENSITIVITY
        rows.append(r)
    # HT regime
    s_ht = np.sqrt(D.second_moment(est, "HT", phat, A))
    pr = D.waterfill(s_ht, n)
    for lam in LAMBDAS:
        pl = D.defensive(pr, n, N, lam)
        add("RA-EST" if lam == 1 else f"DEF-{lam}", "HT", D.av_poisson(mHT, pl), pl)
    # DIFF regime (planner believes phat; truth evaluated with m_diff_true)
    s_d = np.sqrt(np.maximum(D.second_moment(est, "DIFF", phat, A, phat=phat), 1e-300))
    prd = D.waterfill(s_d, n)
    for lam in LAMBDAS:
        pl = D.defensive(prd, n, N, lam)
        add("RA-EST" if lam == 1 else f"DEF-{lam}", "DIFF", D.av_poisson(m_diff_true, pl), pl)
    add("SRSWOR", "DIFF", D.av_srswor(mu - w, v, n))
    if S_hat is not None:
        nh = D.alloc_neyman(st, n, K, S_hat)
        add("ST-NEY-EST", "EXP", D.av_stratified(mu, v, st, nh))
    return rows


def neyman_S_hat(fr_plan, E_plan, bounds, est):
    s = dgp.assign_strata(fr_plan.A, bounds)
    y = E_plan if est == "rate" else fr_plan.A * E_plan
    return np.array([y[s == h].std(ddof=1) if (s == h).sum() > 1 else 0.0 for h in range(K)])


def main():
    t0 = time.time()
    T = Target()
    diag, fixed, eqp, est_rows, qual = [], [], [], [], []
    for dgp_name in ("MAIN", "STRATCAP"):
        infos = P.INFO_LEVELS if dgp_name == "MAIN" else {"STRAT-2000": ("STRAT", 2000)}
        for sigma in SIGMAS:
            for prev in PREVS:
                alpha, p, E = T.scenario(dgp_name, sigma, prev)
                d = dgp.diagnostics(p, T.frame.A)
                d.update({f"mean_p_G{g}": p[T.frame.G == g].mean() for g in range(3)})
                d.update({f"mean_p_S{h}": p[T.strata == h].mean() for h in range(K)})
                d.update(dgp=dgp_name, sigma=sigma, prevalence=prev, alpha=alpha,
                         realized_rate=E.mean(), realized_D=float((T.frame.A * E).sum()),
                         cv_A_sqrt_p=np.std(T.frame.A * np.sqrt(p)) / np.mean(T.frame.A * np.sqrt(p)))
                diag.append(d)
                key = dict(dgp=dgp_name, sigma=sigma, prevalence=prev)
                for n in NS:
                    for est in ESTIMANDS:
                        rows, _ = fixed_designs(T, p, n, est)
                        fixed += [dict(**key, n=n, estimand=est, **r) for r in rows]
                        eqp += [dict(**key, n0=n, estimand=est, **r) for r in eq_precision_rows(T, p, n, est)]
                # estimated designs
                for info, (model, m) in infos.items():
                    keep = (dgp_name, sigma, prev) in MC_KEEP and info in MC_INFOS
                    phats = []
                    for r in range(R_PLAN):
                        rng = np.random.default_rng(seed_for("plan", dgp_name, sigma, prev, info, r))
                        frp, Ep = P.draw_planning(m, dgp_name, sigma, alpha, T.bounds, rng)
                        phat, failed, npos = P.fit_predict(frp, Ep, model, T.frame, T.bounds)
                        q = P.quality(phat, p, E)
                        for est in ESTIMANDS:
                            A = T.frame.A
                            sh_ht = np.sqrt(D.second_moment(est, "HT", phat, A))
                            st_ht = np.sqrt(D.second_moment(est, "HT", p, A))
                            sh_d = np.sqrt(D.second_moment(est, "DIFF", phat, A, phat=phat))
                            st_d = np.sqrt(D.second_moment(est, "DIFF", p, A))
                            c1 = P.score_corr(sh_ht, st_ht); c2 = P.score_corr(sh_d, st_d)
                            qual.append(dict(**key, info=info, rep=r, estimand=est, failed=failed,
                                             n_pos_plan=npos, **q, score_pearson_HT=c1[0],
                                             score_spearman_HT=c1[1], score_pearson_DIFF=c2[0],
                                             score_spearman_DIFF=c2[1]))
                            S_hat = neyman_S_hat(frp, Ep, T.bounds, est) if info in ("COR-2000", "STRAT-2000") else None
                            for n in NS:
                                est_rows += [dict(**key, n=n, estimand=est, **x)
                                             for x in estimated_rows(T, p, E, phat, n, est, info, r, S_hat)]
                        if keep:
                            phats.append(phat.astype(np.float32))
                    if keep:
                        np.savez_compressed(PH / f"{dgp_name}_s{sigma}_p{prev}_{info}.npz", phat=np.array(phats))
                print(f"{dgp_name} sigma={sigma} prev={prev} done {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(diag).to_csv(OUT / "scenario_diagnostics.csv", index=False)
    pd.DataFrame(fixed).to_csv(OUT / "fixed_designs.csv", index=False)
    pd.DataFrame(eqp).to_csv(OUT / "equal_precision_reviews.csv", index=False)
    pd.DataFrame(est_rows).to_parquet(OUT / "estimated_designs_replicates.parquet", index=False)
    pd.DataFrame(qual).to_parquet(OUT / "planning_quality_replicates.parquet", index=False)
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
