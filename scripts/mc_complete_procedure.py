"""COMPLETE-PROCEDURE SENSITIVITY (Problem C-CP; constitution §8 + protocol amendment).

Design-based Monte Carlo on the fixed realized target population (E fixed per scenario).
Estimated designs cycle over the 200 independent planning replicates (rep r uses r mod 200).
Common random numbers: Poisson designs share uniforms; SRSWOR/stratified share random keys.

    .venv/bin/python scripts/mc_complete_procedure.py --reps 2000 --tag pilot
    .venv/bin/python scripts/mc_complete_procedure.py --reps 10000 --tag final
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dbt import cps as C, designs as D, dgp  # noqa: E402
from dbt.setup import N_TARGET, Target, seed_for  # noqa: E402

Z90, Z975 = norm.ppf(0.90), norm.ppf(0.975)
SCEN = {  # id: (dgp, sigma, prev, n)
    "S1": ("MAIN", 0.0, 0.25, 100), "S2": ("MAIN", 0.75, 0.25, 100),
    "S3": ("MAIN", 1.5, 0.05, 100), "S4": ("MAIN", 1.5, 0.05, 400),
    "S5": ("MAIN", 1.5, 0.25, 100), "S6": ("MAIN", 1.5, 0.70, 100),
    "S7": ("MAIN", 1.5, 0.70, 400), "S8": ("STRATCAP", 1.5, 0.25, 100),
}
CPS_SCEN = {"S2", "S5", "S6"}
CPS_PLAN_REPS = 20          # CPS-RA-EST uses the first 20 planning replicates (cost); documented
K = dgp.N_STRATA
OUT = ROOT / "results" / "monte_carlo"; OUT.mkdir(parents=True, exist_ok=True)


def load_phat(dg, sigma, prev, info):
    return np.load(ROOT / "results" / "planning_phat" / f"{dg}_s{sigma}_p{prev}_{info}.npz")["phat"].astype(float)


def summarize(est, var, Ttrue, extra):
    se = np.sqrt(np.maximum(var, 0))
    lb = est - Z90 * se
    cov = lb <= Ttrue
    c2 = (est - Z975 * se <= Ttrue) & (Ttrue <= est + Z975 * se)
    R = est.size
    ph = cov.mean()
    # Wilson 95% CI for coverage
    z = 1.96; den = 1 + z**2 / R
    mid = (ph + z**2 / (2 * R)) / den; hw = z * np.sqrt(ph * (1 - ph) / R + z**2 / (4 * R**2)) / den
    return dict(R=R, true_total=Ttrue, bias=est.mean() - Ttrue, rel_bias=(est.mean() - Ttrue) / Ttrue,
                mc_se_bias=est.std(ddof=1) / np.sqrt(R), emp_var=est.var(ddof=1),
                rmse=np.sqrt(np.mean((est - Ttrue) ** 2)), mean_var_hat=var.mean(),
                cov_lb90=ph, cov_lb90_lo=mid - hw, cov_lb90_hi=mid + hw, cov_2s95=c2.mean(),
                mean_lb_shortfall=np.mean(Ttrue - lb), mean_width95=np.mean(2 * Z975 * se),
                **extra)


def poisson_block(U, Pi, y, w=None):
    """HT (w None) or difference estimator for rows of U (reps x N) with row-specific Pi."""
    I = U < Pi
    if w is None:
        est = (I * y / Pi).sum(1); var = (I * y**2 * (1 - Pi) / Pi**2).sum(1)
    else:
        r = y - w
        est = w.sum(1) + (I * r / Pi).sum(1); var = (I * r**2 * (1 - Pi) / Pi**2).sum(1)
    return est, var, I.sum(1), (I * (y > 0)).sum(1)


def run(reps, tag, seed=20260925, chunk=250):
    T = Target(); A = T.frame.A; N = N_TARGET
    rows = []
    t0 = time.time()
    for sid, (dg, sigma, prev, n) in SCEN.items():
        alpha, p, E = T.scenario(dg, sigma, prev)
        info_c = "STRAT-2000" if dg == "STRATCAP" else "COR-2000"
        ph_c = load_phat(dg, sigma, prev, info_c)
        ph_m = load_phat(dg, sigma, prev, "MIS-G-2000") if dg == "MAIN" else None
        for est_name in ("rate", "dollars"):
            y = E if est_name == "rate" else A * E
            Ttrue = y.sum()
            mu, v = D.moments(est_name, p, A); mHT = v + mu**2
            # ---- fixed pi vectors
            pe = np.full(N, n / N); pa = D.waterfill(A, n); po = D.waterfill(np.sqrt(mHT), n)
            # ---- estimated pi (per planning replicate)
            def est_pis(ph_all, estimator):
                out = []
                for ph in ph_all:
                    s = np.sqrt(np.maximum(D.second_moment(est_name, estimator, ph, A, phat=ph if estimator == "DIFF" else None), 1e-300))
                    out.append(D.waterfill(s, n))
                return np.array(out)
            P_c = est_pis(ph_c, "HT"); P_cd = est_pis(ph_c, "DIFF")
            W_c = np.array([D.working_mean(est_name, ph, A) for ph in ph_c])
            P_m = est_pis(ph_m, "HT") if ph_m is not None else None
            designs = {
                "EQ-P/HT": ("P", lambda r: pe, None), "PPS-A/HT": ("P", lambda r: pa, None),
                "RA-OR/HT": ("P", lambda r: po, None),
                f"RA-EST({info_c})/HT": ("P", lambda r: P_c[r % 200], None),
                f"DEF-0.5({info_c})/HT": ("P", lambda r: 0.5 * pe + 0.5 * P_c[r % 200], None),
                f"EQ-P/DIFF({info_c})": ("P", lambda r: pe, "c"),
                f"RA-EST/DIFF({info_c})": ("P", lambda r: P_cd[r % 200], "c"),
                f"DEF-0.5/DIFF({info_c})": ("P", lambda r: 0.5 * pe + 0.5 * P_cd[r % 200], "c"),
            }
            if P_m is not None:
                designs["RA-EST(MIS-G-2000)/HT"] = ("P", lambda r: P_m[r % 200], None)
                designs["DEF-0.5(MIS-G-2000)/HT"] = ("P", lambda r: 0.5 * pe + 0.5 * P_m[r % 200], None)
            rng = np.random.default_rng(seed_for("mc", seed, sid, est_name))
            acc = {k: dict(est=[], var=[], size=[], nerr=[]) for k in designs}
            for k in ("SRSWOR/EXP", f"SRSWOR/DIFF({info_c})", "ST-PROP/EXP", "ST-EQD/EXP"):
                acc[k] = dict(est=[], var=[], size=[], nerr=[])
            nh_p = D.alloc_prop(T.strata, n, K); nh_e = D.alloc_equal(T.strata, n, K)
            done = 0
            while done < reps:
                c = min(chunk, reps - done); rr = np.arange(done, done + c)
                U = rng.random((c, N))           # CRN for all Poisson designs
                Kkeys = rng.random((c, N))       # CRN for SRSWOR / stratified
                for k, (_, pf, wflag) in designs.items():
                    Pi = np.array([pf(r) for r in rr]) if "EST" in k or "DEF" in k else pf(0)[None, :]
                    Wm = W_c[rr % 200] if wflag else None
                    e_, v_, s_, ne_ = poisson_block(U, Pi, y, Wm)
                    for key, val in zip(("est", "var", "size", "nerr"), (e_, v_, s_, ne_)):
                        acc[k][key].append(val)
                # SRSWOR (EXP and DIFF with COR planning mean)
                sel = np.argpartition(Kkeys, n, axis=1)[:, :n]
                ys = y[sel]; f = n / N
                acc["SRSWOR/EXP"]["est"].append(N * ys.mean(1))
                acc["SRSWOR/EXP"]["var"].append(N**2 * (1 - f) * ys.var(1, ddof=1) / n)
                Wr = W_c[rr % 200]; rs = ys - np.take_along_axis(Wr, sel, 1)
                acc[f"SRSWOR/DIFF({info_c})"]["est"].append(Wr.sum(1) + N * rs.mean(1))
                acc[f"SRSWOR/DIFF({info_c})"]["var"].append(N**2 * (1 - f) * rs.var(1, ddof=1) / n)
                for k in ("SRSWOR/EXP", f"SRSWOR/DIFF({info_c})"):
                    acc[k]["size"].append(np.full(c, n)); acc[k]["nerr"].append((ys > 0).sum(1))
                for k, nh in (("ST-PROP/EXP", nh_p), ("ST-EQD/EXP", nh_e)):
                    e_ = np.zeros(c); v_ = np.zeros(c); ne_ = np.zeros(c)
                    for h in range(K):
                        cols = np.flatnonzero(T.strata == h); Nh = cols.size; m_ = int(nh[h])
                        if m_ >= Nh:
                            e_ += y[cols].sum(); continue
                        s_ = cols[np.argpartition(Kkeys[:, cols], m_, axis=1)[:, :m_]]
                        yh = y[s_]
                        e_ += Nh * yh.mean(1); v_ += Nh**2 * (1 - m_ / Nh) * yh.var(1, ddof=1) / m_
                        ne_ += (yh > 0).sum(1)
                    acc[k]["est"].append(e_); acc[k]["var"].append(v_)
                    acc[k]["size"].append(np.full(c, n)); acc[k]["nerr"].append(ne_)
                done += c
            for k, a in acc.items():
                est = np.concatenate(a["est"]); var = np.concatenate(a["var"])
                size = np.concatenate(a["size"]); nerr = np.concatenate(a["nerr"])
                rows.append(dict(scenario=sid, dgp=dg, sigma=sigma, prevalence=prev, n=n,
                                 estimand=est_name, design=k,
                                 **summarize(est, var, Ttrue, dict(mean_size=size.mean(), sd_size=size.std(),
                                                                  zero_error_freq=float(np.mean(nerr == 0))))))
            # ---- fixed-size CPS sensitivity
            if sid in CPS_SCEN:
                crng = np.random.default_rng(seed_for("cps", seed, sid, est_name))
                for k, pis in (("CPS-RA-OR/HT", [po]), (f"CPS-RA-EST({info_c})/HT", list(P_c[:CPS_PLAN_REPS]))):
                    ests, vars_ = [], []
                    per = int(np.ceil(reps / len(pis)))
                    for piv in pis:
                        d = C.CPS(piv)
                        I = d.sample(crng, per)
                        ests.append((I * y / piv).sum(1)); vars_.append(C.deville_var(y, piv, I))
                    est = np.concatenate(ests)[:reps]; var = np.concatenate(vars_)[:reps]
                    rows.append(dict(scenario=sid, dgp=dg, sigma=sigma, prevalence=prev, n=n,
                                     estimand=est_name, design=k,
                                     **summarize(est, var, Ttrue, dict(mean_size=n, sd_size=0.0,
                                                                      zero_error_freq=np.nan,
                                                                      var_estimator="Deville approx"))))
            print(f"{sid} {est_name} done {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"complete_procedure_{tag}.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--tag", default="pilot"); a = ap.parse_args()
    run(a.reps, a.tag)
