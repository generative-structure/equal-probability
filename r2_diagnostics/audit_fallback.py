"""R2 post-freeze diagnostic (not part of the frozen archive): audit the planning-fit
fallback rule. Regenerates every planning sample with the archived seeds, reproduces the
archived fit, and records separately (i) mathematical separation (LP check, Konis 2007),
(ii) statsmodels warnings, (iii) convergence / |beta|>25 triggers, (iv) the archived
fallback decision. Frozen outputs are read only; nothing in results/ is modified."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy.optimize import linprog
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from dbt import planning as P
from dbt.setup import PREVS, R_PLAN, SIGMAS, Target, seed_for

def separated(X, y):
    """True if some b has (2y-1)*Xb >= 0 for all i and > 0 for at least one i
    (complete or quasi-complete separation => no finite MLE). A b with every
    margin zero is a null-space direction (rank deficiency), not separation.
    LP: max sum (2y-1)Xb subject to the margins >= 0 and |b|<=1; a positive
    optimum requires a strictly positive margin."""
    s = 2 * y - 1
    Xs = X * s[:, None]
    res = linprog(-Xs.sum(0), A_ub=-Xs, b_ub=np.zeros(len(y)),
                  bounds=[(-1, 1)] * X.shape[1], method="highs")
    return bool(res.status == 0 and -res.fun > 1e-7)

def main():
    T = Target(); rows = []
    for dgp_name in ("MAIN", "STRATCAP"):
        infos = P.INFO_LEVELS if dgp_name == "MAIN" else {"STRAT-2000": ("STRAT", 2000)}
        for sigma in SIGMAS:
            for prev in PREVS:
                alpha, p, E = T.scenario(dgp_name, sigma, prev)
                for info, (model, m) in infos.items():
                    for r in range(R_PLAN):
                        rng = np.random.default_rng(seed_for("plan", dgp_name, sigma, prev, info, r))
                        frp, Ep = P.draw_planning(m, dgp_name, sigma, alpha, T.bounds, rng)
                        npos = int(Ep.sum())
                        rec = dict(dgp=dgp_name, sigma=sigma, prevalence=prev, info=info, rep=r,
                                   n_pos=npos, degenerate=npos in (0, m))
                        X = P.design_matrix(frp, P.MODELS[model], T.bounds)
                        rec["rank_deficient"] = np.linalg.matrix_rank(X) < X.shape[1]
                        rec["separated"] = rec["degenerate"] or separated(X, Ep)
                        if model == "STRAT":
                            s = X[:, 1:].argmax(1) + X[:, 1:].any(1)
                            ys = [Ep[s == h] for h in range(5)]
                            rec["strata_all_pos"] = sum(len(v) > 0 and v.min() == 1 for v in ys)
                            rec["strata_all_neg"] = sum(len(v) > 0 and v.max() == 0 for v in ys)
                        warn, conv, bmax = "", np.nan, np.nan
                        if not rec["degenerate"]:
                            with warnings.catch_warnings(record=True) as w:
                                warnings.simplefilter("always")
                                try:
                                    res = sm.GLM(Ep, X, family=sm.families.Binomial()).fit(maxiter=100)
                                    conv, bmax = bool(res.converged), float(np.abs(res.params).max())
                                except Exception as e:
                                    warn = "EXC:" + type(e).__name__
                            warn += ";".join(sorted({type(x.message).__name__ if not isinstance(x.message, str) else "str" for x in w}))
                        rec.update(sm_warning=warn, converged=conv, max_abs_beta=bmax)
                        _, failed, _ = P.fit_predict(frp, Ep, model, T.frame, T.bounds)
                        rec["fallback"] = failed
                        rows.append(rec)
                print(dgp_name, sigma, prev, flush=True)
    df = pd.DataFrame(rows)
    out = Path(__file__).parent / "out"; out.mkdir(exist_ok=True)
    df.to_csv(out / "fallback_audit_replicates.csv", index=False)

if __name__ == "__main__":
    main()
