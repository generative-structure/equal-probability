# Design specification

**Code:**
- `src/dbt/designs.py`: allocation and exact variances;
- `src/dbt/planning.py`: planning models;
- `src/dbt/cps.py`: fixed-size conditional Poisson;
- `scripts/analytic_grid.py`: analytic grid;
- `scripts/mc_complete_procedure.py`: Monte Carlo.

**Notation:**
- N = 20,000.
- n is the expected sample size (Poisson) or the fixed sample size (SRSWOR, stratified, CPS).
- A is the payment amount.
- p is the true risk, and p̂ comes from the independent planning model.
- For the rate estimand, y = E; for dollars, y = A·E.
- μ = E[y | X] and v = Var(y | X).

**Problem labels.** Every row names its complete problem (`PROTOCOL_AMENDMENT_OBJECTIVE_DEPENDENCE.md` §B):
- **A-HT:** direct HT, anticipated variance;
- **A-EXP:** SRSWOR/stratified expansion estimator;
- **B-DIFF:** model-assisted difference estimator;
- **C-CP:** the complete confidence procedure.

## Equal probability

| ID | Design | Estimator | Exact anticipated variance | Variance estimator (Monte Carlo) |
|---|---|---|---|---|
| EQ-P/HT | Poisson, π = n/N | HT Σ_s y/π | Σ E[y²](N/n − 1) | Σ_s y²(1 − π)/π² |
| EQ-P/DIFF(info) | Poisson, π = n/N | Σ_U w + Σ_s (y − w)/π, with w = p̂ (rate) or A·p̂ (dollars) | Σ[v + (μ − w)²](N/n − 1) | Σ_s (y − w)²(1 − π)/π² |
| SRSWOR/EXP | fixed n, without replacement | N·ȳ_s | N²(1 − f)·E[S²]/n | N²(1 − f)s²/n |
| SRSWOR/DIFF(info) | fixed n | Σ_U w + N·(mean of residuals) | N²(1 − f)·E[S²_{y−w}]/n | N²(1 − f)s²_r/n |

E[S²] is the exact model expectation of the finite-population variance for independent units: [Σ(v + μ²) − (Σv + (Σμ)²)/N]/(N − 1).

**DIFF-OR** rows use the oracle working mean w = μ (a benchmark only).

## Conventional amount stratification (documented public practice)

**Strata.** Five strata holding equal total dollars on the target frame (PERM style). Boundaries: $384, $769, $1,385, $2,745. Counts: 12,694 / 3,776 / 2,004 / 1,081 / 445. The boundaries are fixed from amounts only and are never tuned on outcomes.

- **ST-PROP/EXP:** proportional allocation n_h ∝ N_h. Largest-remainder rounding, 2 ≤ n_h ≤ N_h.
- **ST-EQD/EXP:** equal allocation n_h = n/5, the PERM convention. It gives larger π to high-dollar strata.
- **ST-NEY-EST/EXP** (kept separate): Neyman allocation n_h ∝ N_h·Ŝ_h. Ŝ_h is the planning-sample SD of y within stratum, from COR-2000 or STRAT-2000 planning data.
- **Estimator and variance:** Σ_h N_h·ȳ_h, with exact anticipated variance Σ_h N_h²(1 − f_h)E[S_h²]/n_h. The Monte Carlo uses the matching estimator with s_h².

## Size-only unequal probability

**PPS-A/HT:** Poisson with π ∝ A, water-filled. This is the documented VA OIG purchase-amount PPS and penny-sampling practice. It is the direct-HT optimum for dollars **only when p is constant**.

## Objective-aligned (risk-aligned) Poisson designs

Every row is water-filled: π = min(1, λ·score), with Σπ = n solved exactly, never clipped and renormalized. The score depends on the problem:

| Problem | Oracle score (RA-OR) | Estimated score (RA-EST) | AV evaluated with the truth |
|---|---|---|---|
| A-HT, rate | √p | √p̂ | Σ p(1/π − 1) |
| A-HT, dollars | A√p | A√p̂ | Σ A²p(1/π − 1) |
| B-DIFF, rate | √(p(1 − p)) (w = p) | √(p̂(1 − p̂)) (w = p̂) | Σ[p(1 − p) + (p − p̂)²](1/π − 1) |
| B-DIFF, dollars | A√(p(1 − p)) | A√(p̂(1 − p̂)) | Σ A²[p(1 − p) + (p − p̂)²](1/π − 1) |

**DEF-λ[info]:** π(λ) = (1 − λ)(n/N) + λ·π_RA-EST, for λ ∈ {0, .25, .5, .75, 1}.
- The budget holds exactly, and π ≤ 1 holds automatically.
- The floor (1 − λ)n/N is automatic.
- Under DIFF, the working mean is the same w = p̂-based mean for every λ.
- The same mixture of the oracle score (DEF-λ-OR) is reported as a reference.

## Planning models

Unpenalized logistic GLM (statsmodels 0.15.0), fitted to independent superpopulation draws of size m with observed E.

| Model | Covariates |
|---|---|
| COR | z_A, Z, 1{G=1}, 1{G=2}, Z·1{G=2} |
| MIS-I | COR minus the interaction |
| MIS-G | z_A, Z |
| KS | COR + W |
| STRAT | stratum dummies |

**Fallback.** On non-convergence, a non-finite estimate, |β| > 25, or a planning sample with no positives (or all positives), the model falls back to intercept-only p̂. Fallbacks are counted.

## Fixed-size sensitivity (analytic and Monte Carlo)

**AV_fixed_hajek** (analytic, SENSITIVITY): the Hájek (1964) high-entropy approximation to the HT variance under fixed-size conditional Poisson with the same first-order π, in model expectation:

Σc·E[y²]/π² − (Σ(c/π)²v + (Σ(c/π)μ)²)/Σc, with c = π(1 − π).

It reproduces the SRSWOR variance times (N − 1)/N when π = n/N (checked numerically). No optimal fixed-size allocation is derived; the fixed-size versions of the Poisson π vectors are evaluated, not optimized.

**CPS-*/HT** (Monte Carlo): conditional Poisson (rejective, maximum-entropy) sampling with the prescribed π.
- Working odds come from the Chen–Dempster–Liu fixed point.
- First-order π is validated against R `sampling` 2.11 (`UPMEqfromw`/`UPMEpikfromq` reproduce the target to below 1e-8; `UPMEpiktildefrompik` gives proportional odds). Empirical frequencies over 20,000 draws are also checked (`tests/test_cps_vs_R.py`).
- Variance: Deville's (1999) approximate estimator for maximum-entropy designs, labelled as an approximation.
- Certainty units are removed and always included.

## Complete procedure (Problem C-CP)

For every design:
- one-sided 90% lower bound: LB = T̂ − 1.2816·√v̂;
- two-sided 95% interval: T̂ ± 1.96·√v̂;
- v̂ is the design-based estimator listed above.

The same Wald construction is used for all designs, so differences are not artifacts of the interval method. No exact or small-sample-corrected interval is used for any design.

**Decision metric:** the mean lower-bound shortfall T − LB. For a recovery demand at the lower bound, a smaller shortfall means more recovered. It is read only together with coverage.

**Not run:** sequential or stopping procedures (Problem D).
