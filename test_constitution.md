# Test constitution: decision-boundary test for risk-aligned public-audit sampling

**Frozen 2026-09-23, before any design-outcome computation.** The only computations before freezing were the population-only DGP diagnostics (`logs/prefreeze_dgp_diagnostics.log`) and the theorem checks (`core_benchmark_check.md`).

Once the main runs begin, any change to what is below is logged in `CHANGELOG.md` and labelled DIAGNOSTIC, SENSITIVITY or POST HOC.

## 1. Research question

When a public auditor needs a generalizable probability sample, when is equal-probability selection justified? When should information already on the frame change the inclusion probabilities?

The test looks for decision regions, not for a design that "wins".

## 2. Estimands

On a synthetic improper-payment frame of N = 20,000 claims:
- **Rate.** The error count Σ E_i. The rate is the count divided by N, so it has the same design ranking.
- **Dollars.** Improper dollars D = Σ A_i E_i, under an all-or-nothing error.

The two estimands are reported separately and never averaged into one verdict.

## 3. Estimators

- **HT:** direct Horvitz–Thompson under Poisson designs.
- **EXP:** the expansion (HT) estimator under SRSWOR and stratified SRSWOR, i.e. the standard practice estimator.
- **DIFF:** the model-assisted difference estimator Σ_U μ_i + Σ_s (Y_i − μ_i)/π_i. The working mean is μ_i = p̂_i (rate) or A_i p̂_i (dollars), with p̂ taken from the same independent planning model that the design uses; for oracle rows, μ = p. DIFF is reported as a separate regime, so that gains from *using X in the estimator* are kept apart from gains from *using X in the design*.

Every allocation rule is labelled with its estimator. HT rules and DIFF rules are never mixed.

## 4. DGP (`src/dbt/dgp.py`)

**Frame variables:**
- A ~ lognormal(log 250, 1.2), which is skewed;
- Z ~ N(0, 1), a continuous risk-relevant score;
- G ∈ {0, 1, 2} with probabilities (0.6, 0.3, 0.1), a program/provider category;
- W = 0.8 z_A + 0.6(G − Ḡ) + 0.5ε, an irrelevant auxiliary: correlated with A and G, but with no effect on E given A, Z and G.

**Risk model.** logit p = α + σ_η · η₀ˢᵗᵈ, with η₀ = 0.6 z_A + 0.8 Z + b_G[G] + 1.0·Z·1{G=2} and b_G = (0, 0.8, 1.6). The quantity η₀ˢᵗᵈ is η₀ standardized with fixed superpopulation constants. α is solved on the target frame so that mean p equals the prevalence exactly.

**STRATCAP (Control 6).** η₀ is the amount-stratum index (5 equal-dollar strata), so the conventional amount strata carry all the predictable risk.

**Populations.** One fixed target frame (seed 20260923), shared by all scenarios as common random numbers; only p changes. The realized outcomes E (seed 20260924) are used for Monte Carlo coverage. Planning data are independent draws from the same superpopulation.

## 5. Parameter grid (analytic)

| Factor | Levels |
|---|---|
| DGP | MAIN; STRATCAP (Control 6 only) |
| Heterogeneity σ_η (SD of the logit linear predictor) | 0 (none; Control 1), 0.75 (moderate), 1.5 (strong) |
| Prevalence | 0.05, 0.25, 0.70 |
| Main sample n (expected or fixed) | 100, 400 |
| Planning information (MAIN) | see the table below |
| Planning information (STRATCAP) | ORACLE; STRAT-2000 (logit on stratum dummies, m = 2,000) |
| Planning replicates | R_plan = 200 per (σ, prevalence, information level) |
| Defensive λ | 0, 0.25, 0.5, 0.75, 1 |

**Why these prevalences:**
- 0.05 is near national PERM/CERT-type improper-payment rates.
- 0.25 is intermediate.
- 0.70 matches Medicare provider-integrity populations: Edwards et al. (2023) report that over 76% of sampled payments were in error.

**Why these sample sizes:** 100 is typical of provider-extrapolation samples, and 400 of larger program reviews.

**Heterogeneity diagnostics.** These are measured, not labelled: see `results/scenario_diagnostics.csv` (mean p, p10/p50/p90, CV(√p), corr(p, A), group-specific p). Pre-freeze values on the target frame:

| σ_η | CV(√p) at prevalence 0.05 | at 0.25 | at 0.70 | corr(p, A) | Oracle HT-rate gain* |
|---|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | 0 | 0 |
| 0.75 | 0.40 | 0.27 | 0.10 | ≈ 0.27–0.34 | 0.14 / 0.07 / 0.01 |
| 1.5 | 0.82 | 0.48 | 0.19 | ≈ 0.23–0.35 | 0.40 / 0.19 / 0.03 |

\* 1 − (E√p)²/E p (interior, small n/N), at prevalence 0.05 / 0.25 / 0.70.

For dollars, the oracle gain versus equal probability is already 0.73 at σ = 0, because m = A²p.

**Planning information (MAIN):**

| Label | Planning m | Logistic model (unpenalized, statsmodels GLM) | Role |
|---|---:|---|---|
| ORACLE | — | true p | Control 3 |
| COR-10000 | 10,000 | zA + Z + G dummies + Z·1{G=2} (correct form) | Well estimated; Control 4 |
| COR-2000 | 2,000 | correct form | Moderate |
| COR-500 | 500 | correct form | Noisy/small |
| MIS-I-2000 | 2,000 | zA + Z + G dummies (omits the interaction) | Mild misspecification |
| MIS-G-2000 | 2,000 | zA + Z (omits G and the interaction) | Material misspecification; Control 5 |
| MIS-G-10000 | 10,000 | zA + Z | Misspecification not cured by more data |
| KS-2000 | 2,000 | correct form + W | Irrelevant auxiliary; Control 2 |

**Setting A vs B.** In Setting A (existing information) the planning data carry no review cost. In Setting B (a dedicated pilot), m is a review cost, reported separately.

**Fit failure** means non-convergence, perfect separation, or no positive outcome. A failed fit falls back to intercept-only, i.e. constant p̂ and hence equal probability for the rate. Failures are counted.

## 6. Designs (details in `design_specification.md`)

**Equal probability:**
- EQ-P, equal-probability Poisson with π = n/N (HT or DIFF);
- SRSWOR (EXP, or DIFF using the planning model).

**Conventional amount-based:**
- ST-PROP: 5 equal-dollar amount strata, proportional allocation (EXP).
- ST-EQD: the same strata with equal allocation n/5, the PERM-style convention (EXP).
- ST-NEY-EST: kept separate. Neyman allocation on the same strata, with stratum SDs estimated from COR-2000-sized planning data (EXP).
- PPS-A: Poisson with π ∝ A, water-filled (HT). This is size-only unequal probability, as in VA OIG practice and penny sampling.

**Risk-aligned** (Poisson, water-filled caps):
- RA-OR: oracle score (HT: √p or A√p; DIFF: √(p(1−p)) or A√(p(1−p))).
- RA-EST[info]: the same score with p̂ in place of p.
- DEF-λ[info]: π = (1 − λ)(n/N) + λ π_RA-EST. The budget holds exactly; the floor (1 − λ)n/N is automatic.

**Fixed-size sensitivity** (Monte Carlo only, in selected scenarios): conditional Poisson (maximum-entropy) sampling with the RA-OR and RA-EST(COR-2000) π. The HT estimator is used with Deville's approximate variance estimator for maximum-entropy designs, labelled as an approximation. The first-order π is validated against R `sampling` 2.11 and by empirical frequencies.

## 7. Primary metrics

**Analytic** (exact on the fixed target frame, averaged over the 200 planning replicates for estimated designs):
- AV, computed exactly: Poisson designs Σ m_i(1/π_i − 1); SRSWOR N²(1−f)E[S²]/n; stratified Σ_h N_h²(1−f_h)E[S_h²]/n_h. E[S²] is the model expectation of the finite-population variance, computed from μ_i = E Y_i and v_i = Var Y_i.
- Relative variance and efficiency gain, against both **EQ-P/HT** (the theorem benchmark) and **SRSWOR/EXP** (the practice benchmark).
- Equal-precision expected reviews against SRSWOR/EXP at n₀, root-solved with caps (oracle and conventional designs; estimated designs only where stated).
- Design stability: min/max π, max weight 1/π, CV of weights, number of certainty units, SD of Poisson sample size √Σπ(1−π).
- Planning-model quality, on the target frame with the true p:
  - expected Brier score and log loss;
  - calibration-in-the-large (mean p̂ − mean p) and calibration slope (regression of logit p on logit p̂);
  - AUC against realized E (secondary);
  - Pearson and Spearman correlation between the estimated and true allocation scores, per estimand and estimator;
  - the number of planning positives and the fit-failure count.
- The distribution across the 200 planning replicates: mean, 10th and 90th percentile of relative variance, and the share of replicates worse than SRSWOR/EXP.

**Monte Carlo** (selected scenarios): bias, empirical variance, RMSE, one-sided 90% lower-bound coverage (with a 95% Monte Carlo CI), two-sided 95% coverage, mean lower-bound width, the realized-size distribution, and the frequency of zero-event samples.

- Every design uses a z-based (Wald) interval, T̂ − 1.2816√v̂, built on its own design-based variance estimator:
  - Poisson HT: Σ_s y²(1−π)/π²;
  - Poisson DIFF: Σ_s (y−μ)²(1−π)/π²;
  - SRSWOR: N²(1−f)s²/n;
  - stratified: Σ N_h²(1−f_h)s_h²/n_h;
  - conditional Poisson: Deville's approximation.
- The Monte Carlo is design-based: it is conditional on the fixed realized target population, with the planning replicate cycled. Estimated designs take replicate r mod 200, so Monte Carlo draws average over planning variability.

## 8. Monte Carlo scenarios (prespecified)

All are MAIN unless noted.

| ID | σ_η | Prevalence | n | Purpose |
|---|---:|---:|---:|---|
| S1 | 0 | 0.25 | 100 | Homogeneous control |
| S2 | 0.75 | 0.25 | 100 | Moderate |
| S3 | 1.5 | 0.05 | 100 | Low prevalence, strong, small n |
| S4 | 1.5 | 0.05 | 400 | Low prevalence, strong |
| S5 | 1.5 | 0.25 | 100 | Strong |
| S6 | 1.5 | 0.70 | 100 | High-error (Medicare-like) |
| S7 | 1.5 | 0.70 | 400 | High-error, larger n |
| S8 | STRATCAP 1.5 | 0.25 | 100 | Strata capture the risk |

**Designs in the Monte Carlo:**
- EQ-P/HT, SRSWOR/EXP, ST-PROP/EXP, ST-EQD/EXP, PPS-A/HT;
- RA-OR/HT, RA-EST(COR-2000)/HT, RA-EST(MIS-G-2000)/HT, DEF-0.5(COR-2000)/HT, DEF-0.5(MIS-G-2000)/HT;
- SRSWOR/DIFF(COR-2000), EQ-P/DIFF(COR-2000), RA-EST/DIFF(COR-2000), DEF-0.5/DIFF(COR-2000);
- in S8, RA-EST(STRAT-2000) replaces COR-2000;
- fixed-size CPS-RA-OR/HT and CPS-RA-EST(COR-2000)/HT, in S2, S5 and S6 only.

**Replications.** First a 2,000-replicate run to estimate the Monte Carlo SE; then the final count is 10,000 for all eight scenarios. At a coverage of 0.9, SE = √(0.09/10,000) = 0.003, so the 95% Monte Carlo CI is about ±0.006, enough to separate 0.88 from 0.90. Common random numbers: within a replicate, all Poisson designs share the same uniforms U_i, and all stratified and SRSWOR designs share the same random keys.

## 9. Figures (at most 3; the plotted values are also saved as CSV)

1. Oracle value of information: relative variance against σ_η for EQ-P/SRSWOR, stratification, PPS-A and RA-OR, by estimand.
2. Decision boundary: relative variance of RA-EST and DEF-0.5 (against SRSWOR/EXP) across heterogeneity and planning quality, by estimand.
3. The λ frontier: efficiency gain against max weight and lower-bound coverage.

## 10. Robustness and sensitivity (planned)

Fixed-size CPS; KS-2000 (irrelevant W); STRATCAP; ST-NEY-EST; a DIFF regime for every design. Optional, not planned: a mixture-shift transport case, run only if the primary results make it decision-relevant.

## 11. Decision conventions (analysis conventions, not universal thresholds)

- **"Reliably better" than a comparator:** mean relative variance ≤ 0.90, and 90th percentile over planning replicates ≤ 1.00.
- **"Effectively optimal equal probability":** the oracle risk-aligned design improves on the best equal-probability design (EQ-P or SRSWOR, same estimator family) by less than 5% in variance.
- **"Stratification suffices":** the RA-OR (or RA-EST) variance is more than 0.95 × the best conventional stratified or PPS-A design.
- **"Material undercoverage":** the 95% Monte Carlo CI upper limit for 90% lower-bound coverage is below 0.88.

## 12. Stop / narrow rules

These are the handoff's §25, applied as written:
1. The oracle is not weakly better than EQ-P/HT under the theorem's assumptions: stop and debug.
2. The oracle gains materially (> 0.5%) in σ = 0 for the rate/HT: stop and debug.
3. With COR-10000, estimated designs fail to recover useful gains where the oracle gain is large: investigate, and narrow the implementation claim.
4. Conventional stratification captures essentially all gain: report it and narrow the paper.
5. Equal probability is consistently more reliable inferentially: favour SRS in those conditions.
6. Rate and dollar results differ: report both, and do not average them.
