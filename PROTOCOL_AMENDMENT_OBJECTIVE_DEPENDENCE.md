# Protocol amendment: objective dependence and complete-procedure evaluation

**Received 2026-09-23.** It was applied after `test_constitution.md` was frozen (SHA-256 `d6d8dd8f…7071`, in `logs/constitution_freeze.sha256`) and **before any main analytic or Monte Carlo run**. The constitution is not rewritten. This file governs all later analysis and interpretation. Where the two conflict on interpretation or reporting, this file prevails. The frozen grid, designs and seeds are unchanged.

## A. Governing principle

A sampling allocation is evaluated only together with its complete procedure:

audit objective → estimand/decision → estimator → inferential procedure → criterion (loss, precision, stopping) → design.

π ∝ √E[Y²|X] is **the allocation score for the direct-HT anticipated-variance problem** under Poisson sampling. It is not a universal "risk score". Anticipated-variance dominance is not taken to imply dominance in interval width, lower-bound behaviour, sample size to a decision, stopping time, or assurance cost.

## B. Problem labels used in every output

| Label | Estimand | Estimator | Inference | Criterion |
|---|---|---|---|---|
| **A-HT** | error count / improper dollars | direct HT (Poisson) | — | anticipated variance (primary analytic grid) |
| **A-EXP** | same | expansion estimator under SRSWOR / stratified SRSWOR | — | exact anticipated variance (practice comparators) |
| **B-DIFF** | same | model-assisted difference estimator | — | anticipated variance; allocation score from the residual second moment |
| **C-CP** | same | the estimator of the design | Wald one-sided 90% lower bound and two-sided 95% interval, each with its own design-based variance estimator | coverage, lower-bound width (T − LB), interval width, RMSE |
| **D-SEQ** | — | — | — | **Not run.** No established sequential procedure is used in this setting, so none is manufactured. |

## C. Changes to analyses (all labelled)

1. **The COMPLETE-PROCEDURE SENSITIVITY (Problem C-CP) is the previously planned Monte Carlo (constitution §8), relabelled.**
   - It already covers equal probability (EQ-P, SRSWOR), conventional stratification (ST-PROP, ST-EQD), the oracle design (RA-OR), the estimated design (RA-EST), and a defensive mixture (DEF-0.5). It also covers bias, width, coverage and the 90% lower bound.
   - **Addition (per this amendment):** the mean two-sided 95% interval width and the mean lower-bound shortfall T − LB are reported as decision metrics. For a recovery decision, a smaller shortfall means more is recovered.
2. **New diagnostic: `results/objective_ranking_comparison.csv`.**
   - For each Monte Carlo scenario × estimand × estimator family, designs are ranked by anticipated variance, empirical RMSE, interval width, lower-bound shortfall (the decision metric) and max weight.
   - Coverage is reported alongside. Designs whose 90% lower-bound coverage shows material undercoverage (constitution §11) are flagged in the decision-metric ranking, because a short bound that does not cover is not a gain.
3. **New artifact: `objective_dependence_audit.md`.**
4. **Two routes to equal probability.**
   - **Structural**: the allocation-relevant quantity is homogeneous.
   - **Decision/robustness**: heterogeneity exists, but exploiting it with the available information gives insufficient gain, or excessive downside under the complete procedure.

   `decision_boundary_results.md` classifies each equal-probability region by route.
5. **Terminology.**
   - "Risk-aligned" (RA) is used only with its object, for example "HT-rate-aligned" or "DIFF-dollar-aligned".
   - The general framework is called **objective-aligned probability design**.
   - Code and file names are not changed.
6. **Adjudication categories** are replaced by the amendment's §13: SUPPORTS OBJECTIVE-ALIGNED PUBLIC-AUDIT FRAMEWORK / SUPPORTS CONCEPTUAL FRAMEWORK, IMPLEMENTATION NARROWS / UNDERMINES STANDALONE PAPER.
7. **Prohibited inference.** Heterogeneous outcome risk does not imply that using it in selection must improve an audit objective. Results where equal probability wins are interpreted directly and not "repaired".
8. **The terminated project is used only as a methodological boundary.** Its data, populations, methods and financial-statement setting are not used. No MUS comparison is introduced. PPS-A (π ∝ amount) was prespecified in the frozen constitution as documented *public-sector* practice (VA OIG; penny sampling), not as MUS.
