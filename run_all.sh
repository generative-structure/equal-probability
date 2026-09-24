#!/bin/sh
# Reproduces every archived result and the post-freeze (R2) diagnostics.
# Run from the repository root after creating .venv (see README.md).
set -e
PY=.venv/bin/python
export R_LIBS_USER="$PWD/.Rlib"
mkdir -p logs
# frozen pipeline
$PY scripts/core_benchmark_check.py            > logs/core_benchmark_check.log 2>&1
$PY scripts/analytic_grid.py                   > logs/analytic_grid.log 2>&1
$PY scripts/conventional_diff.py               > logs/conventional_diff.log 2>&1
$PY scripts/summarize.py                       > logs/summarize.log 2>&1
$PY scripts/mc_complete_procedure.py --reps 10000 --tag final  > logs/mc_final.log 2>&1
$PY scripts/objective_ranking.py               > logs/objective_ranking.log 2>&1
$PY scripts/posthoc_amount_anchor.py           > logs/posthoc_amount_anchor.log 2>&1
# post-freeze (R2) diagnostics; they write only to r2_diagnostics/out/
$PY r2_diagnostics/audit_fallback.py           > logs/r2_audit_fallback.log 2>&1
$PY r2_diagnostics/audit_summary.py            > logs/r2_audit_summary.log 2>&1
$PY r2_diagnostics/separation_sensitivity.py   > logs/r2_separation_sensitivity.log 2>&1
$PY r2_diagnostics/coverage_by_fit.py          > logs/r2_coverage_by_fit.log 2>&1
$PY r2_diagnostics/hajek_realized.py           > logs/r2_hajek_realized.log 2>&1
$PY r2_diagnostics/tail_uncertainty.py         > logs/r2_tail_uncertainty.log 2>&1
# property tests (test_cps_vs_R.py needs R with the sampling package in .Rlib/)
$PY -m pytest -q tests                         > logs/pytest.log 2>&1
