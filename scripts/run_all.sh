#!/usr/bin/env bash
# Rebuild every result from the raw data. Uses all cores for about an hour; run from the repo root.
set -euo pipefail
PY=.venv/bin/python

$PY -m pytest -q
$PY scripts/01_feedback_and_order.py
$PY scripts/02_environment_check.py
$PY scripts/03_spec_checks.py
$PY scripts/04_recovery_fits.py
RECOVERY_ROUNDS=20 $PY scripts/04_recovery_fits.py
RECOVERY_ROUNDS=40 $PY scripts/04_recovery_fits.py
$PY scripts/05_recovery_report.py
$PY scripts/06_fit_real_data.py
$PY scripts/07_results_report.py
FIT_LAPSE=0.05 $PY scripts/06_fit_real_data.py
REPORT_LAPSE=1 $PY scripts/07_results_report.py
$PY scripts/08_lapse_recovery_check.py
$PY scripts/09_recovery_at_fitted_values.py
$PY scripts/10_robustness.py
