#!/usr/bin/env bash
# =============================================================================
# One-time build step, then hand off to the container command.
#
# The evaluator runs `docker compose up` with the Kaggle CSVs mounted and gets a
# working app - no separate "now run these four scripts" instruction. Each step
# is skipped when its artifact already exists, so a restart takes seconds rather
# than re-running the whole pipeline.
#
# Set SKIP_BUILD=1 to serve whatever artifacts are already on the volume.
# =============================================================================
set -euo pipefail

DATA_DIR="${RAW_DIR:-/app/data/raw}"
MODEL_DIR="${MODEL_DIR:-/app/models}"
SQLITE_PATH="${SQLITE_PATH:-/app/data/credit_risk.db}"
REPORT_DIR="${REPORT_DIR:-/app/reports}"

log() { printf '\n[entrypoint] %s\n' "$1"; }

if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
    log "SKIP_BUILD=1 - serving existing artifacts."
    exec "$@"
fi

if [[ ! -f "${DATA_DIR}/application_train.csv" ]]; then
    log "WARNING: ${DATA_DIR}/application_train.csv not found."
    log "Download the Home Credit Default Risk dataset from Kaggle:"
    log "  https://www.kaggle.com/competitions/home-credit-default-risk/data"
    log "and unzip application_train.csv, application_test.csv, bureau.csv and"
    log "previous_application.csv into the ./data/raw folder mounted here."
    log "Starting the app anyway - it will show setup instructions in the UI."
    exec "$@"
fi

# 1. SQL warehouse (~10s)
if [[ ! -f "${SQLITE_PATH}" ]]; then
    log "Building the SQLite warehouse..."
    python -m src.data.build_database
else
    log "SQLite warehouse present - skipping."
fi

# 2. EDA figures + summary (~15s)
if [[ ! -f "${REPORT_DIR}/eda_summary.json" ]]; then
    log "Running exploratory analysis..."
    python -m src.data.eda_insights
else
    log "EDA summary present - skipping."
fi

# 3. Model, calibrator, metrics, figures, scored predictions (~90s)
if [[ ! -f "${MODEL_DIR}/lgbm_credit_risk.joblib" ]]; then
    log "Training the model (this takes a couple of minutes on first run)..."
    python -m src.ml.train
else
    log "Trained model present - skipping."
fi

# 4. Surrogate policy rules (~5s)
if [[ ! -f "${REPORT_DIR}/business_rules.json" ]]; then
    log "Deriving business rules..."
    python -m src.rules.rule_extractor
else
    log "Business rules present - skipping."
fi

# 5. Fairness and stability report (~10s)
if [[ ! -f "${REPORT_DIR}/fairness_report.json" ]]; then
    log "Building the fairness report..."
    python -m src.monitoring.fairness
else
    log "Fairness report present - skipping."
fi

log "Build complete. Starting: $*"
exec "$@"
