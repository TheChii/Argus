#!/usr/bin/env bash
# ==============================================================================
# Argus Quant — Daily Automated US Close Quantitative Equity Pipeline Runner
# Recommended Schedule: 16:15 ET (20:15 / 21:15 UTC depending on Daylight Saving),
# Monday through Friday (after US equity cash markets close at 16:00 ET).
#
# Crontab entry:
# 15 16 * * 1-5 /home/thechi/Documents/Finance-AI-model/scripts/cron_daily.sh >> /home/thechi/Documents/Finance-AI-model/data/logs/cron_stdout.log 2>&1
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
mkdir -p data/logs

LOG_FILE="data/logs/cron_daily.log"

# Rotate log file if it exceeds 10 MB (keep last 5,000 lines)
if [ -f "$LOG_FILE" ]; then
    FILE_SIZE=$(wc -c < "$LOG_FILE" 2>/dev/null || echo 0)
    if [ "$FILE_SIZE" -gt 10485760 ]; then
        tail -n 5000 "$LOG_FILE" > "${LOG_FILE}.tmp" && mv "${LOG_FILE}.tmp" "$LOG_FILE"
    fi
fi

log() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*" | tee -a "$LOG_FILE"
}

log "=================================================="
log "Starting Automated Daily Pipeline Execution"
log "Working Directory: $PROJECT_ROOT"

# Resolve Python environment
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -f "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    log "ERROR: Python interpreter not found."
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

log "Using Python: $($PYTHON_BIN --version) at $PYTHON_BIN"

# Check if today is Friday (Day 5 in date +%u) for weekly rebalancing
DAY_OF_WEEK=$(date '+%u')
REBALANCE_FLAGS="--auto-rebalance --commit-holdings"
if [ "$DAY_OF_WEEK" -eq 5 ]; then
    log "Today is Friday: Weekly portfolio rebalance & report generation scheduled."
    REBALANCE_FLAGS="--rebalance --commit-holdings"
fi

# Execute the daily loop
log "Executing daily pipeline: $PYTHON_BIN main.py run-daily $REBALANCE_FLAGS"
"$PYTHON_BIN" main.py run-daily $REBALANCE_FLAGS 2>&1 | tee -a "$LOG_FILE"

# Print system health status
log "Pipeline finished. Displaying current system health summary:"
"$PYTHON_BIN" main.py status 2>&1 | tee -a "$LOG_FILE"

log "Automated Daily Pipeline execution finished successfully."
log "=================================================="
