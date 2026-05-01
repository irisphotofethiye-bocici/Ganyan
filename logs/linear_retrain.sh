#!/usr/bin/env bash
set -e
cd /Users/fatihbozdag/Downloads/Cursor-Projects/ganyan
LOG=logs/linear_retrain_$(date +%Y%m%d_%H%M%S).log
echo "=== linear retrain start $(date) ===" | tee -a $LOG

echo "[1/2] linear conditional_logit (90-day)" | tee -a $LOG
uv run ganyan train linear --family conditional_logit --no-all-history >> $LOG 2>&1

echo "[2/2] linear plackett_luce (90-day)" | tee -a $LOG
uv run ganyan train linear --family plackett_luce --no-all-history >> $LOG 2>&1

echo "=== linear retrain done $(date) ===" | tee -a $LOG
