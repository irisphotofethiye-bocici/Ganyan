#!/usr/bin/env bash
set -e
cd /Users/fatihbozdag/Downloads/Cursor-Projects/ganyan
LOG=logs/retrain_$(date +%Y%m%d_%H%M%S).log
echo "=== retrain start $(date) ===" | tee -a $LOG

echo "[1/6] ranker --from 2026-01-26" | tee -a $LOG
uv run ganyan train --from 2026-01-26 --rounds 500 >> $LOG 2>&1

echo "[2/6] value --exclude-agf" | tee -a $LOG
uv run ganyan train --from 2026-01-26 --rounds 500 --exclude-agf >> $LOG 2>&1

echo "[3/6] finish_time" | tee -a $LOG
uv run ganyan train --from 2026-01-26 --rounds 500 --objective finish_time >> $LOG 2>&1

echo "[4/6] specialists" | tee -a $LOG
uv run ganyan train specialists --rounds 500 >> $LOG 2>&1

echo "[5/6] linear conditional_logit" | tee -a $LOG
uv run ganyan train linear --family conditional_logit >> $LOG 2>&1

echo "[6/6] linear plackett_luce" | tee -a $LOG
uv run ganyan train linear --family plackett_luce >> $LOG 2>&1

echo "=== retrain done $(date) ===" | tee -a $LOG
echo $LOG
