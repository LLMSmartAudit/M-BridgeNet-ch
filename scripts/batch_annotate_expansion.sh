#!/usr/bin/env bash
# Batch LLM annotation for 13 expansion events (skips low-post events).
# Run from M-BridgeNet root:
#   OPENAI_API_KEY=sk-... bash scripts/batch_annotate_expansion.sh
# Or source the key file first:
#   source ~/.openclaw/.env && bash scripts/batch_annotate_expansion.sh

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=".venv/bin/python"
MODEL="gpt-5.4-nano"
TAU="0.82"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_LOG="$LOG_DIR/batch_annotate_${TIMESTAMP}.log"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY not set" >&2
  exit 1
fi

# 11 viable events (hua_chenyu and guo_meimei skipped — <50 posts after date filter)
EVENTS=(
  "wang_xing_rescue_001"
  "china_astock_rally_001"
  "nongfu_spring_boycott_001"
  "korea_martial_law_001"
  "hunan_flood_2024_001"
  "china_retirement_reform_001"
  "xizang_earthquake_2025_001"
  "manus_ai_launch_001"
  "pan_zhanle_100m_001"
  "eu_china_ev_tariffs_001"
  "ukraine_kursk_incursion_001"
)

echo "Batch annotation started $(date)" | tee "$SUMMARY_LOG"
echo "Model: $MODEL  tau: $TAU" | tee -a "$SUMMARY_LOG"
echo "Events: ${#EVENTS[@]}" | tee -a "$SUMMARY_LOG"
echo "---" | tee -a "$SUMMARY_LOG"

OK=0; FAIL=0
for EVENT_ID in "${EVENTS[@]}"; do
  EVENT_JSON="data/cphot/raw/${EVENT_ID}.json"
  EVENT_LOG="$LOG_DIR/annotate_${EVENT_ID}.log"

  if [ ! -f "$EVENT_JSON" ]; then
    echo "SKIP $EVENT_ID (no raw JSON)" | tee -a "$SUMMARY_LOG"
    continue
  fi

  echo -n "[$((OK+FAIL+1))/${#EVENTS[@]}] $EVENT_ID ... " | tee -a "$SUMMARY_LOG"
  T0=$(date +%s)

  if OPENAI_API_KEY="$OPENAI_API_KEY" $PYTHON scripts/llm_annotate.py \
      --event "$EVENT_JSON" \
      --model "$MODEL" \
      --tau "$TAU" \
      --resume \
      > "$EVENT_LOG" 2>&1; then
    T1=$(date +%s)
    # Extract bridge count from log
    BRIDGES=$(grep -oE "[0-9]+ bridge" "$EVENT_LOG" | tail -1 | grep -oE "[0-9]+") || BRIDGES="?"
    echo "✓ ${BRIDGES} bridges  ($((T1-T0))s)" | tee -a "$SUMMARY_LOG"
    OK=$((OK+1))
  else
    T1=$(date +%s)
    echo "✗ FAILED ($((T1-T0))s) — see $EVENT_LOG" | tee -a "$SUMMARY_LOG"
    tail -5 "$EVENT_LOG" | tee -a "$SUMMARY_LOG"
    FAIL=$((FAIL+1))
  fi
done

echo "---" | tee -a "$SUMMARY_LOG"
echo "DONE: $OK succeeded, $FAIL failed  $(date)" | tee -a "$SUMMARY_LOG"
