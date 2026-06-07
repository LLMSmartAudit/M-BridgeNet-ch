#!/bin/bash
# Re-crawl Weibo for all 8 Phase 0c events (session now valid)
set -uo pipefail
MC_DIR="/Users/weizhiyuan/Documents/code/vibe-writing/涉军事件知识图谱技术方案资料/07-code/MediaCrawler"
M_DIR="/Users/weizhiyuan/Documents/code/vibe-writing/涉军事件知识图谱技术方案资料/07-code/M-BridgeNet"
LOG_FILE="$M_DIR/logs/phase0c_weibo_recrawl.log"
mkdir -p "$M_DIR/logs"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

EVENTS=(
  "pla_rocket_force_purge_001"
  "taiwan_strait_drill_2025_001"
  "nezha2_movie_001"
  "xiaomi_su7_accident_001"
  "trump_tariff_2025_001"
  "gaokao_reform_2025_001"
  "ai_deepfake_scam_001"
  "covid_jn1_wave_001"
)

for event_id in "${EVENTS[@]}"; do
  crawled_dir="$MC_DIR/data/crawled/$event_id/weibo/jsonl"
  if ls "$crawled_dir"/search_contents_*.jsonl 2>/dev/null | head -1 | grep -q jsonl; then
    log "SKIP  $event_id/weibo — already crawled"
    continue
  fi

  log "START $event_id/weibo"
  cd "$M_DIR"
  .venv/bin/python scripts/phase0b_setup_crawl.py --event "$event_id" --platform weibo >> "$LOG_FILE" 2>&1
  cd "$MC_DIR"
  .venv/bin/python main.py >> "$LOG_FILE" 2>&1 || true
  EXIT_CODE=$?

  COUNT=$(ls "$crawled_dir"/*.jsonl 2>/dev/null | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}' || echo "?")
  log "DONE  $event_id/weibo  (exit=$EXIT_CODE, lines=$COUNT)"
  sleep 5
done
log "=== Weibo re-crawl complete ==="
