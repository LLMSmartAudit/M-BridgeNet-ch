#!/bin/bash
# Phase 0b/0c — sequential crawl orchestrator for all events + platforms
# Run from MediaCrawler directory. Patches config + runs main.py for each.
set -uo pipefail   # -e removed: crawler non-zero exit must not abort the whole run

MC_DIR="/Users/weizhiyuan/Documents/code/vibe-writing/涉军事件知识图谱技术方案资料/07-code/MediaCrawler"
M_DIR="/Users/weizhiyuan/Documents/code/vibe-writing/涉军事件知识图谱技术方案资料/07-code/M-BridgeNet"
LOG_FILE="$M_DIR/logs/phase0b_crawl_log.txt"
mkdir -p "$M_DIR/logs"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

# All events in priority order with their platforms
# Format: event_id:platform1,platform2,...
EVENTS=(
  "zheng_linghua_001:bili,zhihu,douyin"
  "chengdu_49_school_001:weibo,bili,zhihu,douyin"
  "yuhuaying_trafficking_001:weibo,bili,zhihu,douyin"
  "pinduoduo_overwork_001:weibo,bili,zhihu,douyin"
  "dongfang_selection_001:weibo,bili,zhihu,douyin"
  "xiaoyangshu_scandal_001:weibo,bili,zhihu,douyin"
  "luo_yonghao_comeback_001:weibo,bili,zhihu,douyin"
  "xibei_precooked_001:weibo,bili,zhihu"
  "xiaomi_su7_launch_001:weibo,bili,zhihu,douyin"
  "kemu3_dance_001:weibo,bili,zhihu,douyin"
  "beijing_flood_2023_001:weibo,bili,zhihu,douyin"
  "wuhan_univ_cherry_001:weibo,bili,zhihu,douyin"
  "drone_show_event_001:weibo,bili,zhihu,douyin"
  "hetongshu_plagiarism_001:weibo,bili,zhihu,douyin"
  "xinjiang_cotton_001:weibo,bili,zhihu,douyin"
  # ── Phase 0c — military + 2025 hot topics ─────────────────────────────
  "pla_rocket_force_purge_001:weibo,bili,zhihu"
  "taiwan_strait_drill_2025_001:weibo,bili,zhihu,douyin"
  "nezha2_movie_001:weibo,bili,zhihu,douyin"
  "xiaomi_su7_accident_001:weibo,bili,zhihu,douyin"
  "trump_tariff_2025_001:weibo,bili,zhihu,douyin"
  "gaokao_reform_2025_001:weibo,bili,zhihu,douyin"
  "ai_deepfake_scam_001:weibo,bili,zhihu,douyin"
  "covid_jn1_wave_001:weibo,bili,zhihu,douyin"
)

# Skip weibo for zheng_linghua_001 (already done)
# We start from bili (which we already set up to run)

for entry in "${EVENTS[@]}"; do
  event_id="${entry%%:*}"
  platforms="${entry##*:}"

  IFS=',' read -ra plat_list <<< "$platforms"
  for platform in "${plat_list[@]}"; do
    # Check if already crawled
    crawled_dir="$MC_DIR/data/crawled/$event_id/$platform/jsonl"
    if ls "$crawled_dir"/search_contents_*.jsonl 2>/dev/null | head -1 | grep -q jsonl; then
      log "SKIP  $event_id/$platform — already crawled"
      continue
    fi

    log "START $event_id/$platform"

    # Patch config
    cd "$M_DIR"
    .venv/bin/python scripts/phase0b_setup_crawl.py --event "$event_id" --platform "$platform" >> "$LOG_FILE" 2>&1

    # Run crawler (|| true so set -u/-o pipefail don't abort on non-zero exit)
    cd "$MC_DIR"
    .venv/bin/python main.py >> "$LOG_FILE" 2>&1 || true
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
      COUNT=$(ls "$crawled_dir"/*.jsonl 2>/dev/null | head -1 | xargs wc -l 2>/dev/null | awk '{print $1}' || echo "?")
      log "DONE  $event_id/$platform  (exit=$EXIT_CODE, lines=$COUNT)"
    else
      log "FAIL  $event_id/$platform  (exit=$EXIT_CODE)"
    fi

    # Short pause between crawls to avoid rate-limiting
    sleep 5
  done
done

log "=== All Phase 0b crawls complete ==="
