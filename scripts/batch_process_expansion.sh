#!/usr/bin/env bash
# Run convert → annotate → prepare for one expansion event.
# Usage: ./scripts/batch_process_expansion.sh <event_id>
# Example: ./scripts/batch_process_expansion.sh wang_xing_rescue_001
#
# Prerequisites:
#   - All 4 platform crawls for the event must be done
#   - OPENAI_API_KEY must be set in environment
#   - Run from M-BridgeNet root directory

set -euo pipefail

EVENT_ID="${1:?Usage: $0 <event_id>}"
CRAWL_ROOT="../MediaCrawler/data/crawled/${EVENT_ID}"
RAW_OUT="data/cphot/raw/${EVENT_ID}.json"
PROC_OUT="data/cphot/processed/${EVENT_ID}.json"

# ── Date ranges per event ──────────────────────────────────────────────────────
declare -A START_DATE=(
    [wang_xing_rescue_001]="2025-01-01"
    [china_astock_rally_001]="2024-09-24"
    [nongfu_spring_boycott_001]="2024-02-01"
    [korea_martial_law_001]="2024-12-03"
    [hunan_flood_2024_001]="2024-06-15"
    [china_retirement_reform_001]="2024-09-01"
    [xizang_earthquake_2025_001]="2025-01-07"
    [manus_ai_launch_001]="2025-03-06"
    [pan_zhanle_100m_001]="2024-07-27"
    [hua_chenyu_custody_001]="2024-03-01"
    [eu_china_ev_tariffs_001]="2024-06-01"
    [ukraine_kursk_incursion_001]="2024-08-06"
    [guo_meimei_release_001]="2024-04-01"
)
declare -A END_DATE=(
    [wang_xing_rescue_001]="2025-01-20"
    [china_astock_rally_001]="2024-10-15"
    [nongfu_spring_boycott_001]="2024-04-30"
    [korea_martial_law_001]="2024-12-20"
    [hunan_flood_2024_001]="2024-07-10"
    [china_retirement_reform_001]="2024-10-31"
    [xizang_earthquake_2025_001]="2025-01-20"
    [manus_ai_launch_001]="2025-03-31"
    [pan_zhanle_100m_001]="2024-08-15"
    [hua_chenyu_custody_001]="2024-06-30"
    [eu_china_ev_tariffs_001]="2024-09-30"
    [ukraine_kursk_incursion_001]="2024-08-31"
    [guo_meimei_release_001]="2024-06-30"
)

START="${START_DATE[$EVENT_ID]}"
END="${END_DATE[$EVENT_ID]}"

echo "=== Step 2: Convert ==="
.venv/bin/python scripts/convert_mediacrawler.py \
  --inputs \
    "${CRAWL_ROOT}/wb/search/"*.jsonl \
    "${CRAWL_ROOT}/dy/search/"*.jsonl \
    "${CRAWL_ROOT}/zhihu/search/"*.jsonl \
    "${CRAWL_ROOT}/bilibili/search/"*.jsonl \
  --event-id "${EVENT_ID}" \
  --output "${RAW_OUT}" \
  --start "${START}" --end "${END}"

echo ""
echo "=== Post-convert platform check ==="
.venv/bin/python - <<'PYEOF'
import json, sys
path = sys.argv[1] if len(sys.argv) > 1 else None
event_id = sys.argv[2] if len(sys.argv) > 2 else ""
import os
raw_path = f"data/cphot/raw/{os.environ.get('EVENT_ID', '')}.json"
PYEOF
.venv/bin/python -c "
import json, os
path = 'data/cphot/raw/${EVENT_ID}.json'
data = json.load(open(path))
posts = data.get('posts', [])
from collections import Counter
plat = Counter(p['platform'] for p in posts)
print('Platform counts:', dict(plat))
print('Total posts:', len(posts))
if len(posts) < 50:
    print('WARNING: very few posts — consider re-crawling')
"

echo ""
echo "=== Step 3: Annotate (dry run first) ==="
.venv/bin/python scripts/llm_annotate.py \
  --event "${RAW_OUT}" \
  --model gpt-5.5 --dry-run

read -p "Dry run OK? Proceed with full annotation? [y/N] " confirm
if [[ "${confirm,,}" != "y" ]]; then
    echo "Aborted. Re-run manually when ready."
    exit 0
fi

.venv/bin/python scripts/llm_annotate.py \
  --event "${RAW_OUT}" \
  --model gpt-5.5

echo ""
echo "=== Post-annotation bridge check ==="
.venv/bin/python -c "
import json
path = 'data/cphot/raw/${EVENT_ID}.json'
data = json.load(open(path))
bridges = [p for p in data.get('bridge_pairs', []) if p.get('label') == 1]
print(f'Bridge pairs: {len(bridges)} / {len(data.get(\"bridge_pairs\", []))}')
if len(bridges) < 10:
    print('REJECTED: fewer than 10 bridges — do not add to test_real')
    exit(1)
else:
    print('PASSED gate (≥10 bridges) — proceeding to prepare')
"

echo ""
echo "=== Step 4: Prepare training data ==="
.venv/bin/python scripts/prepare_training_data.py \
  --event "${RAW_OUT}" \
  --output data/cphot/processed

echo ""
echo "=== Step 5: Move to test_real ==="
cp "${PROC_OUT}" "data/cphot/processed/test_real/"
cp "data/cphot/processed/${EVENT_ID}_emb.npz" "data/cphot/processed/test_real/" 2>/dev/null || true
echo "Copied to test_real."

echo ""
echo "=== Done: ${EVENT_ID} ==="
echo "Run evaluate.py when 3-4 events are ready:"
echo "  MBRIDGENET_NO_FAISS=1 .venv/bin/python scripts/evaluate.py \\"
echo "    --data data/cphot/processed/test_real \\"
echo "    --checkpoint checkpoints/mlp_v18_fold5.pt \\"
echo "    --use-s1s3-scoring --k 5 10 20 50"
