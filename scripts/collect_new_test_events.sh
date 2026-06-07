#!/usr/bin/env bash
# ============================================================
# Collect 7 new test events for CPHot test set expansion
# Usage: Run each EVENT block manually, one platform at a time.
#        Douyin + Zhihu require interactive browser login on first run.
#
# After all 4 platforms are crawled for an event, run:
#   Step A: convert_mediacrawler.py  (merge into CPHot format)
#   Step B: llm_annotate.py          (LLM annotation)
#   Step C: prepare_training_data.py (compute signals → test_real)
# ============================================================

MEDIACRAWLER=../../MediaCrawler
MBRIDGENET=.
OPENAI_KEY=${OPENAI_API_KEY:-"YOUR_OPENAI_KEY_HERE"}

# ────────────────────────────────────────────────────────────
# EVENT 1: Sam Altman Ousting  (2023-11-17 OpenAI leadership crisis)
# ────────────────────────────────────────────────────────────
EVENT_ID="sam_altman_ousting_001"
KEYWORDS_1="奥特曼被解雇,OpenAI宫斗,山姆奥特曼"
START_1="2023-10-01"
END_1="2024-03-31"

# EVENT 2: Titan Submersible  (2023-06-18 OceanGate implosion)
# ────────────────────────────────────────────────────────────
EVENT_ID_2="titan_submersible_001"
KEYWORDS_2="泰坦号潜艇,泰坦潜水器失踪,泰坦内爆"
START_2="2023-06-01"
END_2="2023-10-31"

# EVENT 3: Paris Olympics Opening Ceremony  (2024-07-26 controversy)
# ────────────────────────────────────────────────────────────
EVENT_ID_3="paris_olympics_ceremony_001"
KEYWORDS_3="巴黎奥运开幕式,巴黎奥运争议,最后的晚餐巴黎"
START_3="2024-06-01"
END_3="2024-10-31"

# EVENT 4: 2024 US Presidential Election  (2024-11-05 Trump wins)
# ────────────────────────────────────────────────────────────
EVENT_ID_4="us_election_2024_001"
KEYWORDS_4="美国大选,特朗普胜选,哈里斯败选"
START_4="2024-09-01"
END_4="2025-02-28"

# EVENT 5: Eileen Gu Identity Controversy  (2022-02 Beijing Olympics)
# ────────────────────────────────────────────────────────────
EVENT_ID_5="eileen_gu_identity_001"
KEYWORDS_5="谷爱凌国籍,谷爱凌身份,谷爱凌双重国籍"
START_5="2022-01-01"
END_5="2022-08-31"

# EVENT 6: Sora Debut  (2024-02-15 OpenAI video model)
# ────────────────────────────────────────────────────────────
EVENT_ID_6="sora_debut_001"
KEYWORDS_6="Sora发布,OpenAI Sora,sora视频生成"
START_6="2024-01-01"
END_6="2024-07-31"

# EVENT 7: TikTok Ban Legislation  (2024-04 US Congress)
# ────────────────────────────────────────────────────────────
EVENT_ID_7="tiktok_ban_001"
KEYWORDS_7="TikTok禁令,抖音美国禁令,TikTok法案"
START_7="2024-01-01"
END_7="2024-08-31"


# ============================================================
# STEP 1: CRAWL (run for each EVENT × PLATFORM combination)
# ============================================================
# In MediaCrawler/config/base_config.py, set:
#   PLATFORM = "wb"   # weibo
#   KEYWORDS = "<keywords for event>"
#   CRAWLER_MAX_NOTES_COUNT = 500   # per keyword
#   START_DAY = "<start>"   # Bilibili only
#   END_DAY   = "<end>"     # Bilibili only
# Then run: cd $MEDIACRAWLER && .venv/bin/python main.py
#
# Platforms: wb (Weibo), zhihu, dy (Douyin), bili (Bilibili)
# Douyin + Zhihu require interactive browser login on first run.

echo "=== CRAWL CHECKLIST ==="
for i in 1 2 3 4 5 6 7; do
    eval "eid=\$EVENT_ID_$i"
    eval "kw=\$KEYWORDS_$i"
    eval "s=\$START_$i"
    eval "e=\$END_$i"
    echo ""
    echo "[$i] $eid"
    echo "    KEYWORDS: $kw"
    echo "    DATE: $s → $e"
    echo "    Platforms to crawl: [ ] wb  [ ] zhihu  [ ] dy  [ ] bili"
done


# ============================================================
# STEP 2: CONVERT  (run after all 4 platforms crawled per event)
# ============================================================
convert_event() {
    local event_id=$1
    local start=$2
    local end=$3

    echo "Converting $event_id ..."
    $MBRIDGENET/.venv/bin/python $MBRIDGENET/scripts/convert_mediacrawler.py \
        --inputs \
            $MEDIACRAWLER/data/wb/search/*.jsonl \
            $MEDIACRAWLER/data/zhihu/search/*.jsonl \
            $MEDIACRAWLER/data/dy/search/*.jsonl \
            $MEDIACRAWLER/data/bili/search/*.jsonl \
        --event-id "$event_id" \
        --output $MBRIDGENET/data/cphot/raw/${event_id}.json \
        --start "$start" --end "$end"
}

# Uncomment to run:
# convert_event "sam_altman_ousting_001"    "2023-10-01" "2024-03-31"
# convert_event "titan_submersible_001"     "2023-06-01" "2023-10-31"
# convert_event "paris_olympics_ceremony_001" "2024-06-01" "2024-10-31"
# convert_event "us_election_2024_001"      "2024-09-01" "2025-02-28"
# convert_event "eileen_gu_identity_001"    "2022-01-01" "2022-08-31"
# convert_event "sora_debut_001"            "2024-01-01" "2024-07-31"
# convert_event "tiktok_ban_001"            "2024-01-01" "2024-08-31"


# ============================================================
# STEP 3: ANNOTATE  (LLM annotation, ~1-2h per event)
# ============================================================
annotate_event() {
    local event_id=$1
    echo "Annotating $event_id ..."
    # Dry run first (5 pairs, no save):
    # OPENAI_API_KEY=$OPENAI_KEY $MBRIDGENET/.venv/bin/python $MBRIDGENET/scripts/llm_annotate.py \
    #     --event $MBRIDGENET/data/cphot/raw/${event_id}.json \
    #     --model gpt-4.1-mini --dry-run

    OPENAI_API_KEY=$OPENAI_KEY $MBRIDGENET/.venv/bin/python $MBRIDGENET/scripts/llm_annotate.py \
        --event $MBRIDGENET/data/cphot/raw/${event_id}.json \
        --model gpt-4.1-mini --tau 0.85 --max-k 1000
}

# Uncomment to run (after convert):
# annotate_event "sam_altman_ousting_001"
# annotate_event "titan_submersible_001"
# annotate_event "paris_olympics_ceremony_001"
# annotate_event "us_election_2024_001"
# annotate_event "eileen_gu_identity_001"
# annotate_event "sora_debut_001"
# annotate_event "tiktok_ban_001"


# ============================================================
# STEP 4: PREPARE TRAINING DATA  (compute s1-s4 + lifecycle phase)
# ============================================================
prepare_event() {
    local event_id=$1
    echo "Preparing $event_id ..."
    $MBRIDGENET/.venv/bin/python $MBRIDGENET/scripts/prepare_training_data.py \
        --event $MBRIDGENET/data/cphot/raw/${event_id}.json \
        --output $MBRIDGENET/data/cphot/processed/test_real
}

# Uncomment to run (after annotate):
# prepare_event "sam_altman_ousting_001"
# prepare_event "titan_submersible_001"
# prepare_event "paris_olympics_ceremony_001"
# prepare_event "us_election_2024_001"
# prepare_event "eileen_gu_identity_001"
# prepare_event "sora_debut_001"
# prepare_event "tiktok_ban_001"


# ============================================================
# STEP 5: EVALUATE on full 15-event test set
# ============================================================
# MBRIDGENET_NO_FAISS=1 $MBRIDGENET/.venv/bin/python $MBRIDGENET/scripts/evaluate.py \
#     --data $MBRIDGENET/data/cphot/processed/test_real \
#     --checkpoint $MBRIDGENET/checkpoints/mlp_v25_fold2.pt \
#     --use-s1s3-scoring \
#     --k 5 10 20 50

echo ""
echo "xuzhou_chained_woman_001 already processed → test_real ✓"
echo "Current test_real count: $(ls data/cphot/processed/test_real/*.json 2>/dev/null | wc -l) events"
