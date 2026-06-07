#!/usr/bin/env python3
"""
Full collection pipeline for 7 new test events.
Runs MediaCrawler for each event × platform, then convert → annotate → prepare.

Usage (from M-BridgeNet/):
    OPENAI_API_KEY=sk-... .venv/bin/python scripts/run_full_collection.py
    OPENAI_API_KEY=sk-... .venv/bin/python scripts/run_full_collection.py --event sam_altman_ousting_001
    OPENAI_API_KEY=sk-... .venv/bin/python scripts/run_full_collection.py --step crawl
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
MC   = ROOT.parent / "MediaCrawler"
PYTHON_MC = MC / ".venv/bin/python"
PYTHON_MB = ROOT / ".venv/bin/python"

RAW_DIR       = ROOT / "data/cphot/raw"
TEST_REAL_DIR = ROOT / "data/cphot/processed/test_real"

# ── Event definitions ───────────────────────────────────────────────────────
EVENTS = [
    {
        "id":       "sam_altman_ousting_001",
        "keywords": "奥特曼被解雇,OpenAI宫斗,山姆奥特曼",
        "start":    "2023-10-01",
        "end":      "2024-03-31",
    },
    {
        "id":       "titan_submersible_001",
        "keywords": "泰坦号潜艇,泰坦潜水器失踪,泰坦内爆",
        "start":    "2023-06-01",
        "end":      "2023-10-31",
    },
    {
        "id":       "paris_olympics_ceremony_001",
        "keywords": "巴黎奥运开幕式,巴黎奥运争议,最后的晚餐巴黎",
        "start":    "2024-06-01",
        "end":      "2024-10-31",
    },
    {
        "id":       "us_election_2024_001",
        "keywords": "美国大选,特朗普胜选,哈里斯败选",
        "start":    "2024-09-01",
        "end":      "2025-02-28",
    },
    {
        "id":       "eileen_gu_identity_001",
        "keywords": "谷爱凌国籍,谷爱凌身份,谷爱凌双重国籍",
        "start":    "2022-01-01",
        "end":      "2022-08-31",
    },
    {
        "id":       "sora_debut_001",
        "keywords": "Sora发布,OpenAI Sora,sora视频生成",
        "start":    "2024-01-01",
        "end":      "2024-07-31",
    },
    {
        "id":       "tiktok_ban_001",
        "keywords": "TikTok禁令,抖音美国禁令,TikTok法案",
        "start":    "2024-01-01",
        "end":      "2024-08-31",
    },
    {
        "id":       "us_kill_chain_001",
        "keywords": "美国斩杀线,美军AI武器,自主杀伤系统",
        "start":    "2023-01-01",
        "end":      "2025-06-30",
    },
    {
        "id":       "zhang_xuefeng_001",
        "keywords": "张雪峰,新闻专业不能报,张雪峰志愿填报",
        "start":    "2023-05-01",
        "end":      "2023-12-31",
    },
    {
        "id":       "zhang_xuefeng_death_001",
        "keywords": "张雪峰猝死,张雪峰去世,张雪峰出事",
        "start":    "2026-03-24",
        "end":      "2026-05-01",
    },
    {
        "id":       "us_china_reckoning_001",
        "keywords": "中美大对账,美国对账,中美对账",
        "start":    "2025-01-01",
        "end":      "2025-06-30",
    },
    {
        "id":       "black_myth_wukong_001",
        "keywords": "黑神话悟空,黑神话带动,悟空游戏影响",
        "start":    "2024-08-01",
        "end":      "2025-01-31",
    },
]

PLATFORMS = ["wb", "zhihu", "bili", "dy"]

# Platform code → MediaCrawler storage subfolder name (used inside SAVE_DATA_PATH)
PLATFORM_DIR = {
    "wb":    "weibo",
    "zhihu": "zhihu",
    "dy":    "douyin",
    "bili":  "bili",
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def run(cmd: list[str], cwd: Path, timeout: int = 5400) -> int:
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    try:
        result = subprocess.run(cmd, cwd=str(cwd), timeout=timeout)
        return result.returncode
    except subprocess.TimeoutExpired:
        print(f"  ⚠ Crawler timed out after {timeout}s — data already saved incrementally, continuing")
        return 0


def set_base_config(platform: str, keywords: str, event_id: str) -> None:
    cfg = MC / "config/base_config.py"
    text = cfg.read_text(encoding="utf-8")
    text = _replace_line(text, "PLATFORM", f'PLATFORM = "{platform}"')
    text = _replace_line(text, "KEYWORDS", f'KEYWORDS = "{keywords}"')
    save_path = MC / f"data/crawled/{event_id}"
    text = _replace_line(text, "SAVE_DATA_PATH", f'SAVE_DATA_PATH = "{save_path}"')
    cfg.write_text(text, encoding="utf-8")


def set_bili_dates(start: str, end: str) -> None:
    cfg = MC / "config/bilibili_config.py"
    text = cfg.read_text(encoding="utf-8")
    text = _replace_line(text, "START_DAY", f'START_DAY = "{start}"')
    text = _replace_line(text, "END_DAY",   f'END_DAY   = "{end}"')
    cfg.write_text(text, encoding="utf-8")


def _replace_line(text: str, key: str, new_line: str) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(key + " ") or stripped.startswith(key + "="):
            lines[i] = new_line
            return "\n".join(lines)
    return text  # key not found — leave unchanged


def report_platform_data(event_id: str, platform: str) -> list[Path]:
    """List jsonl files written by MediaCrawler into the per-event SAVE_DATA_PATH."""
    platform_dir = MC / "data/crawled" / event_id / PLATFORM_DIR[platform]
    files = list(platform_dir.rglob("*.jsonl"))
    print(f"  Found {len(files)} jsonl file(s) in {platform_dir}")
    return files


def collect_staged_inputs(event_id: str) -> list[str]:
    """Return all jsonl files MediaCrawler wrote for an event across all platforms."""
    stage = MC / "data/crawled" / event_id
    files = list(stage.rglob("*.jsonl"))
    return [str(f) for f in sorted(files)]


def already_done(event_id: str, step: str) -> bool:
    """Check if a step has been completed for an event."""
    if step == "crawl":
        return all(platform_done(event_id, p) for p in PLATFORMS)
    if step == "convert":
        return (RAW_DIR / f"{event_id}.json").exists()
    if step == "annotate":
        import json
        raw = RAW_DIR / f"{event_id}.json"
        if not raw.exists():
            return False
        d = json.loads(raw.read_text(encoding="utf-8"))
        pairs = d.get("bridge_pairs", [])
        return len(pairs) > 0
    if step == "prepare":
        return (TEST_REAL_DIR / f"{event_id}.json").exists()
    return False


# ── Per-step functions ───────────────────────────────────────────────────────

def platform_done(event_id: str, platform: str) -> bool:
    """True if this platform already has at least one content jsonl file."""
    platform_name = PLATFORM_DIR[platform]
    src_dir = MC / "data/crawled" / event_id / platform_name / "jsonl"
    return src_dir.exists() and any(src_dir.glob("search_contents_*.jsonl"))


def crawl_event(ev: dict) -> None:
    event_id = ev["id"]
    print(f"\n{'='*60}")
    print(f"CRAWL: {event_id}")
    print(f"{'='*60}")

    any_missing = any(not platform_done(event_id, p) for p in PLATFORMS)
    if not any_missing:
        print(f"  ✓ Already crawled all platforms — skipping")
        return

    # Remove stale SingletonLocks before starting
    for platform_name in PLATFORM_DIR.values():
        lock = MC / f"browser_data/{platform_name}_user_data_dir/SingletonLock"
        if lock.exists():
            lock.unlink()
            print(f"  Removed stale lock: {lock.name}")

    for platform in PLATFORMS:
        if platform_done(event_id, platform):
            files = report_platform_data(event_id, platform)
            print(f"  ✓ {platform}: already crawled ({len(files)} file(s)) — skipping")
            continue

        print(f"\n  Platform: {platform}")
        set_base_config(platform, ev["keywords"], event_id)
        if platform == "bili":
            set_bili_dates(ev["start"], ev["end"])

        rc = run([str(PYTHON_MC), "main.py"], cwd=MC, timeout=5400)
        if rc != 0:
            print(f"  ⚠ Crawler exited with code {rc} for {platform} — continuing")
        time.sleep(3)
        report_platform_data(event_id, platform)

    print(f"  ✓ Crawl complete for {event_id}")


def merge_platform_files(event_id: str, platform_code: str) -> Path | None:
    """Concatenate all search_contents_*.jsonl for a platform into one temp file."""
    platform_name = PLATFORM_DIR[platform_code]
    src_dir = MC / "data/crawled" / event_id / platform_name / "jsonl"
    content_files = sorted(src_dir.glob("search_contents_*.jsonl"))
    if not content_files:
        return None
    tmp = ROOT / f"data/cphot/tmp_{event_id}_{platform_name}.jsonl"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as out:
        for f in content_files:
            out.write(f.read_text(encoding="utf-8"))
    return tmp


def convert_event(ev: dict) -> None:
    event_id = ev["id"]
    print(f"\n{'='*60}")
    print(f"CONVERT: {event_id}")
    print(f"{'='*60}")

    if already_done(event_id, "convert"):
        print(f"  ✓ Already converted — skipping")
        return

    # Build per-platform merged files
    platform_map = {
        "wb":    ("weibo",    "--weibo"),
        "zhihu": ("zhihu",   "--zhihu"),
        "bili":  ("bilibili", "--bilibili"),
        "dy":    ("douyin",   "--douyin"),
    }
    cmd = [str(PYTHON_MB), "scripts/convert_mediacrawler.py",
           "--event-id", event_id,
           "--output", str(RAW_DIR),
           "--start", ev["start"], "--end", ev["end"]]

    tmp_files = []
    found_any = False
    for platform_code, (_, flag) in platform_map.items():
        tmp = merge_platform_files(event_id, platform_code)
        if tmp:
            cmd += [flag, str(tmp)]
            tmp_files.append(tmp)
            found_any = True

    if not found_any:
        print(f"  ✗ No staged files found — run crawl first")
        return

    rc = run(cmd, cwd=ROOT, timeout=300)
    for tmp in tmp_files:
        tmp.unlink(missing_ok=True)  # clean up temp files

    output = RAW_DIR / f"{event_id}.json"
    if rc == 0:
        print(f"  ✓ Converted → {output}")
    else:
        print(f"  ✗ Convert failed (rc={rc})")


def annotate_event(ev: dict) -> None:
    event_id = ev["id"]
    print(f"\n{'='*60}")
    print(f"ANNOTATE: {event_id}")
    print(f"{'='*60}")

    if already_done(event_id, "annotate"):
        print(f"  ✓ Already annotated — skipping")
        return

    raw = RAW_DIR / f"{event_id}.json"
    if not raw.exists():
        print(f"  ✗ {raw} not found — run convert first")
        return

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        print(f"  ✗ OPENAI_API_KEY not set — skipping annotation")
        return

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = openai_key
    cmd = [
        str(PYTHON_MB), "scripts/llm_annotate.py",
        "--event", str(raw),
        "--model", "gpt-5.4-nano",
        "--tau", "0.85",
        "--max-k", "1000",
    ]
    print(f"  Running LLM annotation (this may take 1-2h) ...")
    result = subprocess.run(cmd, cwd=str(ROOT), env=env, timeout=7200)
    if result.returncode == 0:
        print(f"  ✓ Annotated {event_id}")
    else:
        print(f"  ✗ Annotation failed (rc={result.returncode})")


def prepare_event(ev: dict) -> None:
    event_id = ev["id"]
    print(f"\n{'='*60}")
    print(f"PREPARE: {event_id}")
    print(f"{'='*60}")

    if already_done(event_id, "prepare"):
        print(f"  ✓ Already prepared — skipping")
        return

    raw = RAW_DIR / f"{event_id}.json"
    if not raw.exists():
        print(f"  ✗ {raw} not found — run annotate first")
        return

    cmd = [
        str(PYTHON_MB), "scripts/prepare_training_data.py",
        "--event", str(raw),
        "--output", str(TEST_REAL_DIR),
    ]
    rc = run(cmd, cwd=ROOT, timeout=600)
    if rc == 0:
        print(f"  ✓ Prepared → {TEST_REAL_DIR}/{event_id}.json")
    else:
        print(f"  ✗ Prepare failed (rc={rc})")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", default=None, help="Run for a single event ID only")
    parser.add_argument("--step",  default="all",
                        choices=["all", "crawl", "convert", "annotate", "prepare"],
                        help="Run only a specific step")
    args = parser.parse_args()

    events = EVENTS
    if args.event:
        events = [e for e in EVENTS if e["id"] == args.event]
        if not events:
            print(f"Unknown event: {args.event}")
            sys.exit(1)

    steps = ["crawl", "convert", "annotate", "prepare"] if args.step == "all" else [args.step]
    step_fns = {
        "crawl":    crawl_event,
        "convert":  convert_event,
        "annotate": annotate_event,
        "prepare":  prepare_event,
    }

    print(f"\nM-BridgeNet collection pipeline")
    print(f"Events: {[e['id'] for e in events]}")
    print(f"Steps:  {steps}")
    print(f"MC:     {MC}")
    print(f"Out:    {TEST_REAL_DIR}")

    for ev in events:
        for step in steps:
            step_fns[step](ev)

    # Summary
    done = [e["id"] for e in EVENTS if already_done(e["id"], "prepare")]
    print(f"\n{'='*60}")
    print(f"DONE: {len(done)}/7 events fully prepared")
    for event_id in done:
        print(f"  ✓ {event_id}")
    total_test = len(list(TEST_REAL_DIR.glob("*.json")))
    print(f"\ntest_real now has {total_test} events")


if __name__ == "__main__":
    main()
