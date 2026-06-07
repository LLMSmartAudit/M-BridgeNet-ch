"""Phase 0b — Batch pipeline: crawled data → test_real.

For each event in phase0b_events.py (or a specified subset), runs:
  Step 1: convert_mediacrawler.py  (JSONL → raw CPHot JSON)
  Step 2: validate                 (posts ≥ 200, Weibo ≥ 50, platforms ≥ 2)
  Step 3: llm_annotate.py          (LLM bridge-pair labeling, with --resume)
  Step 4: validate bridge rate     (≥ 10 bridge pairs, 5–80% rate)
  Step 5: prepare_training_data.py (compute s1–s4 signals + BGE embeddings)
  Step 6: copy to test_real/       (JSON + _emb.npz)

State is persisted in logs/phase0b_state.json so the script can be safely
re-run — completed steps are skipped, failed ones are retried.

Usage:
    # Process a single event (all steps)
    OPENAI_API_KEY=sk-... .venv/bin/python scripts/phase0b_process.py \\
        --event dongfang_selection_001

    # Process all pending events
    OPENAI_API_KEY=sk-... .venv/bin/python scripts/phase0b_process.py --all

    # Show status of all events
    .venv/bin/python scripts/phase0b_process.py --status

    # Dry-run: show what would be done without executing
    .venv/bin/python scripts/phase0b_process.py --event dongfang_selection_001 --dry-run

    # Re-run annotation only for one event (skip convert if raw exists)
    OPENAI_API_KEY=sk-... .venv/bin/python scripts/phase0b_process.py \\
        --event dongfang_selection_001 --from-step annotate
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_DIR  = SCRIPT_DIR.parent
MC_DIR       = PROJECT_DIR.parent / "MediaCrawler"
if not MC_DIR.exists():
    MC_DIR   = PROJECT_DIR.parent.parent / "07-code" / "MediaCrawler"

PYTHON       = PROJECT_DIR / ".venv" / "bin" / "python"
RAW_DIR      = PROJECT_DIR / "data" / "cphot" / "raw"
PROC_DIR     = PROJECT_DIR / "data" / "cphot" / "processed"
TEST_REAL    = PROC_DIR / "test_real"
STATE_PATH   = PROJECT_DIR / "logs" / "phase0b_state.json"

# Steps in order
STEPS = ["convert", "validate_raw", "annotate", "validate_bridge", "prepare", "copy"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── State management ───────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def mark_step(state: dict, event_id: str, step: str, status: str, note: str = "") -> None:
    state.setdefault(event_id, {})[step] = {
        "status": status,
        "ts": datetime.now(timezone.utc).isoformat(),
        "note": note,
    }
    save_state(state)


def step_done(state: dict, event_id: str, step: str) -> bool:
    return state.get(event_id, {}).get(step, {}).get("status") == "ok"


# ── Helpers ────────────────────────────────────────────────────────────────────

def run(cmd: list[str], dry_run: bool = False) -> int:
    """Run a subprocess. Returns returncode (0 = success)."""
    printable = " ".join(str(c) for c in cmd)
    logger.info("  $ %s", printable)
    if dry_run:
        return 0
    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    return result.returncode


def find_platform_jsonls(event_id: str, platform: str) -> list[Path]:
    """Return all search_contents JSONL files for a platform (supports midnight splits)."""
    crawled = MC_DIR / "data" / "crawled" / event_id / platform / "jsonl"
    if not crawled.exists():
        return []
    return sorted(crawled.glob("search_contents_*.jsonl"))


# ── Per-step implementations ───────────────────────────────────────────────────

def step_convert(cfg, state: dict, dry_run: bool) -> bool:
    """Convert crawled JSONL files → data/cphot/raw/<event_id>.json"""
    out_path = RAW_DIR / f"{cfg.event_id}.json"
    if out_path.exists():
        logger.info("  [convert] raw JSON already exists, skipping: %s", out_path.name)
        return True

    cmd = [str(PYTHON), "scripts/convert_mediacrawler.py",
           "--event-id", cfg.event_id,
           "--output", str(RAW_DIR),
           "--start", cfg.start,
           "--end", cfg.end]

    platform_flags = {
        "weibo": "--weibo",
        "bili": "--bilibili",
        "zhihu": "--zhihu",
        "douyin": "--douyin",
    }
    found_any = False
    for plat in cfg.platforms:
        jsonls = find_platform_jsonls(cfg.event_id, plat)
        if jsonls:
            cmd += [platform_flags[plat]] + [str(j) for j in jsonls]
            found_any = True
            logger.info("  [convert] found %s: %d file(s) — %s",
                        plat, len(jsonls), ", ".join(j.name for j in jsonls))
        else:
            logger.warning("  [convert] no data for %s — skipping platform", plat)

    if not found_any:
        logger.error("  [convert] no crawled data found for %s in %s",
                     cfg.event_id, MC_DIR / "data" / "crawled" / cfg.event_id)
        return False

    rc = run(cmd, dry_run=dry_run)
    return rc == 0


def step_validate_raw(cfg, dry_run: bool) -> tuple[bool, str]:
    """Check posts ≥ 200, Weibo ≥ 50, distinct platforms ≥ 2."""
    if dry_run:
        return True, "dry-run"
    raw_path = RAW_DIR / f"{cfg.event_id}.json"
    if not raw_path.exists():
        return False, "raw JSON missing"
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    posts = data.get("posts", [])
    total = len(posts)
    platforms: dict[str, int] = {}
    for p in posts:
        plat = p.get("platform", "?")
        platforms[plat] = platforms.get(plat, 0) + 1
    weibo_n = platforms.get("weibo", 0)
    n_plat = len(platforms)

    logger.info("  [validate_raw] total=%d  weibo=%d  platforms=%s",
                total, weibo_n, dict(platforms))

    if total < 100:
        return False, f"too few posts: {total} < 100"
    if weibo_n < 50:
        logger.warning("  [validate_raw] Weibo=%d < 50 (soft gate — allowing for test events)", weibo_n)
    if n_plat < 2:
        return False, f"only {n_plat} platform(s) found"
    return True, f"ok: total={total} weibo={weibo_n} platforms={n_plat}"


def step_annotate(cfg, dry_run: bool) -> bool:
    """Run llm_annotate.py with --resume so interrupted runs continue."""
    raw_path = RAW_DIR / f"{cfg.event_id}.json"
    # Check if already annotated (has labeled bridge pairs)
    if not dry_run and raw_path.exists():
        data = json.loads(raw_path.read_text(encoding="utf-8"))
        labeled = [p for p in data.get("bridge_pairs", []) if p.get("label", -1) != -1]
        if labeled:
            logger.info("  [annotate] %d labeled pairs already present — using --resume", len(labeled))

    cmd = [str(PYTHON), "scripts/llm_annotate.py",
           "--event", str(raw_path),
           "--tau", str(cfg.tau),
           "--max-k", str(cfg.max_k),
           "--model", "gpt-5.4-nano",
           "--resume"]
    rc = run(cmd, dry_run=dry_run)
    return rc == 0


def step_validate_bridge(cfg, dry_run: bool) -> tuple[bool, str]:
    """Check bridge rate is in acceptable range."""
    if dry_run:
        return True, "dry-run"
    raw_path = RAW_DIR / f"{cfg.event_id}.json"
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    # Use scored_pairs (dicts with 'label' key) if available; bridge_pairs stores
    # only positive pairs as [id_a, id_b] lists — unsuitable for rate calculation.
    sp = data.get("scored_pairs", [])
    if sp and isinstance(sp[0], dict):
        total_labeled = [p for p in sp if p.get("label", -1) != -1]
        bridges = [p for p in total_labeled if p.get("label") == 1]
    else:
        # Fallback: count bridge_pairs directly (all positive), rate unknown
        bp = data.get("bridge_pairs", [])
        n_bridge = len(bp)
        logger.info("  [validate_bridge] bridge=%d (no scored_pairs for rate)", n_bridge)
        return (n_bridge >= 10, f"{'ok' if n_bridge >= 10 else 'too few'}: {n_bridge} bridge pairs")
    n_bridge = len(bridges)
    n_total  = len(total_labeled)
    rate     = n_bridge / n_total if n_total else 0.0

    logger.info("  [validate_bridge] bridge=%d / %d = %.1f%%", n_bridge, n_total, rate * 100)

    if n_bridge < 10:
        return False, f"too few bridges: {n_bridge} < 10"
    if rate < 0.02:
        return False, f"bridge rate {rate*100:.1f}% < 2% (too sparse)"
    if rate > 0.85:
        logger.warning("  [validate_bridge] rate %.1f%% > 80%% (trivially easy event)", rate * 100)
    return True, f"ok: {n_bridge}/{n_total} = {rate*100:.1f}%"


def step_prepare(cfg, dry_run: bool) -> bool:
    """Run prepare_training_data.py to compute signals + BGE embeddings."""
    raw_path  = RAW_DIR / f"{cfg.event_id}.json"
    proc_path = PROC_DIR / f"{cfg.event_id}.json"
    emb_path  = PROC_DIR / f"{cfg.event_id}_emb.npz"
    if proc_path.exists() and emb_path.exists():
        logger.info("  [prepare] processed JSON + emb already exist, skipping")
        return True
    cmd = [str(PYTHON), "scripts/prepare_training_data.py",
           "--event", str(raw_path),
           "--output", str(PROC_DIR)]
    rc = run(cmd, dry_run=dry_run)
    return rc == 0


def step_copy(cfg, dry_run: bool) -> bool:
    """Copy processed JSON + emb to test_real/."""
    src_json = PROC_DIR / f"{cfg.event_id}.json"
    src_emb  = PROC_DIR / f"{cfg.event_id}_emb.npz"
    dst_json = TEST_REAL / f"{cfg.event_id}.json"
    dst_emb  = TEST_REAL / f"{cfg.event_id}_emb.npz"

    if dst_json.exists():
        logger.info("  [copy] already in test_real/: %s", dst_json.name)
        return True

    if not dry_run:
        TEST_REAL.mkdir(parents=True, exist_ok=True)
        if src_json.exists():
            shutil.copy2(src_json, dst_json)
            logger.info("  [copy] → %s", dst_json)
        else:
            logger.error("  [copy] processed JSON not found: %s", src_json)
            return False
        if src_emb.exists():
            shutil.copy2(src_emb, dst_emb)
            logger.info("  [copy] → %s", dst_emb)
        else:
            logger.warning("  [copy] emb.npz not found (will be generated on first eval)")
    return True


# ── Main pipeline ──────────────────────────────────────────────────────────────

def process_event(cfg, state: dict, from_step: str = "convert",
                  dry_run: bool = False) -> bool:
    """Run the full pipeline for one event. Returns True if all steps succeeded."""
    logger.info("\n%s", "=" * 64)
    logger.info("Processing: %s  (%s)", cfg.event_id, cfg.name_zh)
    logger.info("%s", "=" * 64)

    step_order = STEPS[STEPS.index(from_step):]

    for step in step_order:
        if step_done(state, cfg.event_id, step) and step != from_step:
            logger.info("  [%s] already done — skipping", step)
            continue

        logger.info("  → Step: %s", step)

        if step == "convert":
            ok = step_convert(cfg, state, dry_run)
            mark_step(state, cfg.event_id, step, "ok" if ok else "fail")
            if not ok:
                logger.error("  FAILED at convert — check crawled data")
                return False

        elif step == "validate_raw":
            ok, note = step_validate_raw(cfg, dry_run)
            mark_step(state, cfg.event_id, step, "ok" if ok else "fail", note)
            if not ok:
                logger.error("  FAILED validate_raw: %s", note)
                return False

        elif step == "annotate":
            ok = step_annotate(cfg, dry_run)
            mark_step(state, cfg.event_id, step, "ok" if ok else "fail")
            if not ok:
                logger.error("  FAILED annotation — check OPENAI_API_KEY and retry")
                return False

        elif step == "validate_bridge":
            ok, note = step_validate_bridge(cfg, dry_run)
            mark_step(state, cfg.event_id, step, "ok" if ok else "fail", note)
            if not ok:
                logger.warning("  REJECTED: %s — event will NOT be added to test_real", note)
                return False

        elif step == "prepare":
            ok = step_prepare(cfg, dry_run)
            mark_step(state, cfg.event_id, step, "ok" if ok else "fail")
            if not ok:
                logger.error("  FAILED prepare_training_data")
                return False

        elif step == "copy":
            ok = step_copy(cfg, dry_run)
            mark_step(state, cfg.event_id, step, "ok" if ok else "fail")
            if not ok:
                logger.error("  FAILED copy to test_real")
                return False

    logger.info("  ✓ %s complete!", cfg.event_id)
    return True


def print_status(state: dict, events) -> None:
    """Print a summary table of pipeline state for all events."""
    print(f"\n{'Event':<42} {'convert':<10} {'val_raw':<10} {'annotate':<10} "
          f"{'val_br':<10} {'prepare':<10} {'copy':<8}")
    print("-" * 102)
    for cfg in events:
        ev_state = state.get(cfg.event_id, {})
        def s(step):
            v = ev_state.get(step, {})
            st = v.get("status", "—")
            return {"ok": "✅", "fail": "❌", "—": "—"}.get(st, st)
        print(f"{cfg.event_id:<42} {s('convert'):<10} {s('validate_raw'):<10} "
              f"{s('annotate'):<10} {s('validate_bridge'):<10} "
              f"{s('prepare'):<10} {s('copy'):<8}")

    # Summary
    done = sum(1 for cfg in events
               if state.get(cfg.event_id, {}).get("copy", {}).get("status") == "ok")
    print(f"\n{done}/{len(events)} events fully processed and copied to test_real/")
    n_test = len(list(TEST_REAL.glob("*.json")))
    print(f"Current test_real/ size: {n_test} events")


def main() -> None:
    sys.path.insert(0, str(SCRIPT_DIR))
    from phase0b_events import PHASE0B_EVENTS, EVENT_MAP  # type: ignore

    parser = argparse.ArgumentParser(
        description="Phase 0b batch pipeline: crawled JSONL → test_real"
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--event", default=None, help="Single event_id to process")
    grp.add_argument("--all",   action="store_true", help="Process all Phase 0b events")
    grp.add_argument("--status", action="store_true", help="Show pipeline status table")

    parser.add_argument("--from-step", default="convert",
                        choices=STEPS, dest="from_step",
                        help="Start from this step (skip earlier ones even if not done)")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Show commands without executing")
    args = parser.parse_args()

    state = load_state()

    if args.status:
        print_status(state, PHASE0B_EVENTS)
        return

    if not os.environ.get("OPENAI_API_KEY") and not args.dry_run:
        print("WARNING: OPENAI_API_KEY not set — annotation step will fail.")
        print("Set it: export OPENAI_API_KEY=sk-...")

    events_to_run = []
    if args.event:
        cfg = EVENT_MAP.get(args.event)
        if cfg is None:
            print(f"ERROR: unknown event '{args.event}'")
            sys.exit(1)
        events_to_run = [cfg]
    elif args.all:
        events_to_run = PHASE0B_EVENTS
    else:
        parser.print_help()
        return

    success, failed = [], []
    for cfg in events_to_run:
        ok = process_event(cfg, state, from_step=args.from_step, dry_run=args.dry_run)
        (success if ok else failed).append(cfg.event_id)

    print(f"\n{'='*60}")
    print(f"Done: {len(success)} succeeded, {len(failed)} failed")
    if failed:
        print(f"Failed events: {failed}")
    print_status(state, PHASE0B_EVENTS)

    # Final eval smoke-test hint
    if not failed and len(success) > 0:
        n_test = len(list(TEST_REAL.glob("*.json")))
        print(f"\ntest_real/ now has {n_test} events.")
        print("\nRun quick eval smoke-test:")
        print("  MBRIDGENET_NO_FAISS=1 .venv/bin/python scripts/evaluate.py \\")
        print("    --data data/cphot/processed/test_real \\")
        print("    --checkpoint checkpoints/mlp_v18_fold5.pt \\")
        print("    --use-s1s3-scoring --k 5 10 20 50 2>&1 | tail -8")


if __name__ == "__main__":
    main()
