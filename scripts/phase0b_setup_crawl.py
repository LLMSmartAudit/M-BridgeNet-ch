"""Phase 0b — MediaCrawler config patcher.

Patches MediaCrawler's config files for a given event + platform combination,
prints the crawl command, and (optionally) opens the config file for review.

Usage:
    # Show configs for all events without writing
    python scripts/phase0b_setup_crawl.py --dry-run

    # Set up MediaCrawler for a specific event + platform, then run
    python scripts/phase0b_setup_crawl.py \\
        --event dongfang_selection_001 \\
        --platform weibo

    # Set up and patch all platforms for one event (shows commands in order)
    python scripts/phase0b_setup_crawl.py --event dongfang_selection_001 --all-platforms

After patching, run from the MediaCrawler directory:
    cd ../MediaCrawler && .venv/bin/python main.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Locate the MediaCrawler directory relative to this script
SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MC_DIR      = PROJECT_DIR.parent / "MediaCrawler"

if not MC_DIR.exists():
    # Try sibling directory
    MC_DIR = PROJECT_DIR.parent.parent / "07-code" / "MediaCrawler"

# Platform → MediaCrawler PLATFORM string and output dir
PLATFORM_CODE = {
    "weibo":  "wb",
    "bili":   "bili",
    "zhihu":  "xhs",   # NOTE: zhihu uses its own config
    "douyin": "dy",
}

PLATFORM_CONFIG_FILES = {
    "weibo":  "config/weibo_config.py",
    "bili":   "config/bilibili_config.py",
    "zhihu":  "config/zhihu_config.py",
    "douyin": "config/dy_config.py",
}

# Platform code used in base_config.py PLATFORM field
PLATFORM_BASE_CODE = {
    "weibo":  "wb",
    "bili":   "bili",
    "zhihu":  "zhihu",
    "douyin": "dy",
}


def _patch_file(path: Path, replacements: dict[str, str], dry_run: bool = False) -> None:
    """Apply regex-based replacements to a config file."""
    text = path.read_text(encoding="utf-8")
    for pattern, replacement in replacements.items():
        text, n = re.subn(pattern, replacement, text, flags=re.MULTILINE)
        if n == 0:
            print(f"  WARNING: pattern not found in {path.name}: {pattern!r}")
    if dry_run:
        print(f"  [dry-run] would patch {path}")
    else:
        path.write_text(text, encoding="utf-8")
        print(f"  Patched: {path}")


def setup_crawl(event_id: str, platform: str, dry_run: bool = False) -> None:
    """Patch MediaCrawler config for a given event + platform."""
    from phase0b_events import EVENT_MAP  # type: ignore

    cfg = EVENT_MAP.get(event_id)
    if cfg is None:
        print(f"ERROR: unknown event '{event_id}'. Run with --list to see available events.")
        sys.exit(1)
    if platform not in cfg.platforms:
        print(f"WARNING: platform '{platform}' not in expected platforms {cfg.platforms} for {event_id}")

    save_path = (MC_DIR / "data" / "crawled" / event_id).resolve()
    base_cfg  = MC_DIR / "config" / "base_config.py"
    plat_cfg  = MC_DIR / PLATFORM_CONFIG_FILES[platform]

    print(f"\n{'='*60}")
    print(f"Event    : {event_id}  ({cfg.name_zh})")
    print(f"Platform : {platform}")
    print(f"Keywords : {cfg.keywords}")
    print(f"Dates    : {cfg.start} → {cfg.end}")
    print(f"Output   : {save_path}")
    print(f"{'='*60}")

    # ── Patch base_config.py ──────────────────────────────────────────────
    base_replacements = {
        r'^PLATFORM\s*=\s*".*?"': f'PLATFORM = "{PLATFORM_BASE_CODE[platform]}"',
        r'^KEYWORDS\s*=\s*".*?"': f'KEYWORDS = "{cfg.keywords}"',
        r'^SAVE_DATA_PATH\s*=\s*".*?"': f'SAVE_DATA_PATH = "{save_path}"',
        r'^SAVE_DATA_OPTION\s*=\s*".*?"': 'SAVE_DATA_OPTION = "jsonl"',
        r'^CRAWLER_MAX_NOTES_COUNT\s*=\s*\d+': 'CRAWLER_MAX_NOTES_COUNT = 500',
    }
    _patch_file(base_cfg, base_replacements, dry_run=dry_run)

    # ── Patch per-platform date config ────────────────────────────────────
    date_replacements = {
        r'^START_DAY\s*=\s*".*?"': f'START_DAY = "{cfg.start}"',
        r'^END_DAY\s*=\s*".*?"':   f'END_DAY   = "{cfg.end}"',
    }
    _patch_file(plat_cfg, date_replacements, dry_run=dry_run)

    # ── Print run command ─────────────────────────────────────────────────
    print(f"\nRun command:")
    print(f"  cd {MC_DIR}")
    print(f"  .venv/bin/python main.py")
    print(f"\nExpected output:")
    print(f"  {save_path}/{platform}/jsonl/search_contents_<date>.jsonl")


def list_events() -> None:
    """Print all Phase 0b events in crawl order."""
    from phase0b_events import PHASE0B_EVENTS  # type: ignore
    print(f"\nPhase 0b events ({len(PHASE0B_EVENTS)} total, in recommended crawl order):\n")
    print(f"{'#':<3} {'Group':<6} {'Event ID':<40} {'Platforms':<30} {'Dates'}")
    print("-" * 110)
    for i, ev in enumerate(PHASE0B_EVENTS, 1):
        plat = ",".join(ev.platforms)
        print(f"{i:<3} [{ev.group}]   {ev.event_id:<40} {plat:<30} {ev.start} → {ev.end}")
        print(f"     中文: {ev.name_zh}")
        print(f"     关键词: {ev.keywords}")
        if ev.notes:
            print(f"     注意: {ev.notes}")
        print()


def main() -> None:
    # Add parent dir to path so we can import phase0b_events
    sys.path.insert(0, str(SCRIPT_DIR))

    parser = argparse.ArgumentParser(description="Set up MediaCrawler config for Phase 0b events")
    parser.add_argument("--event", default=None, help="Event ID (e.g. dongfang_selection_001)")
    parser.add_argument("--platform", default=None,
                        choices=["weibo", "bili", "zhihu", "douyin"],
                        help="Platform to crawl")
    parser.add_argument("--all-platforms", action="store_true", dest="all_platforms",
                        help="Show setup commands for all platforms of the event")
    parser.add_argument("--list", action="store_true", help="List all Phase 0b events")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Print what would be patched without writing")
    args = parser.parse_args()

    if args.list:
        list_events()
        return

    if args.event is None:
        parser.print_help()
        return

    if not MC_DIR.exists():
        print(f"ERROR: MediaCrawler not found at {MC_DIR}")
        print("Set MC_DIR in this script to the correct path.")
        sys.exit(1)

    from phase0b_events import EVENT_MAP  # type: ignore
    cfg = EVENT_MAP.get(args.event)
    if cfg is None:
        print(f"ERROR: unknown event '{args.event}'")
        sys.exit(1)

    if args.all_platforms:
        for plat in cfg.platforms:
            setup_crawl(args.event, plat, dry_run=args.dry_run)
            print()
    elif args.platform:
        setup_crawl(args.event, args.platform, dry_run=args.dry_run)
    else:
        # Default: show first platform
        setup_crawl(args.event, cfg.platforms[0], dry_run=args.dry_run)
        print(f"\nHint: run with --all-platforms to see all {len(cfg.platforms)} platform configs")


if __name__ == "__main__":
    main()
