"""Collect real cross-platform posts for a given event keyword.

Queries Bilibili (public), Weibo (cookie), Zhihu (cookie), and optionally
Douyin (Research API) for posts matching the keyword within a time window,
then saves a CPHot-format JSON stub ready for bridge-pair annotation.

Usage:
    # Bilibili only (no auth needed):
    python scripts/collect_data.py \\
        --keyword "南海军事演习" \\
        --start 2024-03-01 --end 2024-03-04 \\
        --event-id e001 --output data/cphot/raw

    # All platforms (provide cookies via env vars):
    export WEIBO_COOKIE="..."
    export ZHIHU_COOKIE="..."
    python scripts/collect_data.py \\
        --keyword "南海军事演习" \\
        --start 2024-03-01 --end 2024-03-04 \\
        --event-id e001 --output data/cphot/raw \\
        --platforms weibo zhihu bilibili

After collection, label bridge pairs with:
    python scripts/annotate_pairs.py --event data/cphot/raw/e001.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from mbridgenet.data.collectors.bilibili import BilibiliCollector
from mbridgenet.data.collectors.weibo import WeiboCollector
from mbridgenet.data.collectors.zhihu import ZhihuCollector
from mbridgenet.data.collectors.douyin import DouyinCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

COLLECTOR_MAP = {
    "bilibili": BilibiliCollector,
    "weibo":    WeiboCollector,
    "zhihu":    ZhihuCollector,
    "douyin":   DouyinCollector,
}


def parse_date(s: str) -> datetime:
    """Accept YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS."""
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Cannot parse date '{s}'. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS."
    )


def deduplicate(posts: list) -> list:
    """Remove exact-duplicate post_ids."""
    seen = set()
    unique = []
    for p in posts:
        if p["post_id"] not in seen:
            seen.add(p["post_id"])
            unique.append(p)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect real cross-platform posts for an event keyword"
    )
    parser.add_argument("--keyword",   required=True,
                        help="Search keyword, e.g. '南海军事演习'")
    parser.add_argument("--start",     required=True, type=parse_date,
                        help="Start datetime (YYYY-MM-DD)")
    parser.add_argument("--end",       required=True, type=parse_date,
                        help="End datetime (YYYY-MM-DD)")
    parser.add_argument("--event-id",  default=None,
                        help="Event ID written into the JSON (default: keyword)")
    parser.add_argument("--output",    default="data/cphot/raw",
                        help="Output directory (default: data/cphot/raw)")
    parser.add_argument("--platforms", nargs="+",
                        default=["bilibili"],
                        choices=list(COLLECTOR_MAP),
                        help="Platforms to collect from (default: bilibili)")
    parser.add_argument("--max-per-platform", type=int, default=200,
                        help="Max posts per platform (default: 200)")
    parser.add_argument("--rate-limit", type=float, default=2.0,
                        help="Seconds between requests (default: 2.0)")
    args = parser.parse_args()

    event_id  = args.event_id or args.keyword
    out_dir   = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path  = out_dir / f"{event_id}.json"

    logger.info("Keyword   : %s", args.keyword)
    logger.info("Window    : %s → %s", args.start.date(), args.end.date())
    logger.info("Platforms : %s", args.platforms)
    logger.info("Output    : %s", out_path)

    all_posts = []

    for platform in args.platforms:
        cls = COLLECTOR_MAP[platform]
        collector = cls(rate_limit_seconds=args.rate_limit)

        logger.info("─── Collecting from %s …", platform)
        try:
            posts = collector.search(
                keyword    = args.keyword,
                start_time = args.start,
                end_time   = args.end,
                max_posts  = args.max_per_platform,
            )
        except NotImplementedError as exc:
            logger.warning("Skipping %s: %s", platform, exc)
            posts = []
        except Exception as exc:
            logger.error("Error collecting from %s: %s", platform, exc)
            posts = []

        # Stamp event_id on every post
        for p in posts:
            p["event_id"] = event_id

        logger.info("%s: %d posts collected", platform, len(posts))
        all_posts.extend(posts)

    all_posts = deduplicate(all_posts)
    all_posts.sort(key=lambda p: p["timestamp"])

    if not all_posts:
        logger.error("No posts collected. Check your keywords, date range, and cookies.")
        sys.exit(1)

    # Count per platform
    from collections import Counter
    plat_counts = Counter(p["platform"] for p in all_posts)
    logger.info("Total posts: %d  %s", len(all_posts), dict(plat_counts))

    # Build CPHot event stub (bridge_pairs empty — fill with annotate_pairs.py)
    event = {
        "event_id":       event_id,
        "keyword":        args.keyword,
        "posts":          all_posts,
        "bridge_pairs":   [],       # fill with: python scripts/annotate_pairs.py
        "hourly_volumes": [],       # optional — fill manually or leave empty
        "scored_pairs":   [],       # filled automatically after annotation
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(event, f, ensure_ascii=False, indent=2)

    logger.info("Saved → %s", out_path)
    logger.info("")
    logger.info("Next step — label bridge pairs:")
    logger.info("  python scripts/annotate_pairs.py --event %s", out_path)


if __name__ == "__main__":
    main()
