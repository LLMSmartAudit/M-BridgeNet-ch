"""Convert MediaCrawler JSONL output files into a CPHot event JSON for M-BridgeNet.

Reads one JSONL file per platform, maps fields to the CPHot post schema,
merges into a single event JSON, and derives hourly_volumes from timestamps.
bridge_pairs and scored_pairs are left empty — fill them with annotate_pairs.py.

Usage:
    python scripts/convert_mediacrawler.py \\
        --bilibili ../MediaCrawler/data/bili/jsonl/search_contents_2026-05-02.jsonl \\
        --weibo    ../MediaCrawler/data/weibo/jsonl/search_contents_2026-05-02.jsonl \\
        --zhihu    ../MediaCrawler/data/zhihu/jsonl/search_contents_2026-05-02.jsonl \\
        --douyin   ../MediaCrawler/data/douyin/jsonl/search_contents_2026-05-02.jsonl \\
        --event-id taiwan_tensions_001 \\
        --output   data/cphot/raw

After conversion, label bridge pairs:
    python scripts/annotate_pairs.py --event data/cphot/raw/taiwan_tensions_001.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Field mapping per platform ────────────────────────────────────────────────

def _clean_html(text: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    text = re.sub(r"<[^>]+>", "", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _to_utc(unix_ts: int) -> datetime:
    return datetime.fromtimestamp(int(unix_ts), tz=timezone.utc)


def _parse_bilibili(raw: dict, event_id: str) -> Optional[Dict]:
    vid = raw.get("video_id")
    if not vid:
        return None
    title = _clean_html(raw.get("title", ""))
    desc  = _clean_html(raw.get("desc", ""))
    text  = f"{title}。{desc}".strip("。").strip() if desc else title
    ts    = raw.get("create_time")
    if not text or not ts:
        return None
    return {
        "post_id":    f"bili_{vid}",
        "text":       text,
        "account_id": str(raw.get("user_id", "")),
        "platform":   "bilibili",
        "event_id":   event_id,
        "timestamp":  _to_utc(ts).isoformat(),
    }


def _parse_weibo(raw: dict, event_id: str) -> Optional[Dict]:
    nid  = raw.get("note_id")
    text = _clean_html(raw.get("content", ""))
    ts   = raw.get("create_time")
    if not nid or not text or not ts:
        return None
    return {
        "post_id":    f"wb_{nid}",
        "text":       text,
        "account_id": str(raw.get("user_id", "")),
        "platform":   "weibo",
        "event_id":   event_id,
        "timestamp":  _to_utc(ts).isoformat(),
    }


def _parse_zhihu(raw: dict, event_id: str) -> Optional[Dict]:
    cid  = raw.get("content_id")
    text = _clean_html(raw.get("content_text", ""))
    ts   = raw.get("created_time")
    if not cid or not text or not ts:
        return None
    return {
        "post_id":    f"zhi_{cid}",
        "text":       text,
        "account_id": str(raw.get("user_id", "")),
        "platform":   "zhihu",
        "event_id":   event_id,
        "timestamp":  _to_utc(ts).isoformat(),
    }


def _parse_douyin(raw: dict, event_id: str) -> Optional[Dict]:
    aid  = raw.get("aweme_id")
    text = _clean_html(raw.get("desc", ""))
    ts   = raw.get("create_time")
    if not aid or not text or not ts:
        return None
    return {
        "post_id":    f"dy_{aid}",
        "text":       text,
        "account_id": str(raw.get("user_id", "")),
        "platform":   "douyin",
        "event_id":   event_id,
        "timestamp":  _to_utc(ts).isoformat(),
    }


PARSERS = {
    "bilibili": _parse_bilibili,
    "weibo":    _parse_weibo,
    "zhihu":    _parse_zhihu,
    "douyin":   _parse_douyin,
}


# ── JSONL loader ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path, platform: str, event_id: str,
               start: Optional[datetime], end: Optional[datetime]) -> List[Dict]:
    parser = PARSERS[platform]
    posts  = []
    skipped = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw  = json.loads(line)
                post = parser(raw, event_id)
            except (json.JSONDecodeError, Exception) as exc:
                logger.warning("%s: parse error — %s", platform, exc)
                skipped += 1
                continue

            if post is None:
                skipped += 1
                continue

            # Date filter
            if start or end:
                ts = datetime.fromisoformat(post["timestamp"])
                if start and ts < start:
                    skipped += 1
                    continue
                if end and ts > end:
                    skipped += 1
                    continue

            # Source-keyword filter (uses raw field before parsing)
            if hasattr(load_jsonl, "_kw_filter") and load_jsonl._kw_filter:
                raw_kw = raw.get("source_keyword", "")
                if not any(kw in raw_kw for kw in load_jsonl._kw_filter):
                    skipped += 1
                    continue

            posts.append(post)

    logger.info("%-10s  %3d posts loaded  (%d skipped)", platform, len(posts), skipped)
    return posts


# ── Hourly volumes ────────────────────────────────────────────────────────────

def derive_hourly_volumes(posts: List[Dict]) -> List[float]:
    """Bin posts into 1-hour buckets relative to the earliest timestamp."""
    if not posts:
        return []

    timestamps = [datetime.fromisoformat(p["timestamp"]) for p in posts]
    t_min = min(timestamps)
    t_max = max(timestamps)
    total_hours = max(1, int((t_max - t_min).total_seconds() / 3600) + 1)

    buckets: Dict[int, int] = defaultdict(int)
    for ts in timestamps:
        hour = int((ts - t_min).total_seconds() / 3600)
        buckets[hour] += 1

    return [float(buckets.get(h, 0)) for h in range(total_hours)]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_date(s: str) -> datetime:
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"Cannot parse date '{s}'. Use YYYY-MM-DD.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MediaCrawler JSONL files to CPHot event JSON"
    )
    parser.add_argument("--bilibili", default=None, nargs="+", help="Bilibili JSONL file(s)")
    parser.add_argument("--weibo",    default=None, nargs="+", help="Weibo JSONL file(s)")
    parser.add_argument("--zhihu",    default=None, nargs="+", help="Zhihu JSONL file(s)")
    parser.add_argument("--douyin",   default=None, nargs="+", help="Douyin JSONL file(s)")
    parser.add_argument("--event-id", default="event_001",
                        help="Event ID written into the JSON (default: event_001)")
    parser.add_argument("--output",   default="data/cphot/raw",
                        help="Output directory (default: data/cphot/raw)")
    parser.add_argument("--start",    default=None, type=parse_date,
                        help="Only keep posts on/after this date (YYYY-MM-DD)")
    parser.add_argument("--end",      default=None, type=parse_date,
                        help="Only keep posts on/before this date (YYYY-MM-DD)")
    parser.add_argument("--source-keywords", default=None,
                        help="Comma-separated list of source_keyword values to keep "
                             "(e.g. '996执法,强制下班,腾讯下班'). Useful when multiple "
                             "events share the same daily JSONL file.")
    args = parser.parse_args()

    # Attach keyword filter to load_jsonl so the inner loop can access it
    if args.source_keywords:
        load_jsonl._kw_filter = [k.strip() for k in args.source_keywords.split(",")]
        logger.info("source_keyword filter: %s", load_jsonl._kw_filter)
    else:
        load_jsonl._kw_filter = None

    # Each platform flag now accepts one or more paths (nargs="+")
    platform_files: dict[str, list[str]] = {
        "bilibili": args.bilibili or [],
        "weibo":    args.weibo    or [],
        "zhihu":    args.zhihu    or [],
        "douyin":   args.douyin   or [],
    }

    if not any(platform_files.values()):
        parser.error("Provide at least one platform JSONL file (--bilibili / --weibo / --zhihu / --douyin).")

    event_id = args.event_id
    all_posts: List[Dict] = []

    for platform, paths in platform_files.items():
        for path in paths:
            p = Path(path)
            if not p.exists():
                logger.warning("File not found, skipping: %s", p)
                continue
            posts = load_jsonl(p, platform, event_id, args.start, args.end)
            all_posts.extend(posts)

    if not all_posts:
        logger.error("No posts loaded. Check file paths and date filters.")
        return

    # Sort by timestamp
    all_posts.sort(key=lambda p: p["timestamp"])

    # Deduplicate by post_id
    seen: set = set()
    unique_posts = []
    for p in all_posts:
        if p["post_id"] not in seen:
            seen.add(p["post_id"])
            unique_posts.append(p)
    all_posts = unique_posts

    hourly_volumes = derive_hourly_volumes(all_posts)

    from collections import Counter
    platform_counts = Counter(p["platform"] for p in all_posts)
    logger.info("Total: %d posts  %s", len(all_posts), dict(platform_counts))
    logger.info("Time span: %s → %s",
                all_posts[0]["timestamp"][:10],
                all_posts[-1]["timestamp"][:10])

    event = {
        "event_id":       event_id,
        "posts":          all_posts,
        "bridge_pairs":   [],   # fill with: python scripts/annotate_pairs.py
        "hourly_volumes": hourly_volumes,
        "scored_pairs":   [],
    }

    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{event_id}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(event, f, ensure_ascii=False, indent=2)

    logger.info("Saved → %s", out_path)
    logger.info("")
    logger.info("Next step — label bridge pairs:")
    logger.info("  python scripts/annotate_pairs.py --event %s", out_path)


if __name__ == "__main__":
    main()
