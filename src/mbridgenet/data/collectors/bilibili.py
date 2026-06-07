"""Bilibili collector — uses the public search API (no auth required).

Bilibili's search endpoint is publicly accessible. We collect video
titles + descriptions as the post text.

No cookie needed for basic keyword search.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List

from mbridgenet.data.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://api.bilibili.com/x/web-interface/search/type"


class BilibiliCollector(BaseCollector):
    """Collect Bilibili videos matching a keyword."""

    PLATFORM = "bilibili"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._session.headers.update({
            "Referer": "https://www.bilibili.com",
            "Origin":  "https://www.bilibili.com",
        })

    def search(
        self,
        keyword: str,
        start_time: datetime,
        end_time: datetime,
        max_posts: int = 100,
    ) -> List[Dict]:
        posts: List[Dict] = []
        page = 1

        while len(posts) < max_posts:
            params = {
                "search_type": "video",
                "keyword":     keyword,
                "page":        page,
                "page_size":   20,
                "order":       "pubdate",   # newest first
            }
            resp = self._get(_SEARCH_URL, params=params)
            if resp is None:
                logger.warning("Bilibili: no response on page %d, stopping.", page)
                break

            try:
                body = resp.json()
                results = body.get("data", {}).get("result") or []
            except Exception as exc:
                logger.warning("Bilibili: JSON parse error: %s", exc)
                break

            if not results:
                break   # no more pages

            for item in results:
                pub_ts = item.get("pubdate", 0)
                ts = datetime.fromtimestamp(pub_ts, tz=timezone.utc)

                if ts < start_time or ts > end_time:
                    continue    # outside requested window

                # Strip HTML tags from title/description
                import re
                clean = re.compile(r"<[^>]+>")
                title = clean.sub("", item.get("title", ""))
                desc  = clean.sub("", item.get("description", ""))
                text  = f"{title}。{desc}".strip("。").strip()

                posts.append(self.make_post(
                    post_id    = f"bili_{item.get('bvid') or item.get('aid', '')}",
                    text       = text,
                    account_id = str(item.get("mid", "")),
                    platform   = "bilibili",
                    timestamp  = ts,
                ))

                if len(posts) >= max_posts:
                    break

            page += 1
            if page > 50:   # Bilibili caps search at ~50 pages
                break

        logger.info("Bilibili: collected %d posts for '%s'", len(posts), keyword)
        return posts
