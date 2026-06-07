"""Zhihu collector — uses the v4 search API with cookie auth.

How to get your cookie:
  1. Open https://www.zhihu.com in Chrome and log in.
  2. Open DevTools → Network → search for anything on Zhihu.
  3. Find a request to www.zhihu.com/api/v4/search_v3 → Headers → Cookie.
  4. Copy the full cookie string and set it via one of:
       export ZHIHU_COOKIE="your cookie string"
       collector.set_cookies("your cookie string")

Collects both answers and articles that match the keyword.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Dict, List

from mbridgenet.data.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://www.zhihu.com/api/v4/search_v3"


class ZhihuCollector(BaseCollector):
    """Collect Zhihu answers and articles matching a keyword.

    Requires cookie auth — set ZHIHU_COOKIE env var or call set_cookies().
    """

    PLATFORM = "zhihu"

    def __init__(self, cookie: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        cookie = cookie or os.environ.get("ZHIHU_COOKIE", "")
        if cookie:
            self.set_cookies(cookie)
            logger.info("Zhihu: cookies loaded (%d chars)", len(cookie))
        else:
            logger.warning(
                "Zhihu: no cookie set — requests will likely be rejected (401/403). "
                "Set ZHIHU_COOKIE env var or pass cookie= to ZhihuCollector()."
            )
        self._session.headers.update({
            "Referer":    "https://www.zhihu.com",
            "x-api-version": "3.0.91",
            "x-app-version": "1.0.0",
        })

    def search(
        self,
        keyword: str,
        start_time: datetime,
        end_time: datetime,
        max_posts: int = 100,
    ) -> List[Dict]:
        posts: List[Dict] = []
        offset = 0
        limit  = 20

        while len(posts) < max_posts:
            params = {
                "t":            "content",
                "q":            keyword,
                "correction":   1,
                "offset":       offset,
                "limit":        limit,
                "lc_idx":       0,
                "show_all_topics": 0,
            }
            resp = self._get(_SEARCH_URL, params=params)
            if resp is None:
                logger.warning("Zhihu: no response at offset %d, stopping.", offset)
                break

            try:
                body  = resp.json()
                items = body.get("data", [])
            except Exception as exc:
                logger.warning("Zhihu: JSON parse error: %s", exc)
                break

            if not items:
                break

            for item in items:
                obj  = item.get("object", {})
                kind = item.get("type", "")

                # Extract text and timestamp depending on content type
                if kind == "answer":
                    created = obj.get("created_time", 0)
                    text    = obj.get("excerpt") or obj.get("content", "")
                    q_title = obj.get("question", {}).get("title", "")
                    text    = f"{q_title}——{text}" if q_title else text
                    uid     = str(obj.get("author", {}).get("id", ""))
                    oid     = f"zhi_ans_{obj.get('id', '')}"

                elif kind == "article":
                    created = obj.get("created", 0)
                    text    = obj.get("excerpt") or obj.get("title", "")
                    uid     = str(obj.get("author", {}).get("id", ""))
                    oid     = f"zhi_art_{obj.get('id', '')}"

                else:
                    continue    # skip questions, topics, etc.

                ts = datetime.fromtimestamp(created, tz=timezone.utc)
                if ts < start_time or ts > end_time:
                    continue

                # Strip HTML
                import re
                text = re.sub(r"<[^>]+>", "", text).strip()

                posts.append(self.make_post(
                    post_id    = oid,
                    text       = text,
                    account_id = uid,
                    platform   = "zhihu",
                    timestamp  = ts,
                ))

                if len(posts) >= max_posts:
                    break

            offset += limit
            # Zhihu paginates up to ~500 items
            if offset >= 500:
                break

        logger.info("Zhihu: collected %d posts for '%s'", len(posts), keyword)
        return posts
