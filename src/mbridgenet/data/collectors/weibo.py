"""Weibo collector — uses the mobile search API with cookie auth.

How to get your cookie:
  1. Open https://m.weibo.cn in Chrome and log in.
  2. Open DevTools → Network → refresh the page.
  3. Click any request to m.weibo.cn → Headers → Request Headers → Cookie.
  4. Copy the full cookie string and set it via one of:
       export WEIBO_COOKIE="your cookie string"
       collector.set_cookies("your cookie string")
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, List
from urllib.parse import quote

from mbridgenet.data.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://m.weibo.cn/api/container/getIndex"


def _parse_weibo_time(raw: str) -> datetime:
    """Parse Weibo's 'Mon Jan 01 00:00:00 +0800 2024' timestamp."""
    try:
        return parsedate_to_datetime(raw).astimezone(timezone.utc)
    except Exception:
        return datetime.now(tz=timezone.utc)


class WeiboCollector(BaseCollector):
    """Collect Weibo posts matching a keyword.

    Requires cookie auth — set WEIBO_COOKIE env var or call set_cookies().
    """

    PLATFORM = "weibo"

    def __init__(self, cookie: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        cookie = cookie or os.environ.get("WEIBO_COOKIE", "")
        if cookie:
            self.set_cookies(cookie)
            logger.info("Weibo: cookies loaded (%d chars)", len(cookie))
        else:
            logger.warning(
                "Weibo: no cookie set — searches will likely return empty results. "
                "Set WEIBO_COOKIE env var or pass cookie= to WeiboCollector()."
            )
        self._session.headers.update({
            "Referer": "https://m.weibo.cn",
            "MWeibo-Pwa": "1",
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
        encoded_kw = quote(keyword)

        while len(posts) < max_posts:
            params = {
                "containerid": f"100103type=1&q={encoded_kw}&t=0",
                "page_type":   "searchall",
                "page":        page,
            }
            resp = self._get(_SEARCH_URL, params=params)
            if resp is None:
                logger.warning("Weibo: no response on page %d, stopping.", page)
                break

            try:
                body  = resp.json()
                cards = body.get("data", {}).get("cards", [])
            except Exception as exc:
                logger.warning("Weibo: JSON parse error: %s", exc)
                break

            if not cards:
                break

            found_any = False
            for card in cards:
                if card.get("card_type") != 9:
                    continue    # only microblog cards
                mblog = card.get("mblog", {})
                if not mblog:
                    continue

                ts = _parse_weibo_time(mblog.get("created_at", ""))
                if ts < start_time or ts > end_time:
                    continue

                # Strip HTML from text
                import re
                text = re.sub(r"<[^>]+>", "", mblog.get("text", "")).strip()

                user = mblog.get("user") or {}
                posts.append(self.make_post(
                    post_id    = f"wb_{mblog.get('id', '')}",
                    text       = text,
                    account_id = str(user.get("id", "")),
                    platform   = "weibo",
                    timestamp  = ts,
                ))
                found_any = True

                if len(posts) >= max_posts:
                    break

            if not found_any:
                break   # exhausted results

            page += 1

        logger.info("Weibo: collected %d posts for '%s'", len(posts), keyword)
        return posts
