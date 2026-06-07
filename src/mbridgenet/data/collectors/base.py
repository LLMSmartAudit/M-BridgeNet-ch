"""Base collector with rate limiting, retries, and a standard post dict schema."""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    """Abstract base for all platform collectors.

    Subclasses implement ``search()`` and set ``PLATFORM``.
    All output dicts follow the CPHot post schema so they can be fed
    directly into ``load_event()`` / ``CPHotDataset``.
    """

    PLATFORM: str = ""

    def __init__(
        self,
        rate_limit_seconds: float = 2.0,
        max_retries: int = 3,
        timeout: int = 15,
    ) -> None:
        self.rate_limit = rate_limit_seconds
        self.max_retries = max_retries
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "zh-CN,zh;q=0.9",
        })
        self._last_request_at: float = 0.0

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _get(self, url: str, **kwargs) -> Optional[requests.Response]:
        """Rate-limited GET with exponential-backoff retries."""
        wait = self.rate_limit - (time.time() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)

        for attempt in range(self.max_retries):
            try:
                resp = self._session.get(url, timeout=self.timeout, **kwargs)
                self._last_request_at = time.time()
                if resp.status_code == 200:
                    return resp
                logger.warning(
                    "%s  HTTP %d  %s  (attempt %d/%d)",
                    self.PLATFORM, resp.status_code, url, attempt + 1, self.max_retries,
                )
            except requests.RequestException as exc:
                logger.warning(
                    "%s  request error: %s  (attempt %d/%d)",
                    self.PLATFORM, exc, attempt + 1, self.max_retries,
                )
            time.sleep(2 ** attempt)

        return None

    def set_cookies(self, cookie_str: str) -> None:
        """Load a cookie string copied from browser DevTools → Network tab."""
        for part in cookie_str.split(";"):
            part = part.strip()
            if "=" in part:
                k, _, v = part.partition("=")
                self._session.cookies.set(k.strip(), v.strip())

    # ── Output schema ─────────────────────────────────────────────────────────

    @staticmethod
    def make_post(
        post_id: str,
        text: str,
        account_id: str,
        platform: str,
        timestamp: datetime,
        event_id: str = "",
    ) -> Dict:
        """Return a post dict matching the CPHot / load_event schema."""
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return {
            "post_id":    post_id,
            "text":       text.strip(),
            "account_id": str(account_id),
            "platform":   platform,
            "event_id":   event_id,
            "timestamp":  timestamp.isoformat(),
        }

    # ── Interface ─────────────────────────────────────────────────────────────

    @abstractmethod
    def search(
        self,
        keyword: str,
        start_time: datetime,
        end_time: datetime,
        max_posts: int = 100,
    ) -> List[Dict]:
        """Collect posts matching ``keyword`` within [start_time, end_time].

        Returns a list of post dicts (CPHot schema).
        """
