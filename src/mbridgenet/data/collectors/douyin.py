"""Douyin collector — stub requiring the official Research API.

Douyin (TikTok China) does not expose a public search API. Options:

  Option A — Official Research API (recommended for academic use)
    Apply at: https://www.douyin.com/falcon/developer/
    Once approved you receive a client_key + client_secret.
    Set env vars:
        DOUYIN_CLIENT_KEY=...
        DOUYIN_CLIENT_SECRET=...
    Then implement OAuth2 + POST /search/general/v1/

  Option B — Third-party aggregators
    Services like RapidAPI host unofficial Douyin/TikTok wrappers.
    Quality and ToS compliance vary — use at your own risk.

  Option C — Manual export
    Douyin Studio (creator dashboard) lets you export your own post data.
    For general keyword search this is not applicable.

This stub raises NotImplementedError with the above guidance so the rest
of the pipeline continues without Douyin data if it is not configured.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, List

from mbridgenet.data.collectors.base import BaseCollector

logger = logging.getLogger(__name__)


class DouyinCollector(BaseCollector):
    """Douyin data collector — requires official API credentials.

    See module docstring for how to obtain access.
    """

    PLATFORM = "douyin"

    def __init__(
        self,
        client_key: str | None = None,
        client_secret: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._client_key    = client_key    or os.environ.get("DOUYIN_CLIENT_KEY", "")
        self._client_secret = client_secret or os.environ.get("DOUYIN_CLIENT_SECRET", "")

        if not self._client_key:
            logger.warning(
                "Douyin: DOUYIN_CLIENT_KEY not set. "
                "Apply for the Research API at https://www.douyin.com/falcon/developer/ "
                "then set DOUYIN_CLIENT_KEY and DOUYIN_CLIENT_SECRET."
            )

    def search(
        self,
        keyword: str,
        start_time: datetime,
        end_time: datetime,
        max_posts: int = 100,
    ) -> List[Dict]:
        if not self._client_key or not self._client_secret:
            logger.warning(
                "Douyin: skipping '%s' — no API credentials. "
                "See src/mbridgenet/data/collectors/douyin.py for setup instructions.",
                keyword,
            )
            return []

        # ── Implement OAuth2 + /search/general/v1/ here once you have credentials ──
        raise NotImplementedError(
            "Douyin search requires official Research API credentials. "
            "Set DOUYIN_CLIENT_KEY + DOUYIN_CLIENT_SECRET and implement "
            "the OAuth2 flow in this method."
        )
