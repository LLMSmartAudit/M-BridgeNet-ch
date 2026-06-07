from __future__ import annotations
import json
import logging
import time
from typing import Any, Optional

import openai

from mbridgenet.schemas import CandidatePair, DebateRecord
from mbridgenet.stage3.prompts import (
    proposer_prompt, challenger_prompt, rebuttal_prompt, judge_prompt,
)

logger = logging.getLogger(__name__)

_RETRY_DELAYS = [5, 15, 30, 60]  # seconds between retries on rate-limit


def _parse_judge_output(raw: str) -> Optional[DebateRecord]:
    """Parse Judge's JSON output into a DebateRecord. Returns None on failure."""
    try:
        # Strip markdown code fences if present
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        data = json.loads(clean)
        return DebateRecord(
            is_bridge=bool(data["is_bridge"]),
            confidence=float(data["confidence"]),
            narrative_relation=data["narrative_relation"],
            deciding_factor=data["deciding_factor"],
            debate_outcome=data["debate_outcome"],
        )
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        logger.warning("Failed to parse Judge output: %s — raw: %.200s", e, raw)
        return None


class MABDDebater:
    """Orchestrates the four-round Multi-Agent Bridge Debate (MABD).

    Uses an OpenAI-compatible client. Retries on rate-limit errors with
    exponential backoff, and paces calls with a inter-call sleep to avoid
    hitting TPM limits.
    """

    def __init__(
        self,
        client: Any,
        model: str = "gpt-5.4-nano",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        call_interval: float = 1.0,  # seconds between API calls
    ):
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._call_interval = call_interval

    def _is_anthropic(self) -> bool:
        """True if the client is an Anthropic client (not OpenAI)."""
        try:
            import anthropic as _ant
            return isinstance(self._client, _ant.Anthropic)
        except ImportError:
            return False

    def _call(self, messages: list[dict]) -> str:
        if self._is_anthropic():
            return self._call_anthropic(messages)
        return self._call_openai(messages)

    def _call_anthropic(self, messages: list[dict]) -> str:
        """Call Anthropic Messages API. Converts OpenAI-style messages dict."""
        import anthropic as _ant
        for attempt, default_delay in enumerate([0] + _RETRY_DELAYS):
            if default_delay:
                logger.info("Rate-limited (Anthropic) — retrying in %ds", default_delay)
                time.sleep(default_delay)
            try:
                # Separate system from user/assistant messages
                system_parts = [m["content"] for m in messages if m["role"] == "system"]
                conv_messages = [m for m in messages if m["role"] != "system"]
                system_text = "\n\n".join(system_parts) if system_parts else None
                kwargs: dict = dict(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    messages=conv_messages,
                )
                if system_text:
                    kwargs["system"] = system_text
                resp = self._client.messages.create(**kwargs)
                time.sleep(self._call_interval)
                return resp.content[0].text
            except _ant.RateLimitError as e:
                if attempt == len(_RETRY_DELAYS):
                    raise
        raise RuntimeError("Unreachable")

    def _call_openai(self, messages: list[dict]) -> str:
        for attempt, default_delay in enumerate([0] + _RETRY_DELAYS):
            if default_delay:
                logger.info("Rate-limited — retrying in %ds (attempt %d)", default_delay, attempt)
                time.sleep(default_delay)
            try:
                # Some models (e.g. gpt-5.5, o-series) do not support temperature;
                # omit it when it equals the default (1) to avoid 400 errors.
                create_kwargs: dict = dict(
                    model=self._model,
                    messages=messages,
                    max_completion_tokens=self._max_tokens,
                )
                if self._temperature != 1:
                    create_kwargs["temperature"] = self._temperature
                resp = self._client.chat.completions.create(**create_kwargs)
                time.sleep(self._call_interval)
                return resp.choices[0].message.content
            except openai.RateLimitError as e:
                if attempt == len(_RETRY_DELAYS):
                    raise
                # Honour retry-after header if the API provided one
                retry_after = getattr(getattr(e, "response", None), "headers", {}).get(
                    "retry-after"
                )
                if retry_after:
                    wait = float(retry_after)
                    logger.info("retry-after header: %.0fs", wait)
                    time.sleep(wait)
        raise RuntimeError("Unreachable")

    def debate(self, pair: CandidatePair) -> Optional[DebateRecord]:
        """Run four-round MABD and return a DebateRecord, or None if parsing fails."""
        ctx = {
            "post_a_id":   pair.post_a.post_id,
            "platform_a":  pair.post_a.platform,
            "timestamp_a": pair.post_a.timestamp.isoformat(),
            "text_a":      pair.post_a.text[:500],
            "post_b_id":   pair.post_b.post_id,
            "platform_b":  pair.post_b.platform,
            "timestamp_b": pair.post_b.timestamp.isoformat(),
            "text_b":      pair.post_b.text[:500],
            "s1": pair.s1, "s2": pair.s2,
            "s3": pair.s3, "s4": pair.s4,
            "s5": getattr(pair, "s5", 0.0),
            "score_S": pair.score_S,
            "phase": pair.phase,
        }

        # Round 1: Proposer
        proposer_out = self._call(proposer_prompt(ctx))

        # Round 2: Challenger
        challenger_out = self._call(challenger_prompt(ctx, proposer_out))

        # Round 3: Rebuttal
        rebuttal_out = self._call(rebuttal_prompt(ctx, proposer_out, challenger_out))

        # Round 4: Judge → structured verdict
        judge_out = self._call(
            judge_prompt(ctx, proposer_out, challenger_out, rebuttal_out)
        )

        return _parse_judge_output(judge_out)
