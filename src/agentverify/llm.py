"""Model provider layer.

OpenAICompatClient talks to any OpenAI-compatible chat endpoint (OpenAI, Ollama, vLLM...).
MockClient is deterministic and offline, used by tests and the smoke test only.
Both return (parsed_json_or_None, CostLog).
"""
from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import re
import time
from pathlib import Path
from typing import Callable, Optional, Protocol

from .config import LLMConfig
from .contracts import CostLog


class VLMClient(Protocol):
    def chat_json(self, system: str, user_text: str,
                  image_path: Optional[str] = None) -> tuple[Optional[dict], CostLog]: ...


_JSON_RE = re.compile(r"\{.*\}", re.S)


def extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of a model reply (handles ```json fences)."""
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None


def image_to_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


class OpenAICompatClient:
    def __init__(self, cfg: Optional[LLMConfig] = None):
        from openai import OpenAI  # lazy import so tests do not need a key
        self.cfg = cfg or LLMConfig.from_env()
        self.client = OpenAI(api_key=self.cfg.api_key, base_url=self.cfg.base_url)

    def chat_json(self, system, user_text, image_path=None):
        content = [{"type": "text", "text": user_text}]
        if image_path:
            content.append({"type": "image_url",
                            "image_url": {"url": image_to_data_url(image_path)}})
        cost = CostLog()
        parsed = None
        for attempt in range(self.cfg.retries + 1):
            t0 = time.time()
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg.model,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": content}],
                )
            except Exception as e:  # network / rate limit
                print(f"[llm error] {type(e).__name__}: {str(e)[:300]}", flush=True)
                time.sleep(2 ** attempt)
                if attempt == self.cfg.retries:
                    cost.llm_calls += 1
                    return None, cost
                continue
            cost.latency_s += time.time() - t0
            cost.llm_calls += 1
            if resp.usage:
                cost.prompt_tokens += resp.usage.prompt_tokens or 0
                cost.completion_tokens += resp.usage.completion_tokens or 0
            parsed = extract_json(resp.choices[0].message.content or "")
            if parsed is not None:
                break
        return parsed, cost


class MockClient:
    """Offline stand-in. NOT a model: outputs are rule/hash based and carry no research meaning."""

    def __init__(self, fn: Optional[Callable[[str, str], dict]] = None):
        self.fn = fn or self._default

    @staticmethod
    def _default(system: str, user_text: str) -> dict:
        h = int(hashlib.md5(user_text.encode()).hexdigest(), 16)
        fake = ("claim" in user_text.lower() and h % 3 != 0) or h % 5 == 0
        src = ["text_false", "image_false", "mismatch"][h % 3] if fake else "real"
        return {"is_fake": fake, "source": src,
                "confidence": round(0.55 + (h % 40) / 100, 2), "reason": "mock"}

    def chat_json(self, system, user_text, image_path=None):
        out = self.fn(system, user_text)
        cost = CostLog(prompt_tokens=len(user_text) // 4, completion_tokens=20, llm_calls=1)
        return out, cost
