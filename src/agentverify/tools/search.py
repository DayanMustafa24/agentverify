"""Evidence tool interface (Phase 1: working interface + cache; used by the Evidence Agent in Phase 2).

Backends: Wikipedia REST search and DuckDuckGo (ddgs). Every call is cached on disk so
repeated experiments are free and reproducible, and search_calls are counted for the cost log.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import requests

from ..contracts import CostLog, EvidenceItem

TRUSTED_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "theguardian.com", "nytimes.com",
    "washingtonpost.com", "npr.org", "aljazeera.com", "dawn.com", "snopes.com",
    "factcheck.org", "politifact.com", "fullfact.org", "afp.com", "wikipedia.org",
}


def domain_of(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def is_trusted(url: str) -> bool:
    d = domain_of(url)
    return any(d == t or d.endswith("." + t) for t in TRUSTED_DOMAINS)


def rerank_by_trust(items: list[EvidenceItem]) -> list[EvidenceItem]:
    return sorted(items, key=lambda e: (not e.trusted,))  # stable: trusted first, original order kept

def wikipedia_backend(query: str, k: int) -> list[dict]:
    import time
    headers = {"User-Agent": "agentverify-student-project/0.1 (YOUR_EMAIL@example.com)"}
    params = {"action": "query", "list": "search", "srsearch": query,
              "srlimit": k, "format": "json"}
    for attempt in range(3):
        r = requests.get("https://en.wikipedia.org/w/api.php", timeout=15,
                         headers=headers, params=params)
        if r.status_code == 429:
            time.sleep(2 * (attempt + 1))
            continue
        r.raise_for_status()
        break
    out = []
    for h in r.json()["query"]["search"]:
        title = h["title"]
        out.append({"title": title, "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
                    "snippet": re.sub(r"<[^>]+>", "", h["snippet"])})
    return out


def ddg_backend(query: str, k: int) -> list[dict]:
    from ddgs import DDGS
    res = DDGS().text(query, max_results=k) or []
    return [{"title": x.get("title", ""), "url": x.get("href", ""), "snippet": x.get("body", "")} for x in res]


class EvidenceTool:
    def __init__(self, backends: Optional[dict[str, Callable]] = None,
                 cache_dir: str | Path = "data/cache/search", k: int = 5):
        self.backends = backends or {"wikipedia": wikipedia_backend, "web": ddg_backend}
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.k = k

    def _key(self, backend: str, query: str) -> Path:
        h = hashlib.md5(f"{backend}|{query}|{self.k}".encode()).hexdigest()
        return self.cache_dir / f"{h}.json"

    def search(self, query: str, backend: str = "wikipedia") -> tuple[list[EvidenceItem], CostLog]:
        cost = CostLog()
        p = self._key(backend, query)
        if p.exists():
            raw = json.loads(p.read_text())
        else:
            try:
                raw = self.backends[backend](query, self.k)
                p.write_text(json.dumps(raw))
                cost.search_calls += 1
            except Exception as e:
                print(f"[search error] {backend}: {type(e).__name__}: {str(e)[:150]}", flush=True)
                raw = []
        items = [EvidenceItem(query=query, title=x["title"], url=x["url"], snippet=x["snippet"],
                              domain=domain_of(x["url"]), trusted=is_trusted(x["url"])) for x in raw]
        return rerank_by_trust(items), cost
