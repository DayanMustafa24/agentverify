"""Evidence agent: writes a query, retrieves Wikipedia + web results, judges the claim against them."""
from ..contracts import AgentVerdict, CostLog, PostInput, SourceLabel
from ..tools.search import rerank_by_trust
from .common import cached_call, failed_verdict

Q_SYS = "You help verify news claims. Reply with one JSON object and nothing else."
Q_USER = """Caption: {text}

Write one short web search query (at most 10 words) that would find evidence to verify the main factual claim.
Return JSON: {{"query": "..."}}"""

J_SYS = "You are an evidence analyst for a fake-news detection team. Reply with one JSON object and nothing else."
J_USER = """Caption: {text}

Retrieved evidence (trusted sources are marked):
{evidence}

Does the evidence support or contradict the caption's main claim?
Return JSON with keys:
"stance": "supports", "contradicts" or "insufficient"
"confidence": number 0-1, probability that your stance is correct
"reason": one or two sentences, mention which evidence item you used"""


def run(post: PostInput, client, cache_dir, tool):
    cost = CostLog()
    q, c = cached_call(cache_dir, "evidence_q", post.post_id, client, Q_SYS, Q_USER.format(text=post.text))
    cost.add(c)
    query = str((q or {}).get("query") or post.text[:100])[:200]
    items = []
    for backend in ("wikipedia", "web"):
        got, c = tool.search(query, backend)
        items += got
        cost.add(c)
    seen, uniq = set(), []
    for it in rerank_by_trust(items):
        if it.url and it.url not in seen:
            seen.add(it.url)
            uniq.append(it)
    items = uniq[:6]
    if not items:
        return AgentVerdict(agent="evidence", confidence=0.5,
                            rationale=f"no evidence retrieved for query: {query}"), cost
    ev_text = "\n".join(f"[{i+1}]{' (trusted)' if e.trusted else ''} {e.title}: {e.snippet[:300]}"
                        for i, e in enumerate(items))
    parsed, c = cached_call(cache_dir, "evidence_j", post.post_id, client, J_SYS,
                            J_USER.format(text=post.text, evidence=ev_text))
    cost.add(c)
    if not parsed:
        v = failed_verdict("evidence", "unparseable judge reply")
        v.evidence = items
        return v, cost
    stance = str(parsed.get("stance", "insufficient")).strip().lower()
    try:
        conf = float(parsed.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    src = SourceLabel.TEXT_FALSE if stance == "contradicts" else SourceLabel.REAL
    return AgentVerdict(agent="evidence", flagged_source=src, confidence=conf, evidence=items,
                        rationale=f"[{stance}] {parsed.get('reason', '')}"[:300]), cost
