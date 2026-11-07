"""Offline check of the agent pipeline with the mock model and fake search backends (no key, no data)."""
import tempfile

from agentverify.agents.orchestrator import run_pipeline
from agentverify.contracts import PostInput, SourceLabel
from agentverify.llm import MockClient
from agentverify.tools.search import EvidenceTool

tmp = tempfile.mkdtemp()
tool = EvidenceTool(backends={"wikipedia": lambda q, k: [{"title": "T", "url": "https://en.wikipedia.org/wiki/T", "snippet": "s"}],
                              "web": lambda q, k: []}, cache_dir=tmp + "/search")
posts = [PostInput(post_id=f"p{i}", text=f"claim number {i}", image_path="x.png",
                   gold_is_fake=bool(i % 2), gold_source=SourceLabel.REAL if i % 2 == 0 else SourceLabel.TEXT_FALSE)
         for i in range(4)]
for flags in ({}, {"use_text": False}, {"use_image": False}, {"use_evidence": False}):
    for p in posts:
        r = run_pipeline(p, MockClient(), tool, tmp + "/cache", trace_path=tmp + "/trace.jsonl", **flags)
        assert r.final.source_label in SourceLabel
print("agent smoke test OK; trace written:", tmp + "/trace.jsonl")
