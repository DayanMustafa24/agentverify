"""Structured contracts shared by every component.

Phase 1 uses PostInput, FinalVerdict, CostLog and RunRecord (baselines).
AgentVerdict and EvidenceItem are defined now so Phase 2 agents plug into the same state.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SourceLabel(str, Enum):
    """Four-way label used by MMFakeBench (real + three forgery sources)."""
    REAL = "real"
    TEXT_FALSE = "text_false"      # textual veracity distortion (TVD)
    IMAGE_FALSE = "image_false"    # visual veracity distortion (VVD)
    MISMATCH = "mismatch"          # cross-modal consistency distortion (CCD)


SOURCE_LABELS = [s.value for s in SourceLabel]


class PostInput(BaseModel):
    post_id: str
    text: str
    image_path: str
    gold_is_fake: Optional[bool] = None
    gold_source: Optional[SourceLabel] = None


class EvidenceItem(BaseModel):
    query: str
    title: str = ""
    url: str = ""
    snippet: str = ""
    domain: str = ""
    trusted: bool = False


class AgentVerdict(BaseModel):
    agent: str                                  # text | image | evidence
    flagged_source: SourceLabel = SourceLabel.REAL
    confidence: float = 0.5
    rationale: str = ""
    evidence: list[EvidenceItem] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def _clip(cls, v: float) -> float:
        return min(1.0, max(0.0, float(v)))


class CostLog(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    search_calls: int = 0
    llm_calls: int = 0

    def add(self, other: "CostLog") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.latency_s += other.latency_s
        self.search_calls += other.search_calls
        self.llm_calls += other.llm_calls


class FinalVerdict(BaseModel):
    post_id: str
    is_fake: bool
    source_label: SourceLabel
    confidence: float = 0.5                    # confidence that the verdict is correct
    explanation: str = ""
    abstain: bool = False
    parse_ok: bool = True

    @field_validator("confidence")
    @classmethod
    def _clip(cls, v: float) -> float:
        return min(1.0, max(0.0, float(v)))


class RunRecord(BaseModel):
    method: str
    post: PostInput
    verdicts: list[AgentVerdict] = Field(default_factory=list)
    final: FinalVerdict
    cost: CostLog = Field(default_factory=CostLog)
