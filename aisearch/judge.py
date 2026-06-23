"""LLM-as-judge と決定的 FakeJudge。

非決定的な LLM 採点を、seed 固定 + N票 + 中央値集約 で再現性ある形に丸める。
集約/パースは純関数として切り出し、決定的にテストする。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median
from typing import Callable, Protocol, runtime_checkable

from .clients import LLMClient


@dataclass
class Judgement:
    score: float
    rationale: str = ""
    votes: tuple[float, ...] = ()


@runtime_checkable
class Judge(Protocol):
    def score(self, task: str, artifact: str, *, seed: int | None = None) -> Judgement: ...


class FakeJudge:
    """決定的な採点器。scorer(task, artifact) -> float を注入。

    既定は artifact 中の 'IMPROVED' マーカー数。同一入力 → 同一スコア。
    """

    def __init__(self, scorer: Callable[[str, str], float] | None = None):
        self._scorer = scorer or (lambda task, art: float(art.count("IMPROVED")))
        self.calls: list[tuple[str, str]] = []

    def score(self, task: str, artifact: str, *, seed: int | None = None) -> Judgement:
        self.calls.append((task, artifact))
        s = float(self._scorer(task, artifact))
        return Judgement(score=s, rationale="fake", votes=(s,))


def parse_score(text: str) -> float:
    """テキストから 0..10 のスコアを取り出す（最初の数値・範囲外は clamp）。"""
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        raise ValueError(f"no score found in judge output: {text!r}")
    return max(0.0, min(10.0, float(m.group())))


def aggregate_votes(votes: list[float]) -> float:
    """票の集約 = 中央値（決定的・外れ値に頑健）。"""
    if not votes:
        raise ValueError("no votes to aggregate")
    return float(median(votes))


class LLMJudge:
    """LLM を採点者に使う。seed 固定 + N票 + 中央値集約で再現性を担保。"""

    def __init__(
        self,
        client: LLMClient,
        *,
        votes: int = 3,
        rubric: str = "Rate the artifact's quality from 0 to 10.",
    ):
        if votes < 1:
            raise ValueError("votes must be >= 1")
        self._client = client
        self._votes = votes
        self._rubric = rubric

    def _prompt(self, task: str, artifact: str) -> str:
        return (
            f"{self._rubric}\nTASK:\n{task}\nARTIFACT:\n{artifact}\n"
            "Respond with a single number 0-10."
        )

    def score(self, task: str, artifact: str, *, seed: int | None = None) -> Judgement:
        prompt = self._prompt(task, artifact)
        raw: list[float] = []
        for i in range(self._votes):
            vote_seed = None if seed is None else seed + i
            resp = self._client.complete(prompt, temperature=0.0, seed=vote_seed)
            raw.append(parse_score(resp.text))
        agg = aggregate_votes(raw)
        return Judgement(score=agg, rationale=f"{self._votes} votes", votes=tuple(raw))
