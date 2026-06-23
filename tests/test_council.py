"""F1 council: 合議生成 propose→critique→aggregate の決定的テスト。"""
from __future__ import annotations

import pytest

from aisearch.clients import FakeLLM
from aisearch.config import Config
from aisearch.council import CouncilError, generate

TASK = "Solve the problem."


# --- 基準1: 決定的に1生成物 + 各段の痕跡をログで検証 ---
def test_generate_deterministic_and_stages_logged():
    cfg = Config(model="fake", council_size=3, budget=10_000_000)
    r1 = generate(TASK, cfg, FakeLLM())
    r2 = generate(TASK, cfg, FakeLLM())
    assert r1.artifact == r2.artifact  # 決定的
    assert len(r1.candidates) == 3
    for stage in ("propose#0:ok", "propose#1:ok", "propose#2:ok", "critique#0:ok", "aggregate:ok"):
        assert stage in r1.log
    assert not r1.truncated


# --- 基準2: council_size=1 と N>1 ---
def test_council_size_one_returns_candidate_directly():
    cfg = Config(model="fake", council_size=1, budget=10_000_000)
    r = generate(TASK, cfg, FakeLLM())
    assert "aggregate:single" in r.log
    assert r.artifact == r.candidates[0]
    assert len(r.candidates) == 1


def test_aggregate_receives_all_candidates_when_n_gt_1():
    cfg = Config(model="fake", council_size=3, budget=10_000_000)
    client = FakeLLM()
    generate(TASK, cfg, client)
    agg_prompts = [p for p in client.calls if p.startswith("[aggregate]")]
    assert len(agg_prompts) == 1
    assert agg_prompts[0].count("CAND:") == 3  # 3 候補すべてが統合に渡る


# --- 基準3: 障害フォールバック / 全滅エラー ---
def test_partial_failure_falls_back_to_survivors():
    cfg = Config(model="fake", council_size=3, budget=10_000_000)
    client = FakeLLM(fail_on=lambda p: "propose#1" in p)
    r = generate(TASK, cfg, client)
    assert len(r.candidates) == 2  # #0, #2 が生存
    assert any("propose#1:FAILED" in line for line in r.log)
    assert r.artifact  # 生存案で成果物を出す


def test_all_proposals_fail_raises():
    cfg = Config(model="fake", council_size=3, budget=10_000_000)
    client = FakeLLM(fail_on=lambda p: p.startswith("[propose"))
    with pytest.raises(CouncilError):
        generate(TASK, cfg, client)


# --- 基準4: budget 超過で打ち切り、部分結果を返す ---
def test_budget_truncation_returns_partial():
    cfg = Config(model="fake", council_size=3, budget=1)
    r = generate(TASK, cfg, FakeLLM())
    assert r.truncated is True
    assert len(r.candidates) < 3  # 全候補を作り切らずに打ち切り
    assert r.artifact  # それでも部分結果は返る


def test_council_threads_distinct_seeds_per_candidate():
    """propose 段に config.seed + i の異なる seed が渡る（実LLM/Ollama の seed 配線確認）。"""
    seen: list[int | None] = []

    class SeedSpy(FakeLLM):
        def complete(self, prompt, *, temperature=0.7, seed=None):
            seen.append(seed)
            return super().complete(prompt, temperature=temperature, seed=seed)

    cfg = Config(model="fake", council_size=3, budget=10_000_000, seed=100)
    generate(TASK, cfg, SeedSpy())
    assert {100, 101, 102} <= set(seen)  # propose#0..2 に異なる seed
