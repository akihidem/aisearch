"""F2 refine: 再帰的自己改善ループの決定的テスト。"""
from __future__ import annotations

from aisearch.clients import FakeLLM
from aisearch.config import Config
from aisearch.judge import FakeJudge, LLMJudge, aggregate_votes
from aisearch.refine import refine

TASK = "Improve the answer."


def _improving_llm() -> FakeLLM:
    """改稿のたびに IMPROVED マーカーが増える決定的レスポンダ。"""
    return FakeLLM(responder=lambda p, i: "IMPROVED " * (p.count("IMPROVED") + 1) + "x")


# --- 基準1: 決定的に収束し、採択スコアは単調改善 ---
def test_refine_monotonic_improvement_and_best_is_max():
    cfg = Config(model="fake", council_size=2, max_iters=3, budget=10_000_000)
    r1 = refine(TASK, cfg, _improving_llm(), FakeJudge())
    r2 = refine(TASK, cfg, _improving_llm(), FakeJudge())
    assert r1.score == r2.score  # 決定的
    accepted = [s.score for s in r1.history if s.reason in ("initial", "accepted")]
    assert accepted == sorted(accepted)  # 単調非減少
    assert len(accepted) >= 2  # 実際に改善が起きている
    # best は history 中の最大スコアと一致
    best = max(r1.history, key=lambda s: s.score)
    assert r1.score == best.score and r1.best_artifact == best.artifact
    # no-improvement step も含め、best を超える step は存在しない（弱いテスト回避）
    assert all(s.score <= r1.score for s in r1.history)


# --- 基準2: 停止条件 3 種が独立に発火 ---
def test_stop_on_max_iters():
    cfg = Config(model="fake", council_size=1, max_iters=2, budget=10_000_000)
    r = refine(TASK, cfg, _improving_llm(), FakeJudge())
    assert r.stop_reason == "max_iters"
    assert len([s for s in r.history if s.reason == "accepted"]) == 2


def test_stop_on_plateau_when_no_improvement():
    cfg = Config(model="fake", council_size=1, max_iters=5, budget=10_000_000)
    flat_judge = FakeJudge(scorer=lambda t, a: 5.0)  # 常に同点
    r = refine(TASK, cfg, _improving_llm(), flat_judge)
    assert r.stop_reason == "plateau"
    assert r.score == 5.0


def test_stop_on_plateau_when_degrades():
    cfg = Config(model="fake", council_size=1, max_iters=5, budget=10_000_000)
    # マーカーが増えるほどスコアが下がる → 改稿で劣化 → 即停止
    degrade_judge = FakeJudge(scorer=lambda t, a: -float(a.count("IMPROVED")))
    r = refine(TASK, cfg, _improving_llm(), degrade_judge)
    assert r.stop_reason == "plateau"
    assert r.score == r.history[0].score  # 初期が最良のまま


def test_stop_on_budget():
    cfg = Config(model="fake", council_size=3, max_iters=3, budget=5)
    r = refine(TASK, cfg, FakeLLM(), FakeJudge())
    assert r.stop_reason == "budget"


# --- 基準3: Judge は seed固定 + N票 + 集約で決定的 ---
def test_judge_aggregation_is_deterministic():
    client = FakeLLM(responder=lambda p, i: "7")
    j = LLMJudge(client, votes=3)
    a = j.score(TASK, "artifact", seed=0)
    b = LLMJudge(FakeLLM(responder=lambda p, i: "7"), votes=3).score(TASK, "artifact", seed=0)
    assert a.score == b.score == 7.0
    assert len(a.votes) == 3
    assert aggregate_votes([2.0, 8.0, 2.0]) == 2.0  # 中央値
