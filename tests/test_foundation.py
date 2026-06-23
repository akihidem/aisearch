"""F0 foundation: クライアント/ジャッジ/Config/コストの決定的テスト。"""
from __future__ import annotations

import inspect

import pytest

from aisearch.clients import (
    ClaudeClient,
    FakeLLM,
    LLMClient,
    LLMResponse,
    MLXClient,
    OllamaClient,
)
from aisearch.config import Config, ConfigError, CostTracker, SearchSpace, make_rng
from aisearch.judge import FakeJudge, Judge, LLMJudge, aggregate_votes, parse_score


# --- 基準1: Fake は決定的（同一入力 → 同一出力） ---
def test_fakellm_deterministic_same_input_same_output():
    a = FakeLLM()
    b = FakeLLM()
    r1 = a.complete("hello world")
    r2 = b.complete("hello world")
    assert r1.text == r2.text
    assert (r1.prompt_tokens, r1.completion_tokens) == (r2.prompt_tokens, r2.completion_tokens)


def test_fakellm_responder_and_failure_injection():
    fll = FakeLLM(responder=lambda p, i: f"resp{i}:{p}")
    assert fll.complete("x").text == "resp0:x"
    assert fll.complete("y").text == "resp1:y"
    boom = FakeLLM(fail_on=lambda p: "BAD" in p)
    boom.complete("ok")  # 成功
    with pytest.raises(RuntimeError):
        boom.complete("this is BAD")
    assert len(boom.calls) == 2  # 失敗呼び出しも記録される


def test_llmresponse_total_tokens():
    r = LLMResponse(text="hi", prompt_tokens=3, completion_tokens=4)
    assert r.total_tokens == 7


def test_fakejudge_deterministic():
    j = FakeJudge()
    assert isinstance(j, Judge)
    assert j.score("t", "IMPROVED IMPROVED x").score == 2.0
    assert j.score("t", "IMPROVED IMPROVED x").score == 2.0


# --- 基準2: 実アダプタが同一インターフェースを実装（契約検査・API非依存） ---
@pytest.mark.parametrize("cls", [FakeLLM, ClaudeClient, OllamaClient, MLXClient])
def test_adapters_satisfy_llmclient_protocol(cls):
    obj = cls()  # 生成にネットワーク/SDK は不要
    assert isinstance(obj, LLMClient)
    assert hasattr(obj, "model") and obj.model
    sig = inspect.signature(cls.complete)
    assert "temperature" in sig.parameters
    assert "seed" in sig.parameters


# --- 基準3: 既定の L0 は API を一切叩かない（このファイルはネットワーク不使用） ---
def test_no_network_in_unit_tests():
    # 実アダプタはインスタンス化のみで、complete() を呼ばない限り通信しない
    c = OllamaClient()
    assert c.model and "localhost" in c._host  # 構築だけで通信は発生しない


# --- 基準4: Config バリデーションと seed 固定の決定性 ---
def test_config_valid_and_to_dict():
    c = Config(model="m", temperature=0.5, council_size=2)
    d = c.to_dict()
    assert d["model"] == "m" and d["council_size"] == 2


def test_config_roles_coerced_to_tuple_and_in_dict():
    c = Config(model="m", roles=["a", "b"])  # list でも tuple 化される
    assert c.roles == ("a", "b")
    assert c.to_dict()["roles"] == ["a", "b"]
    hash(c)  # frozen/hashable（search の評価キャッシュキーに使える）


def test_config_rejects_empty_role_in_roles():
    with pytest.raises(ConfigError):
        Config(model="m", roles=("ok", ""))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": ""},
        {"model": "m", "temperature": 3.0},
        {"model": "m", "temperature": -0.1},
        {"model": "m", "council_size": 0},
        {"model": "m", "budget": 0},
        {"model": "m", "max_iters": -1},
        {"model": "m", "judge_votes": 0},
    ],
)
def test_config_validation_rejects_invalid(kwargs):
    with pytest.raises(ConfigError):
        Config(**kwargs)


def test_make_rng_is_seed_deterministic():
    s1 = [make_rng(7).random() for _ in range(1)]
    a = make_rng(7)
    b = make_rng(7)
    seq_a = [a.random() for _ in range(5)]
    seq_b = [b.random() for _ in range(5)]
    assert seq_a == seq_b
    assert make_rng(8).random() != s1[0]


def test_searchspace_only_produces_valid_configs():
    space = SearchSpace()
    rng = make_rng(0)
    cfgs = [space.sample(rng) for _ in range(50)]
    for c in cfgs:  # 構築できている＝バリデーション通過
        m = space.mutate(c, rng)
        x = space.crossover(c, m, rng)
        assert m.council_size >= 1 and 0.0 <= x.temperature <= 2.0


def test_cost_tracker():
    t = CostTracker(budget=10)
    t.add(4)
    assert t.remaining == 6 and not t.exceeded()
    t.add(6)
    assert t.exceeded() and t.remaining == 0


# --- Judge: パース/集約は純関数として決定的 ---
def test_parse_score_and_clamp():
    assert parse_score("the score is 8.5 out of 10") == 8.5
    assert parse_score("12") == 10.0  # clamp 上限
    assert parse_score("-3") == 0.0  # clamp 下限
    with pytest.raises(ValueError):
        parse_score("no number here")


def test_aggregate_votes_median():
    assert aggregate_votes([1.0, 2.0, 3.0]) == 2.0
    assert aggregate_votes([5.0, 1.0]) == 3.0
    with pytest.raises(ValueError):
        aggregate_votes([])


def test_llmjudge_seed_votes_deterministic():
    client = FakeLLM(responder=lambda p, i: "score: 7")
    j = LLMJudge(client, votes=3)
    jd = j.score("task", "artifact", seed=0)
    assert jd.score == 7.0
    assert len(jd.votes) == 3
    again = LLMJudge(FakeLLM(responder=lambda p, i: "score: 7")).score("task", "artifact", seed=0)
    assert again.score == 7.0


def test_l0_path_makes_no_network(monkeypatch):
    """L0 の生成/改善経路がソケットを一切開かないことを実証（基準 F0-3）。"""
    import socket

    def _blocked(*a, **k):
        raise OSError("network disabled in L0")

    monkeypatch.setattr(socket, "socket", _blocked)
    from aisearch.clients import FakeLLM as _F
    from aisearch.config import Config as _C
    from aisearch.council import generate as _g
    from aisearch.judge import FakeJudge as _J
    from aisearch.refine import refine as _r

    cfg = _C(model="fake", council_size=2, max_iters=2, budget=10_000_000)
    assert _g("t", cfg, _F()).artifact
    improver = _F(responder=lambda p, i: "IMPROVED " * (p.count("IMPROVED") + 1) + "x")
    assert _r("t", cfg, improver, _J()).best_artifact
