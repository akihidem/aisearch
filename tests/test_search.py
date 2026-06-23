"""F3 search: メタ探索の決定的テスト + CLI demo。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aisearch.config import Config, SearchSpace, make_rng
from aisearch.search import build_evaluator, search

ROOT = Path(__file__).resolve().parent.parent
TASK = "meta-search"


def _stub_evaluator(cfg: Config) -> tuple[str, float]:
    """決定的な評価器: 低温・大council・長いモデル名ほど高スコア。"""
    score = (1.0 if cfg.temperature == 0.0 else 0.0) + cfg.council_size * 0.1 + len(cfg.model) * 1e-3
    return f"art::{cfg.model}/{cfg.council_size}", score


# --- 基準1: seed固定 + stub → 決定的に同じ best_config ---
def test_search_reproducible_same_seed():
    r1 = search(TASK, SearchSpace(), _stub_evaluator, generations=5, pop_size=6, seed=0)
    r2 = search(TASK, SearchSpace(), _stub_evaluator, generations=5, pop_size=6, seed=0)
    assert r1.best_config == r2.best_config
    assert r1.best_score == r2.best_score
    assert r1.best_artifact == r2.best_artifact


# --- 基準2: 世代ごとに母集団 best が単調非減少（エリート保存） ---
def test_search_population_best_monotonic_non_decreasing():
    r = search(TASK, SearchSpace(), _stub_evaluator, generations=6, pop_size=8, seed=1)
    assert r.history == sorted(r.history)  # 非減少
    assert r.best_score == max(r.history) == r.history[-1]


def test_search_monotonic_holds_with_noisy_evaluator():
    """呼ぶたびにスコアが下がる敵対的 evaluator。

    評価キャッシュ + エリート保存により世代 best は単調非減少を保つ
    （キャッシュが無ければ後世代のエリート再評価が下振れして崩れる）。
    """
    state = {"n": 0}

    def noisy(cfg: Config) -> tuple[str, float]:
        state["n"] += 1
        return f"art{state['n']}", 1000.0 - state["n"]

    r = search(TASK, SearchSpace(), noisy, generations=6, pop_size=6, seed=2)
    assert r.history == sorted(r.history)  # 単調非減少
    assert r.best_score == r.history[0]  # 初期最良がエリート保存で維持される
    assert r.best_score == r.history[-1]


# --- 基準3: 探索空間から有効な Config のみ生成 ---
def test_search_produces_only_valid_configs():
    space = SearchSpace()
    rng = make_rng(3)
    for _ in range(300):
        a = space.sample(rng)
        b = space.sample(rng)
        child = space.mutate(space.crossover(a, b, rng), rng)
        assert child.council_size >= 1
        assert 0.0 <= child.temperature <= 2.0
        assert child.model in space.models
    # 探索全体でも例外なく完走する
    res = search(TASK, space, _stub_evaluator, generations=4, pop_size=6, seed=9)
    assert res.best_config.model in space.models


def test_search_respects_max_evals_cap():
    calls = {"n": 0}

    def counting(cfg: Config) -> tuple[str, float]:
        calls["n"] += 1
        return "a", float(calls["n"])

    r = search(TASK, SearchSpace(), counting, generations=5, pop_size=6, seed=0, max_evals=3)
    assert calls["n"] <= 3  # コスト天井を超えて評価しない
    assert r.best_config is not None  # best-so-far は返る


def test_build_evaluator_fake_is_deterministic():
    ev = build_evaluator("fake", "task")
    a = ev(Config(model="m", council_size=2, max_iters=1))
    b = ev(Config(model="m", council_size=2, max_iters=1))
    assert a == b


def test_build_evaluator_cli_runs_without_network():
    # runner 注入で subprocess を使わず cli backend を決定的に検証
    runner = lambda argv: '{"is_error": false, "result": "5", "usage": {"input_tokens": 1, "output_tokens": 1}}'
    ev = build_evaluator("cli", "task", model="claude-haiku-4-5-20251001", runner=runner)
    artifact, score = ev(Config(model="m", council_size=2, max_iters=1, judge_votes=1))
    assert score == 5.0 and artifact == "5"


def test_build_evaluator_unknown_raises():
    with pytest.raises(ValueError):
        build_evaluator("bogus", "task")


def test_cli_tui_transport_calls_wrapper_and_wraps_text():
    # claude-cli-run.py(TUIラッパ)経由の runner を tmux 無しで検証。
    # ラッパは本文テキストのみ返す → JSON エンベロープに包まれ judge が採点できる。
    from aisearch.clients import make_tui_runner

    seen = {}

    class _Proc:
        returncode = 0
        stdout = "5"
        stderr = ""

    def fake_run(cmd, to):
        seen["cmd"] = cmd
        seen["to"] = to
        return _Proc()

    runner = make_tui_runner(script="/x/claude-cli-run.py", subprocess_run=fake_run)
    ev = build_evaluator(
        "cli", "task", model="claude-haiku-4-5-20251001", runner=runner
    )
    artifact, score = ev(Config(model="m", council_size=1, max_iters=1, judge_votes=1))
    assert score == 5.0 and artifact == "5"
    # TUI ラッパが呼ばれ、`claude -p` 直叩きではない
    assert seen["cmd"][0] == "/x/claude-cli-run.py"
    assert "-p" not in seen["cmd"] and "--output-format" not in seen["cmd"]
    assert "--model" in seen["cmd"] and "claude-haiku-4-5-20251001" in seen["cmd"]


def test_cli_tui_transport_error_surfaces_as_is_error():
    from aisearch.clients import make_tui_runner

    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "tmux not found"

    runner = make_tui_runner(script="/x/run.py", subprocess_run=lambda cmd, to: _Proc())
    # is_error=True → ClaudeCliClient.complete が RuntimeError を上げる
    ev = build_evaluator("cli", "task", runner=runner)
    with pytest.raises(RuntimeError):
        ev(Config(model="m", council_size=1, max_iters=1, judge_votes=1))


def test_build_evaluator_unknown_transport_raises():
    with pytest.raises(ValueError):
        build_evaluator("cli", "task", transport="carrier-pigeon")


def test_searchspace_evolves_valid_role_rosters():
    space = SearchSpace()
    rng = make_rng(5)
    for _ in range(300):
        a = space.sample(rng)
        b = space.sample(rng)
        child = space.mutate(space.crossover(a, b, rng), rng)
        for cfg in (a, b, child):
            assert cfg.roles  # ロスターは常に非空
            assert all(r in space.roles for r in cfg.roles)  # 既知役割のみ
            assert cfg.role == cfg.roles[0]  # role はロスター先頭に整合


# --- roster長×council_size 連動: 不変条件 C>=L（全 roster 役割が council で使われる） ---
def test_roster_council_size_coupling_invariant_holds():
    space = SearchSpace()
    rng = make_rng(11)
    for _ in range(500):
        a = space.sample(rng)
        b = space.sample(rng)
        cross = space.crossover(a, b, rng)
        mut = space.mutate(cross, rng)
        for cfg in (a, b, cross, mut):
            # 不変条件: roster の全役割が i%len 巡回で最低1回割り当たる
            assert cfg.council_size >= len(cfg.roles), (
                f"C<L 違反: council_size={cfg.council_size} roles={cfg.roles}"
            )


def test_mutate_council_size_gene_never_breaks_coupling():
    # council_size 遺伝子を直接揺らしても roster 長を下回らない
    space = SearchSpace()
    rng = make_rng(7)
    base = space.sample(rng)
    for _ in range(500):
        m = space.mutate(base, rng)
        assert m.council_size >= len(m.roles)


def test_coupling_repairs_when_no_council_size_ge_roster():
    # council_sizes が全て roster 長未満になる病的空間でも不変条件を保つ
    # （フォールバック=roster_len 自体を返し C>=L を最優先）
    space = SearchSpace(roles=("a", "b", "c", "d"), council_sizes=(1, 2))
    rng = make_rng(3)
    for _ in range(500):
        cfg = space.sample(rng)
        assert cfg.council_size >= len(cfg.roles)
        m = space.mutate(cfg, rng)
        assert m.council_size >= len(m.roles)


def test_coupling_backward_compat_single_role_config():
    # roles=() の単一role Config に council_size 変異をかけても壊れない（L=0 無制約）
    space = SearchSpace()
    rng = make_rng(2)
    single = Config(model="m", roles=(), council_size=3)
    for _ in range(200):
        m = space.mutate(single, rng)
        assert m.council_size >= 1  # 単一role経路は従来通り valid


# --- 基準4: CLI --demo が exit 0 で best.json を書き出す ---
def test_cli_demo_writes_best_json(tmp_path):
    out = tmp_path / "best.json"
    proc = subprocess.run(
        [sys.executable, "-m", "aisearch.search", "--demo", "--seed", "0", "--out", str(out)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    for key in ("task", "best_config", "best_artifact", "best_score", "score_history"):
        assert key in data
    assert data["best_config"]["model"]
    assert data["best_config"]["roles"]  # 役割ロスターが探索結果に含まれる
    assert data["best_artifact"]  # 非空の成果物
    assert isinstance(data["best_score"], (int, float))
    assert isinstance(data["score_history"], list) and len(data["score_history"]) > 0
    assert data["score_history"] == sorted(data["score_history"])  # 単調非減少
