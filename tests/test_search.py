"""F3 search: メタ探索の決定的テスト + CLI demo。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from aisearch.config import Config, SearchSpace, make_rng
from aisearch.search import search

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
