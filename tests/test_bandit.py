"""bandit.py（UCB1 評価予算配分）の決定的テスト。"""
from __future__ import annotations

import math
import random

import pytest

from aisearch.bandit import _Arm, search_bandit, ucb1_select
from aisearch.config import Config, SearchSpace, make_rng


def _arms_cfgs(space: SearchSpace, seed: int, n: int):
    """search_bandit と同一手順で腕の config 列を再現（真最良の特定用）。"""
    rng = make_rng(seed)
    return [space.sample(rng) for _ in range(n)]


def _make_noisy_eval(noise_std: float, seed: int):
    """真の平均 = cfg.seed（腕固有・既知）＋ seeded gaussian noise。

    cfg.seed は sample() が rng.randrange(1_000_000) で振るため腕ごとにほぼ一意。
    """
    rnd = random.Random(seed)

    def ev(cfg: Config) -> tuple[str, float]:
        true = float(cfg.seed)
        return (f"art-{cfg.seed}", true + rnd.gauss(0.0, noise_std))

    return ev


# --- UCB1 単体 ---------------------------------------------------------------
def test_ucb1_prefers_unpulled_arm_first():
    arms = [_Arm(cfg=Config(model="m")) for _ in range(3)]
    arms[0].update("a", 100.0)  # 1本だけ引かれている
    # 未 pull の腕（index 1）が +inf 扱いで最優先
    assert ucb1_select(arms, t=2, c=math.sqrt(2)) == 1


def test_ucb1_breaks_ties_to_lowest_index():
    arms = [_Arm(cfg=Config(model="m")) for _ in range(2)]
    arms[0].update("a", 5.0)
    arms[1].update("b", 5.0)  # 同一 mean・同一 pulls → UCB 値同点
    assert ucb1_select(arms, t=3, c=math.sqrt(2)) == 0


def test_ucb1_exploits_higher_mean_when_counts_equal():
    arms = [_Arm(cfg=Config(model="m")) for _ in range(2)]
    arms[0].update("a", 1.0)
    arms[1].update("b", 9.0)  # 同 pulls なら平均が高い方
    assert ucb1_select(arms, t=3, c=math.sqrt(2)) == 1


# --- search_bandit: 予算・配分・同定 -----------------------------------------
def test_budget_is_exactly_consumed():
    space = SearchSpace()
    ev = _make_noisy_eval(noise_std=2.0, seed=1)
    res = search_bandit("t", space, ev, budget=120, seed=0, n_arms=8)
    assert sum(res.pulls) == 120  # 予算ちょうど消費（n_arms < budget）


def test_budget_below_n_arms_no_overspend_and_no_crash():
    space = SearchSpace()
    ev = _make_noisy_eval(noise_std=1.0, seed=2)
    res = search_bandit("t", space, ev, budget=3, seed=0, n_arms=8)
    assert sum(res.pulls) == 3  # 予算 < n_arms でも超過しない
    assert res.best_config is not None
    assert sum(1 for p in res.pulls if p > 0) == 3  # 3 本だけ引かれた


def test_identifies_and_concentrates_on_true_best_arm():
    space = SearchSpace()
    seed, n_arms, budget = 0, 8, 300
    cfgs = _arms_cfgs(space, seed, n_arms)
    true_means = [c.seed for c in cfgs]
    true_best = max(range(n_arms), key=lambda i: true_means[i])

    ev = _make_noisy_eval(noise_std=2.0, seed=7)
    res = search_bandit("t", space, ev, budget=budget, seed=seed, n_arms=n_arms)

    # 真最良腕を返す
    assert res.best_config.seed == cfgs[true_best].seed
    # 予算の最多 pull が真最良腕に充てられる
    assert res.pulls[true_best] == max(res.pulls)
    # 均等配分（budget/n_arms）より明確に多く引く
    assert res.pulls[true_best] > budget / n_arms


def test_deterministic_same_seed():
    space = SearchSpace()
    ev1 = _make_noisy_eval(noise_std=2.0, seed=5)
    ev2 = _make_noisy_eval(noise_std=2.0, seed=5)
    r1 = search_bandit("t", space, ev1, budget=100, seed=3, n_arms=6)
    r2 = search_bandit("t", space, ev2, budget=100, seed=3, n_arms=6)
    assert r1.pulls == r2.pulls
    assert r1.best_config.seed == r2.best_config.seed
    assert r1.best_score == r2.best_score


def test_invalid_args_raise():
    space = SearchSpace()
    ev = _make_noisy_eval(noise_std=1.0, seed=0)
    with pytest.raises(ValueError):
        search_bandit("t", space, ev, budget=0, seed=0, n_arms=4)
    with pytest.raises(ValueError):
        search_bandit("t", space, ev, budget=10, seed=0, n_arms=0)
