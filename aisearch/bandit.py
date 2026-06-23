"""UCB1 バンディットによる評価予算の配分（メタ探索の代替ドライバ）。

進化的 `search()` は世代ごとに母集団を均等評価する。評価が高価（実 LLM/judge）で
かつ noisy なとき、限られた評価予算を有望な config に寄せたい。本モジュールは候補
config 群を「腕(arm)」とみなし、UCB1 で pull（=評価）を配分して最良腕を返す。

限界の明記: 各腕の価値が決定的（再 pull で同値）なら 1 回引けば確定し UCB の旨味は
出ない。真価は evaluator が noisy（votes のばらつき / temperature 抽選）なとき。
本モジュールは「配分機構」を提供し、noise を持つ evaluator と組み合わせて効く。

既存 `search()` とそのキャッシュ単調性には一切触れない独立ドライバ。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from .config import Config, SearchSpace, make_rng

Evaluator = Callable[[Config], "tuple[str, float]"]


@dataclass
class _Arm:
    cfg: Config
    pulls: int = 0
    total: float = 0.0  # スコア総和（mean = total / pulls）
    best_score: float = float("-inf")
    best_artifact: str = ""

    @property
    def mean(self) -> float:
        return self.total / self.pulls if self.pulls else 0.0

    def update(self, artifact: str, score: float) -> None:
        self.pulls += 1
        self.total += score
        if score > self.best_score:
            self.best_score = score
            self.best_artifact = artifact


@dataclass
class BanditResult:
    best_config: Config
    best_artifact: str
    best_score: float  # 最良腕の標本平均
    pulls: list[int] = field(default_factory=list)  # 腕ごとの pull 数
    history: list[float] = field(default_factory=list)  # pull ごとの「その時点の最良腕平均」


def ucb1_select(arms: list[_Arm], t: int, c: float) -> int:
    """UCB1: argmax(mean_i + c*sqrt(ln t / n_i))。同点は最小 index。

    未 pull の腕（n_i==0）は +inf 扱いで最優先（初期は各腕1回ずつ引かれる）。
    """
    best_idx = 0
    best_val = float("-inf")
    ln_t = math.log(t) if t > 0 else 0.0
    for i, arm in enumerate(arms):
        if arm.pulls == 0:
            val = float("inf")
        else:
            val = arm.mean + c * math.sqrt(ln_t / arm.pulls)
        if val > best_val:  # 厳密 > なので同点は先（最小 index）が勝つ
            best_val = val
            best_idx = i
    return best_idx


def search_bandit(
    task: str,
    space: SearchSpace,
    evaluator: Evaluator,
    *,
    budget: int,
    seed: int = 0,
    n_arms: int = 8,
    c: float = math.sqrt(2.0),
) -> BanditResult:
    """UCB1 で評価予算 budget を n_arms 個の候補 config に配分する。

    - 腕集合は space.sample で seed 固定生成（重複 config は許容＝空間が小さい場合の素直な挙動）。
    - 各 pull で evaluator を 1 回呼ぶ（コスト 1）。消費 pull 数は budget を超えない。
    - 返値の best は標本平均が最大の腕。
    """
    if budget < 1:
        raise ValueError("budget must be >= 1")
    if n_arms < 1:
        raise ValueError("n_arms must be >= 1")

    rng = make_rng(seed)
    arms = [_Arm(cfg=space.sample(rng)) for _ in range(n_arms)]

    history: list[float] = []
    for t in range(1, budget + 1):
        idx = ucb1_select(arms, t, c)
        artifact, score = evaluator(arms[idx].cfg)
        arms[idx].update(artifact, score)
        # その時点で平均最大の腕（=暫定 best）の平均を記録
        cur_best = max(arms, key=lambda a: (a.pulls > 0, a.mean))
        history.append(cur_best.mean if cur_best.pulls else 0.0)

    # 最良腕: 平均最大。未 pull の腕は候補外（pulls>0 を優先）。同点は最小 index。
    best = max(range(n_arms), key=lambda i: (arms[i].pulls > 0, arms[i].mean, -i))
    arm = arms[best]
    return BanditResult(
        best_config=arm.cfg,
        best_artifact=arm.best_artifact,
        best_score=arm.mean,
        pulls=[a.pulls for a in arms],
        history=history,
    )
