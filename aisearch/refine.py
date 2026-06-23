"""再帰的自己改善ループ: 生成 → 採点 → 自己批評 → 改稿 → 収束。"""
from __future__ import annotations

from dataclasses import dataclass, field

from .clients import LLMClient
from .config import Config, CostTracker
from .council import generate
from .judge import Judge


@dataclass
class RefineStep:
    iteration: int
    artifact: str
    score: float
    reason: str = ""  # initial / accepted / no-improvement / budget-exceeded


@dataclass
class RefineResult:
    best_artifact: str
    score: float
    history: list[RefineStep] = field(default_factory=list)
    stop_reason: str = ""  # max_iters / plateau / budget


def _self_critique_prompt(task: str, artifact: str) -> str:
    return f"[self-critique] What is weak about this artifact?\nTASK:\n{task}\nARTIFACT:\n{artifact}"


def _revise_prompt(task: str, artifact: str, critique: str) -> str:
    return (
        f"[revise] Improve the artifact using the critique.\n"
        f"TASK:\n{task}\nARTIFACT:\n{artifact}\nCRITIQUE:\n{critique}\n"
        "Return an improved version."
    )


def refine(
    task: str,
    config: Config,
    client: LLMClient,
    judge: Judge,
    *,
    epsilon: float = 1e-9,
) -> RefineResult:
    """council 出力を judge で採点しつつ改稿。改善が止まる/上限/予算で停止。

    停止条件は 3 種:
      - plateau : 改稿で改善しない（劣化含む）→ 即停止
      - max_iters: 反復上限に到達
      - budget  : トークン予算超過
    返る best は history 中の最大スコアと一致する。
    """
    cost = CostTracker(budget=config.budget)

    # iteration 0: 初期生成
    base = generate(task, config, client)
    cost.add(base.tokens)
    artifact = base.artifact
    # Judge は固定 seed（config.seed）で呼ぶ＝評価のランダム性を固定し、反復間で
    # 「動かない物差し」にする。同一 artifact は常に同一スコアになり、plateau 判定が
    # seed ノイズで誤発火しない（基準 F2-3「seed固定+N票集約で同一入力に決定的」）。
    score = judge.score(task, artifact, seed=config.seed).score
    history = [RefineStep(0, artifact, score, "initial")]
    best_artifact, best_score = artifact, score

    stop_reason = "max_iters"
    it = 0
    while it < config.max_iters:
        it += 1
        if cost.exceeded():
            stop_reason = "budget"
            history.append(RefineStep(it, best_artifact, best_score, "budget-exceeded"))
            break

        crit = client.complete(
            _self_critique_prompt(task, best_artifact),
            temperature=config.temperature,
            seed=config.seed + it,
        )
        cost.add(crit.total_tokens)
        rev = client.complete(
            _revise_prompt(task, best_artifact, crit.text),
            temperature=config.temperature,
            seed=config.seed + it,
        )
        cost.add(rev.total_tokens)
        new_artifact = rev.text
        new_score = judge.score(task, new_artifact, seed=config.seed).score

        if new_score <= best_score + epsilon:
            history.append(RefineStep(it, new_artifact, new_score, "no-improvement"))
            stop_reason = "plateau"
            break

        best_artifact, best_score = new_artifact, new_score
        history.append(RefineStep(it, new_artifact, new_score, "accepted"))
    else:
        stop_reason = "max_iters"

    best = max(history, key=lambda s: s.score)
    return RefineResult(
        best_artifact=best.artifact, score=best.score, history=history, stop_reason=stop_reason
    )
