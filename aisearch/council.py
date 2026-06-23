"""複数LLMの合議生成: propose → critique → aggregate。"""
from __future__ import annotations

from dataclasses import dataclass, field

from .clients import LLMClient
from .config import Config, CostTracker


class CouncilError(RuntimeError):
    """合議が成果物を出せなかった（全提案が失敗）。"""


@dataclass
class CouncilResult:
    artifact: str
    candidates: list[str]
    log: list[str] = field(default_factory=list)
    tokens: int = 0
    truncated: bool = False  # budget 超過で打ち切ったか


def _propose_prompt(task: str, role: str, idx: int) -> str:
    return f"[propose#{idx} role={role}] Produce a solution.\nTASK:\n{task}"


def _critique_prompt(task: str, candidate: str) -> str:
    return (
        f"[critique] Critique this candidate for the task.\n"
        f"TASK:\n{task}\nCANDIDATE:\n{candidate}"
    )


def _aggregate_prompt(task: str, pairs: list[tuple[str, str]]) -> str:
    joined = "\n".join(f"- CAND: {c}\n  CRIT: {cr}" for c, cr in pairs)
    return f"[aggregate] Merge the best ideas into one final answer.\nTASK:\n{task}\n{joined}"


def _role_for(config: Config, i: int) -> str:
    """proposer i の役割。roles 指定時はそれを巡回割当、無指定なら単一 role。"""
    if config.roles:
        return config.roles[i % len(config.roles)]
    return config.role


def generate(task: str, config: Config, client: LLMClient) -> CouncilResult:
    """合議生成。client を注入（テストは FakeLLM）。

    - propose: council_size 個の候補を生成（個別の失敗は許容、全滅で CouncilError）
    - critique: 各候補を相互批評
    - aggregate: 統合して1案（council_size==1 はそのまま、budget切れは先頭候補）
    """
    cost = CostTracker(budget=config.budget)
    log: list[str] = []
    candidates: list[str] = []

    # --- propose ---
    for i in range(config.council_size):
        if cost.exceeded():
            log.append(f"budget-exceeded:propose#{i}")
            break
        prompt = _propose_prompt(task, _role_for(config, i), i)
        try:
            resp = client.complete(prompt, temperature=config.temperature, seed=config.seed + i)
        except Exception as e:  # noqa: BLE001 個別候補の障害は許容
            log.append(f"propose#{i}:FAILED:{type(e).__name__}")
            continue
        cost.add(resp.total_tokens)
        candidates.append(resp.text)
        log.append(f"propose#{i}:ok")

    if not candidates:
        raise CouncilError("all proposals failed")

    truncated = cost.exceeded()

    # --- critique ---
    pairs: list[tuple[str, str]] = []
    for i, cand in enumerate(candidates):
        if cost.exceeded():
            log.append(f"budget-exceeded:critique#{i}")
            truncated = True
            pairs.append((cand, ""))
            continue
        try:
            resp = client.complete(
                _critique_prompt(task, cand), temperature=config.temperature, seed=config.seed + i
            )
            cost.add(resp.total_tokens)
            crit = resp.text
            log.append(f"critique#{i}:ok")
        except Exception as e:  # noqa: BLE001
            crit = ""
            log.append(f"critique#{i}:FAILED:{type(e).__name__}")
        pairs.append((cand, crit))

    # --- aggregate ---
    if len(candidates) == 1:
        artifact = candidates[0]
        log.append("aggregate:single")
    elif truncated or cost.exceeded():
        # 予算切れ: 統合呼び出しはせず先頭候補を返す（部分結果）
        artifact = candidates[0]
        truncated = True
        log.append("aggregate:skipped-budget")
    else:
        try:
            resp = client.complete(
                _aggregate_prompt(task, pairs), temperature=config.temperature, seed=config.seed
            )
            cost.add(resp.total_tokens)
            artifact = resp.text
            log.append("aggregate:ok")
        except Exception as e:  # noqa: BLE001
            artifact = candidates[0]
            log.append(f"aggregate:FAILED:{type(e).__name__}")

    return CouncilResult(
        artifact=artifact, candidates=candidates, log=log, tokens=cost.spent, truncated=truncated
    )
