"""モデル/設定のメタ探索（進化的）。最良の (config, artifact) を返す。

evaluator: Config -> (artifact, score) を注入する設計。
- 本番: make_refine_evaluator(task, real_client, real_judge)
- L0  : 決定的な stub evaluator
- CLI --demo: FakeLLM+FakeJudge 経由の refine evaluator（API不要・決定的）
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Callable

from .clients import ClaudeCliClient, FakeLLM, LLMClient, make_tui_runner
from .config import Config, SearchSpace, make_rng
from .judge import FakeJudge, Judge, LLMJudge
from .refine import refine

Evaluator = Callable[[Config], "tuple[str, float]"]


@dataclass
class SearchResult:
    best_config: Config
    best_artifact: str
    best_score: float
    history: list[float] = field(default_factory=list)  # 各世代の母集団 best スコア


def make_refine_evaluator(task: str, client: LLMClient, judge: Judge) -> Evaluator:
    """個体(Config)を refine() で評価する evaluator（本番/demo 用）。"""

    def _eval(cfg: Config) -> tuple[str, float]:
        res = refine(task, cfg, client, judge)
        return res.best_artifact, res.score

    return _eval


def _config_key(cfg: Config) -> tuple:
    return (cfg.model, cfg.temperature, cfg.role, cfg.roles, cfg.council_size, cfg.seed)


def search(
    task: str,
    space: SearchSpace,
    evaluator: Evaluator,
    *,
    generations: int = 5,
    pop_size: int = 6,
    seed: int = 0,
    elite: int = 2,
    max_evals: int | None = None,
) -> SearchResult:
    """進化的メタ探索。

    評価は Config 単位でキャッシュし、エリート保存により母集団 best は単調非減少
    （キャッシュにより noisy/実LLM evaluator でも成立する）。
    """
    if pop_size < 1:
        raise ValueError("pop_size must be >= 1")
    if not (1 <= elite <= pop_size):
        raise ValueError("elite must be in [1, pop_size]")

    rng = make_rng(seed)
    population = [space.sample(rng) for _ in range(pop_size)]

    # 評価は Config 単位でキャッシュ: (1) 同一個体の再評価コストを払わない、
    # (2) エリートが世代をまたいでも同じスコアを保つ → noisy/実LLM evaluator でも
    #     母集団 best が単調非減少になる（基準 F3-2 を実装レベルで担保）。
    cache: dict[Config, tuple[str, float]] = {}
    evals_done = 0  # 実評価(キャッシュミス)回数。max_evals のコスト天井に使う

    def _evaluate(cfg: Config) -> tuple[str, float]:
        nonlocal evals_done
        if cfg not in cache:
            cache[cfg] = evaluator(cfg)
            evals_done += 1
        return cache[cfg]

    best_config: Config | None = None
    best_artifact = ""
    best_score = float("-inf")
    history: list[float] = []

    for _gen in range(generations):
        scored: list[tuple[float, Config, str]] = []
        for cfg in population:
            # コスト天井: 新規評価が上限に達したら既評価(キャッシュ)のみで続行
            if max_evals is not None and cfg not in cache and evals_done >= max_evals:
                continue
            artifact, score = _evaluate(cfg)
            scored.append((score, cfg, artifact))
        if not scored:
            break  # 天井で誰も評価できない → best-so-far を返して終了
        # 決定的ソート: スコア降順、同点は config の決定的キーで安定化
        scored.sort(key=lambda t: (-t[0], _config_key(t[1])))

        gen_best_score, gen_best_cfg, gen_best_art = scored[0]
        history.append(gen_best_score)  # エリート保存 → 非減少
        if gen_best_score > best_score:
            best_score, best_config, best_artifact = gen_best_score, gen_best_cfg, gen_best_art

        # 次世代: エリート + 交叉/突然変異
        elites = [cfg for _, cfg, _ in scored[:elite]]
        next_pop = list(elites)
        parent_pool = elites if len(elites) >= 2 else [c for _, c, _ in scored[: max(2, pop_size)]]
        while len(next_pop) < pop_size:
            a = parent_pool[rng.randrange(len(parent_pool))]
            b = parent_pool[rng.randrange(len(parent_pool))]
            child = space.crossover(a, b, rng)
            child = space.mutate(child, rng)
            next_pop.append(child)
        population = next_pop

    assert best_config is not None
    return SearchResult(
        best_config=best_config,
        best_artifact=best_artifact,
        best_score=best_score,
        history=history,
    )


# --- CLI (--demo: 決定的・API不要) ---------------------------------------------


def _demo_responder(prompt: str, idx: int) -> str:
    """改稿のたびに IMPROVED マーカーが増える決定的レスポンダ。"""
    n = prompt.count("IMPROVED") + 1
    return "IMPROVED " * n + "answer"


def build_demo_evaluator(task: str) -> Evaluator:
    """FakeLLM+FakeJudge で合議+自己改善を回す決定的 evaluator。"""
    client = FakeLLM(responder=_demo_responder)
    judge = FakeJudge()  # 既定 scorer = IMPROVED の数
    return make_refine_evaluator(task, client, judge)


def build_evaluator(
    backend: str,
    task: str,
    *,
    model: str = "claude-haiku-4-5-20251001",
    runner=None,
    transport: str = "direct",
) -> Evaluator:
    """探索の評価器をバックエンド別に構築。

    - "fake": FakeLLM+FakeJudge（決定的・API不要）
    - "cli" : ClaudeCliClient+LLMJudge（実 claude CLI / OAuth・課金あり）。
              runner を注入すれば subprocess 無しで決定的にテストできる。
              transport="tui" なら claude-cli-run.py(対話TUIラッパ)経由で呼び、
              Agent SDK クレジット枠でなく通常サブスク枠から消費する（runner 明示時は優先）。
    """
    if backend == "fake":
        return build_demo_evaluator(task)
    if backend == "cli":
        if runner is None and transport == "tui":
            runner = make_tui_runner()
        elif transport not in ("direct", "tui"):
            raise ValueError(f"unknown transport: {transport!r}")
        client = ClaudeCliClient(model=model, runner=runner)
        judge = LLMJudge(client, votes=1)
        return make_refine_evaluator(task, client, judge)
    raise ValueError(f"unknown backend: {backend!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aisearch.search")
    parser.add_argument("--demo", action="store_true",
                        help="FakeLLM+FakeJudge で決定的に実行（= --backend fake）")
    parser.add_argument("--backend", choices=["fake", "cli"], default="fake",
                        help="fake=決定的/無課金, cli=実 claude CLI(OAuth・課金あり)")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001",
                        help="cli backend のモデル")
    parser.add_argument("--cli-transport", choices=["direct", "tui"], default="direct",
                        help="cli backend の呼び出し経路: direct=claude -p(SDK枠) / "
                             "tui=claude-cli-run.py(対話TUI・サブスク枠でSDKクレジット非消費)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--pop-size", type=int, default=6)
    parser.add_argument("--max-evals", type=int, default=None,
                        help="実評価回数の上限（cli backend のコスト天井）")
    parser.add_argument("--task", type=str, default="Write a haiku about loops.")
    parser.add_argument("--out", type=str, default="best.json")
    args = parser.parse_args(argv)

    backend = "fake" if args.demo else args.backend
    if backend == "cli":
        via = "対話TUIラッパ(SDKクレジット非消費)" if args.cli_transport == "tui" else "claude -p(SDK枠)"
        print(
            f"[backend=cli / model={args.model} / transport={args.cli_transport}] "
            f"実 claude CLI を {via} 経由で使用（OAuth）。"
            " 評価1回 ≈ council+refine の複数呼び出し。--max-evals でコスト天井を。"
            " tui は1呼び出し毎に tmux 起動のため direct より大幅に遅い。",
            file=sys.stderr,
        )

    evaluator = build_evaluator(
        backend, args.task, model=args.model, transport=args.cli_transport
    )
    result = search(
        args.task,
        SearchSpace(),
        evaluator,
        generations=args.generations,
        pop_size=args.pop_size,
        seed=args.seed,
        max_evals=args.max_evals,
    )
    payload = {
        "task": args.task,
        "backend": backend,
        "best_config": result.best_config.to_dict(),
        "best_artifact": result.best_artifact,
        "best_score": result.best_score,
        "score_history": result.history,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(
        f"wrote {args.out}: backend={backend} best_score={result.best_score} "
        f"model={result.best_config.model} council={result.best_config.council_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
