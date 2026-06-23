"""claude CLI バックエンドで実際に「合議 → 再帰的自己改善」を1回走らせる。

APIキー不要（ローカル `claude` CLI の OAuth 認証をそのまま使う）。

  python examples/cli_real_run.py --model claude-haiku-4-5-20251001
  python examples/cli_real_run.py --model claude-opus-4-8 --task "..."
"""
from __future__ import annotations

import argparse

# 関数は submodule から import する（refine/search は同名 submodule と衝突するため）
from aisearch.clients import ClaudeCliClient
from aisearch.config import Config
from aisearch.judge import LLMJudge
from aisearch.refine import refine


def main() -> int:
    ap = argparse.ArgumentParser(prog="cli_real_run")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--task", default="Write a vivid two-line poem about recursion.")
    ap.add_argument("--council-size", type=int, default=2)
    ap.add_argument("--max-iters", type=int, default=1)
    args = ap.parse_args()

    client = ClaudeCliClient(model=args.model)  # judge と共有 → コストも合算される
    judge = LLMJudge(client, votes=1)
    cfg = Config(
        model=args.model,
        council_size=args.council_size,
        roles=("generalist", "contrarian"),
        max_iters=args.max_iters,
        judge_votes=1,
    )

    print(f"[backend=claude CLI / model={args.model}] running council + refine ...")
    result = refine(args.task, cfg, client, judge)

    print("\n=== result ===")
    print("score        :", result.score)
    print("stop_reason  :", result.stop_reason)
    print("history      :", [(s.iteration, round(s.score, 2), s.reason) for s in result.history])
    print(f"total_cost   : ${client.total_cost_usd:.4f}")
    print("\n--- best artifact ---")
    print(result.best_artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
