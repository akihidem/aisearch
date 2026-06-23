"""実LLM smoke（pytest -m integration でのみ実行・L0 からは除外）。

L0(`pytest -m "not integration"`)では収集されるが deselect される。
実行には ANTHROPIC_API_KEY が必要（無ければ skip）。
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_claude_smoke():
    from aisearch.clients import ClaudeClient

    r = ClaudeClient().complete("Reply with the single word: pong", temperature=0.0)
    assert isinstance(r.text, str) and r.text.strip()


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_end_to_end_refine_real():
    from aisearch.config import Config
    from aisearch.clients import ClaudeClient
    from aisearch.judge import LLMJudge
    from aisearch.refine import refine

    client = ClaudeClient()
    judge = LLMJudge(client, votes=1)
    cfg = Config(model="claude-opus-4-8", council_size=2, max_iters=1, judge_votes=1)
    res = refine("Write a one-line haiku about loops.", cfg, client, judge)
    assert res.best_artifact
