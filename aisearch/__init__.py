"""aisearch: 複数LLMの合議生成 × 再帰的自己改善 × メタ探索。

設計の要:
- 実LLM/実Judge は本番経路（clients の各アダプタ, judge.LLMJudge）。
- L0(決定的テスト)は Fake クライアント/ジャッジを *注入* して回す（API非依存・無課金・決定的）。
- 実API挙動は tests/test_integration.py (pytest -m integration) で手動確認、L0 には含めない。

再エクスポートは遅延（PEP 562 __getattr__）にしてある。`python -m aisearch.search`
実行時に submodule を先読みして出る RuntimeWarning を避けるため。
"""
import importlib

_EXPORTS = {
    "LLMClient": "clients", "LLMResponse": "clients", "FakeLLM": "clients",
    "ClaudeClient": "clients", "OllamaClient": "clients", "MLXClient": "clients",
    "Judge": "judge", "Judgement": "judge", "FakeJudge": "judge", "LLMJudge": "judge",
    "Config": "config", "SearchSpace": "config", "CostTracker": "config",
    "make_rng": "config", "ConfigError": "config",
    "generate": "council", "CouncilResult": "council", "CouncilError": "council",
    "refine": "refine", "RefineResult": "refine", "RefineStep": "refine",
    "search": "search", "SearchResult": "search", "make_refine_evaluator": "search",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    mod_name = _EXPORTS.get(name)
    if mod_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(f".{mod_name}", __name__)
    return getattr(mod, name)


def __dir__():
    return sorted(__all__)
