"""LLMClient プロトコルと実アダプタ + 決定的 FakeLLM。

実アダプタ(Claude/Ollama/MLX)は遅延 import で外部依存を読み込むため、
ネットワークや SDK が無い環境でも *インスタンス化と契約検査* は可能。
実呼び出しは L0 では行わない（tests/test_integration.py を参照）。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class LLMClient(Protocol):
    """全アダプタ共通インターフェース。

    seed は決定性のために受け取るが、実API(Claude等)が無視する場合がある。
    """

    model: str

    def complete(
        self, prompt: str, *, temperature: float = 0.7, seed: int | None = None
    ) -> LLMResponse: ...


def _count_tokens(text: str) -> int:
    """決定的・近似のトークン数（実トークナイザ非依存）。"""
    return max(1, len(text.split()))


class FakeLLM:
    """決定的なテスト用クライアント。

    - responder(prompt, call_index) -> text を注入できる。
    - 既定は prompt から決定的に派生したテキスト（同一 prompt → 同一出力）。
    - fail_on(prompt) が True を返すと RuntimeError を送出（障害テスト用）。
    """

    def __init__(
        self,
        model: str = "fake",
        *,
        responder: Callable[[str, int], str] | None = None,
        fail_on: Callable[[str], bool] | None = None,
    ):
        self.model = model
        self._responder = responder
        self._fail_on = fail_on
        self.calls: list[str] = []

    def complete(
        self, prompt: str, *, temperature: float = 0.7, seed: int | None = None
    ) -> LLMResponse:
        idx = len(self.calls)
        self.calls.append(prompt)
        if self._fail_on is not None and self._fail_on(prompt):
            raise RuntimeError(f"FakeLLM induced failure (call {idx})")
        if self._responder is not None:
            text = self._responder(prompt, idx)
        else:
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
            text = f"[{self.model}] {digest}"
        return LLMResponse(
            text=text,
            prompt_tokens=_count_tokens(prompt),
            completion_tokens=_count_tokens(text),
            model=self.model,
        )


class ClaudeClient:
    """Anthropic Claude アダプタ（本番経路・L0では未使用）。

    Claude API に seed パラメタは無いため seed は無視する（temperature のみ反映）。
    """

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        *,
        max_tokens: int = 2048,
        api_key: str | None = None,
    ):
        self.model = model
        self._max_tokens = max_tokens
        self._api_key = api_key

    def complete(
        self, prompt: str, *, temperature: float = 0.7, seed: int | None = None
    ) -> LLMResponse:
        from anthropic import Anthropic  # 遅延 import（L0では未到達）

        client = Anthropic(api_key=self._api_key) if self._api_key else Anthropic()
        msg = client.messages.create(
            model=self.model,
            max_tokens=self._max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            getattr(b, "text", "") for b in msg.content
            if getattr(b, "type", "") == "text"
        )
        usage = msg.usage
        return LLMResponse(
            text=text,
            prompt_tokens=getattr(usage, "input_tokens", _count_tokens(prompt)),
            completion_tokens=getattr(usage, "output_tokens", _count_tokens(text)),
            model=self.model,
        )


class OllamaClient:
    """ローカル Ollama アダプタ（本番経路・L0では未使用）。"""

    def __init__(self, model: str = "gemma4:latest", *, host: str = "http://localhost:11434"):
        self.model = model
        self._host = host

    def complete(
        self, prompt: str, *, temperature: float = 0.7, seed: int | None = None
    ) -> LLMResponse:
        import json as _json
        import urllib.request

        options: dict = {"temperature": temperature}
        if seed is not None:
            options["seed"] = seed
        body = {"model": self.model, "prompt": prompt, "stream": False, "options": options}
        req = urllib.request.Request(
            f"{self._host}/api/generate",
            data=_json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:  # noqa: S310 ローカルホスト固定用途
            data = _json.loads(resp.read().decode("utf-8"))
        text = data.get("response", "")
        return LLMResponse(
            text=text,
            prompt_tokens=data.get("prompt_eval_count", _count_tokens(prompt)),
            completion_tokens=data.get("eval_count", _count_tokens(text)),
            model=self.model,
        )


class MLXClient:
    """Mac Studio の MLX サーバ アダプタ（本番経路・L0では未使用）。

    mlx_lm.server の OpenAI 互換 /v1/completions を叩く。
    """

    def __init__(self, model: str = "mlx-local", *, host: str = "http://ssms:8080"):
        self.model = model
        self._host = host

    def complete(
        self, prompt: str, *, temperature: float = 0.7, seed: int | None = None
    ) -> LLMResponse:
        import json as _json
        import urllib.request

        body = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": 2048,
        }
        req = urllib.request.Request(
            f"{self._host}/v1/completions",
            data=_json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            data = _json.loads(resp.read().decode("utf-8"))
        choice = data["choices"][0]
        text = choice.get("text", "")
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", _count_tokens(prompt)),
            completion_tokens=usage.get("completion_tokens", _count_tokens(text)),
            model=self.model,
        )


class ClaudeCliClient:
    """ローカルの `claude` CLI を headless(-p) で叩くアダプタ（OAuth 認証・APIキー不要）。

    Claude Code のサブスク認証をそのまま使うので ANTHROPIC_API_KEY は不要。
    `claude -p --output-format json --model <model> <prompt>` を実行し、JSON の
    `result`(本文) と `usage`(tokens)、`total_cost_usd`(累計コスト) を取り出す。

    - CLI に温度/seed のフラグは無いため temperature/seed は無視する。
    - runner((argv) -> stdout) を注入すれば subprocess 無しで決定的にテストできる。
    """

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        *,
        command: str = "claude",
        timeout: int = 180,
        extra_args: list[str] | None = None,
        runner: Callable[[list[str]], str] | None = None,
    ):
        self.model = model
        self._command = command
        self._timeout = timeout
        self._extra_args = list(extra_args or [])
        self._runner = runner
        self.total_cost_usd = 0.0  # 呼び出しごとに累積

    def _build_args(self) -> list[str]:
        return [
            self._command,
            "-p",
            "--output-format",
            "json",
            "--model",
            self.model,
            *self._extra_args,
        ]

    def _invoke(self, argv: list[str]) -> str:
        if self._runner is not None:
            return self._runner(argv)
        import subprocess

        proc = subprocess.run(argv, capture_output=True, text=True, timeout=self._timeout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"claude CLI failed (rc={proc.returncode}): {(proc.stderr or proc.stdout)[:300]}"
            )
        return proc.stdout

    def complete(
        self, prompt: str, *, temperature: float = 0.7, seed: int | None = None
    ) -> LLMResponse:
        import json as _json

        argv = self._build_args() + [prompt]
        raw = self._invoke(argv)
        try:
            data = _json.loads(raw)
        except _json.JSONDecodeError as e:
            raise RuntimeError(f"claude CLI returned non-JSON output: {raw[:200]!r}") from e
        if data.get("is_error"):
            raise RuntimeError(f"claude CLI error: {data.get('result') or data}")
        text = (data.get("result") or "").strip()
        self.total_cost_usd += float(data.get("total_cost_usd") or 0.0)
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            prompt_tokens=usage.get("input_tokens", _count_tokens(prompt)),
            completion_tokens=usage.get("output_tokens", _count_tokens(text)),
            model=self.model,
        )


def make_tui_runner(
    *,
    script: str | None = None,
    cwd: str | None = None,
    timeout: int = 300,
    subprocess_run: Callable[[list[str], int], object] | None = None,
) -> Callable[[list[str]], str]:
    """ClaudeCliClient 用 runner: `claude -p` の代わりに claude-cli-run.py(対話TUIラッパ)
    を呼び、その plain-text 応答を ClaudeCliClient が期待する JSON エンベロープに包む。

    対話TUI(entrypoint=cli)経由なので Agent SDK クレジット枠を食わず通常のサブスク枠から
    消費される（理由は ~/.claude/scripts/claude-cli-run.py の docstring / memory 参照）。
    TUI ラッパは本文テキストのみ返すため usage/cost は空。`-p`/`--output-format json` は
    使わず、ラッパ独自の `--model/--cwd/--timeout` を渡す。

    subprocess_run((cmd, timeout) -> proc) を注入すれば tmux 無しで決定的にテストできる。
    """
    import json as _json
    import os
    import re
    from pathlib import Path

    runner_script = script or os.environ.get("CLAUDE_CLI_RUN") or str(
        Path.home() / ".claude" / "scripts" / "claude-cli-run.py"
    )
    # claude-cli-run.py は完了検知に CCRUN_DONE_<hex> sentinel を使う。モデルが
    # markdown 装飾付き(🎯 **CCRUN_DONE_..**)で書くとラッパの strip をすり抜けて
    # 本文に漏れることがある → 防御的に除去（前後の装飾・空白ごと）。
    _sentinel_re = re.compile(r"[`*🎯\s]*CCRUN_DONE_[0-9a-fA-F]+[`*\s]*")

    def _default_run(cmd: list[str], to: int):
        import subprocess

        return subprocess.run(cmd, capture_output=True, text=True, timeout=to)

    _run = subprocess_run or _default_run

    def runner(argv: list[str]) -> str:
        # argv = ["claude","-p","--output-format","json","--model",M, ..., prompt]
        model = argv[argv.index("--model") + 1] if "--model" in argv else None
        prompt = argv[-1]
        cmd = [runner_script]
        if model:
            cmd += ["--model", model]
        if cwd:
            cmd += ["--cwd", cwd]
        cmd += ["--timeout", str(timeout), prompt]
        proc = _run(cmd, timeout + 60)
        if getattr(proc, "returncode", 1) != 0:
            err = (getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "")[:500]
            return _json.dumps({"is_error": True, "result": f"claude-cli-run failed: {err}"})
        text = _sentinel_re.sub("", proc.stdout or "").strip()
        return _json.dumps({"is_error": False, "result": text, "usage": {}})

    return runner
