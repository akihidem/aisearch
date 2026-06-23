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
