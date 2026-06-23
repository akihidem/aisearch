"""Config と探索空間、コスト計上、seed ユーティリティ。"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Sequence


class ConfigError(ValueError):
    pass


# 探索空間の既定値（実モデル名は本番アダプタが解決する）
DEFAULT_MODELS = ("claude-opus-4-8", "claude-sonnet-4-6", "gemma4:latest", "mlx-local")
DEFAULT_ROLES = ("generalist", "critic", "domain-expert", "contrarian")


@dataclass(frozen=True)
class Config:
    """合議＋自己改善 1 個体の設定。frozen=決定的探索のため不変。"""

    model: str
    temperature: float = 0.7
    role: str = "generalist"  # roles 未指定時の単一役割（後方互換のフォールバック）
    roles: tuple[str, ...] = ()  # 合議の役割ロスター（proposer に巡回割当）
    council_size: int = 3
    budget: int = 100_000  # トークン予算
    max_iters: int = 3  # refine の最大反復
    judge_votes: int = 3
    seed: int = 0

    def __post_init__(self):
        if not self.model:
            raise ConfigError("model must be non-empty")
        if not (0.0 <= self.temperature <= 2.0):
            raise ConfigError(f"temperature out of range [0,2]: {self.temperature}")
        if self.council_size < 1:
            raise ConfigError(f"council_size must be >= 1: {self.council_size}")
        if self.budget <= 0:
            raise ConfigError(f"budget must be > 0: {self.budget}")
        if self.max_iters < 0:
            raise ConfigError(f"max_iters must be >= 0: {self.max_iters}")
        if self.judge_votes < 1:
            raise ConfigError(f"judge_votes must be >= 1: {self.judge_votes}")
        # roles は frozen のため object.__setattr__ で tuple 化（list 受け取りも許容）
        object.__setattr__(self, "roles", tuple(self.roles))
        if any(not isinstance(r, str) or not r for r in self.roles):
            raise ConfigError(f"roles must be non-empty strings: {self.roles!r}")

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "role": self.role,
            "roles": list(self.roles),
            "council_size": self.council_size,
            "budget": self.budget,
            "max_iters": self.max_iters,
            "judge_votes": self.judge_votes,
            "seed": self.seed,
        }


@dataclass
class SearchSpace:
    """探索する設定空間。sample/mutate/crossover は常に valid な Config を返す。"""

    models: Sequence[str] = DEFAULT_MODELS
    roles: Sequence[str] = DEFAULT_ROLES
    temperature_choices: Sequence[float] = (0.0, 0.3, 0.7, 1.0)
    council_sizes: Sequence[int] = (1, 2, 3, 5)

    def _sample_roster(self, rng: random.Random) -> tuple[str, ...]:
        """1..len(roles) 個の重複なし役割ロスター（順序もランダム）。"""
        k = rng.randint(1, len(self.roles))
        return tuple(rng.sample(list(self.roles), k))

    def _council_size_for(self, roster_len: int, rng: random.Random) -> int:
        """roster 長 L 以上の council_size を選ぶ（不変条件 C>=L の連動の要）。

        council_sizes から L 以上の候補を抽選。候補が皆無なら不変条件を最優先し
        roster_len 自体を返す（C>=L を必ず保証。set 内維持より不変条件が上位契約）。
        roster_len==0（単一role の roles=() 互換経路）は全候補が有効。
        """
        valid = [c for c in self.council_sizes if c >= roster_len]
        return rng.choice(valid) if valid else roster_len

    def sample(self, rng: random.Random) -> Config:
        roster = self._sample_roster(rng)
        return Config(
            model=rng.choice(list(self.models)),
            temperature=rng.choice(list(self.temperature_choices)),
            role=roster[0],
            roles=roster,
            council_size=self._council_size_for(len(roster), rng),
            seed=rng.randrange(1_000_000),
        )

    def mutate(self, cfg: Config, rng: random.Random) -> Config:
        gene = rng.choice(["model", "temperature", "roles", "council_size"])
        if gene == "model":
            return replace(cfg, model=rng.choice(list(self.models)))
        if gene == "temperature":
            return replace(cfg, temperature=rng.choice(list(self.temperature_choices)))
        if gene == "roles":
            roster = self._sample_roster(rng)
            # 既存 council_size が連動条件を満たせば温存（単一遺伝子変異の精神）、
            # 満たさなければ最小限の修復で C>=L を回復。
            csize = (
                cfg.council_size
                if cfg.council_size >= len(roster)
                else self._council_size_for(len(roster), rng)
            )
            return replace(cfg, roles=roster, role=roster[0], council_size=csize)
        # council_size 変異も既存 roster 長以上にクランプ（roles=() は L=0 で無制約）。
        return replace(cfg, council_size=self._council_size_for(len(cfg.roles), rng))

    def crossover(self, a: Config, b: Config, rng: random.Random) -> Config:
        roster = rng.choice([a.roles, b.roles]) or (a.role,)
        # 親の council_size を継ぐが、合成 roster に対し C<L なら不変条件へ修復。
        csize = rng.choice([a.council_size, b.council_size])
        if csize < len(roster):
            csize = self._council_size_for(len(roster), rng)
        return Config(
            model=rng.choice([a.model, b.model]),
            temperature=rng.choice([a.temperature, b.temperature]),
            role=roster[0],
            roles=roster,
            council_size=csize,
            budget=a.budget,
            max_iters=a.max_iters,
            judge_votes=a.judge_votes,
            seed=rng.randrange(1_000_000),
        )


@dataclass
class CostTracker:
    budget: int
    spent: int = 0

    def add(self, tokens: int) -> None:
        self.spent += tokens

    @property
    def remaining(self) -> int:
        return self.budget - self.spent

    def exceeded(self) -> bool:
        return self.spent >= self.budget


def make_rng(seed: int) -> random.Random:
    """seed 固定の RNG。同一 seed → 同一系列。"""
    return random.Random(seed)
