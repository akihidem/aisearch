"""Run the Q6 Goodhart sweep over aisearch's real meta-search.

    python3 -m experiments.q6_goodhart.run
"""

from __future__ import annotations

import hashlib
import json

from aisearch.config import Config, SearchSpace
from aisearch.search import search


# ---- deterministic "world": Config -> latent (correctness c, verbosity v) ---- #
def _skill(model: str) -> float:
    h = int(hashlib.sha256(model.encode()).hexdigest(), 16) % 100
    return 0.45 + 0.50 * (h / 99.0)          # per-model competence in [0.45, 0.95]


def latent(cfg: Config) -> tuple[float, float]:
    skill = _skill(cfg.model)
    review = min(1.0, 0.45 + 0.10 * cfg.council_size + 0.04 * len(cfg.roles))
    # correctness: best near temperature 0.3, high temp HURTS accuracy
    c = skill * review * (1.0 - 0.55 * abs(cfg.temperature - 0.3))
    c = max(0.0, min(1.0, c))
    # verbosity: RISES with temperature (and a bit with council size); gameable, skill-free
    v = max(0.0, min(1.0, 0.20 + 0.55 * cfg.temperature + 0.06 * cfg.council_size))
    return c, v


def make_evaluator(beta: float):
    """Evaluator(cfg) -> (artifact, score). score = soft judge = (1-β)c + βv."""
    def ev(cfg: Config):
        c, v = latent(cfg)
        score = (1.0 - beta) * c + beta * v
        return (f"c={c:.4f} v={v:.4f}", round(score, 6))
    return ev


def true_quality(cfg: Config) -> float:
    return latent(cfg)[0]


def _spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(a):
        order = sorted(range(len(a)), key=lambda i: a[i])
        r = [0.0] * len(a)
        i = 0
        while i < len(a):                     # average ties
            j = i
            while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    vy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return round(cov / (vx * vy), 3) if vx and vy else 0.0


def run(betas=(0.0, 0.2, 0.4, 0.6, 0.8), generations=8, pop_size=10, seed=0) -> dict:
    space = SearchSpace()
    # true optimum (β=0 IS the true objective, but compute over a fixed broad pool too)
    true_opt = search("task", space, make_evaluator(0.0),
                      generations=generations, pop_size=pop_size, seed=seed)
    true_best_q = round(true_quality(true_opt.best_config), 4)

    rows = []
    for b in betas:
        ev = make_evaluator(b)
        res = search("task", space, ev, generations=generations, pop_size=pop_size, seed=seed)
        # rank-corr between the soft score and TRUE quality over a fresh broad sample
        import random
        rng = random.Random(seed + 1)
        pool = [space.sample(rng) for _ in range(60)]
        soft_scores = [ev(c)[1] for c in pool]
        true_scores = [true_quality(c) for c in pool]
        rows.append({
            "beta": b,
            "soft_best_true_quality": round(true_quality(res.best_config), 4),
            "true_quality_gap_vs_true_opt": round(true_best_q - true_quality(res.best_config), 4),
            "soft_best_temperature": res.best_config.temperature,
            "rank_corr_soft_vs_true": _spearman(soft_scores, true_scores),
        })
    return {"true_opt_quality": true_best_q,
            "true_opt_temperature": true_opt.best_config.temperature,
            "sweep": rows}


def main() -> int:
    print(json.dumps(run(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
