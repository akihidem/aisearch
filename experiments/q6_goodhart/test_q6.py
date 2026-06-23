"""Tests for the Q6 Goodhart sweep — deterministic, threshold behaviour, seed-robust."""

from __future__ import annotations

from .run import latent, make_evaluator, run, true_quality
from aisearch.config import Config


def _cfg(**kw):
    base = dict(model="m", temperature=0.3, role="r", roles=("r",), council_size=3)
    base.update(kw)
    return Config(**base)


def test_world_has_the_intended_tradeoff():
    # higher temperature -> more verbosity but less correctness
    c_lo, v_lo = latent(_cfg(temperature=0.3))
    c_hi, v_hi = latent(_cfg(temperature=1.0))
    assert v_hi > v_lo          # temp raises verbosity
    assert c_hi < c_lo          # temp hurts correctness


def test_beta0_equals_true_objective():
    r = run(betas=(0.0,), generations=6, pop_size=8, seed=0)
    row = r["sweep"][0]
    assert row["true_quality_gap_vs_true_opt"] == 0.0
    assert row["rank_corr_soft_vs_true"] == 1.0


def test_goodhart_threshold_emerges():
    r = run(betas=(0.0, 0.4, 0.8), generations=8, pop_size=10, seed=0)
    gaps = {row["beta"]: row["true_quality_gap_vs_true_opt"] for row in r["sweep"]}
    corrs = {row["beta"]: row["rank_corr_soft_vs_true"] for row in r["sweep"]}
    assert gaps[0.0] == 0.0                       # no bias -> no loss
    assert gaps[0.8] > 0.1                        # high bias -> true quality lost
    assert corrs[0.0] > corrs[0.8]               # rank-corr degrades with bias


def test_threshold_robust_across_seeds():
    # the qualitative flip (low beta safe, high beta lost) holds for several seeds
    for s in (0, 1, 7):
        r = run(betas=(0.0, 0.8), generations=8, pop_size=10, seed=s)
        lo = r["sweep"][0]["true_quality_gap_vs_true_opt"]
        hi = r["sweep"][1]["true_quality_gap_vs_true_opt"]
        assert lo == 0.0 and hi > 0.1, f"seed {s}: lo={lo} hi={hi}"
