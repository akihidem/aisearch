"""Run the Q3 mixed-council precision probe.

    python3 -m experiments.q3_council_precision.run
"""

from __future__ import annotations

import json
import os
import shutil

from aisearch.clients import ClaudeCliClient

# Resolve `claude` robustly: background shells often lack ~/.npm-global/bin on PATH.
_CLAUDE = (shutil.which("claude")
           or os.path.expanduser("~/.npm-global/bin/claude"))

REVIEWERS = ["haiku", "sonnet", "opus"]   # capability gradient (soshiki: stronger over-flags)

# (label, code, has_flaw). FLAWLESS = correct FOR VALID INPUTS; a YES is a false
# positive (over-flag). These are edge-case-tempting (no guard for invalid inputs)
# to induce the soshiki precision failure (strong reviewers flag non-bugs).
ARTIFACTS = [
    ("divide", "def divide(a, b):\n    return a / b", False),
    ("first", "def first(xs):\n    return xs[0]", False),
    ("get_item", "def get_item(d, k):\n    return d[k]", False),
    ("sqrt", "def sqrt(x):\n    return x ** 0.5", False),
    ("to_int", "def to_int(s):\n    return int(s)", False),
    # flawed (recall check): a YES is correct — clear logic bug for valid inputs
    ("is_even_bug", "def is_even(n):\n    return n % 2 == 1", True),
    ("add_bug", "def add(a, b):\n    return a - b", True),
    ("avg_bug", "def average(xs):\n    return sum(xs) / (len(xs) - 1)", True),
]

# Strict reviewer, NO protective clause -> induces over-flagging (the soshiki
# precision-failure regime). Structured VERDICT line so even verbose reviewers
# (opus) are parseable.
PROMPT = ("Carefully review this Python function for any bug, issue, or problem. "
          "Be thorough. Finish your reply with a final line in EXACTLY this format:\n"
          "VERDICT: BUG   (if it has any problem at all)\n"
          "VERDICT: OK    (if it is completely fine)\n\n")


def _flag(text: str) -> bool:
    """Flag = reviewer says BUG. Read the structured VERDICT line (last wins)."""
    verdict = None
    for line in text.strip().splitlines():
        u = line.strip().upper()
        if u.startswith("VERDICT:") or "VERDICT:" in u:
            if "BUG" in u:
                verdict = True
            elif "OK" in u:
                verdict = False
    if verdict is not None:
        return verdict
    # fallback if the model ignored the format
    t = text.lower()
    return ("bug" in t or "issue" in t or "problem" in t) and "no bug" not in t


def main() -> int:
    clients = {m: ClaudeCliClient(model=m, command=_CLAUDE) for m in REVIEWERS}
    # flags[model][artifact_label] = bool(flagged)
    flags = {m: {} for m in REVIEWERS}
    raw = {m: {} for m in REVIEWERS}
    for label, code, _ in ARTIFACTS:
        for m in REVIEWERS:
            resp = clients[m].complete(PROMPT + code, temperature=0.0)
            flags[m][label] = _flag(resp.text)
            raw[m][label] = resp.text.strip().replace("\n", " ")[:70]

    flawless = [a for a in ARTIFACTS if not a[2]]
    flawed = [a for a in ARTIFACTS if a[2]]

    def fp(model_flags):   # false-positive rate on flawless
        return round(sum(model_flags[l] for l, _, _ in flawless) / len(flawless), 3)

    def recall(model_flags):
        return round(sum(model_flags[l] for l, _, _ in flawed) / len(flawed), 3)

    # council = majority vote across the 3 reviewers, per artifact
    council = {}
    for label, _, _ in ARTIFACTS:
        votes = sum(flags[m][label] for m in REVIEWERS)
        council[label] = votes >= 2

    # correlation of over-flags: how many flawless artifacts flagged by >=2 reviewers
    correlated = sum(1 for l, _, _ in flawless
                     if sum(flags[m][l] for m in REVIEWERS) >= 2)

    per_model = {m: {"fp_overflag": fp(flags[m]), "recall": recall(flags[m])}
                 for m in REVIEWERS}
    council_res = {"fp_overflag": fp(council), "recall": recall(council)}
    worst_fp = max(per_model[m]["fp_overflag"] for m in REVIEWERS)

    verdict = ("council REDUCES over-flag FP (precision saved)"
               if council_res["fp_overflag"] < worst_fp - 1e-9 else
               "council does NOT reduce FP (over-flaggers carry majority / correlated)")

    print(json.dumps({
        "reviewers": REVIEWERS,
        "per_model": per_model,
        "council_majority": council_res,
        "worst_single_fp": worst_fp,
        "flawless_flagged_by_2plus (correlated over-flags)": correlated,
        "n_flawless": len(flawless),
        "verdict": verdict,
        "raw_samples": raw,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
