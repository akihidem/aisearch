"""Q3 — does a mixed council offset the 'stronger reviewer over-flags' precision failure?

soshiki-genron's oversight pilot found supervision fails on PRECISION (strong
models flag non-bugs), not recall. Q3: does majority-vote council reduce the
false-positive (over-flag) rate, or do the over-flaggers carry the majority?

We give real reviewers (Claude CLI: haiku<sonnet<opus capability gradient)
FLAWLESS code (a YES = false positive) plus a few FLAWED ones (recall check), and
compare each single reviewer's FP to the council majority's FP — and whether the
over-flags are correlated (same artifact flagged by several) or independent.
relations OPEN-QUESTIONS Q3.
"""
