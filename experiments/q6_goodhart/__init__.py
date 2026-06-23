"""Q6 — does meta-search over a soft LLM-judge fitness improve TRUE quality, or Goodhart it?

Uses aisearch's real `search()` / `SearchSpace` / `Config` (public API only),
fully deterministic (no LLM, no ollama). A Config maps to two latent features:
correctness c (TRUE quality) and verbosity v (a gameable feature LLM-judges
over-reward). The soft judge scores (1-β)·c + β·v. Sweeping β answers Q6: at
what judge-bias does the meta-search's chosen config stop being truly good?
relations OPEN-QUESTIONS Q6.
"""
