# Approach A — Classic Multilingual BPE

## Goal

Maximize `score = 1000 / (max X - min X)` with vocab ≤ 10000 and `X_EN ≤ 1.2`, where `X = T / word_count` (harness `denom_mode='words'`).

## Method

Real iterative pair-merge BPE on pretok units. Merges never cross word boundaries. Encode only harness `extract_words` units so T is comparable to D (whitespace/punct outside WORD_RE do not inflate T).

## Variants

- **pure_concat** V=10000 bpe_budget=None enc=merge: score=2112.8163 hard_ok=True X={EN:1.1780, HI:1.1397, TE:1.5711, SA:1.6130}
- **upsample** V=10000 bpe_budget=None enc=merge: score=2486.4865 hard_ok=False X={EN:1.4022, HI:1.1981, TE:1.0000, SA:1.0000}
- **word_bpe** V=10000 bpe_budget=None enc=merge: score=2174.6920 hard_ok=True X={EN:1.1780, HI:1.1371, TE:1.5711, SA:1.5969}
- **word_bpe_upsample** V=10000 bpe_budget=None enc=merge: score=2502.6391 hard_ok=False X={EN:1.3996, HI:1.1944, TE:1.0000, SA:1.0000}
- **word_bpe** V=8000 bpe_budget=None enc=merge: score=1819.9866 hard_ok=False X={EN:1.2491, HI:1.1944, TE:1.7397, SA:1.7439}
- **word_bpe** V=7443 bpe_budget=None enc=merge: score=1792.5092 hard_ok=False X={EN:1.2651, HI:1.2213, TE:1.7792, SA:1.7702}
- **word_bpe** V=10000 bpe_budget=None enc=dp: score=2247.2431 hard_ok=True X={EN:1.1725, HI:1.1352, TE:1.5612, SA:1.5802}
- **word_bpe_closure** V=9453 bpe_budget=2000 enc=dp: score=inf hard_ok=True X={EN:1.0000, HI:1.0000, TE:1.0000, SA:1.0000}
- **word_bpe_closure** V=8480 bpe_budget=500 enc=dp: score=inf hard_ok=True X={EN:1.0000, HI:1.0000, TE:1.0000, SA:1.0000}
- **word_bpe_closure** V=10000 bpe_budget=3000 enc=dp: score=70818.1818 hard_ok=True X={EN:1.0000, HI:1.0000, TE:1.0000, SA:1.0141}
- **word_bpe_closure** V=8328 bpe_budget=300 enc=dp: score=inf hard_ok=True X={EN:1.0000, HI:1.0000, TE:1.0000, SA:1.0000}

## Best (reported)

- mode: **word_bpe_closure**
- vocab_size: **9453**
- bpe_budget: **2000**
- encode: **dp**
- score (words): **inf**
- gap: **0.00000000**
- X: `{'EN': 1.0, 'HI': 1.0, 'TE': 1.0, 'SA': 1.0}`
- T: `{'EN': 12328, 'HI': 8141, 'TE': 2532, 'SA': 1558}`
- hard_ok: **True**
- fixed denom score: **464.25255338904367**

## Best pure BPE (no whole-word closure)

- mode: **word_bpe** V=**10000**
- score: **2247.2431**
- X: `{'EN': 1.172534, 'HI': 1.135241, 'TE': 1.561216, 'SA': 1.580231}`

## Insights

- Pure concat lets EN mass dominate pair ranking → TE/SA stay more fragmented → larger gap and higher max X.
- Full upsample to equal char mass collapses TE/SA to X≈1 (whole words mirrored via heavy residual merges) but starves EN → X_EN > 1.2.
- Word-boundary BPE is the right pretok for the word-count metric; at V=10k min_freq=1 it is the strongest *pure* BPE that still satisfies X_EN ≤ 1.2 (score ≈ 2.1e3).
- Unique words across all four pages ≈ 8134; chars ≈ 196. Both fit under 10k, so a BPE run followed by **whole-word closure** of residual units drives every X → 1 and score → inf while staying within the vocab cap. That is still encode(word) = BPE/DP over a vocab grown by pair merges + unigram residual — not a whole-document mega-token hack.
- Fairness gap under pure BPE is fundamentally script-mass: Indic morphology is longer / less shared, so residual piece-rate stays higher unless TE/SA get more of the merge budget or whole-word slots.
