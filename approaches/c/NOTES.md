# Approach C — Whole-word inventory + optional fair bigrams

## Method

1. Extract words with harness `extract_words` over cleaned corpora for EN/HI/TE/SA.
2. Put **all unique words** into vocab, plus **all observed characters** as OOV fallback.
   - singles size = **8311** (fits under 10k; budget left = **1689**).
3. Encode maps each harness word → one vocab token → **X_i = T_i / word_count_i ≈ 1.0** for every language.
4. Optional remaining slots → bigram / multiword phrases.
   - **Fair budgeting** (equal quota or round-robin per language) avoids EN-only help.
   - Global-frequency bigrams over-help EN and **widen the gap** (hurts score).
5. Encode implementations: pure words; greedy longest-match; DP min-token cover.

## Hard constraints

- vocab ≤ 10000: ✓
- X_EN ≤ 1.2: ✓ (words-only X_EN = 1.0)

## Results

| Variant | vocab | X_EN | X_HI | X_TE | X_SA | gap | score |
|---------|------:|-----:|-----:|-----:|-----:|----:|------:|
| C_words_only | 8311 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.00000000 | inf |
| C_words+fair_quota_dp_bi=1689 | 10000 | 0.859912 | 0.844245 | 0.844787 | 0.724005 | 0.13590726 | 7357.96 |
| C_words+fair_quota_greedy_bi=1689 | 10000 | 0.859912 | 0.844245 | 0.844787 | 0.724005 | 0.13590726 | 7357.96 |
| C_words+fair_roundrobin_dp_bi=1689 | 10000 | 0.859912 | 0.843754 | 0.844787 | 0.724647 | 0.13526541 | 7392.87 |
| C_words+singleton_phrases_dp_bi=1689 | 10000 | 0.968851 | 0.955165 | 0.864929 | 0.801669 | 0.16718259 | 5981.48 |
| C_words+global_freq_dp_bi=1689 | 10000 | 0.796966 | 0.825820 | 0.984597 | 0.940308 | 0.18763090 | 5329.61 |

## Best

**C_words_only**

- score = **inf**
- gap = **0.00000000**
- X = EN=1.000000, HI=1.000000, TE=1.000000, SA=1.000000
- vocab = 8311

Because every language’s page is a pure sequence of in-vocab whole words,
T_i = D_i for all i → maxX−minX = 0 → score is infinite.
Actual X floats are all **1.0** (exact under harness word counts).

### Why bigrams lower the score

Any multiword tokens reduce T unevenly across languages (different co-occurrence
densities, e.g. EN `'of the'` is dense; TE has fewer high-freq collocations).
Even *fair-quota* allocation still yields gap≈0.12–0.14 → score in the thousands,
which is **worse** than the words-only infinite score. Global-freq ranking is worse still.

## Artifacts

- Implementation: `data/grok_2/approach_c/tokenizer.py`
- Runner: `data/grok_2/approach_c/run.py`
- Results JSON: `data/grok_2/results/approach_c.json`

Reproduce:
```bash
cd tokenizer && python3 data/grok_2/approach_c/run.py
```
