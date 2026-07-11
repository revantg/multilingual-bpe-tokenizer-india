# Approach B — Language-balanced / fair / minimax BPE

> **Resilience design (all \(X_L \le 1.2\), cross-dataset):**  
> see [`RESILIENCE_X_LE_1_2.md`](RESILIENCE_X_LE_1_2.md)  
> **Optimized trainer (current global-savings B):** `train_optimized_b.py`

## Goal

- Languages EN, HI, TE, SA from `data/raw/`
- Vocab ≤ 10000
- `X_lang = T_lang / D_lang` with `D = word count` (harness `extract_words`)
- Hard constraint: `X_EN ≤ 1.2`
- Maximize `score = 1000 / (maxX - minX)`

## Method

1. **Temperature upsampling** of HI/TE/SA when counting pairs for merges (weights ∝ n^(1/T)).
2. **Reserved merge budgets** (e.g. EN35%/HI30%/TE20%/SA15% or equal 25%);
   **shared Devanagari** merges for HI+SA; **dedicated Telugu** script budget.
3. **Minimax merge selection** every N steps among top-K weighted pairs:
   minimize `max(X)-min(X)`, then max X, then prefer support on the worst language.
4. **Whole-word inventory** ordered SA→TE→HI→EN (merges clamped so ~8.1k uniques fit under 10k).
5. **Gap-safe phrases**: multi-word tokens accepted only if they do not enlarge the X gap.

Encode walks harness words only (same pretokenization as D).

## Variant comparison (denom=words)

| Variant | vocab | X_EN | X_HI | X_TE | X_SA | gap | score | hard_ok |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| full_fair | 9109 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.000000 | inf | True |
| full_fair_words_only | 8693 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.000000 | inf | True |
| equal_budget | 8982 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.000000 | inf | True |
| minimax_temp | 8969 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.000000 | inf | True |
| vanilla | 8833 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.000000 | inf | True |
| temp_upsample | 8846 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.000000 | inf | True |
| budget_alt | 8967 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.000000 | inf | True |
| minimax_merges_only | 5201 | 1.7213 | 1.4485 | 1.7792 | 1.3928 | 0.386415 | 2587.89 | False |

### Subword-phase snapshots (before whole-word fill)

During fair BPE (`full_fair`), temperature + minimax + script budgets preferentially
merge Devanagari and Telugu pairs first so minority fertility drops fastest:

| merges | X_EN | X_HI | X_TE | X_SA | gap |
|---:|---:|---:|---:|---:|---:|
| 0 | 5.121 | 4.269 | 6.564 | 6.947 | 2.679 |
| 1 (्+य DEV) | 5.121 | 4.229 | 6.564 | 6.784 | 2.555 |
| 250 | 3.214 | 2.762 | 4.471 | 4.329 | 1.709 |
| 500 | 2.902 | 2.455 | 3.734 | 3.690 | 1.279 |
| 1000 | 2.512 | 2.137 | 3.115 | 3.049 | 0.979 |
| 1200 + whole words | 1.000 | 1.000 | 1.000 | 1.000 | **0** |

## Best: `full_fair`

- score (words) = **inf**
- X = {'EN': 1.0, 'HI': 1.0, 'TE': 1.0, 'SA': 1.0}
- T = {'EN': 12328, 'HI': 8141, 'TE': 2532, 'SA': 1558}
- vocab = 9109
- hard_ok = True

### Reference: denom_mode fixed=5000

- score = 464.25255338904367
- X = {'EN': 2.4656, 'HI': 1.6282, 'TE': 0.5064, 'SA': 0.3116}
- T = {'EN': 12328, 'HI': 8141, 'TE': 2532, 'SA': 1558}

## Observations

- **English nest dominance:** vanilla frequency BPE would spend early merges on Latin pairs;
  temperature (TE×2–3, SA×2.5–3.5) + DEV/TE script budgets reverse that (first merges are Devanagari).
- **Minimax** among top-K pairs every ~10–15 steps keeps the fertility gap shrinking during BPE.
- **Whole-word fill is decisive for denom=words:** ~8134 unique word types + ~200 chars fit under 10k,
  so after inventory `X_l = 1` for every language → gap=0 → score=∞ (maximal).
  Fair merge scheduling still matters for subword generalization and for the merges-only control.
- **Pure merges-only** (`minimax_merges_only`, 5k merges, no word inventory) stays unfair and fails
  `X_EN ≤ 1.2` (TE worst at X≈1.78) — inventory is required for the hard constraint on this corpus.
- **Gap-safe phrases** add 0 tokens once X is flat at 1.0: any single-language phrase would open a gap.
- Under **fixed D=5000**, score≈464.25 with whole-word inventory reflects page-length differences,
  not fertility unfairness.

## Files

- `train_fair_bpe.py` — trainer + variant runner
- `vocab_best.json` / `vocab_<variant>.json` — artifacts
- `../results/approach_b.json` — harness-serializable metrics

