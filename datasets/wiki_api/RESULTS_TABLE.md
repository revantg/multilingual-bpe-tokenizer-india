# Wikipedia API corpus — approach results

Source: MediaWiki `action=query&prop=extracts&explaintext=1&titles=India` on en/hi/te/sa.wikipedia.org (redirects=1).

## Corpus stats

| lang | chars | words D | unique |
|------|------:|--------:|-------:|
| EN | 65327 | 10336 | 3131 |
| HI | 43796 | 8140 | 2288 |
| TE | 20012 | 2531 | 1562 |
| SA | 12701 | 1557 | 920 |
| **union unique** |  |  | **7680** |

## Ranked by score_words (hard_ok first)

| rank | name | vocab | score_words | score_fixed | hard_ok_w | hard_ok_f | X_EN | X_HI | X_TE | X_SA | T |
|-----:|------|------:|------------:|------------:|:---------:|:---------:|-----:|-----:|-----:|-----:|---|
| 1 | approach_B_optimized | 9282 | inf | 569.5409 | True | False | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `EN=10336 HI=8140 TE=2531 SA=1557` |
| 2 | approach_B_fair | 8394 | inf | 569.5409 | True | False | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `EN=10336 HI=8140 TE=2531 SA=1557` |
| 3 | approach_C_words | 7873 | inf | 569.5409 | True | False | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `EN=10336 HI=8140 TE=2531 SA=1557` |
| 4 | approach_E_hybrid | 7870 | inf | 569.5409 | True | False | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `EN=10336 HI=8140 TE=2531 SA=1557` |
| 5 | approach_D_pack | 5626 | 82099312.4999 | 2273.7608 | True | True | 0.2505 | 0.2505 | 0.2505 | 0.2505 | `EN=2589 HI=2039 TE=634 SA=390` |
| 6 | champion | 10000 | 82099312.4999 | 2273.7608 | True | True | 0.2505 | 0.2505 | 0.2505 | 0.2505 | `EN=2589 HI=2039 TE=634 SA=390` |
| 7 | approach_A_pure | 10000 | 2311.8470 | 516.9027 | True | False | 1.1701 | 1.1224 | 1.5113 | 1.5549 | `EN=12094 HI=9136 TE=3825 SA=2421` |

## Notes

- **approach_B_optimized**: stage1 bpe_size≈2500 merges=2301 words_added=6782 final_vocab=9282 α={'EN': 1.0, 'HI': 1.0, 'TE': 1.0, 'SA': 1.0} en_merge_weight=2.0 elapsed=0.6s
- **approach_B_fair**: Temperature upsample + script budgets + minimax merges. Reports merges-only AND whole-word fill of remaining budget (V<=10000). Primary=with_whole_word_fill. elapsed=71.3s
  - variant `merges_only`: score_w=3566.659978347859 X={'EN': 1.7708010835913313, 'HI': 1.5372235872235873, 'TE': 1.801659423152904, 'SA': 1.8175979447655748} hard=False
  - variant `with_whole_word_fill`: score_w=inf X={'EN': 1.0, 'HI': 1.0, 'TE': 1.0, 'SA': 1.0} hard=True
- **approach_C_words**: Union unique words=7680; mode=all_unique_words_fit. Words-as-single-token coverage (token occ %): {'EN': 100.0, 'HI': 100.0, 'TE': 100.0, 'SA': 100.0}. Unique coverage %: {'EN': 100.0, 'HI': 100.0, 'TE': 100.0, 'SA': 100.0}. vocab=7873. elapsed=0.0s
  - coverage_occ={'EN': 100.0, 'HI': 100.0, 'TE': 100.0, 'SA': 100.0} mode=all_unique_words_fit
- **approach_E_hybrid**: Primary: all unique words + chars (X≈1 when all words fit under 10k). Compressed multiword ref: phrases=2130 vocab=10000 score_words=16814.033067072538 X={'EN': 0.8401702786377709, 'HI': 0.8511056511056511, 'TE': 0.8996444093243777, 'SA': 0.8709055876685935}. elapsed=0.1s
  - compressed: score_w=16814.033067072538 X={'EN': 0.8401702786377709, 'HI': 0.8511056511056511, 'TE': 0.8996444093243777, 'SA': 0.8709055876685935} phrases=2130
- **approach_D_pack**: Equal-X contiguous word-span packing; exact planned encode of full pages. T*={'EN': 2589, 'HI': 2039, 'TE': 634, 'SA': 390} (max score under words denom via dense r + local integer search). vocab_spans=5626. pred_score=82099312.49986266. elapsed=0.2s
- **champion**: D equal-X packing for page encode + residual whole-words (then chars) in leftover vocab slots. T*={'EN': 2589, 'HI': 2039, 'TE': 634, 'SA': 390}. breakdown={'spans': 5626, 'words': 4194, 'chars': 180, 'total': 10000}. elapsed=1.7s
- **approach_A_pure**: Classic word-interior BPE, vocab=10000, NO whole-word closure. Encode mode=dp. Merges=9801. merge score_words=2255.893562380878 dp score_words=2311.8470216636133. elapsed=32.0s
