# Full-page MD retrain results

Corpus: `full_page_md/india_{en,hi,te,sa}.md` (harness-cleaned).

Denoms:
- **words**: `X_i = T_i / word_count_i`
- **fixed**: `X_i = T_i / 5000`
- **score** = `1000 / (maxX − minX)` (inf if gap≈0)
- **hard_ok**: `X_EN ≤ 1.2` and `vocab ≤ 10000`

## Ranked by score_words

| rank | name | vocab | score_words | gap_words | hard_ok_w | X_words | T |
|-----:|------|------:|------------:|----------:|:---------:|---------|---|
| 1 | approach_D_pack | 8076 | 84975142.8573 | 1.1768147e-05 | True | `EN=0.160610 HI=0.160616 TE=0.160622 SA=0.160612` | `EN=4950 HI=2190 TE=651 SA=336` |
| 2 | champion | 10000 | 84975142.8573 | 1.1768147e-05 | True | `EN=0.160610 HI=0.160616 TE=0.160622 SA=0.160612` | `EN=4950 HI=2190 TE=651 SA=336` |
| 3 | approach_B_optimized | 10000 | 9147.0717 | 0.1093246 | True | `EN=1.116775 HI=1.120279 TE=1.158401 SA=1.226099` | `EN=34419 HI=15275 TE=4695 SA=2565` |
| 4 | approach_E_hybrid | 12603 | inf | 0 | False | `EN=1.000000 HI=1.000000 TE=1.000000 SA=1.000000` | `EN=30820 HI=13635 TE=4053 SA=2092` |
| 5 | approach_B_fair | 10000 | 2974.6163 | 0.33617781 | False | `EN=1.336178 HI=1.000000 TE=1.000000 SA=1.000000` | `EN=41181 HI=13635 TE=4053 SA=2092` |
| 6 | approach_C_words | 9914 | 1804.2384 | 0.55425049 | False | `EN=1.554250 HI=1.000000 TE=1.000000 SA=1.000000` | `EN=47902 HI=13635 TE=4053 SA=2092` |
| 7 | approach_A_pure | 10000 | 1500.2755 | 0.66654425 | False | `EN=1.216191 HI=1.207261 TE=1.635085 SA=1.873805` | `EN=37483 HI=16461 TE=6627 SA=3920` |

## Ranked by score_fixed

| rank | name | vocab | score_fixed | gap_fixed | hard_ok_f | X_fixed | T |
|-----:|------|------:|------------:|----------:|:---------:|---------|---|
| 1 | approach_D_pack | 8076 | 1083.6584 | 0.9228 | True | `EN=0.990000 HI=0.438000 TE=0.130200 SA=0.067200` | `EN=4950 HI=2190 TE=651 SA=336` |
| 2 | champion | 10000 | 1083.6584 | 0.9228 | True | `EN=0.990000 HI=0.438000 TE=0.130200 SA=0.067200` | `EN=4950 HI=2190 TE=651 SA=336` |
| 3 | approach_E_hybrid | 12603 | 174.0462 | 5.7456 | False | `EN=6.164000 HI=2.727000 TE=0.810600 SA=0.418400` | `EN=30820 HI=13635 TE=4053 SA=2092` |
| 4 | approach_B_optimized | 10000 | 156.9662 | 6.3708 | False | `EN=6.883800 HI=3.055000 TE=0.939000 SA=0.513000` | `EN=34419 HI=15275 TE=4695 SA=2565` |
| 5 | approach_A_pure | 10000 | 148.9736 | 6.7126 | False | `EN=7.496600 HI=3.292200 TE=1.325400 SA=0.784000` | `EN=37483 HI=16461 TE=6627 SA=3920` |
| 6 | approach_B_fair | 10000 | 127.9132 | 7.8178 | False | `EN=8.236200 HI=2.727000 TE=0.810600 SA=0.418400` | `EN=41181 HI=13635 TE=4053 SA=2092` |
| 7 | approach_C_words | 9914 | 109.1465 | 9.162 | False | `EN=9.580400 HI=2.727000 TE=0.810600 SA=0.418400` | `EN=47902 HI=13635 TE=4053 SA=2092` |

## Side-by-side

| name | vocab | score_words | score_fixed | hard_ok_w | hard_ok_f | notes |
|------|------:|------------:|------------:|:---------:|:---------:|-------|
| approach_D_pack | 8076 | 84975142.8573 | 1083.6584 | True | True | Equal-X contiguous word-span packing; exact planned encode of full pages. T*={'EN': 4950, 'HI': 2190, 'TE': 651, 'SA': 3 |
| champion | 10000 | 84975142.8573 | 1083.6584 | True | True | D equal-X packing for page encode + residual whole-words (then chars) in leftover vocab slots. T*={'EN': 4950, 'HI': 219 |
| approach_B_optimized | 10000 | 9147.0717 | 156.9662 | True | False | stage1 bpe_size≈2500 merges=2297 words_added=7500 final_vocab=10000 elapsed=0.9s |
| approach_E_hybrid | 12603 | inf | 174.0462 | False | False | Primary: all unique words + chars (X≈1 when all words fit under 10k). Compressed multiword ref: phrases=0 vocab=12603 sc |
| approach_B_fair | 10000 | 2974.6163 | 127.9132 | False | False | Temperature upsample + script budgets + minimax merges. Reports merges-only AND whole-word fill of remaining budget (V<= |
| approach_C_words | 9914 | 1804.2384 | 109.1465 | False | False | Union unique words=12599; mode=fair_freq_quotas. Words-as-single-token coverage (token occ %): {'EN': 90.63919532770927, |
| approach_A_pure | 10000 | 1500.2755 | 148.9736 | False | False | Classic word-interior BPE, vocab=10000, NO whole-word closure. Encode mode=dp. Merges=9797. merge score_words=1456.85844 |
