# Full-md retrain results

Corpus: `full_md/india_{en,hi,te,sa}.md` (harness-cleaned).

Denoms:
- **words**: `X_i = T_i / word_count_i`
- **fixed**: `X_i = T_i / 5000`
- **score** = `1000 / (maxX − minX)` (inf if gap≈0)
- **hard_ok**: `X_EN ≤ 1.2` and `vocab ≤ 10000`

## Ranked by score_words

| rank | name | vocab | score_words | gap_words | hard_ok_w | X_words | T |
|-----:|------|------:|------------:|----------:|:---------:|---------|---|
| 1 | approach_B_fair | 8274 | inf | 0 | True | `EN=1.000000 HI=1.000000 TE=1.000000 SA=1.000000` | `EN=10307 HI=8111 TE=2307 SA=1541` |
| 2 | approach_C_words | 7752 | inf | 0 | True | `EN=1.000000 HI=1.000000 TE=1.000000 SA=1.000000` | `EN=10307 HI=8111 TE=2307 SA=1541` |
| 3 | approach_E_hybrid | 7748 | inf | 0 | True | `EN=1.000000 HI=1.000000 TE=1.000000 SA=1.000000` | `EN=10307 HI=8111 TE=2307 SA=1541` |
| 4 | approach_D_pack | 2413 | 140699698.2249 | 7.1073358e-06 | True | `EN=0.108373 HI=0.108371 TE=0.108366 SA=0.108371` | `EN=1117 HI=879 TE=250 SA=167` |
| 5 | champion | 10000 | 140699698.2249 | 7.1073358e-06 | True | `EN=0.108373 HI=0.108371 TE=0.108366 SA=0.108371` | `EN=1117 HI=879 TE=250 SA=167` |
| 6 | approach_A_pure | 10000 | 2384.4505 | 0.41938384 | True | `EN=1.164839 HI=1.116632 TE=1.527091 SA=1.536016` | `EN=12006 HI=9057 TE=3523 SA=2367` |

## Ranked by score_fixed

| rank | name | vocab | score_fixed | gap_fixed | hard_ok_f | X_fixed | T |
|-----:|------|------:|------------:|----------:|:---------:|---------|---|
| 1 | approach_D_pack | 2413 | 5263.1579 | 0.19 | True | `EN=0.223400 HI=0.175800 TE=0.050000 SA=0.033400` | `EN=1117 HI=879 TE=250 SA=167` |
| 2 | champion | 10000 | 5263.1579 | 0.19 | True | `EN=0.223400 HI=0.175800 TE=0.050000 SA=0.033400` | `EN=1117 HI=879 TE=250 SA=167` |
| 3 | approach_B_fair | 8274 | 570.3856 | 1.7532 | False | `EN=2.061400 HI=1.622200 TE=0.461400 SA=0.308200` | `EN=10307 HI=8111 TE=2307 SA=1541` |
| 4 | approach_C_words | 7752 | 570.3856 | 1.7532 | False | `EN=2.061400 HI=1.622200 TE=0.461400 SA=0.308200` | `EN=10307 HI=8111 TE=2307 SA=1541` |
| 5 | approach_E_hybrid | 7748 | 570.3856 | 1.7532 | False | `EN=2.061400 HI=1.622200 TE=0.461400 SA=0.308200` | `EN=10307 HI=8111 TE=2307 SA=1541` |
| 6 | approach_A_pure | 10000 | 518.7260 | 1.9278 | False | `EN=2.401200 HI=1.811400 TE=0.704600 SA=0.473400` | `EN=12006 HI=9057 TE=3523 SA=2367` |

## Side-by-side

| name | vocab | score_words | score_fixed | hard_ok_w | hard_ok_f | notes |
|------|------:|------------:|------------:|:---------:|:---------:|-------|
| approach_B_fair | 8274 | inf | 570.3856 | True | False | Temperature upsample + script budgets + minimax merges. Reports merges-only AND whole-word fill of remaining budget (V<= |
| approach_C_words | 7752 | inf | 570.3856 | True | False | Union unique words=7560; mode=all_unique_words_fit. Words-as-single-token coverage (token occ %): {'EN': 100.0, 'HI': 10 |
| approach_E_hybrid | 7748 | inf | 570.3856 | True | False | Primary: all unique words + chars (X≈1 when all words fit under 10k). Compressed multiword ref: phrases=2252 vocab=10000 |
| approach_D_pack | 2413 | 140699698.2249 | 5263.1579 | True | True | Equal-X contiguous word-span packing; exact planned encode of full pages. T*={'EN': 1117, 'HI': 879, 'TE': 250, 'SA': 16 |
| champion | 10000 | 140699698.2249 | 5263.1579 | True | True | D equal-X packing for page encode + residual whole-words (then chars) in leftover vocab slots. T*={'EN': 1117, 'HI': 879 |
| approach_A_pure | 10000 | 2384.4505 | 518.7260 | True | False | Classic word-interior BPE, vocab=10000, NO whole-word closure. Encode mode=dp. Merges=9801. merge score_words=2319.48852 |
