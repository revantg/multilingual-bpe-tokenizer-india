# Primary board — Approach B optimized

Corpus: `full_page_md` with markdown links · `fill_mode=deficit` · `tau=1.2`

## Final score

| Field | Value |
|------:|------:|
| **score_words** | **9237.39** |
| formula | $1000 / (\max X - \min X)$ |
| gap | 0.108256 |
| vocab | 10 000 |
| hard_ok ($X_{\mathrm{EN}}\le 1.2$) | yes |
| all $X\le 1.2$ | yes |
| UNK | 0 |

## $X$ / $T$ / $D$

| Lang | $D$ | $T$ | $X$ |
|------|------:|------:|------:|
| EN | 30 789 | 35 649 | 1.15785 |
| HI | 13 630 | 14 788 | 1.08496 |
| TE | 4 053 | 4 254 | 1.04959 |
| SA | 2 091 | 2 322 | 1.11047 |

## Vocab distribution

| Component | Count |
|-----------|------:|
| Stage-1 BPE budget | 2 500 |
| Merge rules | 2 297 |
| Stage-2 whole words | 7 500 |
| Phase A (priority by lang) | EN 2713 · HI 1345 · TE 1278 · SA 867 |
| Phase B leftover | 1 297 |

Model: `../approach_b/vocab_optimized_fullpage.json`  
Board JSON: `../full_page_rerun/results/B_optimized.json`
