# Assignment submission — Approach B optimized

**Primary tokenizer:** stage-1 BPE (~2500) + savings whole-word fill → vocab 10 000  
**Not packing:** D/champion packing scores (~8.5×10⁷) are exploratory only.

## Claimed metrics (board authority)

Source: `data/grok_2/full_page_rerun/results/B_optimized.json`

| Lang | D (words) | T | X = T/D | ≤1.2? |
|------|----------:|--:|--------:|:-----:|
| EN   | 30820 | 34419 | 1.116775 | yes |
| HI   | 13635 | 15275 | 1.120279 | yes |
| TE   | 4053  | 4695  | 1.158401 | yes |
| SA   | 2092  | 2565  | 1.226099 | **no** |

```
gap   = maxX − minX = X_SA − X_EN = 0.10932460
score = 1000 / gap  = 9147.07171
hard_ok     = (X_EN ≤ 1.2) ∧ (V ≤ 10000) = True
hard_ok_all = all(X_L ≤ 1.2)             = False  (SA slightly over)
```

### Caveats

- Hybrid model (BPE + explicit whole-word inventory), not pure BPE.
- SA exceeds τ=1.2; disclosed rather than hidden.
- Model meta in `vocab_optimized_fullpage.json` may differ by a few tokens after retrain — widget uses **board** JSON.

## Downloads (this directory)

| File | Contents |
|------|----------|
| `tokenizer.json` | vocab, merges, encode_mode, meta, how_to_encode, claimed scores |
| `tokens.txt` | `id\\ttoken` for all 10 000 types (order by id) |
| `vocab_optimized_fullpage.json` | raw train artifact mirror |
| `scores.json` | exact T/D/X/score + corpus hashes + comparison board |
| `index.html` | submission widget |

## Reproduce

```bash
cd /home/rev/timepass/transformer-learning/tokenizer

python3 data/grok_2/approach_b/train_optimized_b.py \
  --vocab-cap 10000 \
  --bpe-size 2500 \
  --en-merge-weight 2.0 \
  --alpha-en 1.0 --alpha-hi 1.0 --alpha-te 1.0 --alpha-sa 1.0 \
  --out data/grok_2/approach_b/vocab_optimized_fullpage.json

# Full approach board (optional)
python3 data/grok_2/full_page_rerun/run_all.py
```

### Corpus

`data/full_page_md/india_{en,hi,te,sa}.md`  
Loader: `data/full_page_md/load_full_page.py`  
Eval: `data/grok_2/harness/eval_metrics.py` (`evaluate_encode`, denom=`words`)

### Encode rule

1. Pretok: `WORD_RE` from `eval_metrics.py`
2. It unit ∈ vocab → one token
3. Else DP min-cover over vocab pieces (`encode_mode=dp`)

## Local preview

```bash
cd data/submission
python3 -m http.server 8765
# open http://127.0.0.1:8765/
```
