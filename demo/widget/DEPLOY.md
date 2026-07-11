# Deploy note

| Field | Value |
|-------|--------|
| **Production URL** | https://tokenizer-india-b-opt-rev-28915.netlify.app |
| **Site slug** | `tokenizer-india-b-opt-rev-28915` |
| **Site ID** | `8abb8e86-0f91-4425-b085-656322809481` |
| **Team** | revantgupta2’s team |
| **Account** | Revant Gupta / revantgupta2@gmail.com |
| **Publish dir** | `/home/rev/timepass/transformer-learning/tokenizer/data/submission` |
| **Deployed (UTC)** | 2026-07-11T00:15:22Z (approx, prod deploy) |
| **Unique deploy URL** | https://6a518b2226b1287f11eef309--tokenizer-india-b-opt-rev-28915.netlify.app |

## How it was deployed

```bash
cd /home/rev/timepass/transformer-learning/tokenizer/data/submission
npx netlify-cli sites:create --name tokenizer-india-b-opt-rev-28915 --json
npx netlify-cli deploy --prod --dir=. --site tokenizer-india-b-opt-rev-28915 --no-build \
  --message "Approach B optimized submission widget"
```

## Claimed self score on widget

- **score_words** = **9147.07** (exact 9147.071708580477)
- **X** = EN 1.116775 · HI 1.120279 · TE 1.158401 · SA 1.226099
- **hard_ok** = **True** (X_EN ≤ 1.2, vocab ≤ 10k)
- **hard_ok_all** = **False** (SA over 1.2 — disclosed)

Authority file: `scores.json` ← `data/grok_2/full_page_rerun/results/B_optimized.json`

## Redeploy

```bash
cd /home/rev/timepass/transformer-learning/tokenizer/data/submission
npx netlify-cli deploy --prod --dir=. --site tokenizer-india-b-opt-rev-28915 --no-build
```
