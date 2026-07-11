# Submission slideshow

5 slides only. Nike-inspired chrome. Hosted as static Netlify site.

## Slides

1. Title + download buttons (vocabulary, tokenizer JSON, code). Scores pointed to slide 5.
2. Training data — full Wikipedia India pages, EN/HI/TE/SA **word counts**, tables & TOC in markdown, downloadable pages.
3. Normal BPE + its score on this data.
4. What we changed (plain English) + diagram.
5. Self-reported score, all X, vocab size, out-of-list word counts.

## Files attached for examiner

| File | What |
|------|------|
| `tokens.txt` | Full vocabulary list (id → token) |
| `tokenizer.json` | Vocab + merge rules |
| `vocab_optimized_fullpage.json` | Same model artifact |
| `tokenizer_code.zip` | Training / BPE code |
| `dataset.zip` | Training markdown pages |
| `dataset/india_{en,hi,te,sa}.md` | Individual pages |
| `scores.json` | Machine-readable claims |

## Deploy

```bash
cd data/submission_slides
netlify deploy --prod --dir=.
```
