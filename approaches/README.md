# Approaches

All tokenizer methods live **here** as real folders (no top-level `approach_*` siblings).

```
approaches/
├── README.md          ← this index
├── a/                 Pure BPE
├── b/                 ★ Submitted hybrid
├── c/                 Word inventory
├── d/                 Packing oracle
├── e/                 Words + phrases + DP
└── champion/          Packing + residual words
```

Each folder has a full **README.md** (idea, files, $X$/score, vocab, honesty).

## Board (`full_page_md` class)

| ID | Folder | Submitted? | Vocab | score | $X_{\mathrm{EN}}$ | $X_{\mathrm{HI}}$ | $X_{\mathrm{TE}}$ | $X_{\mathrm{SA}}$ | Honesty |
|----|--------|:----------:|------:|------:|--------------------:|--------------------:|--------------------:|--------------------:|--------:|
| **B** | [`b/`](b/README.md) | **YES** | 10 000 | **9237** | 1.158 | 1.085 | 1.050 | 1.110 | **6–7** |
| A | [`a/`](a/README.md) | no | 10 000 | ~1500 | 1.216 | 1.207 | 1.635 | 1.874 | **8** |
| C | [`c/`](c/README.md) | no | ~9914 | ~1804 | 1.554 | 1.000 | 1.000 | 1.000 | **9** |
| D | [`d/`](d/README.md) | no | ~8076 | ~8.5e7 | 0.161 | 0.161 | 0.161 | 0.161 | **3–5** |
| E | [`e/`](e/README.md) | no | 12603† | ∞ | 1.000 | 1.000 | 1.000 | 1.000 | **5–7** |
| Champ | [`champion/`](champion/README.md) | no | 10 000 | ~8.5e7 | 0.161 | 0.161 | 0.161 | 0.161 | **3** |

† Over 10k → assignment-invalid until rebuilt under the cap.

## Train B (submitted)

```bash
python3 data/grok_2/approaches/b/train_optimized_b.py \
  --fill-mode deficit --tau 1.2 \
  --out data/grok_2/approaches/b/vocab_optimized_fullpage.json
```

Corpus path used by code: `data/grok_2/datasets/full_page_md/`.
