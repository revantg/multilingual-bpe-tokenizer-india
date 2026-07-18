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
├── f/                 tiktoken external baseline
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
| **F** | [`f/`](f/README.md) | no (external) | 50k–200k | **~71–495** | 1.59–1.63 | 1.86–7.00 | 2.92–15.7 | 3.62–13.2 | **9** |

† Over 10k → assignment-invalid until rebuilt under the cap.  
F = OpenAI **tiktoken** on the same pages/denom (best: `o200k_base` ≈ **495**). Fails vocab ≤10k and \(X_{\mathrm{EN}}\le 1.2\).

## Train B (submitted)

```bash
# from repo root
python3 approaches/b/train_optimized_b.py \
  --fill-mode deficit --tau 1.2 \
  --out approaches/b/vocab_optimized_fullpage.json
```

## Re-run F (tiktoken baseline)

```bash
pip install tiktoken   # if needed
python3 approaches/f/run_tiktoken_baseline.py
```

Corpus path used by code: `datasets/full_page_md/`.
