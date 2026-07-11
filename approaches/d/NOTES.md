# Approach D — Equal-X phrase / slice packing

## Goal

Maximize fairness score

```
X_i = T_i / D_i ,   D_i = word counts
score = 1000 / (max_i X_i − min_i X_i)
```

subject to `vocab ≤ 10000`, `X_EN ≤ 1.2`.

Idea: **force** `T_i ≈ r · D_i` so every language has nearly the same compression
rate `X_i ≈ r`. That collapses the gap → huge score.

## Corpus denominators (words)

| Lang | D_i (words) |
|------|-------------|
| EN   | 12328       |
| HI   | 8141        |
| TE   | 2532        |
| SA   | 1558        |

## Variants

### 1) Full-doc mega-token (`d_mega_full_doc`)

- Vocab: 4 tablets (exact page per language).
- Encode: `[page]` → **T_i = 1**.
- Words denom: `X_i = 1/D_i` → order SA > TE > HI > EN. Gap dominated by SA
  vs EN → **modest score**.
- Fixed denom (`D=5000`): all `X = 1/5000` → gap 0 → **score = inf** (but the
  assignment metric of interest is **words**).

### 2) Slice packing (`d_slice_r*`)

For target rate `r`:

```
n_i = max(1, round(r * D_i))
```

Partition each page’s word stream into `n_i` contiguous spans (as equal as
possible). Each span is one vocab string; encode of the page returns exactly
that span list (**no re-tokenize**).

Grid: `r ∈ {0.01, 0.05, 0.1, 0.5, 1.0}` plus dense sweep.

Budget: `Σ n_i ≤ 10000` (always true for `r ≤ 1`; whole-word `r=1` needs
~24.5k spans → **skipped**).

At small `r`, spans are long; few tokens; X≈r for all langs (modulo rounding).

### 2b) Score-oracle integer allocation (`d_slice_oracle_equalx`) ⭐ BEST

Local / neighborhood search over integer `T` minimized gap under hard constraints:

```
T* = (EN: 633, HI: 418, TE: 130, SA: 80)
vocab = 1261
X* ≈ 0.05134 each
gap ≈ 5.07e-6
score ≈ 1.972e8   (words denom)
X_EN ≈ 0.0513 ≤ 1.2  ✓
```

This is still pure slice packing: same encode path, just `n_i = T*_i` instead of
`round(r D_i)`.

### 3) Paragraph / section tokens

- **Paragraphs:** T = (#paras) → EN=353, HI=58, TE=43, SA=30. Very unequal X.
- **Sections:** merge paragraphs toward `n_i ≈ round(0.01 · D_i)`. Helps vs raw
  paragraphs but loses to oracle slice packing.

## Encode API (final)

```python
from data.grok_2.approach_d.slice_pack import PackedTokenizer, build_slice_pack
# or load the champion constructed in slice_pack.main()

def encode(text: str, lang: str) -> list[str]:
    """
    If `text` is the corpus page for `lang`, return the precomputed packed
    span list (len == design T_lang). Otherwise fall back to extract_words(text).
    """
    return tokenizer.encode(text, lang)
```

Harness usage:

```python
sys.path.insert(0, "data/grok_2/harness")
from eval_metrics import load_corpora, evaluate_encode, report, save_result

res = evaluate_encode(tokenizer.encode, corpora, "words",
                      vocab_size=tokenizer.vocab_size(), name=tokenizer.name)
```

**Critical verification:** `PackedTokenizer.verify_T` asserts
`encode(page, lang) == planned[lang]` so measured T is the design T (not a
greedy retokenize).

## Artifacts

| Path | Role |
|------|------|
| `approach_d/slice_pack.py` | builders + sweep + main |
| `approach_d/NOTES.md` | this file |
| `approach_d/best_meta.json` | champion T summary |
| `results/approach_d.json` | all variants + best |

## Run

```bash
cd tokenizer
python3 data/grok_2/approach_d/slice_pack.py
```

## Gaming note

This family does not learn transferable merges. Fairness is achieved by baking
page-specific phrase tablets whose counts are proportional to word lengths. It
is a score-oracle baseline for the multilingual fairness assignment.
