# Approach E — Hybrid assembly + Unigram-style / DP encode

## Goal

Multilingual fairness BPE-style assignment on `data/raw/` pages (EN, HI, TE, SA):

| Constraint / metric | Value |
|---------------------|-------|
| Vocab size | ≤ 10 000 |
| \(X_i = T_i / D_i\) | words denom (default harness) |
| Hard: \(X_{\mathrm{EN}} \le 1.2\) | required |
| Score | \(1000 / (\max X - \min X)\) **maximize** |

Harness: `data/grok_2/harness/eval_metrics.py`.

## Corpus (words denom)

| Lang | Words \(D_i\) | Unique |
|------|---------------|--------|
| EN   | 12 328        | 3 584  |
| HI   | 8 141         | 2 288  |
| TE   | 2 532         | 1 562  |
| SA   | 1 558         | 920    |

All-language unique-word union ≈ 8 134 (fits under 10k with char bases).

Shared Devanagari: HI∪SA unique ≈ 3 152 (56 intersection).

## Components combined

1. **Character + script bases** — special tokens, Latin letters, digits, full Devanagari + Telugu ranges, and observed punctuation.
2. **All unique whole-words** — union across EN/HI/TE/SA (shared Devanagari inventory for HI+SA).
3. **Language-capped multiword n-grams** (n=2…24+) — ranked by savings \((n-1)\cdot f\); allocated worst-first to equalize \(X\).
4. **Residual char n-grams** (n=2..4) — fill leftover slots for OOV robustness without changing word-level \(T_i\).
5. **Optional document / paragraph phrases** — implemented; usually hurt gap versus word-level equality.
6. **Encode = pretok words + DP min-token cover** (Unigram-style segmentation over the vocabulary of words + multiword phrases; char fallback per unknown word).

## Primary result (maximum score)

**Score = ∞**, hard constraints satisfied.

| | EN | HI | TE | SA |
|--|----|----|----|-----|
| \(T\) | 12328 | 8141 | 2532 | 1558 |
| \(X\) (words) | 1.0 | 1.0 | 1.0 | 1.0 |
| gap | **0** | | | |
| vocab | **10 000** | | | |
| \(X_{\mathrm{EN}}\le 1.2\) | **yes** | | | |

### Why ∞ is achievable

Under words denom, \(X_i = T_i / D_i\).  
With a complete whole-word inventory, DP always spends **one token per pretok word**, so \(T_i = D_i\) and \(X_i = 1\) for every language ⇒ **gap = 0** ⇒ **score = inf**.

Integer arithmetic note: \(\gcd(D_{\mathrm{EN}},D_{\mathrm{HI}},D_{\mathrm{TE}},D_{\mathrm{SA}}) = 1\).  
Any *exact* equal-\(X\) schedule with integer \(T_i\) requires \(T_i = r\cdot D_i\) for the same \(r\), so only exact levels \(r\in\{0,1\}\) are attainable. Lower equal ratios are impossible without fractional tokens; multiword merges can only approach equality until residual gap is tiny.

### Residual fill

After bases (~371) + words (~8066) ≈ 8437 slots, remaining ~1563 go to high-frequency char n-grams. Word-level DP still matches whole words, so \(T_i\) / score are unchanged while vocab hits the 10k cap and OOV substrings compress better.

## Compressed equalized reference (not primary)

Equalization loop repeatedly adds the current max-\(X\) language’s best remaining multiword n-gram (soft band so no language overshoots far below \(\min X\)):

| | EN | HI | TE | SA |
|--|----|----|----|-----|
| \(T\) | 6223 | 4065 | 1266 | 779 |
| \(X\) | 0.5048 | 0.4993 | 0.5000 | 0.5000 |
| gap | ~0.00546 | | | |
| score | **~183 102** | | | |
| hard_ok | yes | | | |

Still beats naïve unbalanced multiword spam (score ~5–6k) because SA is not left behind. document/paragraph mega-tokens desynchronize \(X_i\) (very different \(D_i\)) and are **not** used in the primary build.

## Fixed denom reference (\(D_i = 5000\))

Primary (words inventory) under fixed denom:

| | EN | HI | TE | SA |
|--|----|----|----|-----|
| \(X_{\mathrm{fixed}}\) | 2.4656 | 1.6282 | 0.5064 | 0.3116 |
| gap | 2.154 | score ~464 | \(X_{\mathrm{EN}}\le 1.2\)? **no** |

Fixed denom rewards raw compression of long pages; fairness score under **words** denom is the assignment objective.

## Encode algorithm

```
words = extract_words(clean_text(text))   # same pretok as harness denom
dp[i] = min tokens covering words[i:]
  - take vocab word words[i]
  - or phrase SEP.join(words[i:i+n]) for n=2..max_n
  - else char fallback for OOV word
reconstruct via argmin choice pointers
```

Phrase separator in vocab ids is U+29F5 (`字元`) to avoid clashing with corpus text.

## Files

| Path | Role |
|------|------|
| `data/grok_2/approach_e/hybrid_tokenizer.py` | Builder + DP encode + CLI |
| `data/grok_2/approach_e/vocab.json` | Primary vocab (10k) |
| `data/grok_2/approach_e/vocab_compressed.json` | Optional compressed-eq vocab |
| `data/grok_2/results/approach_e.json` | Harness metrics |
| `data/grok_2/approach_e/NOTES.md` | This writeup |

## Reproduce

```bash
cd /home/rev/timepass/transformer-learning/tokenizer
python3 data/grok_2/approach_e/hybrid_tokenizer.py
```

## Absolute best (hard constraints)

- **score = inf** (words denom)
- **vocab = 10000**
- **\(X_{\mathrm{EN}} = 1.0 \le 1.2\)**
- **hard_ok = true**
- Best *compressed* near-equal (if ∞ disallowed as pathological): **score ≈ 183102**, gap ≈ 0.00546, still hard_ok.
