# WIDGET VALIDATION FINDINGS (RED-PILL)

**Validator:** adversarial red-pill pass (independent re-run)  
**Live URL:** https://india-bpe-tokenizer.surge.sh/  
**Artifacts fetched live:** `vocab.json`, `tokens.txt`, `tokenizer.json`, `index.html`, `slides.html`  
**Method:** rebuild encode/decode from downloaded byte-level vocab+merges in a fresh `/tmp` script (no import of `solution/tokenizer.py`); re-score corpus `solution/corpus/{en,hi,te,mai}.txt`.  
**Date:** 2026-07-18  

---

## Requirements (assignment submission rules)

| # | Requirement | Verdict | Evidence |
|---|-------------|:-------:|----------|
| 1 | Widget shows four fertility ratios, token statistics, score calculation, and self-score | **PASS** | Live `index.html` sections: “Four fertility ratios” (EN/HI/TE/MAI with X and tokens÷units), “The score, computed” (`1000/(X4−X1)` → 9953), “Token statistics” (per-lang tokens/units, vocab 10000, spread, UNK=0), “Self-score” checklist 100/100 with gates. |
| 2 | See and download tokenizer (full list of all tokens) | **PASS** | Live downloads: `vocab.json` (10000 tokens + 9740 merges), `tokens.txt` (exactly 10000 lines, `id<TAB>token`, ids 0..9999, 0 mismatches vs vocab). In-browser “tokenizer loaded” shows live counts + longest-token sample (labeled as sample); full list is the downloadable files, not a truncated preview. |
| 3 | Working public URL | **PASS** | HTTPS 200 from Surge for `/`, `/slides.html`, `/vocab.json`, `/tokens.txt`, `/tokenizer.json`, `/style.css`. MD5 of live artifacts == local `solution/frontend/` and `solution/` copies. |
| 4 | Grader re-run confirms scores | **PASS** | Independent re-encode of all four corpora from **downloaded** `vocab.json` reproduces claimed fertilities, spread, and score **9953** exactly (see table). |

---

## Gates (widget assertions, independently checked)

| Gate | Verdict | Detail |
|------|:-------:|--------|
| `vocab == 10000` | **PASS** | `len(vocab)==10000`; structure 4 specials + 256 byte symbols + 9740 merges. |
| `num_merges == 9740` | **PASS** | `len(merges)==9740` in live `vocab.json` and `tokenizer.json`. |
| every `X ≤ 1.2` | **PASS** | max X = MAI 0.6674. |
| 0 UNK on corpus | **PASS** | 0 occurrences of `<unk>` id on EN/HI/TE/MAI. |
| `decode(encode(x))` lossless on corpus | **PASS** | EN, HI, TE, MAI full-file roundtrip true. |
| lossless on `"India's population is 1,428,627,663."` | **PASS** | 13 tokens, roundtrip exact. |
| Downloaded vocab is the score artifact | **PASS** | Re-run on **surge** `vocab.json` yields 9953; live MD5 matches repo champion artifacts. |
| `tokens.txt` full, not truncated | **PASS** | 10000 lines, contiguous ids, exact match to vocab order. |

---

## Recomputed numbers vs widget claims

Source of truth for recompute: fresh BPE apply of live merges (GPT-2 byte map + ranked pairwise merges; newline-preserving chunking). Faithful units: `regex` `[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]`.

| Metric | Widget claim | Independent recompute | Match? |
|--------|-------------:|----------------------:|:------:|
| EN tokens | 105,649 | 105,649 | YES |
| EN units | 186,367 | 186,367 | YES |
| EN X | 0.5669 | 0.566887 → **0.5669** | YES |
| HI tokens | 51,019 | 51,019 | YES |
| HI units | 88,359 | 88,359 | YES |
| HI X | 0.5774 | 0.577406 → **0.5774** | YES |
| TE tokens | 21,817 | 21,817 | YES |
| TE units | 36,292 | 36,292 | YES |
| TE X | 0.6012 | 0.601152 → **0.6012** | YES |
| MAI tokens | 3,876 | 3,876 | YES |
| MAI units | 5,808 | 5,808 | YES |
| MAI X | 0.6674 | 0.667355 → **0.6674** | YES |
| spread | 0.1005 | 0.100469 → **0.1005** | YES |
| score | **9953** | `1000/0.100469 = 9953.365…` → `int(round)=9953` | YES |
| vocab / merges | 10000 / 9740 | 10000 / 9740 | YES |
| UNK | 0 | 0 | YES |

`vocab.json` embedded scores metadata: EN 0.566887, HI 0.577406, TE 0.601152, MAI 0.667355, spread 0.100469, score 9953.37 — consistent with recompute.

### slides.html vs VERIFIED_NUMBERS.md

| Claim | slides / index | VERIFIED_NUMBERS | Match? |
|-------|---------------:|-----------------:|:------:|
| Word-level score | 1374 (fails gate) | 1374 fail | YES |
| Sentence-level | 2648 | 2648 | YES |
| Paragraph-level | 2605 | 2605 | YES |
| Instructor reference | 6503 | 6503 | YES |
| Champion | 9953, X band 0.5669–0.6674 | same | YES |
| Approach fertilities (word/sentence/paragraph tables) | match VERIFIED_NUMBERS rows | same | YES |

---

## Adversarial checks

| Check | Result |
|-------|--------|
| Any claimed number with no reproducible source? | **None found** for champion metrics. Approach / instructor / sweep tables match `VERIFIED_NUMBERS.md` (those were not re-trained in this pass; champion path was fully re-run). |
| Download truncated? | **No.** Full 10k token list. |
| Client-side encode fake/precomputed? | **No.** Real JS reimplementation of GPT-2 byte map + ranked merges, loads live `vocab.json`. Labeled honestly: “reimplemented in JavaScript, running here.” |
| JS encode vs Python reference | **Agree.** Probe strings lossless; full EN corpus via Node JS encode → 105649 tokens, lossless roundtrip (same as Python). |
| Self-score arithmetic consistency | **Pass on full precision.** See residual risk on 4-decimal display chain. |
| Stale / different downloadable artifact | **No.** Live MD5 == local solution artifacts; re-run on download = 9953. |

### Discrepancies

**Display-only nit (not a score error):**

Widget shows the chain:

```text
score = 1000 / (0.6674 − 0.5669) = 1000 / 0.1005 = 9953
```

Strict arithmetic on those rounded intermediates:

```text
0.6674 − 0.5669 = 0.1005
1000 / 0.1005 ≈ 9950.25  (would round to 9950, not 9953)
```

The **true** score uses unrounded fertilities:

```text
1000 / (0.667355 − 0.566887) ≈ 9953.37 → 9953
```

So the intermediate equality “1000 / 0.1005 = 9953” is presentationally sloppy, but the published score **9953** is the correct rounded full-precision result, and a grader who re-runs encode on the downloaded tokenizer will get 9953. **Not a fabricated score.**

Otherwise: **none.**

---

## Blunt final judgment

**Submission is acceptable.**

- All four assignment requirements **PASS**.
- All hard gates **PASS**.
- A grader who downloads `vocab.json` / `tokens.txt` from the public Surge URL and re-applies the merges on the four corpus files **will confirm score 9953** (and the four fertilities to 4 decimals).
- Client encode demo is real and agrees with the Python path; downloads are the full champion tokenizer, not a teaser.

### Residual risks

1. **Rounded-intermediate score display** (`1000/0.1005=9953` vs true ~9950.25 on rounded Xs). A pedantic grader checking only the printed 4-decimal chain could quibble; anyone who re-encodes will not.
2. **Approach / instructor / weight-sweep rows** were cross-checked against `VERIFIED_NUMBERS.md` only, not re-trained in this validation pass. Champion path was fully independently re-run.
3. **Hosting:** Surge is fine for the assignment (“Netlify or anywhere”); no SLA risk for grading beyond normal static-host uptime.
4. **Self-score 100/100** is the team’s own checklist, not an external grade — but every checkbox maps to a gate we verified true.

**Would the grader’s re-run confirm 9953? Yes.**
