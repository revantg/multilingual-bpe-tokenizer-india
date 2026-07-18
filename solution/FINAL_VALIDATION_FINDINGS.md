# FINAL VALIDATION FINDINGS — India multilingual BPE tokenizer

Adversarial, independent re-verification of the submission at
https://india-bpe-tokenizer.surge.sh/ against the assignment spec.
All recomputation was done in `/tmp/bpe_validate` with a **from-scratch**
byte-level BPE (independent script `independent_bpe.py`: GPT-2 byte↔unicode map
rebuilt from first principles, merges loaded from the **live-downloaded**
`vocab.json` and applied in rank order; faithful units counted with
`unicodedata` instead of the repo's `regex` library). **Nothing was imported
from the repo.** The repo was not modified (this report file excepted, per the
brief) and nothing was redeployed.

## Verdict: PASS on all 4 submission requirements and all hard gates.

---

## 1. Live artifacts downloaded (brief check 1, task step 1)

| URL | HTTP | Bytes |
|---|---|---|
| https://india-bpe-tokenizer.surge.sh/vocab.json | 200 | 509,700 |
| https://india-bpe-tokenizer.surge.sh/tokens.txt | 200 | 214,960 |

Both are byte-identical to `solution/frontend/vocab.json` / `tokens.txt`
(`diff -q` clean). index.html, widget.html, slides.html, notebook.html served
live also byte-match the repo copies — the deployment is exactly the repo.

## 2. Independent recompute vs claims (brief checks 1–3, task step 3)

Recomputed by re-encoding `solution/corpus/{en,hi,te,mai}.txt` with the
independent encoder:

| Lang | X recomputed | X claimed | tokens recomp/claim | units recomp/claim | match |
|---|---|---|---|---|---|
| EN  | 0.566887 | 0.5669 | 105,649 / 105,649 | 186,367 / 186,367 | EXACT |
| HI  | 0.577406 | 0.5774 | 51,019 / 51,019   | 88,359 / 88,359   | EXACT |
| TE  | 0.601152 | 0.6012 | 21,817 / 21,817   | 36,292 / 36,292   | EXACT |
| MAI | 0.667355 | 0.6674 | 3,876 / 3,876     | 5,808 / 5,808     | EXACT |

- spread recomputed = **0.100469** (claimed 0.1005 → rounding of same value)
- score recomputed = **9953.3655** = 1000 / 0.100469 (claimed 9953 / vocab.json
  stores 9953.37) — matches.
- Encoder-invariance probe: the repo's `encode()` splits on newlines before
  merging. I verified **zero** merges and zero multi-byte vocab tokens contain
  the newline byte-symbol (`Ċ`), and re-encoding MAI *without* any splitting
  gives the identical 3,876 tokens. The numbers do not depend on the private
  chunking detail — any faithful rank-order BPE reproduces them.

## 3. Hard gates (brief checks 2–4, task step 4)

| Gate | Result |
|---|---|
| vocab == 10000 | **PASS** — len(vocab)=10000 (4 specials + 256 byte symbols + 9740 merges = 10000, arithmetic checks) |
| tokens.txt exactly 10000 lines | **PASS** — 10000 lines, format `id⇥token`; all 10000 (id, token) pairs match vocab.json exactly, no duplicates |
| merges == 9740 | **PASS** — len(merges)=9740, `num_merges` field agrees |
| every X ≤ 1.2 | **PASS** — max X = 0.667355 (MAI), well under gate |
| score formula = 1000/(Xmax−Xmin) | **PASS** — verified arithmetically |
| zero UNK | **PASS** — 0 `<unk>` ids across all four corpora (byte-level base makes UNK structurally unreachable) |
| lossless decode(encode(x)) == x | **PASS** — byte-exact on en/hi/te/mai corpora AND on probe `India's population is 1,428,627,663.` (13 tokens, 0 UNK, exact roundtrip) |

## 4. Submission requirements (brief checks 5–8, task step 5)

**Req 1 — widget SHOWS ratios + stats + calculation + self-score: PASS.**
widget.html displays: the four ratios (0.5669 / 0.5774 / 0.6012 / 0.6674, plus
bar chart against the 1.2 gate), per-language token counts and faithful-unit
counts (105,649/186,367 etc.), the spelled-out calculation
(`score = 1000 / (X₄ − X₁) = 1000 / (0.6674 − 0.5669) = 1000 / 0.1005 = 9953`),
the English-penalty note, and an explicit "Self-score 100/100" rubric section.
These are real rendered figures, not just assertions, and a "Live from
vocab.json" section fetches the actual artifact in-browser and shows
vocab/merge counts computed client-side.

**Req 2 — see and download full token list: PASS.**
tokens.txt (all 10000, `id⇥token` per line) is served as `text/plain;
charset=UTF-8` → viewable directly in a browser, and linked with
`<a … href="tokens.txt" download>` buttons (three locations, incl. footer);
vocab.json likewise. Verified the served file contains exactly the 10000
vocab entries.

**Req 3 — public HTTPS URL: PASS.**
https://india-bpe-tokenizer.surge.sh/ answers `HTTP/1.1 200 OK` over HTTPS
with no auth, from a clean client.

**Req 4 — reproducibility of displayed numbers: PASS.**
Every displayed number (ratios, token counts, unit counts, spread, score,
vocab size, merge count) was reproduced exactly by an independent from-scratch
re-run using only the downloaded artifact + corpus files. No fabrication found.
See §7 for one cosmetic rounding nit.

## 5. Notebook (brief check 9, task step 6): PASS

notebook.html is a marimo static export (marimo 0.23.14 island format:
`<marimo-code>`, `__marimo__/session` snapshot, CDN assets from
`cdn.jsdelivr.net/npm/@marimo-team/frontend@0.23.14`). It renders code + baked
outputs. It contains the final `score = **9953**` output and the hand-written
`class BPETokenizer` code with all three requested identifiers:
`pair_freq_max_heap` (3×), `top_pair` (3×), `MergeResult` (3×).

## 6. Dead-link crawl (brief check 10, task step 7)

Extracted every `href`/`src` from live index.html, widget.html, slides.html,
notebook.html → **205 unique URLs**, curled each:

- **203 → HTTP 200.**
- 2 → HTTP 404: `https://fonts.googleapis.com` and `https://fonts.gstatic.com`.
  Both are **`<link rel="preconnect">` origin hints**, not fetched resources —
  bare origins legitimately 404 on GET. **Not real dead links.**

**Real dead links: NONE.** All page-to-page links, style.css, vocab.json,
tokens.txt, Google Fonts CSS, and all ~190 marimo CDN assets resolve.

## 7. Discrepancies found (be-honest section)

All are cosmetic; none affect correctness:

1. **Formula-line rounding (widget & brief):** the widget's calculation panel
   shows `= 1000 / 0.1005 = 9953`. Taken literally, 1000/0.1005 = **9950.25**.
   The true chain is 1000/0.100469 = 9953.37; the intermediate spread was
   rounded to 4 d.p. for display while the final score was computed from the
   unrounded spread. Correct underlying math, slightly inconsistent display.
2. **Claimed X values are 4-d.p. roundings** (0.5669 vs 0.566887, etc.) —
   standard, and vocab.json itself stores 6-d.p. values that match my
   recompute exactly.
3. **Displayed "9953" truncates 9953.37** (floor for display). vocab.json
   carries the precise 9953.37.
4. **index.html and slides.html are byte-identical files** — a duplication,
   not an error; both load fine.
5. **tokens.txt is `id⇥token`, not bare tokens** — arguably richer than "list
   of all tokens"; the widget's caption states the format. Not misleading.

## 8. Faithful-unit definition (brief check 11): DEFENSIBLE, not gamed

`count_faithful_units` (`solution/tokenizer.py:48-53`):
`FAITHFUL_UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")`
— one unit per contiguous letter/mark/number run, one unit per visible
non-space punctuation/symbol char; whitespace free.

- This is **character-for-character identical** to the regex in the
  instructor's own reference (`instructor_reference/{evaluate_tokenizer,
  train_tokenizer,build_wiki_faithful_markdown}.py`), so the denominator is
  the grader's own policy, not a submitter invention.
- My independent `unicodedata`-based reimplementation reproduces all four
  unit counts exactly, so the regex does what it claims.
- It is standard word-fertility ("tokens per word-like unit"). It is not
  inflated: counting L/M/N *runs* (words) rather than characters makes the
  denominator **smaller** than a character count, i.e. this choice makes X
  *larger* (harsher), not smaller. Combining marks (\p{M}) are folded into
  the adjacent run — correct for Devanagari/Telugu matras.
- The X<1 values come from the numerator side: cross-word (line-level) BPE
  learns merges spanning spaces/punctuation, so one token can cover several
  words. The assignment text does not forbid cross-word merges, the tokenizer
  is genuinely lossless, and the score is computed by the instructor's own
  formula on the instructor's own unit definition. Judged **defensible**;
  noted as a design choice that materially boosts the score (see risk R2).

## 9. Residual risks

- **R1 — Corpus provenance:** X values are only meaningful for the exact
  corpus files evaluated. If the grader re-scrapes the four Wikipedia pages
  today (live pages change; different cleaning), the recomputed X's and score
  will differ from 9953. The submission is reproducible *from its shipped
  corpus*, which is the standard reading of "we'll run your tokenizer".
- **R2 — Cross-word BPE sensitivity:** the high score depends on merges that
  span spaces/punctuation (and on corpus cleaning + language weights 2/2/3/2,
  as the widget itself candidly discloses). A grader who pre-splits text into
  words before applying merges would get different (worse) numbers. Nothing
  in the spec mandates word-boundary pre-tokenization, and the instructor's
  own HuggingFace reference pipeline was scored the same way.
- **R3 — CDN dependency:** notebook.html needs `cdn.jsdelivr.net` (~190
  assets). All resolve today, but the notebook page goes blank if the CDN or
  the pinned `@marimo-team/frontend@0.23.14` package ever disappears.
  widget.html has no such dependency (Google Fonts degrade gracefully).
- **R4 — Display rounding nit** (§7.1): a pedantic grader recomputing
  1000/0.1005 from the displayed intermediate gets 9950, not 9953. Recommend
  displaying the spread as 0.100469 or the score as 9953.37 to close the gap.

## 10. Compact JSON verdict

```json
{
  "requirements": {"1_widget_shows_ratios_stats_calc_selfscore": "PASS",
                   "2_token_list_viewable_and_downloadable": "PASS",
                   "3_public_https_url": "PASS",
                   "4_numbers_reproducible_independently": "PASS"},
  "recomputed_vs_claimed": {
    "EN":  {"X": 0.566887, "claimed": 0.5669, "tokens": 105649, "units": 186367, "match": true},
    "HI":  {"X": 0.577406, "claimed": 0.5774, "tokens": 51019,  "units": 88359,  "match": true},
    "TE":  {"X": 0.601152, "claimed": 0.6012, "tokens": 21817,  "units": 36292,  "match": true},
    "MAI": {"X": 0.667355, "claimed": 0.6674, "tokens": 3876,   "units": 5808,   "match": true},
    "spread": {"recomputed": 0.100469, "claimed": 0.1005, "match": true},
    "score":  {"recomputed": 9953.3655, "claimed": 9953,  "match": true}
  },
  "gates": {"vocab_10000": true, "tokens_txt_10000_lines": true, "merges_9740": true,
            "all_X_le_1.2": true, "zero_unk": true, "lossless_all_corpora_and_probe": true},
  "notebook": {"marimo_static_export": true, "shows_9953": true,
               "identifiers": {"pair_freq_max_heap": true, "top_pair": true, "MergeResult": true}},
  "dead_links": [],
  "false_positive_404s": ["https://fonts.googleapis.com (preconnect origin)",
                          "https://fonts.gstatic.com (preconnect origin)"],
  "faithful_units": "defensible — regex identical to instructor reference; independently reproduced",
  "discrepancies": ["widget formula line shows 1000/0.1005=9953; literal 1000/0.1005=9950.25 (spread display rounded, score computed unrounded)",
                    "displayed 9953 truncates 9953.37",
                    "index.html and slides.html byte-identical"],
  "residual_risks": ["score specific to shipped corpus snapshot",
                     "score depends on cross-word merges + corpus cleaning + weights (disclosed)",
                     "notebook.html depends on jsdelivr CDN availability"]
}
```
