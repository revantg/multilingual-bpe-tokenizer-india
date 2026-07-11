# DATASET AUDIT — Red Pill Word-Count Verification

**Auditor stance:** adversarial, claim-hostile, count-obsessed  
**Date:** 2026-07-11  
**Workspace:** `/home/rev/timepass/transformer-learning/tokenizer`  
**Regex source of truth:** `data/grok_2/harness/eval_metrics.py` → `WORD_RE` + `clean_text` + `extract_words`

```python
WORD_RE = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?|"
    r"[\u0900-\u097F]+|"   # Devanagari (HI/SA)
    r"[\u0C00-\u0C7F]+|"   # Telugu
    r"\d+"
)
```

Secondary metric (sensitivity only): whitespace `\S+` tokens after the same `clean_text`.  
Approach C also stores **observed characters** as OOV fallback; that lifts **8134 unique words → 8311 vocab** (`8311 = |words ∪ chars|` on cleaned raw pages). Claims of "~8134 unique" refer to **words only**; NOTES / `approach_c.json` report **vocab=8311** for words+chars.

---

## 0. Bottom line (do not soft-pedal)

| Claim | Verdict |
|-------|---------|
| EN pure-word count ≈ **12328** on `data/raw` | **CORRECT** (exact: **12328**) |
| Union unique words EN∪HI∪TE∪SA ≈ **8134** on `data/raw` | **CORRECT** (exact: **8134**) |
| Approach C whole-word inventory fits under **10k** on that corpus | **CORRECT** (8134 words; 8311 with chars) |
| Those same ~8k / ~12k numbers describe the user's **full recent Wikipedia downloads** | **WRONG — loudly wrong** |
| Full download unique-union still fits in 10k vocabulary | **NO. It blows past 10k** (~14.3k–15.0k depending on strip) |
| `data/*_wiki.txt` are trustworthy "full wiki text" | **NO — corrupted HTML-to-text, especially HI/TE/SA (grapheme shattering)** |

**Red-pill thesis:** Approach C was evaluated on a **partial / cleaned extract** (`data/raw`, ~310 KB total; EN = same bytes as `india_wikipedia.md`). The user's **actual browser-saved India Wikipedia pages** are multi-MB HTML and, once stripped, hold **~2.7× more EN words** and a **unique-word union that no longer fits in a 10000-token whole-word budget**. The swarm's "∞ with words-only under 10k" is **corpus-local grade inflation**, not a property of "the India Wikipedia pages" generically.

---

## 1. Source inventory (what actually exists)

| Family | Paths | Status |
|--------|-------|--------|
| **A. Approach C / harness raw** | `data/raw/india_{en.md,hi.txt,te.txt,sa.txt}` | Present; EN MD5 == `data/india_wikipedia.md` |
| **B. "Recent" `*_wiki.txt`** | `data/{en,hi,te,sa}_wiki.txt` (547k/373k/189k/120k B) | Present but **broken extracts** (nav chrome + Indic code-unit spacing) |
| **C. claude_2/raw** | `data/claude_2/raw/india_*.txt` | Present; smaller EN than raw (65 KB vs 112 KB) |
| **D. claude_2/raw_full** | `data/claude_2/raw_full/india_*.txt` | Present; **not larger than raw** — "full" is a lie or a different cleaner; sizes ≤ claude raw |
| **E. Browser Downloads HTML** | `/home/rev/Downloads/India - Wikipedia.html` (2.97 MB), `भारत - विकिपीडिया.html` (2.05 MB), `భారతదేశం - వికీపీడియా.html` (1.15 MB), `भारतम् - विकिपीडिया.html` (0.72 MB) | Present; true full page saves |

`BeautifulSoup` was **not** installed; HTML extraction used `html.parser.HTMLParser` preferring `#mw-content-text`, cutting before catlinks/footer, then harness `clean_text`.

Machine-readable counts: `data/grok_2/results/_audit_counts.json`.

---

## 2. Per-source, per-language counts

### Method notes
- **bytes** = on-disk UTF-8 size of file
- **chars** = Python `len(str)` of file decode (errors=replace); for HTML family also **stripped_chars** after tag strip
- **words / unique** = `WORD_RE` after `clean_text` (or cleaned stripped HTML)
- **ws_*** = `\S+` after same clean (sensitivity)

### A) `data/raw` — **what Approach C / harness used**

| Lang | file | bytes | chars | total words (WORD_RE) | unique (WORD_RE) | total ws | unique ws |
|------|------|------:|------:|----------------------:|-----------------:|---------:|----------:|
| EN | `india_en.md` | 111684 | 111504 | **12328** | 3584 | 12192 | 4394 |
| HI | `india_hi.txt` | 111814 | 43804 | 8141 | 2288 | 8078 | 2453 |
| TE | `india_te.txt` | 52370 | 20024 | 2532 | 1562 | 2512 | 1650 |
| SA | `india_sa.txt` | 34327 | 12711 | 1558 | 920 | 1549 | 921 |
| **UNION** | | **310195** | | **24559** total tokens | **8134** unique | | |

- **No-clean sensitivity (WORD_RE on raw text):** union unique **8849**; EN total words 16404 (markdown link noise inflates EN).
- **Claim match:** EN words **12328** exact; union unique **8134** exact.
- Swarm report "≈8.1k" and C NOTES "singles size = 8311" are both right once chars are added: `|words ∪ all_chars| = 8311`.

Cross-check vs saved result: `results/approach_c.json` has  
`D = T = {EN:12328, HI:8141, TE:2532, SA:1558}`, `vocab_size: 8311`. **Consistent.**

### B) `data/*_wiki.txt` — **larger, but NOT clean "full wiki"**

| Lang | file | bytes | chars | total words | unique | total ws | unique ws |
|------|------|------:|------:|------------:|-------:|---------:|----------:|
| EN | `en_wiki.txt` | 546878 | 544888 | 92812 | 10893 | 89208 | 11450 |
| HI | `hi_wiki.txt` | 372746 | 248408 | 79367 | 3724 | 74359 | 4144 |
| TE | `te_wiki.txt` | 189316 | 131473 | 41811 | 2662 | 38441 | 3071 |
| SA | `sa_wiki.txt` | 120419 | 81013 | 27084 | 1901 | 23826 | 2245 |
| **UNION** | | **1229359** | | | **13788** unique | | |

**Corruption evidence (red pill):**
- Every file contains Wikipedia chrome: `bodyContent`, "Jump to content", MediaWiki search chrome.
- **HI/TE/SA are grapheme-shattered:** ~39–48% of whitespace tokens are **single characters** (`"भ रत व क प ड य …"` instead of coherent words). That **invents** thousands of fake "words" under `WORD_RE` (single Devanagari/Telugu code points are matches).
- EN is less shattered (~5% single-char tokens) but still nav-bloated HTML dump prose.
- Size ratios vs raw (~3.3–4.9×) therefore do **not** mean "3–5× more real content."
- Overlap check: raw EN unique vs wiki ~98.8% overlap; raw HI ⊂ wiki? only **15.2%** — cleanup/encoding broke Indic serial identity.

**Use these numbers only as "what that broken file contains," not as article word counts.**

### C) `data/claude_2/raw`

| Lang | bytes | chars | total words | unique | ws total | ws unique |
|------|------:|------:|------------:|-------:|---------:|----------:|
| EN | 65467 | 65327 | 10336 | 3131 | 10121 | 3774 |
| HI | 111798 | 43796 | 8140 | 2288 | 8077 | 2453 |
| TE | 52342 | 20012 | 2531 | 1562 | 2511 | 1650 |
| SA | 34305 | 12701 | 1557 | 920 | 1548 | 921 |
| **UNION unique** | | | | **7680** | | |

Almost the raw HI/TE/SA, **shorter EN** (more-aggressively cleaned English extract). Still **fits 10k**.

### D) `data/claude_2/raw_full` — name is fraudulent

| Lang | bytes | chars | total words | unique |
|------|------:|------:|------------:|-------:|
| EN | 65097 | 64963 | 10306 | 3122 |
| HI | 111325 | 43531 | 8108 | 2276 |
| TE | 48324 | 18318 | 2306 | 1453 |
| SA | 33840 | 12380 | 1391 | 918 |
| **UNION unique** | | | | **7575** |

**Smaller than `data/raw` and slightly smaller than `claude_2/raw`.** Not a "full" mirror of Downloads or of English WP. Fits 10k easily. Do not cite as larger ground truth.

### E) `/home/rev/Downloads/*` Wikipedia HTML — **user's real full pages**

#### E1) Preferred strip: `#mw-content-text` via HTMLParser + clean_text

| Lang | HTML bytes | stripped chars | total words | unique |
|------|-----------:|---------------:|------------:|-------:|
| EN | 2969425 | 218062 | **32802** | **7764** |
| HI | 2049514 | 95758 | 15540 | 4135 |
| TE | 1148250 | 41327 | 5679 | 2587 |
| SA | 717627 | 19396 | 2225 | 1334 |
| **UNION unique** | | | | **14252** |

#### E2) Full-page naive tag strip (more chrome leakage)

| Lang | total words | unique |
|------|------------:|-------:|
| EN | 34640 | 8159 |
| HI | 16990 | 4553 |
| TE | 6792 | 3092 |
| SA | 2944 | 1814 |
| **UNION unique** | | **15021** |

**Compare to raw Approach C EN:**  
Downloads EN stripped words **32802** vs raw **12328** → **~2.66×** more EN words on the real save.  
HI: 15540 vs 8141 ≈ **1.91×**. TE: 5679 vs 2532 ≈ **2.24×**. SA: 2225 vs 1558 ≈ **1.43×**.

**Unique-union:** **14252–15021 ≫ 10000.** Whole-word inventory **cannot** place every unique word in a 10k vocab on the full downloads.

---

## 3. Union unique comparison table (main exhibit)

| Source family | EN words | EN unique | Union unique (WORD_RE) | Fits ≤10k whole-word vocab? |
|---------------|---------:|----------:|-----------------------:|:----------------------------|
| **A. `data/raw` (Approach C)** | **12328** | 3584 | **8134** | **YES** (headroom 1866 words; 1689 after +chars→8311) |
| B. `data/*_wiki.txt` (corrupted) | 92812† | 10893† | **13788**† | **NO** (†counts polluted by chrome + shattered Indic) |
| C. `claude_2/raw` | 10336 | 3131 | **7680** | YES |
| D. `claude_2/raw_full` | 10306 | 3122 | **7575** | YES (not full) |
| E. Downloads HTML (main content strip) | **32802** | 7764 | **14252** | **NO — blows past 10k by ~4.3k** |
| E'. Downloads HTML (full naive strip) | 34640 | 8159 | **15021** | **NO — ~5k over** |

Relative to raw union 8134:

| Family | Δ unique vs raw | Relative |
|--------|----------------:|----------|
| claude raw_full | −559 | 0.93× |
| claude raw | −454 | 0.94× |
| raw | 0 | 1.00× |
| wiki_txt | +5654 | 1.69× (junk-inflated) |
| Downloads strip | +6118 | **1.75×** |
| Downloads fullstrip | +6887 | **1.85×** |

---

## 4. Explicit Q&A (assigned questions)

### Are the earlier claims (~8134 unique union, EN ~12328 words) CORRECT for `data/raw`?

**YES. Exact.**
- EN total words (harness clean + WORD_RE) = **12328**
- Union unique words = **8134**
- Per-lang uniques ≈ 3584 / 2288 / 1562 / 920
- Approach C's reported **8311** = **8134 words + residual character types** present in cleaned pages (`|words ∪ all chars in cleaned text| = 8311`). Notes were slightly loose calling 8311 "singles" but consistent with what the runner serializes.

### Are they WRONG for the user's larger recent dataset (`*_wiki.txt` / Downloads)?

**YES — completely wrong as descriptors of those corpora.**
- On **Downloads (true HTML saves)**: EN words ~**33k**, union unique ~**14.3k–15.0k**.
- On **`*_wiki.txt`**: would *appear* even larger on EN totals (92k) and ~13.8k unique, but files are **shredded HTML extractions**; they must **not** be used as fair word denoms without re-extraction.
- Neither family matches 8134 / 12328.

### If Approach C used only `data/raw` under 10k unique — does the FULL download unique-union still fit in 10k or blow past?

**It blows past 10k.**  
Full Downloads main-content unique-union ≈ **14252**. Cap 10000 ⇒ at least **~4252 unique page-words would be OOV** under a pure whole-word inventory (more if you also reserve slots for digits/punct/chars/merges). **C's ∞ score is not transferable to the full saved Wikipedia pages without changing the problem (larger vocab, subwords, or truncation).**

---

## 5. What Approach C actually trained/scored on (forensic)

| Fact | Evidence |
|------|----------|
| Corpus path | `eval_metrics.RAW` → `data/raw/` with `LANG_FILES` |
| EN file identity | `data/raw/india_en.md` **byte-identical** to `data/india_wikipedia.md` (MD5 `50dabee142714b4ccfce8f77891ded5f`) |
| EN page is **partial relative to browser full article** | Clean EN ~112 KB / ~12.3k words vs Downloads strip ~218 KB / ~32.8k words |
| Non-EN raw files look like **real cleaned article text** | HI starts `# भारत`, TE `# భారతదేశం`, SA `# भारतम्` |
| `*_wiki.txt` is a **worse** later dump, not a superseding clean full-text | Chrome strings + space-separated unicode code units |
| `claude_2/raw_full` is **not** the full Downloads and not larger than harness raw | File sizes prove it |

**Shredded partial page judgment:**
- Harness `data/raw` is **not HTML trash**, but for **EN it is a truncated/cleaned export**, not the full live article the user saved under Downloads (~2.66× fewer words).
- HI/TE/SA raw may also be incomplete vs Downloads (1.4–2.2× fewer words) even when visually coherent.
- Calling the experiment "India Wikipedia tokenization fairness" without saying **"on these four ~partial cleaned extracts totaling ~310 KB"** is **under-disclosure**. The ∞ whole-word win **depends** on that under-size corpus.

---

## 6. Sensitivity: counting definition matters (but not enough to save the claim)

| Definition on `data/raw` | EN total | Union unique |
|--------------------------|---------:|-------------:|
| Harness clean + WORD_RE (**official**) | 12328 | **8134** |
| No clean + WORD_RE | 16404 | 8849 |
| Clean + `\S+` | 12192 | EN-unique 4394 (ws) |
| Downloads strip + WORD_RE | 32802 | **14252** |

Even the **harshest** scrub of Downloads that this audit tried still leaves union **>14k**. You cannot "regex your way" back under 10k without dropping large content regions or switching to subword units.

---

## 7. Implications for the swarm narrative

1. **Approach C score=∞ is correct *on raw*.** Identity encode over an 8134-type closed inventory is a valid exploit of the words-denom metric **when** types ≤ 10k.
2. **That exploit dies on full pages.** Unique-union **14k+** means pure whole-word X=1 is **forbidden** by the hard vocab cap. C becomes "top-10k most frequent types + char fallback" → X_i > 1 unevenly.
3. **Leaderboard packaging that sells "works on India Wikipedia"** should be rewritten as **"works on the cleaned partial extracts in `data/raw`"**.
4. **`data/*_wiki.txt` must not be used as D_i or vocab inventory** until re-parsed; they will **fake-inflate** Devanagari/Telugu word counts via code-unit splitting.
5. **True next experiment corpus** = re-extract text from the four Downloads HTML files (or MediaWiki API), then recompute unique-union, then re-run A–E. Expect:
   - whole-word vocab fits → fails under 10k
   - packing oracles still game T on known pages
   - pure BPE relative importance increases
6. **Champion residual "8134 words"** in hybrid vocab breakdown is raw-derived; on full data that residual alone already exceeds budget before spans.

---

## 8. Reproduction

```bash
cd /home/rev/timepass/transformer-learning/tokenizer
python3 - <<'PY'
import sys
sys.path.insert(0, "data/grok_2/harness")
from eval_metrics import load_corpora, extract_words
corpora = load_corpora()
union = set()
for L, t in corpora.items():
    w = extract_words(t)
    print(L, "words", len(w), "unique", len(set(w)))
    union |= set(w)
print("UNION", len(union))
PY
```

Also see: `data/grok_2/results/_audit_counts.json`.

---

## 9. Final punchlines

- **Claims 8134 / 12328: TRUE on `data/raw`. FALSE as "the dataset the user actually downloaded."**
- **Approach C's "fits in 10k" is an artifact of incomplete pages.**
- **Full Downloads unique-union ≈ 14.3k → 10k whole-word inventory is mathematically impossible without OOV.**
- **`raw_full` is not full; `*_wiki.txt` is shredded; browser HTML is the honest large source.**
- Stop laundering partial-page ∞ scores as a solution to multilingual Wikipedia tokenization.
