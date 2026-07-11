# RED PILL review: HTML → markdown conversion (`full_md`)

**Date:** 2026-07-11  
**Converter:** `/home/rev/timepass/transformer-learning/tokenizer/data/grok_2/full_rerun/html_to_md.py`  
**Outputs:** `/home/rev/timepass/transformer-learning/tokenizer/data/full_md/india_{en,hi,te,sa}.md`, `STATS.json`, `load_full.py`  
**Inputs:** browser-saved Wikipedia HTML under `/home/rev/Downloads/`

---

## Verdict

**YES — ready to retrain A–E on this corpus**, with eyes open.

Not because the converter captured the "full" Downloads pages in the earlier-audit sense (it didn't). Ready because:

1. The markdown is **real article prose**, not shredded chrome.
2. Counts match prior **cleaned article-body** corpora (`claude_2/raw_full` ≈ identical; `data/raw` slightly larger only because EN paste was fatter).
3. Union unique **7560** fits cleanly under a **10k whole-word budget** with ~2.4k headroom — the point of a controlled retrain.
4. The earlier **~32k EN / ~14k union** number was a **generous main-content strip that keeps reference warehouses, navboxes, galleries, tables**. It is *not* a fair "article body" denom. This conversion intentionally kills that mass. That is a product decision, not a silent gutting bug.

If the experiment goal is *"whole-word inventory on real full Wikipedia article text including refs"* → **NO**, wrong corpus; go back to a keep-refs strip.  
If the goal is *"comparable cleaned India article prose EN/HI/TE/SA, fit ≤10k"* → **YES**.

---

## 1. Sample quality (start / middle / end)

### EN (`india_en.md`, 65 221 B, 273 lines)

| Region | What you get |
|--------|----------------|
| **Start** | Clean lede: "India, officially the Republic of India…", full intro paras, then `## Etymology` / `## History` with continuous multi-sentence prose. |
| **Middle** | Around states list: real section structure (`#### States` / `#### Union territories`) with bullet names — listy but legitimate article content, not nav chrome. |
| **End** | Culture → Visual art → Mathematics → Music → Dance → Clothing → Cuisine → **Sports**. Stops on cricket paragraph. **No** See also / Notes / References dump. |

**Prose quality:** High. Long paragraphs (avg ~118 words/para, 0 short stubs). Continuous narrative history/gov/economy/culture. Not shredded.

**Structural nits (cosmetic, not corpus-breaking):**
- Parent `## History`, `## Government and politics`, `## Culture` sit above subsections with no orphan body text of their own (Wikipedia style; empty-ish parent H2 only).
- Stops cleanly before back-matter.

### HI (`india_hi.md`, 111 485 B, 256 lines)

| Region | What you get |
|--------|----------------|
| **Start** | Solid Devanagari lede + named sections `नामोत्पत्ति`, `इतिहास`. |
| **Middle** | Real politics/military prose; some **near-duplicate paragraphs** under `सैन्य शक्ति` that look like WP board / merge residue, not converter shredding. |
| **End** | Culture, festivals, cinema, cuisine — ends coherently on food. |

**Prose quality:** Real HI article text. Not chrome soup. Slightly more "wiki mess" than EN editorially (duplicates), but tokenizer-usable.

### TE (`india_te.md`, 48 424 B, 177 lines)

| Region | What you get |
|--------|----------------|
| **Start** | Clean lede; etymology; history. |
| **Middle** | Transport subsections with real Telugu sentences (rail from Mumbai–Thane 1853, etc.). |
| **End** | National symbols list + trivia (`अल्प विषयాలు`) — Wikpedia-list style, but genuine body. |

**Nits:**
- Stray orphan `**` near history opener (emphasis mismatch).
- Zero-width / ZWJ chars inside one heading (`ప్రాచీన భార‌తంలో…`) — HTML artifact, may add weird "words" if you don't NFC, but mass is tiny.
- Shorter average para (~44 words) than EN — article is genuinely smaller/listier on TE WP, not just over-stripped.

### SA (`india_sa.md`, 34 042 B, 116 lines)

| Region | What you get |
|--------|----------------|
| **Start** | Classical Sanskrit lede on भारतम / भारतगणराज्यम्. Real prose. |
| **Middle** | Thin; some sections are stub/list (`राजनीतिः` is **empty between header and next header**; `अर्थव्यवस्था` is **one sentence**). |
| **End** | Culture / states / national song content — some **redundancy** (same Jan Gan Man narrative appears in `राष्ट्रगीतम्` and `शताब्दोत्सवः`). |

**Prose quality:** Real SA wiki text, genuinely short and uneven. Converter did not invent that thinness; SA WP "India" is not EN WP length.

**Bottom line on sample review:** Four coherent article-body markdowns. **Not shredded. Not chrome. Not reference dumps.** SA has empty/stub politics; TE has a couple of focus glitches. Liveable for tokenizer training.

---

## 2. Rough HTML reality check — over-stripped?

### Replicating the earlier "32k" strip (stdlib, `#mw-content-text`)

| Corpus | EN words | EN unique | Union unique |
|--------|---------:|----------:|-------------:|
| **Generous plaintext** (main content, **keep tables**, drop only script/style/svg/…) | **32 787** | 7 760 | **14 254** |
| Same but **drop tables** | 30 237 | 7 093 | 12 237 |
| **This converter → full_md** (clean_text + WORD_RE) | **10 307** | 3 122 | **7 560** |
| Earlier audit "E1 preferred strip" | 32 802 | 7 764 | 14 252 |

Matches the old audit almost exactly. So the 32k figure is **real and reproducible** — and almost entirely **not** what ordinary article-reader prose is.

### Where EN words go (kept-vs-dropped diagnostic on EN main HTML)

Word-RE mass classified under the converter's drop rules (before Stop @ "See also"):

| Drop reason | ~words |
|-------------|-------:|
| `mw-references` / reference lists | **13 727** |
| `refbegin` | **4 487** |
| `role=navigation` / nav | 2 182 |
| `mw-empty-elt` | 1 841 |
| bare `<table>` | 1 235 |
| galleries | 999 |
| navbox styles / figures / hatnotes / … | ~1k combined |

**Kept (pre-stop) ≈ 10.9k raw words → after stop + clean ≈ 10.3k.**  
About **80 of 107 words in stopped See-also+outward** outline — basically zero body after stop. The stop is working as designed: drop notes/refs/see-also/external.

**Section coverage EN (H2s present in both HTML and MD):**

```
Etymology, History, Geography, Government and politics,
Economy, Demographics, Culture
(+ MD ends Sports under Culture)
```

HTML also has: `See also | Notes | References | Bibliography | External links` — correctly **omitted**.

**Did we "cut at See also too early"?**  
**No.** "See also" is the *last* real content H2. All major article sections above it are present with full-looking prose. Word density by section (MD):

| Section | words |
|---------|------:|
| Intro | 672 |
| Etymology | 211 |
| History | 2 241 |
| Geography | 1 303 |
| Gov/pol | 1 591 |
| Economy | 686 |
| Demographics | 1 176 |
| Culture | 2 427 |

That is a **whole EN article body**, not a half-leaked lede.

### Ratio story (full_md vs generous strip)

| Lang | full_md words | generous words | retention |
|------|-------------:|---------------:|----------:|
| EN | 10 307 | 32 787 | **~31%** |
| HI | 8 111 | 15 535 | **~52%** |
| TE | 2 307 | 5 677 | **~41%** |
| SA | 1 541 | 2 225 | **~69%** |

**EN loss looks dramatic because EN WP is citation-nuclear.** Reference lists + bibliography catalogues alone are ~18k WORD_RE tokens — almost two full article bodies. Those words are **citation chrome**, not narrative used for fair X_i demoms if the claim is "tokenize Wikipedia article about India".

**Comparison to other "clean" families (same harness):**

| Family | EN | HI | TE | SA | Union |
|--------|---:|---:|---:|---:|------:|
| **full_md (this)** | **10307** | **8111** | **2307** | **1541** | **7560** |
| claude_2/raw_full | 10306 | 8108 | 2306 | 1391 | 7575 |
| claude_2/raw | 10336 | 8140 | 2531 | 1557 | 7680 |
| data/raw (Approach C era) | 12328 | 8141 | 2532 | 1558 | 8134 |
| Downloads generous | 32787 | 15535 | 5677 | 2225 | 14254 |

**This conversion is essentially `claude_2/raw_full` re-derived from Downloads HTML**, not a new larger ground truth. vs `data/raw`, EN is **~2k words thinner** (raw EN had more link/markdown residue and slightly longer paste). Downstream: union under 10k **more** comfortable than C's 8134.

### Blunt call on over-strip

- **Relative to "every visible text node under mw-content-text":** yes, intentionally over-stripped (~3× less EN).  
- **Relative to "readable India article for tokenizer demoms":** **not gutted.** History through Sports is there.  
- **Relative to goal of testing under 10k after full HTML save:** this is the *only* strip among the audits that still fits. The "14k union" corpus makes pure whole-word X=1 **impossible** without OOV/top-k truncation.

---

## 3. Harness re-count (confirmed)

Command equivalent: `clean_text` + `extract_words` from `grok_2/harness/eval_metrics.py` on `full_md/*.md`.

| Lang | total_words | unique_words | chars (raw) | chars_cleaned |
|------|------------:|-------------:|------------:|--------------:|
| EN | **10307** | **3122** | 65043 | 64856 |
| HI | **8111** | **2276** | 43678 | 43527 |
| TE | **2307** | **1453** | 18402 | 18301 |
| SA | **1541** | **904** | 12572 | 12495 |
| **UNION** | | **7560** | | |
| union > 10000? | | **False** | | |

**Matches `STATS.json` exactly** (and the harness numbers the user reported).  
On-disk md is bitwise-equivalent to a fresh `html_to_md.html_file_to_markdown` pass on the EN HTML (10307 words again). Converter is stable.

`load_full.py` is a thin loader wrapping the same `clean_text` — good for retrain scripts.

---

## 4. One-shot generous EN plaintext (stdlib) comparison

Already done above; short form:

| Method | EN words | EN unique |
|--------|---------:|----------:|
| Full-page-ish main content, keep tables, no chrome class filtering | **32 787** | 7 760 |
| Same, drop tables | 30 237 | 7 093 |
| Converter markdown after clean | **10 307** | 3 122 |

**Δ ≈ 22 480 EN words removed** of which ~**18k are refs/refbegin** and the rest nav/tables/galleries/empty-elts. The 10k result is "article narrative"; the 32k result is "article + apparatus".

---

## 5. Ready to retrain? **YES**

### Use this corpus if you want

- Controlled re-run of A–E under **hard_ok: vocab ≤ 10k and X_en ≤ 1.2**.
- Fair multi-script astronomy of EN Latin + HI Devanagari + TE Telugu + SA Devanagari on **comparable article spines**.
- Continuity with prior swarm numbers (~7.5–8.1k union; Approach C ∞ on raw was corpus-local grade inflation relative to *Downloads-as-stripped-blobs*, but **not** relative to *Downloads-as-article-body*).

### Do **not** claim

- "Trained on full Wikipedia Downloads HTML." You trained on **cleaned article body extracted from** those HTML saves.
- "Whole-word inventory still covers every unique word on the raw main-content strip." That is **false** (14k ≫ 10k). Only true for this cleaned body (7.5k < 10k).

### Known limitations that retrain must live with

1. **Denom imbalance is structural:** EN 10.3k words vs SA 1.5k. Fairness score will always reward upsampling / budget-per-lang approaches over naive global BPE.
2. **SA stub sections** (`राजनीतिः` empty) shrink SA D further; don't interpret TE/SA X as "the languages are harder," some of it is "the articles are shorter."
3. **TE orphan `**`, ZWJ in a heading** — noise floor, not training blocker.
4. **Citation numbers stripped** (`[1]`) — good; if any approach overfit to digit soup on old raws, this corpus removes that cheat.
5. This is **not** a harder corpus than `data/raw`. It is **slightly easier** on unique-union (7560 vs 8134). Any approach that scored ∞ on raw should still score ∞ *here* if it puts every word in vocab — retrain is a **replication**, not a stress test against the 14k wall.

### If you want the stress test instead

Reconvert with:
- keep reference *section prose* out but **do** allow more tables / untouch list prose, **or**
- main-content strip without `mw-references` drop and raise vocab to 15–16k / admit OOV.

That is a **different experiment**.

---

## 6. Operational checklist for retrain

```text
corpora   = full_md/load_full.py :: load_corpora(cleaned=True)
D_i       = len(extract_words(corpora[lang]))   # words mode
stats     = full_md/STATS.json                  # already validated
union     = 7560  →  whole-word inventory still has ~2440 free slots for merges/special/chars
```

Suggested compare baselines in results/:
- Approach C words-only ∞ reincarnation on full_md (sanity: still under 10k).
- A/B/D/E re-eval with same harness — prove ranking still holds when EN is 10307 not 12328.

---

## Bottom line one-liner

**Prose is real and complete through Sports; the 3× EN shrink vs the old 32k audit is almost all refs/nav/tables, not missing article sections; harness counts match STATS; ready to retrain under 10k — just don't market it as "full Downloads unique inventory."**
