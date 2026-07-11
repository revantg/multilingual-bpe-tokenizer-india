# Mess map — this greasy `data/` folder vs intended sandbox

You asked agents to keep work **inside one folder**. Here is what actually happened.

## Intended layout

| Folder | Role |
|--------|------|
| `data/grok_2/` | **Only** experimental sandbox for this agent line |
| `data/submission* /` | Final Netlify deliverables |
| `data/full_page_md` etc. | Corpora only |

## What this session's transcript shows

- Unique paths touched (transcript): **105**
- Inside sandbox `grok_2/`: **48**
- Submission packs: **17**
- Corpus trees: **27**
- Scatter outside sandbox: **40**

### Major creations from this line of work (honest)

**Kept in sandbox (good):** trainers, models, demos, resilience notes under `grok_2/`.

**Deliverables (expected):** `submission/`, `submission_slides/`.

**Corpus pollution / intentional regen (borderline):**
- `full_page_md/*` regenerated with markdown links + stats/cross-eval logs
- `wiki_api/*` entire tree (fetch + run_all approaches)
- `full_md/*` cleaner conversion
- `datasets/` index + HTML dump symlinks
- `raw/` sometimes updated

**Loose / other agent dirt already in `data/` (not all from this session):**
- `.mimocode/` (3441 files) — parallel agent or older tree
- `__pycache__/` (1 files) — parallel agent or older tree
- `bpe_merges.json` (43436 bytes) — loose file at data root
- `claude/` (9 files) — parallel agent or older tree
- `claude_2/` (17 files) — parallel agent or older tree
- `clean_convert.py` (11030 bytes) — loose file at data root
- `clean_mediawiki.py` (12556 bytes) — loose file at data root
- `clean_wiki_md.py` (1108 bytes) — loose file at data root
- `codex/` (2089 files) — parallel agent or older tree
- `conftest.py` (340 bytes) — loose file at data root
- `en_wiki.txt` (546878 bytes) — loose file at data root
- `grok/` (3 files) — parallel agent or older tree
- `hi_wiki.txt` (372746 bytes) — loose file at data root
- `india_wikipedia.md` (111684 bytes) — loose file at data root
- `india_wikipedia.md.bak` (113347 bytes) — loose file at data root
- `india_wikipedia.raw` (308906 bytes) — loose file at data root
- `mimo_2/` (1 files) — parallel agent or older tree
- `sa_wiki.txt` (120419 bytes) — loose file at data root
- `sitecustomize.py` (402 bytes) — loose file at data root
- `te_wiki.txt` (189316 bytes) — loose file at data root
- `verify.py` (8286 bytes) — loose file at data root
- `vocab.json` (216574 bytes) — loose file at data root
- `vocab_aggressive_10.json` (537111 bytes) — loose file at data root
- `vocab_aggressive_5.json` (333411 bytes) — loose file at data root
- `vocab_aggressive_6.json` (371992 bytes) — loose file at data root
- `vocab_aggressive_8.json` (454298 bytes) — loose file at data root
- `vocab_bpe.json` (263409 bytes) — loose file at data root
- `vocab_multilingual.json` (879086 bytes) — loose file at data root
- `vocab_optimized.json` (85483 bytes) — loose file at data root
- `vocab_optimized_v2.json` (373326 bytes) — loose file at data root
- `vocab_overfitted.json` (1085233 bytes) — loose file at data root
- `vocab_word_bigram.json` (126320 bytes) — loose file at data root

## Agent + rules added

| Path | Purpose |
|------|---------|
| `tools/session_file_audit.py` | Transcript → file inventory agent |
| `data/grok_2/AGENTS.md` | Isolation rule for future work |
| `data/grok_2/_session_audit/SESSION_FILES.md` | This run's detailed path list |
| `data/grok_2/_session_audit/session_files.json` | Machine-readable same |

## Cleanup suggestion (optional, ask first)

1. Leave `grok_2/`, `submission/`, `submission_slides/`, `full_page_md/` as canonical.
2. Archive or delete parallel agent dirs (`claude/`, `claude_2/`, `codex/`, `mimo_2/`, old `grok/`) if unused.
3. Move or delete loose `data/vocab_*.json`, `en_wiki.txt`, legacy root scripts into `_archive/`.
4. Keep `wiki_api/` only if you still need API corpus; otherwise archive under `grok_2/archive/wiki_api/`.

I will **not** move/delete bulk data without an explicit OK.
