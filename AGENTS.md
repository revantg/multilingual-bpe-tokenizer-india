# Agent isolation rules — `data/grok_2/`

All experimental work stays **inside this directory**.  
**No symlinks to files outside `grok_2/`.**

## Layout (only top-level folders)

| Folder | Write? | Purpose |
|--------|:------:|---------|
| `approaches/` | YES | Methods `a`…`e`, `champion` |
| `datasets/` | YES | Corpora + HTML→MD builders + board JSONs |
| `demo/` | YES | `slides/`, `widget/`, local `lab/` |
| `harness/` | YES | Shared metrics |
| `results/` | YES | PRIMARY score + early boards |
| `archive/` | YES | Dead vocabs |
| `docs/` | YES | Structure notes |
| `_session_audit/` | YES | Session file inventory |

## Rules

1. New code → `approaches/<id>/` or `datasets/build_*/` or `demo/lab/`.
2. Training data → `datasets/` only (copies live here).
3. Netlify artifacts → `demo/slides` and `demo/widget`.
4. Do not create `data/grok_2/approach_b` at top level again — use `approaches/b`.
5. No `ln -s` to `data/full_page_md`, `data/submission`, etc.
6. If you need an outer path temporarily, **copy** into `grok_2` then work.

## Path conventions in Python

```python
GROK2 = Path(__file__).resolve().parents[2]  # from approaches/<id>/file.py
# harness:        GROK2 / "harness"
# BPE engine:     GROK2 / "approaches" / "a"
# corpus:         GROK2 / "datasets" / "full_page_md"
# shipped model:  GROK2 / "approaches" / "b" / "vocab_optimized_fullpage.json"
```
