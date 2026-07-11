# Datasets (inside `grok_2` only)

All training text and conversion tooling for this agent lives **here**.
Nothing under `datasets/` is a symlink to `data/` outside `grok_2`.

## Map

```
datasets/
├── README.md                 ← this file
├── full_page_md/             ★ PRIMARY training corpus
│   ├── india_{en,hi,te,sa}.md
│   ├── load_full_page.py
│   └── STATS.json
├── full_md/                  stricter / smaller MD conversion
├── wiki_api/                 MediaWiki API plaintext extracts
├── raw/                      early harness extracts
│
├── build_full_page/          how full_page_md was produced + A–E boards
│   ├── html_to_md_fullpage.py
│   ├── run_all.py            retrain/eval all approaches on this corpus
│   ├── results/              A.json, B_optimized.json, …
│   └── reviews/
│
└── build_full_md/            older conversion pipeline + boards
    ├── html_to_md.py
    ├── run_all.py
    └── results/
```

## What `build_*` is (not mystery)

| Folder | Meaning |
|--------|---------|
| **`build_full_page/`** | Scripts that turn downloaded Wikipedia HTML into **`full_page_md/`**, plus metrics after training each approach on that corpus. Formerly called `full_page_rerun/`. |
| **`build_full_md/`** | Older HTML→MD path targeting **`full_md/`**. Formerly called `full_rerun/`. |

They are **dataset builders + score dumps**, not separate “mystery approaches.”

## Primary corpus stats (`full_page_md`, harness-cleaned words)

| Lang | File | Raw chars | Clean chars | Words \(D\) | Unique | MD links |
|------|------|----------:|------------:|------------:|-------:|---------:|
| EN | `india_en.md` | 351 678 | 191 158 | 30 789 | 7 015 | 2 784 |
| HI | `india_hi.md` | 227 643 | 75 783 | 13 630 | 3 589 | 1 182 |
| TE | `india_te.md` | 107 838 | 29 835 | 4 053 | 2 155 | 613 |
| SA | `india_sa.md` | 69 784 | 16 971 | 2 091 | 1 275 | 388 |
| | **union unique** | | | | **12 635** | |

Union unique **> 10k** → cannot store every type as one token under the vocab cap.

## Load in code

```python
import sys
from pathlib import Path
G2 = Path("data/grok_2")  # or absolute
sys.path.insert(0, str(G2 / "harness"))
sys.path.insert(0, str(G2 / "datasets" / "full_page_md"))
from load_full_page import load_corpora
corpora = load_corpora()  # EN/HI/TE/SA cleaned strings
```

## Regenerating pages

```bash
# from repo root (Downloads HTML paths are configured inside the script)
python3 data/grok_2/datasets/build_full_page/html_to_md_fullpage.py
```

Outputs **into** `data/grok_2/datasets/full_page_md/` only.
