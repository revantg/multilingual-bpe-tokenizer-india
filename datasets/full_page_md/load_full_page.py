#!/usr/bin/env python3
"""Load full_page_md India corpora (EN/HI/TE/SA) with harness cleaning."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

FULL_PAGE_MD = Path(__file__).resolve().parent
HARNESS = FULL_PAGE_MD.parent.parent / "harness"  # grok_2/harness
sys.path.insert(0, str(HARNESS))

from eval_metrics import clean_text  # noqa: E402

LANG_FILES = {
    "EN": "india_en.md",
    "HI": "india_hi.md",
    "TE": "india_te.md",
    "SA": "india_sa.md",
}


def load_corpora(raw_dir: Optional[Path] = None, cleaned: bool = True) -> Dict[str, str]:
    """Return EN/HI/TE/SA strings from full_page_md/*.md (optionally harness-cleaned)."""
    raw_dir = raw_dir or FULL_PAGE_MD
    out: Dict[str, str] = {}
    for lang, fname in LANG_FILES.items():
        text = (raw_dir / fname).read_text(encoding="utf-8")
        out[lang] = clean_text(text) if cleaned else text
    return out


if __name__ == "__main__":
    from eval_metrics import extract_words

    corpora = load_corpora()
    union = set()
    for lang, text in corpora.items():
        w = extract_words(text)
        union |= set(w)
        print(f"{lang}: words={len(w)} unique={len(set(w))} chars={len(text)}")
    print(f"UNION unique={len(union)}  >10000={len(union) > 10000}")
