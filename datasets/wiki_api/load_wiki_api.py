#!/usr/bin/env python3
"""Load India page plaintext fetched via MediaWiki extracts API."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent
LANG_TO_FILE = {
    "EN": "en.txt",
    "HI": "hi.txt",
    "TE": "te.txt",
    "SA": "sa.txt",
}


def load_corpora(raw_dir: Optional[Path] = None) -> Dict[str, str]:
    raw_dir = raw_dir or ROOT
    out: Dict[str, str] = {}
    for lang, fname in LANG_TO_FILE.items():
        p = raw_dir / fname
        if not p.exists():
            raise FileNotFoundError(p)
        # API already returns explanatory plaintext; keep as-is (no markdown stripping)
        out[lang] = p.read_text(encoding="utf-8").strip()
    return out


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT.parent / "grok_2" / "harness"))
    from eval_metrics import LANGS, extract_words

    c = load_corpora()
    for L in LANGS:
        w = extract_words(c[L])
        print(f"{L}: chars={len(c[L])} words={len(w)} unique={len(set(w))}")
    union = set()
    for L in LANGS:
        union.update(extract_words(c[L]))
    print(f"union unique words={len(union)}")
