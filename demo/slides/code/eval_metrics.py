#!/usr/bin/env python3
"""
Shared evaluation harness for multilingual fairness score.

X_i = T_i / D_i
score = 1000 / (max X - min X)   (inf if gap ~ 0)

Denom modes:
  words  — D_i = word count of page (default; fairest fairness metric)
  unique — D_i = unique word count
  fixed  — D_i = 5000 for all languages
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

ROOT = Path(__file__).resolve().parents[1]  # grok_2/
RAW = ROOT.parent / "raw"  # data/raw/
LANGS = ("EN", "HI", "TE", "SA")
LANG_FILES = {
    "EN": "india_en.md",
    "HI": "india_hi.txt",
    "TE": "india_te.txt",
    "SA": "india_sa.txt",
}
WORD_RE = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?|"
    r"[\u0900-\u097F]+|"
    r"[\u0C00-\u0C7F]+|"
    r"\d+"
)


def clean_text(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[\^[^\]]+\]", " ", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[|*_`]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_words(text: str) -> List[str]:
    return WORD_RE.findall(text)


def load_corpora(raw_dir: Optional[Path] = None) -> Dict[str, str]:
    raw_dir = raw_dir or RAW
    out = {}
    for lang, fname in LANG_FILES.items():
        p = raw_dir / fname
        out[lang] = clean_text(p.read_text(encoding="utf-8"))
    return out


def denominators(
    corpora: Dict[str, str], mode: str = "words"
) -> Dict[str, float]:
    words = {k: extract_words(v) for k, v in corpora.items()}
    if mode == "words":
        return {k: float(len(w)) for k, w in words.items()}
    if mode == "unique":
        return {k: float(len(set(w))) for k, w in words.items()}
    if mode == "fixed":
        return {k: 5000.0 for k in corpora}
    raise ValueError(f"unknown denom mode: {mode}")


def evaluate_encode(
    encode_fn: Callable[[str, str], Sequence],
    corpora: Optional[Dict[str, str]] = None,
    denom_mode: str = "words",
    vocab_size: Optional[int] = None,
    name: str = "approach",
) -> Dict:
    """
    encode_fn(text, lang) -> token sequence (or list of ids/tokens).
    """
    corpora = corpora or load_corpora()
    D = denominators(corpora, denom_mode)
    T: Dict[str, int] = {}
    X: Dict[str, float] = {}
    for lang in LANGS:
        toks = encode_fn(corpora[lang], lang)
        T[lang] = len(toks)
        X[lang] = T[lang] / D[lang] if D[lang] else float("inf")
    xs = list(X.values())
    gap = max(xs) - min(xs)
    score = float("inf") if gap < 1e-15 else 1000.0 / gap
    x_en = X["EN"]
    hard_ok = x_en <= 1.2 and (vocab_size is None or vocab_size <= 10000)
    return {
        "name": name,
        "denom_mode": denom_mode,
        "T": T,
        "D": D,
        "X": X,
        "gap": gap,
        "score": score,
        "vocab_size": vocab_size,
        "X_en_le_1_2": x_en <= 1.2,
        "vocab_ok": vocab_size is None or vocab_size <= 10000,
        "hard_ok": hard_ok,
        "sorted_X": sorted(((k, X[k]) for k in LANGS), key=lambda kv: -kv[1]),
    }


def report(result: Dict) -> str:
    lines = [
        f"=== {result['name']} (denom={result['denom_mode']}) ===",
        f"vocab={result.get('vocab_size')}  hard_ok={result['hard_ok']}  "
        f"X_en<=1.2={result['X_en_le_1_2']}  vocab_ok={result['vocab_ok']}",
    ]
    for lang in LANGS:
        lines.append(
            f"  {lang}: T={result['T'][lang]:7d}  D={result['D'][lang]:8.1f}  "
            f"X={result['X'][lang]:.6f}"
        )
    sc = result["score"]
    sc_s = "inf" if sc == float("inf") else f"{sc:.4f}"
    lines.append(f"  gap={result['gap']:.8f}  score={sc_s}")
    order = " > ".join(f"{k}({v:.4f})" for k, v in result["sorted_X"])
    lines.append(f"  order: {order}")
    return "\n".join(lines)


def save_result(result: Dict, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(result)
    if serializable["score"] == float("inf"):
        serializable["score"] = "inf"
    # JSON-safe keys
    for key in ("T", "D", "X"):
        serializable[key] = {k: float(v) for k, v in result[key].items()}
    serializable["sorted_X"] = [
        [k, float(v)] for k, v in result["sorted_X"]
    ]
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


if __name__ == "__main__":
    corpora = load_corpora()
    for mode in ("words", "unique", "fixed"):
        D = denominators(corpora, mode)
        words = {k: extract_words(v) for k, v in corpora.items()}
        # trivial whole-word oracle so T = W for words mode

        def enc(text, lang, _w=words):
            return extract_words(text)

        r = evaluate_encode(enc, corpora, mode, vocab_size=10000, name="whole-word-oracle")
        print(report(r))
        print()
