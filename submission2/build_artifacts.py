"""Build the downloadable widget artifacts (vocab.json + tokens.txt) from the
trained tokenizer.json + metrics.json. Run after train.py.

    python build_artifacts.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
tok = json.loads((ROOT / "tokenizer.json").read_text(encoding="utf-8"))
met = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))

vocab = tok["vocab"]
merges = tok["merges"]
r = met["ratios"]

payload = {
    "assignment": "ERA V5 Assignment 2 - multilingual BPE tokenizer",
    "vocab_size": len(vocab),
    "num_merges": len(merges),
    "languages": ["English", "Hindi", "Telugu", "Maithili"],
    "weights": met["weights"],
    "scores": {
        "EN": r["EN"],
        "HI": r["HI"],
        "TE": r["TE"],
        "MAI": r["MAI"],
        "spread": met["spread"],
        "score": met["adjusted_score"],
    },
    "vocab": vocab,
    "merges": merges,
}

(ROOT / "vocab.json").write_text(
    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
)

# tokens.txt: one "id<TAB>token" line per vocab entry, ordered by id.
id_to_token = {i: t for t, i in vocab.items()}
lines = [f"{i}\t{id_to_token[i]}" for i in range(len(vocab))]
(ROOT / "tokens.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(
    f"wrote vocab.json ({len(vocab)} tokens, {len(merges)} merges) + "
    f"tokens.txt; score={met['adjusted_score']} spread={met['spread']}"
)
