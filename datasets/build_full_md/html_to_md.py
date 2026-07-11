#!/usr/bin/env python3
"""
Convert saved Wikipedia HTML (EN/HI/TE/SA India articles) into clean UTF-8 markdown.

Pure Python stdlib (html.parser). Extracts main article body, strips chrome
(navboxes, reference lists, scripts, styles, infoboxes), keeps prose.

Writes:
  data/full_md/india_{en,hi,te,sa}.md
  data/full_md/STATS.json
  data/full_md/load_full.py
"""
from __future__ import annotations

import html as html_lib
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

GROK2 = Path(__file__).resolve().parents[2]  # grok_2
DATA = GROK2  # keep name for compat
FULL_MD = GROK2 / "datasets" / "full_md"
HARNESS_DIR = Path(__file__).resolve().parents[2] / "harness"
sys.path.insert(0, str(HARNESS_DIR))

INPUTS = {
    "EN": Path("/home/rev/Downloads/India - Wikipedia.html"),
    "HI": Path("/home/rev/Downloads/भारत - विकिपीडिया.html"),
    "TE": Path("/home/rev/Downloads/భారతదేశం - వికీపీడియా.html"),
    "SA": Path("/home/rev/Downloads/भारतम् - विकिपीडिया.html"),
}
OUT_FILES = {
    "EN": "india_en.md",
    "HI": "india_hi.md",
    "TE": "india_te.md",
    "SA": "india_sa.md",
}

# HTML void elements — never emit end tags; must not affect skip-depth.
VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)

# Entire subtree dropped (and not void-only).
SKIP_TREE_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        "iframe",
        "svg",
        "math",
        "figure",
        "figcaption",
        "audio",
        "video",
        "object",
        "canvas",
        "form",
        "button",
        "select",
        "option",
        "textarea",
        "table",  # drop tables (infoboxes, data tables, chrome)
    }
)

# Class/id tokens that mark chrome to drop (matched as whole class tokens or id).
DROP_CLASS_EXACT = frozenset(
    {
        "navbox",
        "navbox-styles",
        "navbox-inner",
        "infobox",
        "sidebar",
        "vertical-navbox",
        "sistersitebox",
        "metadata",
        "noprint",
        "nomobile",
        "mw-editsection",
        "mw-authority-control",
        "reflist",
        "references",
        "mw-references",
        "refbegin",
        "refend",
        "hatnote",
        "navigation-not-searchable",
        "shortdescription",
        "ambox",
        "tmbox",
        "ombox",
        "cmbox",
        "fmbox",
        "imbox",
        "portal",
        "portal-bar",
        "mw-empty-elt",
        "gallerybox",
        "catlinks",
        "printfooter",
        "mw-indicators",
        "thumb",
        "thumbinner",
        "tmulti",
        "gallery",
        "navbox-abovebelow",
        # note: do NOT drop "mw-heading" — modern Vector wraps real h2/h3 in it
        "mw-cite-backlink",
        "reference-text",
        "toc",
        "toccolours",
        "mw-indicator",
        "sidebar-content",
    }
)

DROP_ID_EXACT = frozenset(
    {
        "toc",
        "mw-toc-heading",
        "coordinates",
        "siteSub",
        "contentSub",
        "jump-to-nav",
        "catlinks",
    }
)

# Drop subtree if any class substring matches (for compound class names).
DROP_CLASS_SUBSTR = (
    "navbox",
    "infobox",
    "reflist",
    "shortdescription",
    "sistersite",
    "authority-control",
    "mw-editsection",
    "mbox-",
)

STOP_SECTION_RE = re.compile(
    r"^\s*("
    r"See\s+also|Notes|References|External\s+links|Bibliography|Further\s+reading|"
    r"Citations|Sources|Works\s+cited|Footnotes|Gallery|"
    # Hindi
    r"इन्हें\s+भी\s+देखें|टिप्पणी|टिप्पणियाँ|टिप्पणियां|सन्दर्भ|संदर्भ|"
    r"बाहरी\s+कड़ियाँ|बाह्य\s+कड़ियाँ|बाहरी\s+कडियाँ|ग्रंथसूची|आगे\s+पढ़ें|"
    r"संदर्भ\s+ग्रंथ|सहायक\s+ग्रंथ|संदर्भ\s+सूची|चित्र\s*दीर्घा|चित्रदीर्घा|"
    # Telugu (including condensed ఇవికూడా without spaces)
    r"ఇవి\s*కూడా\s*చూడండి|మూలాలు|ఆధారాలు|గమనికలు|వనరులు|బయటి\s*లింకులు|"
    r"ఇంకా\s*చదవండి|గ్రంథ\s*సూచి|చిత్రమాలిక|చిత్ర\s*మాలిక|"
    # Sanskrit
    r"अपि\s+पश्यतु|इदम्\s+अपि\s+पश्यतु|टिप्पण्यः|सन्दर्भाः|बाह्यसम्पर्काः|"
    r"सहाय्यग्रन्थाः|सहायकग्रन्थाः|आधारग्रन्थाः?|अधिकपठनम्|उल्लेखाः|टिप्पणीः|"
    r"चित्रशाला"
    r")\s*:?\s*$",
    re.I | re.U,
)

INLINE_EMPHASIS = {
    "b": "**",
    "strong": "**",
    "i": "*",
    "em": "*",
}


def _attr_dict(attrs) -> Dict[str, str]:
    return {k: (v or "") for k, v in attrs}


def _classes(attrs: Dict[str, str]) -> List[str]:
    return (attrs.get("class") or "").split()


def should_drop_element(tag: str, attrs: Dict[str, str]) -> bool:
    if tag in SKIP_TREE_TAGS:
        return True
    if (attrs.get("role") or "") == "navigation":
        return True
    cid = attrs.get("id") or ""
    if cid in DROP_ID_EXACT:
        return True
    classes = _classes(attrs)
    for c in classes:
        if c in DROP_CLASS_EXACT:
            return True
        cl = c.lower()
        for sub in DROP_CLASS_SUBSTR:
            if sub in cl:
                return True
    # "references" id used by list of refs
    if cid.lower() in ("references", "notes", "bibliography", "external_links"):
        # Do not drop whole article; section write-stop handles these.
        pass
    return False


def is_ref_sup(attrs: Dict[str, str]) -> bool:
    classes = set(_classes(attrs))
    ctype = attrs.get("typeof") or ""
    if classes & {"reference", "mw-ref", "Reference"}:
        return True
    if "mw:Extension/ref" in ctype or "mw:Extension/references" in ctype:
        return True
    return False


def extract_main_html(raw: str) -> Tuple[str, str]:
    """Return (title_text, main_inner_html)."""
    title = ""
    m = re.search(r'id="firstHeading"[^>]*>(.*?)</h1>', raw, flags=re.I | re.S)
    if m:
        title = re.sub(r"<[^>]+>", "", m.group(1))
        title = html_lib.unescape(re.sub(r"\s+", " ", title)).strip()
        # Separately, some skins leak UI labels into firstHeading text nodes
        for noise in (
            r"संक्षिप्त URL.*$",
            r"Get shortened URL.*$",
            r"Short URL.*$",
            r"చిన్న URL.*$",
        ):
            title = re.sub(noise, "", title, flags=re.I).strip()

    start = raw.find('id="mw-content-text"')
    if start < 0:
        start = raw.find('id="bodyContent"')
    if start < 0:
        raise ValueError("Could not find mw-content-text / bodyContent")

    po = raw.find("mw-parser-output", start)
    if 0 <= po < start + 8000:
        div_start = raw.rfind("<", start, po + 1)
        start = div_start if div_start >= 0 else po
    else:
        gt = raw.find(">", start)
        start = gt + 1 if gt >= 0 else start

    end_candidates = []
    for marker in (
        'id="catlinks"',
        'class="printfooter"',
        'id="mw-data-after-content"',
        'id="footer"',
        'class="mw-footer"',
    ):
        i = raw.find(marker, start)
        if i > start:
            tag = raw.rfind("<", start, i)
            end_candidates.append(tag if tag > start else i)
    end = min(end_candidates) if end_candidates else len(raw)
    return title, raw[start:end]


class WikiHTMLToMarkdown(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._skip_depth = 0
        self._suppress_ref = 0
        self._heading_level = 0
        self._heading_buf: List[str] = []
        self._stopped = False
        self._list_stack: List[str] = []
        self._in_p = False
        self._in_pre = False
        self._emit_mode: Optional[str] = None
        self._block_buf: List[str] = []
        self._bold = 0
        self._italic = 0

    # ---- emitters ----
    def _write_inline(self, s: str) -> None:
        if not s or self._stopped or self._skip_depth or self._suppress_ref:
            return
        if self._heading_level:
            self._heading_buf.append(s)
            return
        self._block_buf.append(s)

    def _flush_block(self, kind: str = "p") -> None:
        if self._stopped:
            self._block_buf = []
            return
        text = "".join(self._block_buf)
        self._block_buf = []
        if kind == "pre":
            text = text.strip("\n")
            if text.strip():
                self.parts.append("```\n" + text + "\n```\n\n")
            return
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n+", " ", text).strip()
        # residual bracket citations
        text = re.sub(r"\[\s*\d+\s*\]", "", text)
        text = re.sub(r"\[\s*[a-z]\s*\]", "", text, flags=re.I)
        text = re.sub(r" {2,}", " ", text).strip()
        if not text:
            return
        if kind == "li":
            indent = "  " * max(0, len(self._list_stack) - 1)
            bullet = "-" if (self._list_stack and self._list_stack[-1] == "ul") else "1."
            self.parts.append(f"{indent}{bullet} {text}\n")
        else:
            self.parts.append(text + "\n\n")

    def _emit_heading(self, level: int, text: str) -> None:
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(
            r"\[\s*(edit|सम्पादन|संपादित|మార్చు|সম্পাদনা)\s*\]",
            "",
            text,
            flags=re.I,
        ).strip()
        if not text:
            return
        if STOP_SECTION_RE.match(text):
            self._stopped = True
            return
        level = max(1, min(6, level))
        self.parts.append("#" * level + " " + text + "\n\n")

    # ---- tag handlers ----
    def _enter_skip(self) -> None:
        self._skip_depth += 1

    def _start(self, tag: str, attrs) -> None:
        if self._stopped:
            return
        tag = tag.lower()
        ad = _attr_dict(attrs)

        # Void tags: never affect skip stack except ignore their content (none).
        if tag in VOID_TAGS:
            if tag == "br":
                if not self._skip_depth and not self._suppress_ref:
                    self._write_inline("\n")
            elif tag == "hr":
                if not self._skip_depth and not self._suppress_ref and not self._heading_level:
                    if self._block_buf and self._in_p:
                        self._flush_block("p")
                        self._in_p = False
                    self.parts.append("---\n\n")
            return

        if self._skip_depth:
            self._skip_depth += 1
            return

        if should_drop_element(tag, ad):
            self._enter_skip()
            return

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            if self._in_p and self._block_buf:
                self._flush_block("p")
            self._in_p = False
            self._heading_level = int(tag[1])
            self._heading_buf = []
            return

        if tag == "p":
            if self._in_p and self._block_buf:
                self._flush_block("p")
            self._in_p = True
            self._emit_mode = "p"
            self._block_buf = []
            return

        if tag in ("ul", "ol"):
            if self._in_p and self._block_buf:
                self._flush_block("p")
                self._in_p = False
            self._list_stack.append(tag)
            return

        if tag == "li":
            self._emit_mode = "li"
            self._block_buf = []
            return

        if tag == "pre":
            if self._in_p and self._block_buf:
                self._flush_block("p")
                self._in_p = False
            self._in_pre = True
            self._emit_mode = "pre"
            self._block_buf = []
            return

        if tag == "sup" and is_ref_sup(ad):
            self._suppress_ref += 1
            return

        if tag == "sup":
            # Non-ref superscript: still often citation-ish; keep text without markers
            return

        if tag == "span":
            if is_ref_sup(ad):
                self._enter_skip()
                return
            return

        if tag == "a":
            return  # keep anchor text only

        if tag in INLINE_EMPHASIS:
            self._write_inline(INLINE_EMPHASIS[tag])
            return

        if tag == "blockquote":
            if self._in_p and self._block_buf:
                self._flush_block("p")
            self._in_p = True
            self._emit_mode = "p"
            self._block_buf = ["> "]
            return

        if tag == "dd":
            # definition list value as paragraph-ish
            self._emit_mode = "p"
            if not self._block_buf:
                self._block_buf = []
            return

        if tag == "dt":
            self._emit_mode = "p"
            self._block_buf = []
            return

    def _end(self, tag: str) -> None:
        tag = tag.lower()

        if tag in VOID_TAGS:
            return

        if self._skip_depth:
            self._skip_depth = max(0, self._skip_depth - 1)
            return

        if self._stopped:
            return

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag[1])
            text = "".join(self._heading_buf)
            self._heading_level = 0
            self._heading_buf = []
            self._emit_heading(level, text)
            return

        if tag == "p":
            if self._in_p:
                self._flush_block("p")
            self._in_p = False
            self._emit_mode = None
            return

        if tag in ("ul", "ol"):
            if self._list_stack:
                self._list_stack.pop()
            if self.parts and not self.parts[-1].endswith("\n\n"):
                self.parts.append("\n")
            self._emit_mode = None
            return

        if tag == "li":
            self._flush_block("li")
            self._emit_mode = None
            return

        if tag == "pre":
            if self._in_pre:
                self._flush_block("pre")
            self._in_pre = False
            self._emit_mode = None
            return

        if tag == "sup":
            if self._suppress_ref:
                self._suppress_ref = max(0, self._suppress_ref - 1)
            return

        if tag in INLINE_EMPHASIS:
            self._write_inline(INLINE_EMPHASIS[tag])
            return

        if tag == "blockquote":
            if self._in_p:
                self._flush_block("p")
            self._in_p = False
            return

        if tag in ("dt", "dd"):
            if self._block_buf:
                self._flush_block("p")
            self._emit_mode = None
            return

    def handle_starttag(self, tag: str, attrs) -> None:
        self._start(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        self._end(tag)

    def handle_startendtag(self, tag: str, attrs) -> None:
        # XHTML-style void: only start path, explicitly no end
        self._start(tag, attrs)
        # If _start entered skip for a non-void (shouldn't for startend), balance
        tag = tag.lower()
        if tag not in VOID_TAGS and self._skip_depth and should_drop_element(tag, _attr_dict(attrs)):
            # rare self-closing skip container — undo
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._stopped or self._skip_depth or self._suppress_ref:
            return
        if not data:
            return
        self._write_inline(data)

    def handle_entityref(self, name: str) -> None:
        self.handle_data(html_lib.unescape(f"&{name};"))

    def handle_charref(self, name: str) -> None:
        self.handle_data(html_lib.unescape(f"&#{name};"))

    def get_markdown(self) -> str:
        if self._block_buf:
            kind = self._emit_mode or "p"
            self._flush_block("li" if kind == "li" else ("pre" if kind == "pre" else "p"))
        md = "".join(self.parts)
        md = re.sub(r"\*\*\s*\*\*", "", md)
        md = md.replace("****", "")
        md = re.sub(r"[ \t]+\n", "\n", md)
        md = re.sub(r"\[\d+\]", "", md)
        md = re.sub(r"\[[a-z]\]", "", md, flags=re.I)
        # clean spacing only on non-newline runs
        md = re.sub(r"[^\S\n]{2,}", " ", md)
        md = re.sub(r" +([,.;:!?])", r"\1", md)
        md = re.sub(r"\n{3,}", "\n\n", md)
        return md.strip() + "\n"


def html_file_to_markdown(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    title, fragment = extract_main_html(raw)
    parser = WikiHTMLToMarkdown()
    parser.feed(fragment)
    parser.close()
    body = parser.get_markdown()
    if title:
        first = body.split("\n", 1)[0] if body else ""
        if not (first.startswith("# ") and title[: min(12, len(title))] in first):
            body = f"# {title}\n\n{body}"
    return body


def compute_stats(md_paths: Dict[str, Path]) -> dict:
    from eval_metrics import clean_text, extract_words

    per_lang = {}
    all_unique: Set[str] = set()
    for lang, p in md_paths.items():
        raw = p.read_text(encoding="utf-8")
        cleaned = clean_text(raw)
        words = extract_words(cleaned)
        uniq = set(words)
        all_unique |= uniq
        per_lang[lang] = {
            "path": str(p),
            "bytes": len(raw.encode("utf-8")),
            "chars": len(raw),
            "chars_cleaned": len(cleaned),
            "total_words": len(words),
            "unique_words": len(uniq),
        }
    return {
        "languages": per_lang,
        "union_unique_words": len(all_unique),
        "union_gt_10000": len(all_unique) > 10000,
    }


LOAD_FULL_PY = r'''#!/usr/bin/env python3
"""Load full_md India corpora (EN/HI/TE/SA) with harness cleaning."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

FULL_MD = Path(__file__).resolve().parent
HARNESS = FULL_MD.parent / "grok_2" / "harness"
sys.path.insert(0, str(HARNESS))

from eval_metrics import clean_text  # noqa: E402

LANG_FILES = {
    "EN": "india_en.md",
    "HI": "india_hi.md",
    "TE": "india_te.md",
    "SA": "india_sa.md",
}


def load_corpora(raw_dir: Optional[Path] = None, cleaned: bool = True) -> Dict[str, str]:
    """Return EN/HI/TE/SA strings from full_md/*.md (optionally harness-cleaned)."""
    raw_dir = raw_dir or FULL_MD
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
'''


def main() -> int:
    FULL_MD.mkdir(parents=True, exist_ok=True)
    md_paths: Dict[str, Path] = {}
    for lang, src in INPUTS.items():
        if not src.is_file():
            print(f"ERROR: missing input {src}", file=sys.stderr)
            return 1
        print(f"Converting {lang}: {src.name} ...")
        md = html_file_to_markdown(src)
        out = FULL_MD / OUT_FILES[lang]
        out.write_text(md, encoding="utf-8")
        md_paths[lang] = out
        print(f"  wrote {out}  ({len(md.encode('utf-8'))} bytes, {len(md)} chars)")

    loader = FULL_MD / "load_full.py"
    loader.write_text(LOAD_FULL_PY, encoding="utf-8")
    print(f"wrote {loader}")

    stats = compute_stats(md_paths)
    stats_path = FULL_MD / "STATS.json"
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {stats_path}")
    print("\n=== STATS ===")
    for lang in ("EN", "HI", "TE", "SA"):
        s = stats["languages"][lang]
        print(
            f"  {lang}: bytes={s['bytes']} chars={s['chars']} "
            f"words={s['total_words']} unique={s['unique_words']}"
        )
    print(
        f"  UNION unique={stats['union_unique_words']}  "
        f">10000={stats['union_gt_10000']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
