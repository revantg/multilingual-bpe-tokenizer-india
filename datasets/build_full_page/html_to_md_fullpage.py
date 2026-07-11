#!/usr/bin/env python3
"""
Convert FULL Wikipedia HTML pages to markdown with MINIMAL stripping.

Goal: keep essentially all page language inside #mw-content-text / mw-parser-output:
article body, infobox text, tables, lists, captions, References/See also/etc.

Remove only: HTML tags, script/style/SVG noise, pure UI chrome outside content
(extraction region) and light in-content edit chrome (mw-editsection).

Stdlib only (html.parser).

Writes:
  data/full_page_md/india_{en,hi,te,sa}.md
  data/full_page_md/STATS.json
  data/full_page_md/load_full_page.py
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
FULL_PAGE_MD = GROK2 / "datasets" / "full_page_md"
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

# Only drop true non-language noise subtrees (MINIMAL).
SKIP_TREE_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        "iframe",
        "svg",
        "math",
        "audio",
        "video",
        "object",
        "canvas",
        "map",
    }
)

# Drop only pure edit/UI chrome classes — NOT infobox, tables, reflist, thumbs.
DROP_CLASS_EXACT = frozenset(
    {
        "mw-editsection",
        "mw-editsection-bracket",
        "mw-editsection-divider",
        "mw-editsection-visualeditor",
        "mw-empty-elt",
        "noprint",  # often pure UI badges
        "mw-indicators",
        "mw-indicator",
        "printfooter",
        "mw-cite-backlink",  # caret backlinks "^" only
    }
)

DROP_ID_EXACT = frozenset(
    {
        "coordinates",  # geo microformat chrome often duplicated
        "jump-to-nav",
        "mw-toc-heading",  # optional; keep actual #toc list if present — skip only heading id fluff
    }
)

DROP_CLASS_SUBSTR = (
    "mw-editsection",
    "vector-toc-",  # sticky ToC chrome widgets if nested in content dump
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


def _style_hidden(attrs: Dict[str, str]) -> bool:
    style = (attrs.get("style") or "").lower().replace(" ", "")
    if "display:none" in style or "visibility:hidden" in style:
        return True
    if (attrs.get("hidden") is not None) and attrs.get("hidden") != "false":
        # presence of hidden attribute
        return True
    if (attrs.get("aria-hidden") or "").lower() == "true":
        return True
    return False


def should_drop_element(tag: str, attrs: Dict[str, str]) -> bool:
    if tag in SKIP_TREE_TAGS:
        return True
    if _style_hidden(attrs):
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
    # pure form chrome occasionally in content
    if tag in ("button", "select", "option", "textarea", "input"):
        return True
    return False


def is_ref_sup(attrs: Dict[str, str]) -> bool:
    classes = set(_classes(attrs))
    ctype = attrs.get("typeof") or ""
    if classes & {"reference", "mw-ref", "Reference"}:
        return True
    if "mw:Extension/ref" in ctype:
        return True
    return False


def extract_main_html(raw: str) -> Tuple[str, str]:
    """Return (title_text, main_inner_html) from #mw-content-text preferred region."""
    title = ""
    m = re.search(r'id="firstHeading"[^>]*>(.*?)</h1>', raw, flags=re.I | re.S)
    if m:
        title = re.sub(r"<[^>]+>", "", m.group(1))
        title = html_lib.unescape(re.sub(r"\s+", " ", title)).strip()
        for noise in (
            r"संक्षिप्त URL.*$",
            r"Get shortened URL.*$",
            r"Short URL.*$",
            r"చిన్న URL.*$",
            r"लघु URL.*$",
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
        'id="mw-navigation"',  # start of skin chrome after content on some dumps
    ):
        i = raw.find(marker, start)
        if i > start:
            tag = raw.rfind("<", start, i)
            end_candidates.append(tag if tag > start else i)
    end = min(end_candidates) if end_candidates else len(raw)
    return title, raw[start:end]


# Wikipedia origin per language (used to absolutize relative /wiki links)
WIKI_ORIGIN = {
    "EN": "https://en.wikipedia.org",
    "HI": "https://hi.wikipedia.org",
    "TE": "https://te.wikipedia.org",
    "SA": "https://sa.wikipedia.org",
}


def _escape_md_link_text(text: str) -> str:
    """Escape characters that would break [text](url) markdown."""
    return text.replace("\\", "\\\\").replace("]", "\\]")


def _escape_md_link_url(url: str) -> str:
    # spaces and parens are common failure modes in raw hrefs
    url = url.replace(" ", "%20").replace("(", "%28").replace(")", "%29")
    return url


def resolve_href(href: str, wiki_origin: str) -> str:
    """Turn relative Wikipedia hrefs into absolute URLs; leave others as-is."""
    href = (href or "").strip()
    if not href:
        return ""
    low = href.lower()
    if low.startswith("javascript:") or low.startswith("mailto:") or low.startswith("data:"):
        return ""
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("/"):
        return wiki_origin.rstrip("/") + href
    if href.startswith("#"):
        # fragment-only: keep as fragment (page-local)
        return href
    # relative path without leading slash
    return wiki_origin.rstrip("/") + "/" + href.lstrip("./")


class WikiHTMLToMarkdownFull(HTMLParser):
    """Convert article HTML fragment to markdown, preserving structure + tables + links."""

    def __init__(self, wiki_origin: str = "https://en.wikipedia.org") -> None:
        super().__init__(convert_charrefs=True)
        self.wiki_origin = wiki_origin.rstrip("/") or "https://en.wikipedia.org"
        self.parts: List[str] = []
        self._skip_depth = 0
        self._suppress_ref = 0  # inline [1] cite markers only
        self._heading_level = 0
        self._heading_buf: List[str] = []
        self._list_stack: List[str] = []
        self._in_p = False
        self._in_pre = False
        self._emit_mode: Optional[str] = None
        self._block_buf: List[str] = []
        # open <a> stack: each entry is [href_raw_or_resolved, text_parts]
        self._link_stack: List[List] = []
        # table state
        self._table_depth = 0
        self._table_rows: List[List[str]] = []
        self._cur_row: List[str] = []
        self._cur_cell: List[str] = []
        self._in_cell = False
        self._cell_is_header = False

    def _write_inline(self, s: str) -> None:
        if not s or self._skip_depth or self._suppress_ref:
            return
        # Nested text inside <a> accumulates as link label first
        if self._link_stack:
            self._link_stack[-1][1].append(s)
            return
        if self._heading_level:
            self._heading_buf.append(s)
            return
        if self._in_cell and self._table_depth:
            self._cur_cell.append(s)
            return
        self._block_buf.append(s)

    def _normalize_inline(self, text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n+", " ", text).strip()
        # residual bare bracket citations like [12] — do NOT strip markdown links [12](url)
        text = re.sub(r"\[\s*\d+\s*\](?!\()", "", text)
        text = re.sub(r" {2,}", " ", text).strip()
        return text

    def _flush_block(self, kind: str = "p") -> None:
        text = "".join(self._block_buf)
        self._block_buf = []
        if kind == "pre":
            text = text.strip("\n")
            if text.strip():
                self.parts.append("```\n" + text + "\n```\n\n")
            return
        text = self._normalize_inline(text)
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
            r"\[\s*(edit|सम्पादन|संपादित|संपादन|మార్చు|সম্পাদনা|সম্পাদন)\s*\]",
            "",
            text,
            flags=re.I,
        ).strip()
        if not text:
            return
        # DO NOT stop at See also / References — keep full page language
        level = max(1, min(6, level))
        self.parts.append("#" * level + " " + text + "\n\n")

    def _flush_cell(self) -> None:
        if not self._in_cell:
            return
        cell = self._normalize_inline("".join(self._cur_cell))
        # pipe tables need escaped pipes
        cell = cell.replace("|", "\\|")
        if self._cell_is_header and cell:
            cell = f"**{cell}**"
        self._cur_row.append(cell)
        self._cur_cell = []
        self._in_cell = False
        self._cell_is_header = False

    def _flush_row(self) -> None:
        self._flush_cell()
        if self._cur_row:
            # keep even sparse rows (infobox label/value)
            self._table_rows.append(self._cur_row)
        self._cur_row = []

    def _emit_table(self) -> None:
        self._flush_row()
        rows = self._table_rows
        self._table_rows = []
        if not rows:
            return
        # Prefer markdown pipe if rectangular-ish; else line-broken cells
        max_cols = max(len(r) for r in rows)
        if max_cols <= 1:
            for r in rows:
                line = " | ".join(c for c in r if c)
                if line:
                    self.parts.append(line + "\n")
            self.parts.append("\n")
            return

        # Pad short rows
        norm = [r + [""] * (max_cols - len(r)) for r in rows]
        # Drop fully empty rows
        norm = [r for r in norm if any(c.strip() for c in r)]
        if not norm:
            return

        def pipe_row(r: List[str]) -> str:
            return "| " + " | ".join(r) + " |"

        self.parts.append(pipe_row(norm[0]) + "\n")
        self.parts.append("| " + " | ".join("---" for _ in range(max_cols)) + " |\n")
        for r in norm[1:]:
            self.parts.append(pipe_row(r) + "\n")
        self.parts.append("\n")

    def _enter_skip(self) -> None:
        self._skip_depth += 1

    def _start(self, tag: str, attrs) -> None:
        tag = tag.lower()
        ad = _attr_dict(attrs)

        if tag in VOID_TAGS:
            if self._skip_depth or self._suppress_ref:
                return
            if tag == "br":
                self._write_inline("\n")
            elif tag == "hr":
                if self._table_depth:
                    return
                if self._block_buf and self._in_p:
                    self._flush_block("p")
                    self._in_p = False
                self.parts.append("---\n\n")
            elif tag == "img":
                # keep alt/title text if present (captions sometimes only on img)
                alt = (ad.get("alt") or ad.get("title") or "").strip()
                if alt and not alt.lower().endswith((".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp")):
                    self._write_inline(alt + " ")
            return

        if self._skip_depth:
            self._skip_depth += 1
            return

        if should_drop_element(tag, ad):
            self._enter_skip()
            return

        if tag == "table":
            # nested tables: flush pending text, accumulate rows at outermost write only
            if self._in_p and self._block_buf and not self._table_depth:
                self._flush_block("p")
                self._in_p = False
            if self._table_depth == 0:
                # starting outer table
                self._table_rows = []
                self._cur_row = []
                self._cur_cell = []
            self._table_depth += 1
            return

        if tag == "tr" and self._table_depth:
            if self._table_depth == 1:
                self._flush_row()
                self._cur_row = []
            return

        if tag in ("td", "th") and self._table_depth:
            if self._table_depth == 1:
                self._flush_cell()
                self._in_cell = True
                self._cell_is_header = tag == "th"
                self._cur_cell = []
            return

        if tag in ("thead", "tbody", "tfoot", "caption") and self._table_depth:
            if tag == "caption" and self._table_depth == 1:
                # treat caption as a pseudo single-cell row later via block buf path
                self._in_cell = True
                self._cell_is_header = True
                self._cur_cell = []
            return

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            if self._table_depth:
                # headings inside tables: treat as cell text
                return
            if self._in_p and self._block_buf:
                self._flush_block("p")
            self._in_p = False
            self._heading_level = int(tag[1])
            self._heading_buf = []
            return

        if tag == "p":
            if self._table_depth and self._in_cell:
                # paragraph inside cell — soft break
                self._write_inline(" ")
                return
            if self._in_p and self._block_buf:
                self._flush_block("p")
            self._in_p = True
            self._emit_mode = "p"
            self._block_buf = []
            return

        if tag in ("ul", "ol"):
            if self._table_depth and self._in_cell:
                return
            if self._in_p and self._block_buf:
                self._flush_block("p")
                self._in_p = False
            self._list_stack.append(tag)
            return

        if tag == "li":
            if self._table_depth and self._in_cell:
                self._write_inline("; ")
                return
            self._emit_mode = "li"
            self._block_buf = []
            return

        if tag == "pre":
            if self._table_depth:
                return
            if self._in_p and self._block_buf:
                self._flush_block("p")
                self._in_p = False
            self._in_pre = True
            self._emit_mode = "pre"
            self._block_buf = []
            return

        if tag == "sup" and is_ref_sup(ad):
            # light strip of try-cite superscript markers (keep References body)
            self._suppress_ref += 1
            return

        if tag in ("span", "div", "section", "article", "main", "aside", "nav", "header", "footer"):
            # Keep text; only drop if should_drop already handled.
            # Role=navigation inside content (TOC) — still page text (headings list);
            # keep it for "full page" fullness.
            return

        if tag == "a":
            # Collect inner text, emit [label](url) on </a>
            href = (ad.get("href") or "").strip()
            self._link_stack.append([href, []])
            return

        if tag in INLINE_EMPHASIS:
            self._write_inline(INLINE_EMPHASIS[tag])
            return

        if tag == "blockquote":
            if self._table_depth:
                return
            if self._in_p and self._block_buf:
                self._flush_block("p")
            self._in_p = True
            self._emit_mode = "p"
            self._block_buf = ["> "]
            return

        if tag == "dd":
            if self._table_depth:
                self._write_inline(" ")
                return
            self._emit_mode = "p"
            return

        if tag == "dt":
            if self._table_depth:
                self._write_inline(" ")
                return
            self._emit_mode = "p"
            self._block_buf = []
            return

        if tag in ("figure", "figcaption"):
            # keep textual caption content
            return

        # figure images etc. already handled voids
        return

    def _end(self, tag: str) -> None:
        tag = tag.lower()

        if tag in VOID_TAGS:
            return

        if self._skip_depth:
            self._skip_depth = max(0, self._skip_depth - 1)
            return

        if tag == "table":
            if self._table_depth:
                if self._table_depth == 1:
                    self._emit_table()
                self._table_depth = max(0, self._table_depth - 1)
            return

        if tag == "tr" and self._table_depth == 1:
            self._flush_row()
            return

        if tag in ("td", "th") and self._table_depth == 1:
            self._flush_cell()
            return

        if tag == "caption" and self._table_depth == 1:
            # emit caption as its own leading row
            self._flush_cell()
            if self._cur_row:
                self._table_rows.insert(0, self._cur_row)
                self._cur_row = []
            return

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            if self._table_depth:
                return
            level = int(tag[1])
            text = "".join(self._heading_buf)
            self._heading_level = 0
            self._heading_buf = []
            self._emit_heading(level, text)
            return

        if tag == "p":
            if self._table_depth and self._in_cell:
                self._write_inline(" ")
                return
            if self._in_p:
                self._flush_block("p")
            self._in_p = False
            self._emit_mode = None
            return

        if tag in ("ul", "ol"):
            if self._table_depth and self._in_cell:
                return
            if self._list_stack:
                self._list_stack.pop()
            if self.parts and not self.parts[-1].endswith("\n\n"):
                self.parts.append("\n")
            self._emit_mode = None
            return

        if tag == "li":
            if self._table_depth and self._in_cell:
                return
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

        if tag == "a":
            if not self._link_stack:
                return
            href_raw, parts = self._link_stack.pop()
            label = "".join(parts)
            # collapse whitespace inside the label only
            label_norm = re.sub(r"[ \t\n]+", " ", label).strip()
            url = resolve_href(href_raw, self.wiki_origin)
            if not label_norm:
                return  # empty anchors (icons etc.)
            if not url:
                # no usable href → keep visible text only
                self._write_inline(label_norm)
                return
            md = f"[{_escape_md_link_text(label_norm)}]({_escape_md_link_url(url)})"
            self._write_inline(md)
            return

        if tag == "blockquote":
            if self._table_depth:
                return
            if self._in_p:
                self._flush_block("p")
            self._in_p = False
            return

        if tag in ("dt", "dd"):
            if self._table_depth:
                return
            if self._block_buf:
                self._flush_block("p")
            self._emit_mode = None
            return

    def handle_starttag(self, tag: str, attrs) -> None:
        self._start(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        self._end(tag)

    def handle_startendtag(self, tag: str, attrs) -> None:
        self._start(tag, attrs)
        tag_l = tag.lower()
        if tag_l not in VOID_TAGS and self._skip_depth and should_drop_element(
            tag_l, _attr_dict(attrs)
        ):
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_depth or self._suppress_ref:
            return
        if not data:
            return
        self._write_inline(data)

    def handle_entityref(self, name: str) -> None:
        self.handle_data(html_lib.unescape(f"&{name};"))

    def handle_charref(self, name: str) -> None:
        self.handle_data(html_lib.unescape(f"&#{name};"))

    def get_markdown(self) -> str:
        if self._table_depth:
            self._emit_table()
            self._table_depth = 0
        if self._block_buf:
            kind = self._emit_mode or "p"
            self._flush_block(
                "li" if kind == "li" else ("pre" if kind == "pre" else "p")
            )
        md = "".join(self.parts)
        md = re.sub(r"\*\*\s*\*\*", "", md)
        md = md.replace("****", "")
        md = re.sub(r"[ \t]+\n", "\n", md)
        # strip bare float citations [12] but never markdown links [12](url)
        md = re.sub(r"\[\d+\](?!\()", "", md)
        # preserve newlines; collapse horizontal space only
        md = re.sub(r"[^\S\n]{2,}", " ", md)
        md = re.sub(r" +([,.;:!?])", r"\1", md)
        md = re.sub(r"\n{3,}", "\n\n", md)
        return md.strip() + "\n"


def html_file_to_markdown(path: Path, lang: str = "EN") -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    title, fragment = extract_main_html(raw)
    origin = WIKI_ORIGIN.get(lang.upper(), WIKI_ORIGIN["EN"])
    parser = WikiHTMLToMarkdownFull(wiki_origin=origin)
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


LOAD_FULL_PAGE_PY = r'''#!/usr/bin/env python3
"""Load full_page_md India corpora (EN/HI/TE/SA) with harness cleaning."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

FULL_PAGE_MD = Path(__file__).resolve().parent
HARNESS = FULL_PAGE_MD.parent / "grok_2" / "harness"
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
'''


def main() -> int:
    FULL_PAGE_MD.mkdir(parents=True, exist_ok=True)
    md_paths: Dict[str, Path] = {}
    for lang, src in INPUTS.items():
        if not src.is_file():
            print(f"ERROR: missing input {src}", file=sys.stderr)
            return 1
        print(f"Converting {lang}: {src.name} ...")
        md = html_file_to_markdown(src, lang=lang)
        out = FULL_PAGE_MD / OUT_FILES[lang]
        out.write_text(md, encoding="utf-8")
        md_paths[lang] = out
        print(f"  wrote {out}  ({len(md.encode('utf-8'))} bytes, {len(md)} chars)")

    loader = FULL_PAGE_MD / "load_full_page.py"
    loader.write_text(LOAD_FULL_PAGE_PY, encoding="utf-8")
    print(f"wrote {loader}")

    stats = compute_stats(md_paths)
    stats_path = FULL_PAGE_MD / "STATS.json"
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {stats_path}")
    print("\n=== STATS ===")
    for lang in ("EN", "HI", "TE", "SA"):
        s = stats["languages"][lang]
        print(
            f"  {lang}: bytes={s['bytes']} chars={s['chars']} "
            f"cleaned={s['chars_cleaned']} words={s['total_words']} "
            f"unique={s['unique_words']}"
        )
    print(
        f"  UNION unique={stats['union_unique_words']}  "
        f">10000={stats['union_gt_10000']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
