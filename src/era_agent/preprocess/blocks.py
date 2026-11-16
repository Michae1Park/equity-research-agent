import json
import re
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from era_agent.preprocess.forms import form_spec

ITEM_LABEL = re.compile(r"^items?\s+\d{1,2}[a-c]?\b", re.I)
TEXT_BLOCK_TAGS = {"ix:nonnumeric", "ix:continuation"}
BLOCK_TAGS = {"div", "p", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "table", "center"}
INTERNAL_LINK = re.compile("^#")
COVER_TAGS = [
    "TradingSymbol",
    "DocumentFiscalYearFocus",
    "DocumentPeriodEndDate",
    "EntityRegistrantName",
    "DocumentType",
]


@dataclass
class Block:
    id: str = ""
    kind: str = "paragraph"
    text: str = ""
    rows: list[list[str]] = field(default_factory=list)
    emph: str = "-"
    runin: int = 0
    xbrl: str | None = None
    xbrl_start: str | None = None
    pb: bool = False
    link: bool = False
    page: int = 0

    def features(self) -> str:
        parts = [
            f"emph=runin:{self.runin}" if self.emph == "runin" else f"emph={self.emph}",
            f"chars={len(self.text)}",
        ]
        if ITEM_LABEL.match(self.text):
            parts.append("item_label")
        if self.xbrl_start:
            parts.append(f"xbrl={self.xbrl_start}")
        if self.pb:
            parts.append("pb")
        if self.link:
            parts.append("link")
        return " ".join(parts)


def read_blocks(html: str) -> list[Block]:
    soup = BeautifulSoup(html, "lxml")
    blocks: list[Block] = []
    _walk(soup.body or soup, blocks, _continuation_names(soup))
    page = 1
    for i, b in enumerate(blocks, start=1):
        b.id = f"b{i:04d}"
        b.page = page
        page += b.pb
    return blocks


def cover_metadata(full_filing: Path) -> dict:
    html = full_filing.read_text(encoding="utf-8")
    tags: dict = {}
    for name in COVER_TAGS:
        m = re.search(rf'name="dei:{name}"[^>]*>(.*?)</ix:', html, re.S)
        tags[name] = _clean(BeautifulSoup(m.group(1), "lxml").get_text(" ")) if m else None
    form_spec(tags["DocumentType"])
    metadata = json.loads((full_filing.parent / "metadata.json").read_text())
    return {
        "ticker": tags["TradingSymbol"],
        "registrant": tags["EntityRegistrantName"],
        "fiscal_year": int(tags["DocumentFiscalYearFocus"]),
        "period_end": metadata.get("period_end_date"),
        "accession": metadata.get("accession_no"),
        "source_url": metadata.get("source_url"),
    }


def _walk(element: Tag, blocks: list[Block], names: dict[str, str]) -> None:
    for child in element.children:
        if isinstance(child, Comment):
            continue
        if isinstance(child, NavigableString):
            text = _clean(str(child))
            if text and element.find(BLOCK_TAGS):
                blocks.append(Block(text=text))
            continue
        if not isinstance(child, Tag) or _hidden(child):
            continue
        style = (child.get("style") or "").replace(" ", "").lower()
        if "page-break-before:always" in style and blocks:
            blocks[-1].pb = True
        if child.name == "table":
            if table := _table_block(child, names):
                blocks.append(table)
        elif child.find(BLOCK_TAGS):
            _walk(child, blocks, names)
        elif text := _clean(child.get_text(" ")):
            blocks.append(_paragraph_block(child, text, names))
        if "page-break-after:always" in style and blocks:
            blocks[-1].pb = True


def _table_block(tag: Tag, names: dict[str, str]) -> Block | None:
    rows = [cells for tr in tag.find_all("tr") if (cells := _row_cells(tr))]
    if not rows:
        return None
    xbrl, xbrl_start = _xbrl_context(tag, names)
    return Block(
        kind="table",
        text="\n".join(" | ".join(r) for r in rows),
        rows=rows,
        xbrl=xbrl,
        xbrl_start=xbrl_start,
        link=bool(tag.find("a", href=INTERNAL_LINK)),
    )


def _paragraph_block(tag: Tag, text: str, names: dict[str, str]) -> Block:
    emph, runin = _emphasis(tag, text)
    xbrl, xbrl_start = _xbrl_context(tag, names)
    return Block(
        text=text,
        emph=emph,
        runin=runin,
        xbrl=xbrl,
        xbrl_start=xbrl_start,
        link=_is_link(tag, text),
    )


def _hidden(tag: Tag) -> bool:
    return "display:none" in tag.get("style", "").replace(" ", "").lower()


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _row_cells(tr: Tag) -> list[str]:
    cells: list[str] = []
    for td in tr.find_all(["td", "th"]):
        text = _clean(td.get_text(" "))
        text = re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", text))
        if not text:
            continue
        if cells and (re.fullmatch(r"[)%]+|\)?%", text) or cells[-1] in {"$", "(", "$("}):
            cells[-1] += text
        else:
            cells.append(text)
    return cells


def _is_link(tag: Tag, text: str) -> bool:
    link = tag.find("a", href=INTERNAL_LINK)
    return bool(link) and len(_clean(link.get_text(" "))) >= len(text) / 2


def _emphasis(tag: Tag, text: str) -> tuple[str, int]:
    strings = [s for s in tag.find_all(string=True) if s.strip()]
    runs = [
        (emph, "".join(map(str, group)))
        for emph, group in groupby(strings, key=lambda s: _emphasis_of(s, tag))
    ]
    styles = {emph for emph, run in runs if re.search(r"\w", run)}
    if len(styles) == 1 and "-" not in styles:
        return styles.pop(), 0
    if len(runs) > 1 and runs[0][0] != "-":
        lead = _clean(runs[0][1])
        if lead and text.startswith(lead) and any(emph == "-" for emph, _ in runs[1:]):
            return "runin", len(lead)
    return "-", 0


def _emphasis_of(node: NavigableString, block: Tag) -> str:
    bold = italic = False
    for parent in node.parents:
        if parent is block.parent:
            break
        style = (parent.get("style") or "").replace(" ", "").lower()
        bold |= parent.name in ("b", "strong") or bool(
            re.search(r"font-weight:(bold|[6-9]00)", style)
        )
        italic |= parent.name in ("i", "em") or "font-style:italic" in style
    return ("B" if bold else "") + ("I" if italic else "") or "-"


def _continuation_names(soup: BeautifulSoup) -> dict[str, str]:
    names: dict[str, str] = {}
    for tag in soup.find_all("ix:nonnumeric", attrs={"name": re.compile("TextBlock$")}):
        next_id = tag.get("continuedat")
        while next_id and next_id not in names:
            names[next_id] = tag["name"]
            continuation = soup.find("ix:continuation", attrs={"id": next_id})
            next_id = continuation.get("continuedat") if continuation else None
    return names


def _xbrl_context(tag: Tag, names: dict[str, str]) -> tuple[str | None, str | None]:
    enclosing = [(t, name) for t in [tag, *tag.parents] if (name := _text_block_name(t, names))]
    if enclosing:
        innermost = enclosing[0][1]
        outer_tag, outer_name = enclosing[-1]
        is_start_tag = outer_tag.name == "ix:nonnumeric"
        starts_here = is_start_tag and _first_text(outer_tag) is _first_text(tag)
        return innermost, (_text_block_kind(outer_name) if starts_here else None)
    for child in tag.find_all("ix:nonnumeric"):
        if name := _text_block_name(child, names):
            return name, _text_block_kind(name)
    return None, None


def _text_block_name(tag, names: dict[str, str]) -> str | None:
    if not isinstance(tag, Tag) or tag.name not in TEXT_BLOCK_TAGS:
        return None
    name = tag.get("name") if tag.name == "ix:nonnumeric" else names.get(tag.get("id", ""))
    return name if name and name.endswith("TextBlock") else None


def _text_block_kind(name: str) -> str:
    local = name.split(":")[-1]
    if "Policy" in local:
        return "policy"
    return "table" if "Table" in local or local.startswith("Schedule") else "note"


def _first_text(tag: Tag):
    return next((s for s in tag.find_all(string=True) if s.strip()), None)
