import os
import re
from functools import lru_cache

TOKENIZER: str
TARGET: int
PARENT: int
CHILD: int
CHILD_OVERLAP: int
CANDIDATE_SIZE: int
CAPTION_CHARS: int
TABLE_CHARS: int
OVERLAP_CHARS: int
YEAR = re.compile(r"\b(20[0-3]\d)\b")
SENTENCE_END = re.compile(r"[.;]\s+")
ABBREVIATION = re.compile(r"(?:^|[\s(])(?:U\.S|U\.K|Inc|Corp|Co|Ltd|No|Nos|e\.g|i\.e|vs)\.$")
NUMERIC_CELL = re.compile(r"\(?\$?\s*\(?[\d,]*\.?\d+\)?\s*%?\)?")
BARE_YEAR = re.compile(r"(19|20)\d\d")


def configure(chunking: dict) -> None:
    global TOKENIZER, TARGET, PARENT, CHILD, CHILD_OVERLAP, CANDIDATE_SIZE
    global CAPTION_CHARS, TABLE_CHARS, OVERLAP_CHARS
    TOKENIZER = chunking["tokenizer"]
    TARGET = chunking["target"]
    PARENT, CHILD, CHILD_OVERLAP = chunking["parent"], chunking["child"], chunking["child_overlap"]
    CANDIDATE_SIZE = chunking["candidate_size"]
    CAPTION_CHARS, TABLE_CHARS = chunking["caption_chars"], chunking["table_chars"]
    OVERLAP_CHARS = chunking["overlap_chars"]
    tokenizer.cache_clear()


@lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)


def tokens(text: str) -> int:
    return len(tokenizer().encode(text, add_special_tokens=False))


def subsections(blocks: list[dict]) -> list[dict]:
    groups: list[dict] = []
    for b in blocks:
        if b["kind"] == "heading":
            if groups and not groups[-1]["body"]:
                groups[-1]["headings"].append(b)
            else:
                groups.append({"headings": [b], "body": []})
        else:
            if not groups:
                groups.append({"headings": [], "body": []})
            groups[-1]["body"].append(b)
    return groups


def chunk_path(blocks: list[dict]) -> list[str]:
    paths = [b["heading_path"] for b in blocks]
    keep = [
        p
        for b, p in zip(blocks, paths, strict=True)
        if not (b["kind"] == "heading" and any(len(q) > len(p) and q[: len(p)] == p for q in paths))
    ]
    return list(os.path.commonprefix(keep or paths))


def make_chunk(section: dict, kind: str, text: str, blocks: list[dict], **extra) -> dict:
    printed = [b["printed_page"] for b in blocks if b.get("printed_page")]
    chunk = {
        "item": section["item"],
        "item_title": section["title"],
        "heading_path": chunk_path(blocks),
        "kind": kind,
        "text": text,
        "block_ids": [b["id"] for b in blocks],
        "pages": [blocks[0]["page"], blocks[-1]["page"]],
        "printed_pages": [printed[0], printed[-1]] if printed else None,
        "tokens": tokens(text),
    }
    if kind == "table":
        table = next(b for b in blocks if b["kind"] == "table")
        top = "\n".join(text.splitlines()[:6])
        chunk |= {
            "units": table.get("units"),
            "years_covered": sorted({int(y) for y in YEAR.findall(top)}),
        }
    return chunk | extra


def assign_ids(document: dict, chunks: list[dict]) -> list[dict]:
    filing = document["filing"]
    seq: dict[str, int] = {}
    for c in chunks:
        n = seq.get(c["item"], 0)
        seq[c["item"]] = n + 1
        c["id"] = f"{filing['ticker']}-FY{filing['fiscal_year']}-{c['item']}-{n:03d}"
    return chunks


def caption(blocks: list[dict], i: int) -> list[dict]:
    out: list[dict] = []
    for b in reversed(blocks[max(0, i - 2) : i]):
        if b["kind"] == "table" or len(b["text"]) > CAPTION_CHARS:
            break
        out.insert(0, b)
    return out


def captions(section: dict, body: list[dict]) -> set[str]:
    index = {b["id"]: i for i, b in enumerate(section["blocks"])}
    return {
        c["id"]
        for b in body
        if b["kind"] == "table"
        for c in caption(section["blocks"], index[b["id"]])
    }


def table_chunks(section: dict, i: int, children: bool = False) -> list[dict]:
    blocks = section["blocks"]
    cap = caption(blocks, i)
    header = header_rows(blocks[i]["rows"])
    rows = ["| " + " | ".join(r) + " |" for r in blocks[i]["rows"]]
    head = [c["text"] for c in cap] + rows[:header]
    groups: list[list[int]] = [[]]
    for n in range(header, len(rows)):
        if (
            groups[-1]
            and len("\n".join(head + [rows[k] for k in groups[-1]] + [rows[n]])) > TABLE_CHARS
        ):
            groups.append([])
        groups[-1].append(n)
    sources = cap + [blocks[i]]
    labelled = row_children(blocks[i]["rows"]) if children else {}
    path = chunk_path(sources)
    out = []
    for g in groups:
        text = "\n".join(head + [rows[n] for n in g])
        extra = {}
        if children:
            kids = [("table_title", "\n".join(head))] + [
                ("row", labelled[n]) for n in g if n in labelled
            ]
            extra["children"] = [make_child(kind, t, path) for kind, t in kids]
        out.append(make_chunk(section, "table", text, sources, **extra))
    return out


def _is_value(cell: str) -> bool:
    return bool(NUMERIC_CELL.fullmatch(cell.strip())) and not BARE_YEAR.fullmatch(cell.strip())


def header_rows(rows: list[list[str]]) -> int:
    for n, r in enumerate(rows):
        if any(_is_value(c) for c in r[1:]) or (len(r) == 1 and r[0].endswith(":")):
            return n
    return min(1, len(rows))


def row_children(rows: list[list[str]]) -> dict[int, str]:
    header = rows[: header_rows(rows)]
    widest = max(header, key=len) if header else []
    data_width = max((len(r) for r in rows[len(header) :]), default=0)
    columns = widest[1:] if widest and len(widest) >= data_width else widest
    years = [c for c in columns if YEAR.search(c)]

    out, group = {}, ""
    for n, r in enumerate(rows[len(header) :], start=len(header)):
        label, values = r[0].strip(), [c.strip() for c in r[1:]]
        if not any(values):
            group = label.rstrip(":")
            continue
        name = f"{group} › {label}" if group else label
        names = years if len(values) == len(years) < len(columns) else columns[-len(values) :]
        if len(names) == len(values):
            cells = "; ".join(f"{c} = {v}" for c, v in zip(names, values, strict=True))
        else:
            cells = " ".join(values)
        out[n] = f"{name}: {cells}"
        if label.lower().startswith("total"):
            group = ""
    return out


def make_child(kind: str, text: str, path: list[str]) -> dict:
    return {"kind": kind, "text": text, "tokens": tokens(text), "heading_path": path}


def sentences(text: str) -> list[str]:
    pieces, start = [], 0
    for m in SENTENCE_END.finditer(text):
        if not ABBREVIATION.search(text[max(start, m.start() - 8) : m.start() + 1]):
            pieces.append(text[start : m.end()])
            start = m.end()
    return pieces + ([text[start:]] if start < len(text) else [])


def split_at_sentences(text: str, limit: int) -> list[str]:
    if tokens(text) <= limit:
        return [text]
    parts, current = [], ""
    for sentence in (s.strip() for s in sentences(text)):
        if current and tokens(f"{current} {sentence}") > limit:
            parts.append(current)
            current = ""
        current = f"{current} {sentence}".strip()
    return parts + ([current] if current else [])
