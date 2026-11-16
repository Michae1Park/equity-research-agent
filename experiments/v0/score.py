import bisect
import math
import os
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent / "testdata"))

import ground_truth as gtb  # noqa: E402

WHOLE_BLOCK_MIN_COVERAGE = 0.95
MIN_LINE_CHARS = 8
FULL_PARENT_FILL_RATIO = 0.8

ChunkCoverage = dict[int, set[int]]


def key(text: str) -> str:
    return re.sub(r"[^0-9a-z]", "", text.lower())


@dataclass
class RefBlock:
    text: str
    key: str
    item: str
    path: list[str]
    heading: bool
    table: dict | None
    locatable: int
    unit: str | None = None
    label: bool = False


class Reference:
    def __init__(self, html_path: Path, gt: dict):
        soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "lxml")
        self.furniture: set[str] = set()
        self.blocks = self._read_blocks(soup, gt)
        self._assign_risk_units(gt)
        self._mark_labels()
        self.stream = "".join(b.key for b in self.blocks)
        self.starts = list(accumulate((len(b.key) for b in self.blocks[:-1]), initial=0))
        self._occurrence_cache: dict[str, list[int]] = {}

    def _read_blocks(self, soup: BeautifulSoup, gt: dict) -> list[RefBlock]:
        headings = {i["item"]: i["headings"] for i in gt["items"]}
        tables = iter(gt["tables"])
        blocks: list[RefBlock] = []
        item, item_headings, path = None, [], []
        for b in gtb.blocks(soup.body):
            is_paragraph = b["kind"] == "paragraph"
            text = b["text"] if is_paragraph else "\n".join(" | ".join(r) for r in b["rows"])
            if is_paragraph and any(p.match(text) for p in gtb.FURNITURE):
                self.furniture.add(key(text))
                continue
            m = gtb.ITEM.match(text) if is_paragraph and len(text) <= 150 else None
            if m:
                item, path = m.group(1).upper(), []
                item_headings = list(headings.get(item, []))
                continue
            if item is None:
                continue
            heading = False
            if (
                is_paragraph
                and item_headings
                and key(text).startswith(key(item_headings[0]["text"]))
            ):
                h = item_headings.pop(0)
                path, heading = h["path"], key(text) == key(h["text"])
            blocks.append(
                RefBlock(
                    text=text,
                    key=key(text),
                    item=item,
                    path=[key(p) for p in path],
                    heading=heading,
                    table=None if is_paragraph else next(tables),
                    locatable=_locatable(text),
                )
            )
        return blocks

    def _assign_risk_units(self, gt: dict) -> None:
        units = iter((gt["risk_factors"] or {"units": []})["units"])
        expected = next(units, None)
        current, pending = "preamble", []
        for b in (b for b in self.blocks if b.item == "1A"):
            if b.heading and expected and b.key == key(expected["heading"]):
                current = expected.get("id") or expected["kind"]
                expected = next(units, None)
                for p in pending:
                    p.unit = current
                pending = []
                b.unit = current
            elif b.heading:
                pending.append(b)
            else:
                b.unit = current
        for p in pending:
            p.unit = current

    def _mark_labels(self) -> None:
        counts = Counter(b.key for b in self.blocks)
        for b in self.blocks:
            b.label = b.table is None and not b.heading and len(b.text) <= 40 and counts[b.key] >= 3

    def locate(self, chunk_text: str, anchor: int) -> list[tuple[int, int]]:
        lines = [k for line in chunk_text.split("\n") if len(k := key(line)) >= MIN_LINE_CHARS]
        offsets = [0, *accumulate(len(k) for k in lines[:-1])]
        found = [(j, k) for j, k in enumerate(lines) if self._occurrences(k)]
        if not found:
            return []
        pj, pk = min(found, key=lambda jk: len(self._occurrences(jk[1])))
        occ = self._occurrences(pk)
        chunk_start = next((p for p in occ if p >= anchor), occ[-1]) - offsets[pj]
        spans = []
        for j, k in found:
            at = min(self._occurrences(k), key=lambda p: abs(p - (chunk_start + offsets[j])))
            spans.append((at, at + len(k)))
        return spans

    def _occurrences(self, k: str) -> list[int]:
        if k not in self._occurrence_cache:
            positions, at = [], self.stream.find(k)
            while at >= 0:
                positions.append(at)
                at = self.stream.find(k, at + 1)
            self._occurrence_cache[k] = positions
        return self._occurrence_cache[k]

    def block_at(self, pos: int) -> int:
        return bisect.bisect_right(self.starts, pos) - 1


def _locatable(text: str) -> int:
    long_lines = (n for line in text.split("\n") if (n := len(key(line))) >= MIN_LINE_CHARS)
    return sum(long_lines) or len(key(text))


def score_all(
    html_path: Path, gt: dict, document: dict, chunks: dict[str, list[dict]], chunking: dict
) -> dict:
    ref = Reference(Path(html_path), gt)
    out = {"preprocessing": headings_score(document, gt)}
    for method, cs in chunks.items():
        parents = method == "parent_child"
        target = chunking["parent"] if parents else chunking["target"]
        out[method] = score_method(ref, gt, cs, target, parents)
    return out


def score_method(ref: Reference, gt: dict, chunks: list[dict], target: int, parents: bool) -> dict:
    covers = _locate_chunks(ref, chunks)
    sizes = sorted(c["tokens"] for c in chunks)
    return {
        "chunks": len(chunks),
        "unlocated": sum(not cover for cover in covers),
        "tokens": {
            "mean": round(statistics.mean(sizes)),
            "p95": sizes[int(0.95 * (len(sizes) - 1))],
            "max": sizes[-1],
        },
        **_criterion_1(ref, chunks, covers),
        **_criterion_2(ref, gt, chunks, covers, target),
        "3_ratio": _criterion_3(chunks, target) if parents else None,
        **_sanity_checks(ref, chunks, covers),
    }


def _locate_chunks(ref: Reference, chunks: list[dict]) -> list[ChunkCoverage]:
    covers, anchor = [], 0
    for c in chunks:
        spans = ref.locate(c["text"], anchor)
        cover: ChunkCoverage = {}
        for s, e in spans:
            for pos in range(s, e):
                cover.setdefault(ref.block_at(pos), set()).add(pos)
        covers.append(cover)
        if spans:
            anchor = spans[0][0]
    return covers


def _is_whole(ref: Reference, cover: ChunkCoverage, i: int) -> bool:
    return len(cover.get(i, ())) >= WHOLE_BLOCK_MIN_COVERAGE * max(1, ref.blocks[i].locatable)


def _criterion_1(ref: Reference, chunks: list[dict], covers: list[ChunkCoverage]) -> dict:
    item_ok = 0
    classes = dict.fromkeys(["exact", "merged", "coarse", "wrong"], 0)
    located = [(c, cover) for c, cover in zip(chunks, covers, strict=True) if cover]
    for c, cover in located:
        item_ok += all(ref.blocks[i].item == c["item"] for i in cover)
        classes[path_class(ref, c, sorted(cover), cover)] += 1
    return {
        "1a_item": pct(item_ok, len(located)),
        "1b_path": {k: pct(v, len(located)) for k, v in classes.items()},
    }


def path_class(ref: Reference, chunk: dict, idx: list[int], cover: ChunkCoverage) -> str:
    idx = [i for i in idx if not ref.blocks[i].label] or idx
    paths = [tuple(ref.blocks[i].path) for i in idx]
    keep = [
        p
        for i, p in zip(idx, paths, strict=True)
        if not (ref.blocks[i].heading and any(len(q) > len(p) and q[: len(p)] == p for q in paths))
    ] or paths
    mine = tuple(key(p) for p in chunk["heading_path"])
    distinct = set(keep)
    if len(distinct) == 1:
        return "exact" if mine == keep[0] else "wrong"
    common = os.path.commonprefix(keep)
    if mine != common:
        return "wrong"
    below = [q for q in distinct if len(q) > len(common)]
    members = [
        i for i, b in enumerate(ref.blocks) if any(tuple(b.path[: len(q)]) == q for q in below)
    ]
    return "merged" if all(_is_whole(ref, cover, i) for i in members) else "coarse"


def _criterion_2(
    ref: Reference, gt: dict, chunks: list[dict], covers: list[ChunkCoverage], target: int
) -> dict:
    units = [
        u for u in (gt["risk_factors"] or {"units": []})["units"] if u["kind"] == "risk_factor"
    ]
    rf_units = [
        {ref.blocks[i].unit for i in cover if ref.blocks[i].item == "1A"} for cover in covers
    ]
    rf_units = [us for us in rf_units if us]
    pure = sum(len(us) == 1 for us in rf_units)

    intact, fragmentation, carried, later = 0, [], 0, 0
    small = [u for u in units if u["tokens"] <= target]
    for u in units:
        blocks = [i for i, b in enumerate(ref.blocks) if b.unit == u["id"]]
        touching = [
            (c, cover)
            for c, cover in zip(chunks, covers, strict=True)
            if any(i in cover for i in blocks)
        ]
        fragmentation.append(len(touching) / math.ceil(u["tokens"] / target))
        covered = sum(len(set().union(*(cv.get(i, set()) for _, cv in touching))) for i in blocks)
        if (
            u["tokens"] <= target
            and len(touching) == 1
            and covered >= WHOLE_BLOCK_MIN_COVERAGE * sum(len(ref.blocks[i].key) for i in blocks)
        ):
            intact += 1
        pieces = [p for c, _ in touching for p in (c.get("children") or [c])]
        for p in pieces[1:]:
            later += 1
            carried += key(u["heading"]) in key(p["text"]) or bool(
                p.get("heading_path") and key(p["heading_path"][-1]) == key(u["heading"])
            )
    return {
        "2a_purity": pct(pure, len(rf_units)),
        "2a_chunks": len(rf_units),
        "2b_intact": pct(intact, len(small)),
        "2b_of": len(small),
        "2c_fragmentation": round(statistics.mean(fragmentation), 2),
        "2d_heading": pct(carried, later) if later else None,
        "2d_of": later,
    }


def _criterion_3(chunks: list[dict], parent: int) -> dict:
    prose = [c for c in chunks if c["kind"] == "prose"]
    ratios = [
        c["tokens"] / statistics.mean(k["tokens"] for k in c["children"])
        for c in prose
        if c["tokens"] >= FULL_PARENT_FILL_RATIO * parent and c.get("children")
    ]
    return {
        "full_parents": len(ratios),
        "median": round(statistics.median(ratios), 2) if ratios else None,
        "pct_4_6": pct(sum(4 <= r <= 6 for r in ratios), len(ratios)),
        "small_parents_pct": pct(
            sum(c["tokens"] < FULL_PARENT_FILL_RATIO * parent for c in prose), len(prose)
        ),
    }


def _sanity_checks(ref: Reference, chunks: list[dict], covers: list[ChunkCoverage]) -> dict:
    covered = set().union(*(positions for cover in covers for positions in cover.values()))
    noise = sum(
        key(line) in ref.furniture for c in chunks for line in c["text"].split("\n") if key(line)
    )
    data = [i for i, b in enumerate(ref.blocks) if (b.table or {}).get("kind") == "data"]
    tables_ok = sum(
        any(
            _is_whole(ref, cover, i)
            and scale(c.get("units")) == scale(ref.blocks[i].table["units"])
            for c, cover in zip(chunks, covers, strict=True)
        )
        for i in data
    )
    return {
        "coverage": pct(len(covered), len(ref.stream)),
        "duplication": round(
            sum(len(s) for cv in covers for s in cv.values()) / len(ref.stream), 2
        ),
        "noise": noise,
        "tables": f"{tables_ok}/{len(data)}",
    }


def headings_score(document: dict, gt: dict) -> dict:
    got = [
        (s["item"], key(b["text"]), b["level"])
        for s in document["sections"]
        for b in s["blocks"]
        if b["kind"] == "heading"
    ]
    want = [(i["item"], key(h["text"]), h["level"]) for i in gt["items"] for h in i["headings"]]
    pool = list(want)
    matched = level_ok = 0
    for item, k, level in got:
        hit = next((w for w in pool if w[0] == item and w[1] == k), None)
        if hit:
            pool.remove(hit)
            matched += 1
            level_ok += hit[2] == level
    return {
        "headings": len(got),
        "gt_headings": len(want),
        "precision": pct(matched, len(got)),
        "recall": pct(matched, len(want)),
        "level": pct(level_ok, matched),
        "items": [s["item"] for s in document["sections"]],
        "dropped": document["dropped"],
        "review": len(document["review"]),
    }


def scale(units: str | None) -> str | None:
    m = re.search(r"(millions|thousands|billions)", units or "", re.I)
    return m.group(1).lower() if m else None


def pct(n: int, d: int) -> float | None:
    return round(100 * n / d, 1) if d else None
