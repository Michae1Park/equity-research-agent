import json
import re
from collections import Counter
from dataclasses import dataclass, field, replace

import openai

from era_agent.preprocess.blocks import Block
from era_agent.preprocess.validation import (
    ITEMS,
    Context,
    Violation,
    check_block,
    keep_role_fields,
    missing_labels,
)
from era_agent.prompts import PROMPTS

DROPPED_ROLES = {"cover", "toc", "part_heading", "furniture", "index_table"}
STATEMENT_KINDS = {"statement", "notes_header", "auditor_report"}
PRINTED_PAGE = re.compile(r"(?<![\w-])([A-Z]-\d{1,3}|\d{1,3})\s*$")
ITEM_PREFIX = re.compile(
    r"^items?\s+\d{1,2}[a-c]?(\s*(,|and|&)\s*\d{1,2}[a-c]?)*\s*[.:]?\s*", re.I
)


def load_prompts() -> tuple[str, str, str]:
    text = (PROMPTS / "structure.md").read_text(encoding="utf-8")
    system, templates = text.split("<!-- END SYSTEM PROMPT", 1)
    window = templates.split("## Template: window", 1)[1].split("## Template: repair", 1)[0]
    repair = templates.split("## Template: repair", 1)[1]
    return system.strip(), window.strip(), repair.strip()


@dataclass
class Outline:
    texts: dict[str, str]
    item: str | None = None
    items: list[str] = field(default_factory=list)
    path: list[tuple[int, str]] = field(default_factory=list)

    def advance(self, label: dict) -> None:
        if label["role"] == "item_heading":
            self.item = label.get("item")
            if self.item in ITEMS:
                self.items.append(self.item)
            self.path = []
        elif label["role"] == "heading" and isinstance(label.get("level"), int):
            level = min(label["level"], len(self.path) + 1)
            self.path = self.path[: level - 1] + [(level, label["id"])]

    def path_text(self) -> str:
        return " › ".join(f"L{level} {self.texts.get(i, i)}" for level, i in self.path) or "(none)"

    def copy(self) -> "Outline":
        return replace(self, items=list(self.items), path=list(self.path))


class Labeler:
    def __init__(self, model: str, base_url: str, filing: dict, labeling: dict):
        self.client = openai.OpenAI(base_url=base_url, api_key="not-needed", timeout=1800)
        self.model = model
        self.filing = filing
        self.window = labeling["window"]
        self.context = labeling["context"]
        self.repair_rounds = labeling["repair_rounds"]
        self.system, self.window_template, self.repair_template = load_prompts()
        schema = json.loads((PROMPTS / "structure.schema.json").read_text())
        self.response_format = {
            "type": "json_schema",
            "json_schema": {"name": "labels", "schema": schema},
        }
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.log: list[dict] = []

    def label(self, blocks: list[Block]) -> dict[str, dict]:
        labels: dict[str, dict] = {}
        outline = Outline({b.id: b.text[:80] for b in blocks})
        for start in range(0, len(blocks), self.window):
            window = blocks[start : start + self.window]
            context = blocks[max(0, start - self.context) : start]
            got, violations = self._label_window(window, context, labels, outline)
            for b in window:
                missing = {"id": b.id, "role": "body", "uncertain": True}
                label = keep_role_fields(b, got.get(b.id) or missing)
                if any(b.id in v.ids for v in violations):
                    label["unresolved"] = True
                labels[b.id] = label
                outline.advance(label)
        return labels

    def _label_window(
        self, window: list[Block], context: list[Block], labels: dict, outline: Outline
    ) -> tuple[dict[str, dict], list[Violation]]:
        user = self.window_template.format(
            company=self.filing["registrant"],
            ticker=self.filing["ticker"],
            fiscal_year=self.filing["fiscal_year"],
            open_item=outline.item or "none",
            open_path=outline.path_text(),
            context="\n".join(_context_line(b, labels[b.id]) for b in context) or "(none)",
            blocks="\n".join(_input_line(b) for b in window),
        )
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": user}]
        got = _by_id(self._call(messages))
        violations = validate(window, got, outline)
        rounds = [list(map(str, violations))]
        for _ in range(self.repair_rounds):
            if not violations:
                break
            flagged = {i for v in violations for i in v.ids}
            lines = [
                f"{_input_line(b)}\n    your label: {json.dumps(got.get(b.id))}"
                for b in window
                if b.id in flagged
            ]
            repair = self.repair_template.format(
                violations="\n".join(f"- {v}" for v in violations), blocks="\n".join(lines)
            )
            messages += [
                {"role": "assistant", "content": json.dumps({"labels": list(got.values())})},
                {"role": "user", "content": repair},
            ]
            fixed = _by_id(self._call(messages))
            got.update({i: label for i, label in fixed.items() if i in flagged})
            violations = validate(window, got, outline)
            rounds.append(list(map(str, violations)))
        self.log.append({"start": window[0].id, "end": window[-1].id, "violations": rounds})
        return got, violations

    def _call(self, messages: list[dict]) -> dict:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=14000,
            response_format=self.response_format,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += r.usage.prompt_tokens
        self.usage["completion_tokens"] += r.usage.completion_tokens
        try:
            return json.loads(r.choices[0].message.content)
        except json.JSONDecodeError:
            return {"labels": []}


def _input_line(b: Block) -> str:
    if b.kind == "table":
        rows = b.rows if len(b.rows) <= 6 else b.rows[:3] + b.rows[-1:]
        return (
            f"{b.id} | TABLE rows={len(b.rows)} cols={max(len(r) for r in b.rows)} "
            + f"{b.features()} | "
            + " ‖ ".join(" | ".join(r) for r in rows)
        )
    text = b.text if len(b.text) <= 300 else f"{b.text[:200]} … {b.text[-60:]}"
    return f"{b.id} | {b.features()} | {text}"


def _context_line(b: Block, label: dict) -> str:
    extra = " ".join(f"{k}={label[k]}" for k in ("item", "level", "kind") if k in label)
    return f"{b.id} | {label['role']} {extra} | {b.text[:100]}"


def _by_id(reply: dict) -> dict[str, dict]:
    return {lb["id"]: lb for lb in reply.get("labels", []) if isinstance(lb, dict) and "id" in lb}


def validate(window: list[Block], got: dict[str, dict], outline: Outline) -> list[Violation]:
    out = missing_labels(window, got)
    outline = outline.copy()
    previous: dict = {}
    for b in window:
        label = got.get(b.id)
        if label:
            ctx = Context(
                item=outline.item, items=outline.items, depth=len(outline.path), previous=previous
            )
            out += check_block(b, label, ctx)
            outline.advance({**label, "id": b.id})
        previous = label or {}
    return out


def assemble(blocks: list[Block], labels: dict[str, dict], filing: dict) -> dict:
    printed = _printed_pages(blocks, labels)
    sections: list[dict] = []
    dropped: Counter[str] = Counter()
    review: list[str] = []
    for b in blocks:
        label = labels[b.id]
        role = label["role"]
        if label.get("uncertain") or label.get("unresolved"):
            review.append(b.id)
        if role == "item_heading":
            text = " ".join(b.rows[0]) if b.kind == "table" else b.text
            title = ITEM_PREFIX.sub("", text) or text
            sections.append({"item": label.get("item"), "title": title, "blocks": []})
        elif not sections:
            dropped["cover"] += 1
        elif role in DROPPED_ROLES:
            dropped[role] += 1
        else:
            pages = {"page": b.page, "printed_page": printed.get(b.page)}
            sections[-1]["blocks"] += [d | pages for d in _document_blocks(b, label)]
    _move_statements_into_item8(sections)
    _add_heading_paths(sections)
    return {"filing": filing, "sections": sections, "dropped": dict(dropped), "review": review}


def _printed_pages(blocks: list[Block], labels: dict[str, dict]) -> dict[int, str]:
    printed: dict[int, str] = {}
    for b in blocks:
        if labels[b.id]["role"] == "furniture" and (m := PRINTED_PAGE.search(b.text)):
            printed[b.page] = m.group(1)
    return printed


def _document_blocks(b: Block, label: dict) -> list[dict]:
    role = label["role"]
    if role == "heading" and label.get("run_in") and b.emph == "runin":
        return [
            _heading(f"{b.id}.h", b.text[: b.runin], label, b),
            _paragraph(f"{b.id}.b", b.text[b.runin :].strip(), b),
        ]
    if role == "heading":
        return [_heading(b.id, b.text, label, b)]
    if role == "data_table":
        return [
            {
                "id": b.id,
                "kind": "table",
                "text": "\n".join("| " + " | ".join(r) + " |" for r in b.rows),
                "rows": b.rows,
                "units": label.get("units"),
                "units_from": label.get("units_from"),
                "xbrl": b.xbrl,
            }
        ]
    if role == "layout_table":
        row_labels = label.get("rows") or [{"role": "body"}] * len(b.rows)
        out = []
        for k, (row, row_label) in enumerate(zip(b.rows, row_labels, strict=False)):
            if row_label.get("role") == "heading":
                out.append(_heading(f"{b.id}.r{k}", " ".join(row), row_label, b))
            elif row_label.get("role") == "body":
                out.append(_paragraph(f"{b.id}.r{k}", " ".join(row), b))
        return out
    return [_paragraph(b.id, b.text, b)]


def _heading(id_: str, text: str, label: dict, b: Block) -> dict:
    return {
        "id": id_,
        "kind": "heading",
        "text": text,
        "level": label.get("level") or 1,
        "heading_kind": label.get("kind") or "other",
        "xbrl": b.xbrl,
    }


def _paragraph(id_: str, text: str, b: Block) -> dict:
    return {"id": id_, "kind": "paragraph", "text": text, "xbrl": b.xbrl}


def _move_statements_into_item8(sections: list[dict]) -> None:
    item8 = next((s for s in sections if s["item"] == "8"), None)
    if item8 is None or any(b.get("heading_kind") == "statement" for b in item8["blocks"]):
        return
    for section in sections[sections.index(item8) + 1 :]:
        for i, b in enumerate(section["blocks"]):
            if b.get("heading_kind") in STATEMENT_KINDS:
                item8["blocks"] += section["blocks"][i:]
                del section["blocks"][i:]
                return


def _add_heading_paths(sections: list[dict]) -> None:
    for section in sections:
        path: list[str] = []
        for b in section["blocks"]:
            if b["kind"] == "heading":
                b["level"] = max(1, min(b["level"], len(path) + 1))
                path = path[: b["level"] - 1] + [b["text"]]
            b["heading_path"] = list(path)
