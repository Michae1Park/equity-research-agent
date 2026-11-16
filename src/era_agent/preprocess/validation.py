from collections.abc import Callable
from dataclasses import dataclass, field

from era_agent.preprocess.blocks import ITEM_LABEL, Block
from era_agent.preprocess.forms import TEN_K, FormSpec

ITEMS = list(TEN_K.items)
TABLE_ONLY_ROLES = {"data_table", "layout_table", "index_table"}
TABLE_ROLES = TABLE_ONLY_ROLES | {"toc", "furniture", "cover"}
ROLE_FIELDS = {
    "item_heading": {"item"},
    "heading": {"level", "kind", "run_in"},
    "data_table": {"units", "units_from"},
    "layout_table": {"rows"},
}
ITEM8_KINDS = {"statement", "note", "notes_header"}


@dataclass
class Violation:
    rule: str
    ids: list[str]
    text: str

    def __str__(self) -> str:
        return f"{self.rule}: {self.text}"


@dataclass
class Context:
    item: str | None
    items: list[str]
    depth: int
    previous: dict
    form: FormSpec = field(default=TEN_K)


Check = Callable[[Block, dict, Context], str | None]


@dataclass
class Rule:
    name: str
    role: str | None
    check: Check


RULES: list[Rule] = []


def rule(name: str, role: str | None = None) -> Callable[[Check], Check]:
    def register(check: Check) -> Check:
        RULES.append(Rule(name, role, check))
        return check

    return register


def check_block(b: Block, label: dict, ctx: Context) -> list[Violation]:
    out = []
    for r in RULES:
        if r.role in (None, label.get("role")) and (text := r.check(b, label, ctx)):
            out.append(Violation(r.name, [b.id], text))
    return out


def missing_labels(window: list[Block], got: dict[str, dict]) -> list[Violation]:
    missing = [b.id for b in window if b.id not in got]
    if not missing:
        return []
    text = f"no label for {', '.join(missing)}; give exactly one label per block"
    return [Violation("V1", missing, text)]


@rule("V4")
def role_fits_block(b: Block, label: dict, ctx: Context) -> str | None:
    role = label.get("role")
    fits = role in TABLE_ROLES if b.kind == "table" else role not in TABLE_ONLY_ROLES
    if role == "item_heading" and is_item_table(b):
        fits = True
    if not fits:
        return f"{b.id} is a {b.kind}, which can't be `{role}`"
    return None


def is_item_table(b: Block) -> bool:
    return b.kind == "table" and len(b.rows) == 1 and bool(ITEM_LABEL.match(" ".join(b.rows[0])))


@rule("V4", role="item_heading")
def item_is_valid(b: Block, label: dict, ctx: Context) -> str | None:
    if label.get("item") not in ctx.form.items:
        return f"{b.id}: item_heading needs a valid `item`"
    return None


@rule("V2", role="item_heading")
def items_in_order(b: Block, label: dict, ctx: Context) -> str | None:
    item = label.get("item")
    items = ctx.form.items
    if item not in items or not ctx.items:
        return None
    last = ctx.items[-1]
    if items.index(item) <= items.index(last):
        return f"{b.id}: Item {item} after Item {last}; Items appear once, in order"
    return None


def _well_formed(heading: dict) -> bool:
    return isinstance(heading.get("level"), int) and bool(heading.get("kind"))


@rule("V4", role="heading")
def heading_has_level_and_kind(b: Block, label: dict, ctx: Context) -> str | None:
    if not _well_formed(label):
        return f"{b.id}: a heading needs `level` and `kind`"
    return None


@rule("V5", role="heading")
def levels_step_down(b: Block, label: dict, ctx: Context) -> str | None:
    if _well_formed(label) and label["level"] > ctx.depth + 1:
        jump = f"level {label['level']} jumps from {ctx.depth}"
        return f"{b.id}: {jump}; go down one level at a time"
    return None


@rule("V6", role="heading")
def risk_kinds_in_item_1a(b: Block, label: dict, ctx: Context) -> str | None:
    kind = label.get("kind") or ""
    if kind.startswith("risk_") and ctx.item != "1A":
        return f"{b.id}: `{kind}` outside Item 1A (open Item: {ctx.item})"
    return None


@rule("V6", role="heading")
def statement_kinds_in_item_8(b: Block, label: dict, ctx: Context) -> str | None:
    kind = label.get("kind")
    if kind in ITEM8_KINDS and ctx.item not in ctx.form.statement_homes:
        return f"{b.id}: `{kind}` outside Item 8 (open Item: {ctx.item})"
    return None


@rule("V4", role="layout_table")
def layout_rows_match(b: Block, label: dict, ctx: Context) -> str | None:
    if len(label.get("rows") or []) != len(b.rows):
        return f"{b.id}: `rows` needs one label per row ({len(b.rows)} rows)"
    return None


@rule("V7")
def note_at_xbrl_note(b: Block, label: dict, ctx: Context) -> str | None:
    is_note = label.get("role") == "heading" and label.get("kind") == "note"
    follows_note = ctx.previous.get("kind") == "note"
    if b.xbrl_start == "note" and not (is_note or follows_note):
        return f"{b.id}: an XBRL note starts here; it should be (or follow) a `note` heading"
    return None


def keep_role_fields(b: Block, label: dict) -> dict:
    keep = {"id", "role", "uncertain"} | ROLE_FIELDS.get(label.get("role", "body"), set())
    return {k: v for k, v in label.items() if k in keep} | {"id": b.id}
