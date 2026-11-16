import time
from dataclasses import dataclass
from pathlib import Path

from era_agent.generation.answers import finalize, unparsed
from era_agent.models import Model, parse_json
from era_agent.prompts import PROMPTS
from era_agent.retrieval.index import ITEM_NAMES
from era_agent.store import ChunkRecord, Store

PROMPT = PROMPTS / "answer.md"


def corpus_table(store: Store) -> str:
    lines = ["Corpus (filer's fiscal-year label: period end date):"]
    by_company: dict[str, list[str]] = {}
    names = {}
    for f in store.filings():
        names[f.ticker] = f.company_name
        by_company.setdefault(f.ticker, []).append(f"FY{f.fiscal_year}: {f.period_end_date}")
    for ticker, years in by_company.items():
        lines.append(f"- {ticker} {names[ticker]} — " + "; ".join(years))
    return "\n".join(lines)


def format_chunks(chunks: list[ChunkRecord], store: Store) -> str:
    parts = []
    for c in chunks:
        filing = store.filing(c.ticker, c.fiscal_year)
        period = f", period ended {filing.period_end_date}" if filing else ""
        item = f"Item {c.item} {ITEM_NAMES.get(c.item, '')}".strip()
        units = f", {c.units}" if c.units else ""
        parts.append(
            f'<chunk id="{c.id}">\n{c.ticker} FY{c.fiscal_year} 10-K{period}, {item}'
            f" ({c.kind}{units})\n{c.text}\n</chunk>"
        )
    return "\n\n".join(parts)


def build_prompt(question: str, chunks: list[ChunkRecord], store: Store, prompt: Path = PROMPT):
    system = prompt.read_text().replace("{corpus}", corpus_table(store))
    excerpts = format_chunks(chunks, store) if chunks else "(none)"
    return system, f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"


@dataclass
class Generation:
    answer: dict
    replies: list[str]
    seconds: float
    input_tokens: int
    output_tokens: int


def generate(
    model: Model, system: str, user: str, store: Store, max_tokens: int = 4000
) -> Generation:
    messages = [{"role": "user", "content": user}]
    replies, seconds, tokens_in, tokens_out = [], 0.0, 0, 0
    raw = None
    for _ in range(2):
        start = time.perf_counter()
        response = model.complete(messages, system=system, max_tokens=max_tokens)
        seconds += time.perf_counter() - start
        tokens_in += response.input_tokens
        tokens_out += response.output_tokens
        replies.append(response.text)
        try:
            raw = parse_json(response.text)
            break
        except ValueError:
            messages += [
                response.message(),
                {"role": "user", "content": "Reply again with only the JSON object."},
            ]
    answer = finalize(raw if raw is not None else unparsed(replies[-1]), store)
    return Generation(answer, replies, round(seconds, 3), tokens_in, tokens_out)
