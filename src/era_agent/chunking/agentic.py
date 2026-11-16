import json
import re

from era_agent.chunking import common
from era_agent.chunking.common import assign_ids, make_chunk, table_chunks, tokenizer, tokens


def agentic(document: dict, model: str, base_url: str) -> list[dict]:
    chunker = _slumber_chunker(model, base_url)
    chunks: list[dict] = []
    for section in document["sections"]:
        text, spans = render_markdown(section)
        index = {b["id"]: i for i, b in enumerate(section["blocks"])}
        cuts: list[int] = []
        for s, e in _segments(text, spans):
            if tokens(text[s:e]) <= common.TARGET:
                cuts.append(s)
            else:
                cuts += [s + p.start_index for p in chunker.chunk(text[s:e])]
        cuts = _glue_headings(text, cuts)
        for start, end in zip(cuts, cuts[1:] + [len(text)], strict=True):
            touched = [(b, s) for b, s, e in spans if s < end and e > start]
            prose = [b for b, _ in touched if b["kind"] != "table"]
            body = _strip_markdown(text[start:end], section)
            if prose and body:
                chunks.append(make_chunk(section, "prose", body, prose))
            for b, s in touched:
                if b["kind"] == "table" and start <= s < end:
                    chunks += table_chunks(section, index[b["id"]])
    return assign_ids(document, chunks)


HEADING_LINE = re.compile(r"#{1,7} ")


def _segments(text: str, spans: list[tuple[dict, int, int]]) -> list[tuple[int, int]]:
    starts, run = [0], None
    for b, s, _ in spans:
        run = (s if run is None else run) if b["kind"] == "heading" else None
        if b.get("heading_kind") == "risk_factor" and run > starts[-1]:
            starts.append(run)
    return list(zip(starts, starts[1:] + [len(text)], strict=True))


def _glue_headings(text: str, cuts: list[int]) -> list[int]:
    out: list[int] = []
    for c in cuts:
        while c > 0 and text[c - 2 : c] == "\n\n":
            prev = text.rfind("\n\n", 0, c - 2) + 2 if "\n\n" in text[: c - 2] else 0
            if prev < (out[-1] if out else 0) or not HEADING_LINE.match(text, prev):
                break
            c = prev
        if not out or c > out[-1]:
            out.append(c)
    return out


def _slumber_chunker(model: str, base_url: str):
    from chonkie import RecursiveLevel, RecursiveRules, SlumberChunker

    candidates = RecursiveRules(
        levels=[
            RecursiveLevel(delimiters=["\n\n"]),
            RecursiveLevel(delimiters=[". ", "; ", "? ", "! "]),
            RecursiveLevel(whitespace=True),
        ]
    )
    return SlumberChunker(
        genie=_genie(model, base_url),
        tokenizer=tokenizer(),
        chunk_size=common.TARGET,
        rules=candidates,
        candidate_size=common.CANDIDATE_SIZE,
        min_characters_per_chunk=1,
        extract_mode="json",
        verbose=False,
    )


def _item_title(section: dict) -> str:
    return f"Item {section['item']}. {section['title']}"


def render_markdown(section: dict) -> tuple[str, list[tuple[dict, int, int]]]:
    parts = [f"# {_item_title(section)}"]
    spans: list[tuple[dict, int, int]] = []
    pos = len(parts[0]) + 2
    tables = 0
    for b in section["blocks"]:
        if b["kind"] == "heading":
            line = "#" * (b["level"] + 1) + " " + b["text"]
        elif b["kind"] == "table":
            tables += 1
            line = f"[TABLE {tables}]"
        else:
            line = b["text"]
        spans.append((b, pos, pos + len(line)))
        parts.append(line)
        pos += len(line) + 2
    return "\n\n".join(parts), spans


def _strip_markdown(text: str, section: dict) -> str:
    lines = [re.sub(r"^#{1,7} ", "", ln) for ln in text.strip().split("\n\n")]
    drop = {_item_title(section), ""}
    return "\n".join(
        ln for ln in lines if ln not in drop and not re.fullmatch(r"\[TABLE \d+\]", ln)
    )


def _genie(model: str, base_url: str):
    from chonkie.genie import OpenAIGenie

    class LocalGenie(OpenAIGenie):
        def _create(self, prompt: str, **kw):
            return self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                **kw,
            )

        def generate(self, prompt: str) -> str:
            return self._create(prompt, max_tokens=64).choices[0].message.content

        def generate_json(self, prompt: str, schema) -> dict:
            fmt = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            }
            r = self._create(prompt, max_tokens=64, response_format=fmt)
            return json.loads(r.choices[0].message.content)

    return LocalGenie(model=model, base_url=base_url, api_key="not-needed")
