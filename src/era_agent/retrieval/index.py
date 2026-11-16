import logging
from pathlib import Path

from era_agent.corpus import COMPANIES, company_by_ticker
from era_agent.preprocess.forms import TEN_K, form_spec
from era_agent.retrieval.embeddings import Embedder
from era_agent.store import ChildRecord, ChunkRecord, FilingRecord, Store, id_prefix

log = logging.getLogger(__name__)

ITEM_NAMES = TEN_K.item_names
COMPANY_NAMES = {c.ticker: c.name for c in COMPANIES}


def chunk_header(chunk: ChunkRecord | ChildRecord, company_name: str) -> str:
    spec = form_spec(getattr(chunk, "form", "10-K"))
    item = spec.item_names.get(chunk.item, "")
    return (
        f"{company_name} ({chunk.ticker}) fiscal {chunk.fiscal_year} {spec.form}, "
        f"Item {chunk.item} {item}"
    )


def embedding_text(chunk: ChunkRecord | ChildRecord) -> str:
    return chunk_header(chunk, COMPANY_NAMES.get(chunk.ticker, chunk.ticker)) + "\n" + chunk.text


def child_text(child: dict) -> str:
    path = child.get("heading_path") or []
    if not path or child["text"].startswith(path[-1]):
        return child["text"]
    return " › ".join(path) + "\n" + child["text"]


def meta_prefix(meta: dict) -> str:
    return id_prefix(
        meta["ticker"],
        meta["fiscal_year"],
        meta.get("fiscal_period", "FY"),
        meta.get("form", "10-K"),
    )


def chunk_ids(chunks: list[dict], prefix: str) -> list[tuple[str, int]]:
    seq: dict[str, int] = {}
    ids = []
    for c in chunks:
        n = seq[c["item"]] = seq.get(c["item"], -1) + 1
        ids.append((f"{prefix}-{c['item']}-{n:03d}", n))
    return ids


def load(path: Path, filings: list[tuple[dict, dict, list[dict]]],
         append: bool = False) -> dict[str, int]:
    if not append:
        path.unlink(missing_ok=True)
    store = Store(path)
    if append:
        held = {f.accession_no for f in store.filings()}
        filings = [f for f in filings if f[0]["accession_no"] not in held]
    if any(c.get("children") for _, _, chunks in filings for c in chunks):
        store.enable_children()
    counts = {"filings": len(filings), "chunks": 0, "children": 0}
    for meta, document, chunks in filings:
        ticker, year = meta["ticker"], meta["fiscal_year"]
        company = company_by_ticker(ticker)
        prefix = meta_prefix(meta)
        filing = FilingRecord(
            meta["accession_no"],
            ticker,
            company.name,
            year,
            meta["period_end_date"],
            meta["filing_date"],
            meta.get("source_url") or meta["documents"][0]["url"],
            form=meta.get("form", "10-K"),
            fiscal_period=meta.get("fiscal_period", "FY"),
            accepted_at=meta.get("accepted_at"),
            retrieved_at=meta.get("retrieved_at"),
        )
        sections: dict[str, str] = {}
        for s in document["sections"]:
            sections.setdefault(s["item"], s["title"])
        records, children = [], []
        for (chunk_id, n), c in zip(chunk_ids(chunks, prefix), chunks, strict=True):
            records.append((_chunk_record(chunk_id, ticker, year, c), n))
            children += [
                ChildRecord(
                    f"{chunk_id}-c{j:02d}",
                    chunk_id,
                    ticker,
                    year,
                    c["item"],
                    k["kind"],
                    child_text(k),
                )
                for j, k in enumerate(c.get("children") or [])
            ]
        store.add_filing(filing, company.cik, list(company.aliases), sections, records, children)
        counts["chunks"] += len(records)
        counts["children"] += len(children)
    store.rebuild_search_index()
    return counts


def _chunk_record(chunk_id: str, ticker: str, year: int, c: dict) -> ChunkRecord:
    return ChunkRecord(
        id=chunk_id,
        ticker=ticker,
        fiscal_year=year,
        item=c["item"],
        kind=c["kind"],
        text=c["text"],
        units=c.get("units"),
        years_covered=c.get("years_covered") or [],
        pages=c.get("pages"),
        printed_pages=c.get("printed_pages"),
    )


def add_pages(store: Store, filings: list[tuple[dict, list[dict]]]) -> dict[str, int]:
    loaded = {(f.ticker, f.fiscal_year) for f in store.filings()}
    counts = {"updated": 0, "mismatched": 0}
    for meta, chunks in filings:
        if (meta["ticker"], meta["fiscal_year"]) not in loaded:
            continue
        prefix = meta_prefix(meta)
        for (chunk_id, _), c in zip(chunk_ids(chunks, prefix), chunks, strict=True):
            updated = store.set_pages(chunk_id, c["text"], c.get("pages"), c.get("printed_pages"))
            counts["updated" if updated else "mismatched"] += 1
    return counts


def embed_corpus(store: Store, embedder: Embedder, batch: int = 256) -> int:
    if store.embedding_model() != embedder.name:
        dimensions = len(embedder.embed_query("dimension probe"))
        store.reset_embeddings(embedder.name, dimensions)
    pending = store.chunks_without_embeddings()
    for start in range(0, len(pending), batch):
        group = pending[start : start + batch]
        vectors = embedder.embed_documents([embedding_text(c) for c in group])
        store.add_embeddings([c.id for c in group], vectors)
        log.info("Embedded %d/%d chunks", min(start + batch, len(pending)), len(pending))
    children = store.children_without_embeddings() if store.has_children() else []
    for start in range(0, len(children), batch):
        group = children[start : start + batch]
        vectors = embedder.embed_documents([embedding_text(c) for c in group])
        store.add_child_embeddings([c.id for c in group], vectors)
        log.info("Embedded %d/%d children", min(start + batch, len(children)), len(children))
    return len(pending) + len(children)
