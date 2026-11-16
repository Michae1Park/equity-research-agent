import json
import multiprocessing
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from era_agent.chunking import chunk, is_current
from era_agent.preprocess import preprocess
from era_agent.retrieval.embeddings import GPU_FREE_MB, get_embedder
from era_agent.retrieval.index import add_pages, embed_corpus, load
from era_agent.store import Store

STAGES = ("chunk", "load", "append", "embed", "pages", "facts", "market", "kpis")
KPIS = Path("output/extraction/v4.8-releases-all/extractions.json")


def filing_dirs(
    data_dir: Path, names: list[str] | None = None, pattern: str = "FY*"
) -> list[Path]:
    root = data_dir / "raw" / "filings"
    return [root / n for n in names] if names else sorted(root.glob(f"*/{pattern}"))


def filing_name(filing: Path) -> str:
    return f"{filing.parent.name}-{filing.name}"


def chunk_all(filings: list[Path], out: Path, settings: dict, methods, jobs: int) -> list[str]:
    failed = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(jobs, mp_context=context) as pool:
        work = {f: pool.submit(_chunk_logged, f, out, settings, list(methods)) for f in filings}
        for filing, future in work.items():
            ok, minutes = future.result()
            status = "ok" if ok else f"FAILED, see {out / filing_name(filing) / 'chunk.log'}"
            print(f"{filing_name(filing)}: {status} in {minutes:.1f} min", flush=True)
            if not ok:
                failed.append(filing_name(filing))
    return failed


def _chunk_logged(filing: Path, out: Path, settings: dict, methods: list[str]):
    target = out / filing_name(filing)
    target.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with open(target / "chunk.log", "a") as log, redirect_stdout(log), redirect_stderr(log):
        try:
            chunk_one(filing, target, settings, methods)
            ok = True
        except Exception:
            traceback.print_exc()
            ok = False
    return ok, (time.time() - t) / 60


def chunk_one(filing: Path, target: Path, settings: dict, methods) -> None:
    meta = json.loads((filing / "metadata.json").read_text())
    if not (target / "document.json").exists():
        if "form" in meta:
            from era_agent.corpus import company_by_ticker
            from era_agent.preprocess.rules import preprocess_filing

            preprocess_filing(filing, target, company_by_ticker(meta["ticker"]).name)
        else:
            html = next(filing.glob("*.htm"))
            preprocess(html, html, target, settings["llm"], settings["labeling"])
    todo = [m for m in methods if not is_current(target, m, settings["chunking"])]
    if todo:
        chunk(target, settings["llm"], settings["chunking"], todo)


def load_db(method: str, filings: list[Path], out: Path, append: bool = False) -> Path:
    path = out / f"{method}.db"
    rows = []
    for filing in filings:
        source = out / filing_name(filing)
        rows.append(
            (
                json.loads((filing / "metadata.json").read_text()),
                json.loads((source / "document.json").read_text()),
                json.loads((source / f"chunks-{method}.json").read_text()),
            )
        )
    counts = load(path, rows, append=append)
    print(f"{method}: {len(filings)} filings, {counts} → {path}")
    return path


def embed_db(method: str, out: Path, retrieval: dict) -> None:
    store = Store(out / f"{method}.db")
    t = time.time()
    embedder = get_embedder(
        retrieval["embedder"], retrieval["device"], retrieval.get("gpu_free_mb", GPU_FREE_MB)
    )
    n = embed_corpus(store, embedder)
    print(f"{method}: embedded {n} in {time.time() - t:.0f} s")


def pages_db(method: str, filings: list[Path], out: Path) -> None:
    store = Store(out / f"{method}.db")
    rows = [
        (
            json.loads((filing / "metadata.json").read_text()),
            json.loads((out / filing_name(filing) / f"chunks-{method}.json").read_text()),
        )
        for filing in filings
    ]
    print(f"{method}: pages for {add_pages(store, rows)} → {out / f'{method}.db'}")


def facts_db(method: str, filings: list[Path], out: Path, since: str) -> None:
    from era_agent.asof import AsOf
    from era_agent.facts.xbrl import company_facts, filing_facts

    store = Store(out / f"{method}.db", AsOf.today())
    store.enable_facts()
    counts = {"companyfacts": 0, "inline": 0}
    for ticker in sorted({f.parent.name for f in filings}):
        raw = filings[0].parents[2] / "edgar" / ticker / "companyfacts.json"
        counts["companyfacts"] += store.add_facts(
            company_facts(ticker, json.loads(raw.read_text()), since)
        )
    for filing in filings:
        if filing.name.endswith(("-10K", "-10Q")):
            counts["inline"] += store.add_facts(filing_facts(filing))
    print(f"{method}: facts {counts} → {out / f'{method}.db'}")


def kpis_db(method: str, out: Path, extractions: Path) -> None:
    from era_agent.asof import AsOf

    store = Store(out / f"{method}.db", AsOf.today())
    store.enable_facts()
    filed = {
        (f.ticker, f"{f.fiscal_year}{f.fiscal_period}"): f.filing_date
        for f in store.filings()
        if f.form == "ER"
    }
    n, missing = 0, []
    for r in json.loads(extractions.read_text()).values():
        if (r["ticker"], r["period"]) not in filed:
            missing.append(f"{r['ticker']}-{r['period']}")
            continue
        n += store.add_extractions(
            r["ticker"], r["period"], filed[r["ticker"], r["period"]], r["kpis"], r["guidance"]
        )
    print(f"{method}: kpis and guidance {n} rows → {out / f'{method}.db'}"
          + (f" (no release filing: {', '.join(missing)})" if missing else ""))


def build(filings: list[Path], out: Path, settings: dict, methods, stages, jobs: int = 8) -> None:
    if "chunk" in stages and (failed := chunk_all(filings, out, settings, methods, jobs)):
        sys.exit(f"{len(failed)} filings failed: {', '.join(failed)}")
    if "load" in stages:
        for method in methods:
            load_db(method, filings, out)
    if "append" in stages:
        for method in methods:
            load_db(method, filings, out, append=True)
    if "embed" in stages:
        for method in methods:
            embed_db(method, out, settings["retrieval"])
    if "pages" in stages:
        for method in methods:
            pages_db(method, filings, out)
    if "market" in stages:
        from era_agent.acquisition.market import market_rows
        from era_agent.asof import AsOf

        for method in methods:
            store = Store(out / f"{method}.db", AsOf.today())
            store.enable_facts()
            rows = market_rows(filings[0].parents[2])
            store.add_market(rows)
            print(f"{method}: market {len(rows)} rows → {out / f'{method}.db'}")
    if "facts" in stages:
        from era_agent.corpus import WINDOW_START

        for method in methods:
            facts_db(method, filings, out, WINDOW_START)
    if "kpis" in stages:
        for method in methods:
            kpis_db(method, out, Path(settings.get("build", {}).get("kpis", KPIS)))
