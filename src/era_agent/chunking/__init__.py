import json
from pathlib import Path

from era_agent.chunking import agentic, parent_child, rule_based
from era_agent.chunking.common import configure

METHODS = ("rule", "parent_child", "agentic")


def run_all(
    document: dict, chunking: dict, model: str, base_url: str, methods=METHODS
) -> dict[str, list[dict]]:
    configure(chunking)
    run = {
        "rule": lambda: rule_based.rule_based(document),
        "parent_child": lambda: parent_child.parent_child(document),
        "agentic": lambda: agentic.agentic(document, model, base_url),
    }
    return {m: run[m]() for m in methods}


def chunk(out: Path, llm: dict, chunking: dict, methods=METHODS) -> None:
    document = json.loads((out / "document.json").read_text())
    log_path = out / "chunks-log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    for method, chunks in run_all(
        document, chunking, llm["model"], llm["base_url"], methods
    ).items():
        (out / f"chunks-{method}.json").write_text(json.dumps(chunks, indent=1, ensure_ascii=False))
        log[method] = chunking
        print(f"{method}: {len(chunks)} chunks")
    log_path.write_text(json.dumps(log, indent=1, ensure_ascii=False))


def is_current(out: Path, method: str, chunking: dict) -> bool:
    log_path = out / "chunks-log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    return log.get(method) == chunking and (out / f"chunks-{method}.json").exists()
