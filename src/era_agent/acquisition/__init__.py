import json
import logging
from dataclasses import asdict
from pathlib import Path

from era_agent.acquisition.edgar import EdgarClient, EdgarError, fetch_company
from era_agent.corpus import Company


def fetch(
    companies: list[Company],
    fiscal_years: tuple[int, ...],
    data_dir: Path,
    user_agent: str,
    refresh: bool = False,
) -> bool:
    client = EdgarClient(user_agent)
    manifest_path = data_dir / "raw" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    ok = True
    for company in companies:
        try:
            filings = fetch_company(client, company, fiscal_years, data_dir, refresh)
        except EdgarError as error:
            logging.error("%s", error)
            ok = False
            continue
        manifest[company.ticker] = [asdict(f) for f in filings]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return ok
