import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

from bs4 import BeautifulSoup

from era_agent.acquisition.companies import filing_rows
from era_agent.acquisition.edgar import ARCHIVE_URL, DATA_URL, EdgarClient, EdgarError
from era_agent.corpus import Company

log = logging.getLogger(__name__)

FORMS = ("10-K", "10-Q", "ER", "DEF14A")
FOLDER_FORM = {"10-K": "10K", "10-Q": "10Q", "ER": "ER", "DEF14A": "DEF14A"}
INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{accession_dashed}-index.htm"


@dataclass(frozen=True)
class Document:
    name: str
    type: str
    url: str


@dataclass
class FilingV4:
    ticker: str
    cik: int
    form: str
    accession_no: str
    filing_date: str
    accepted_at: str
    period_end_date: str
    fiscal_year: int
    fiscal_period: str
    documents: list[Document] = field(default_factory=list)
    retrieved_at: str | None = None

    @property
    def period(self) -> str:
        if self.fiscal_period == "FY":
            return f"FY{self.fiscal_year}"
        return f"{self.fiscal_year}{self.fiscal_period}"

    @property
    def folder(self) -> str:
        return f"{self.period}-{FOLDER_FORM[self.form]}"


def periods_by_accession(company_facts: dict) -> dict[str, tuple[int, str]]:
    seen: dict[str, Counter] = {}
    for taxonomy in company_facts.get("facts", {}).values():
        for concept in taxonomy.values():
            for rows in concept.get("units", {}).values():
                for row in rows:
                    if row.get("form") in ("10-K", "10-Q") and row.get("fy") and row.get("fp"):
                        seen.setdefault(row["accn"], Counter())[(row["fy"], row["fp"])] += 1
    return {accession: counts.most_common(1)[0][0] for accession, counts in seen.items()}


def quarter_before(day: str) -> tuple[date, int, int]:
    d = date.fromisoformat(day)
    ends = [date(d.year - 1, 12, 31)] + [date(d.year, m, e) for m, e in ((3, 31), (6, 30), (9, 30))]
    end = max(e for e in ends if e < d)
    return end, end.year, (end.month - 1) // 3 + 1


def resolve(
    company: Company, rows: list[dict], company_facts: dict
) -> tuple[list[FilingV4], list[str]]:
    periods = periods_by_accession(company_facts)
    filings, problems = [], []
    for row in rows:
        form, accession = row["form"], row["accessionNumber"]
        common = dict(
            ticker=company.ticker,
            cik=company.cik,
            accession_no=accession,
            filing_date=row["filingDate"],
            accepted_at=row.get("acceptanceDateTime", ""),
        )
        if form in ("10-K", "10-Q"):
            if accession not in periods:
                problems.append(f"{company.ticker} {form} {accession}: no XBRL fiscal period")
                continue
            fy, fp = periods[accession]
            filings.append(
                FilingV4(
                    form=form,
                    period_end_date=row["reportDate"],
                    fiscal_year=fy,
                    fiscal_period=fp,
                    **common,
                )
            )
        elif form == "8-K" and "2.02" in row.get("items", ""):
            end, year, quarter = quarter_before(row["filingDate"])
            filings.append(
                FilingV4(
                    form="ER",
                    period_end_date=end.isoformat(),
                    fiscal_year=year,
                    fiscal_period=f"Q{quarter}",
                    **common,
                )
            )
        elif form in ("DEF 14A", "DEFC14A"):
            year = int(row["filingDate"][:4]) - 1
            filings.append(
                FilingV4(
                    form="DEF14A",
                    period_end_date=f"{year}-12-31",
                    fiscal_year=year,
                    fiscal_period="FY",
                    **common,
                )
            )

    by_key: dict[tuple[str, str], list[FilingV4]] = {}
    for f in filings:
        by_key.setdefault((f.form, f.period), []).append(f)
    unique = []
    for (form, period), group in sorted(by_key.items()):
        if len(group) > 1 and form != "DEF14A":
            accessions = [f.accession_no for f in group]
            problems.append(f"{company.ticker} {form} {period}: {len(group)} filings {accessions}")
        unique.append(min(group, key=lambda f: f.filing_date))

    reports = {f.period_end_date for f in unique if f.form != "ER"}
    for f in unique:
        if f.form == "ER" and f.period_end_date not in reports:
            problems.append(
                f"{company.ticker} ER {f.period} ({f.accession_no}): no 10-Q or 10-K for the "
                f"period ending {f.period_end_date}"
            )
    return unique, problems


def exhibits(index_html: str, base: str) -> list[Document]:
    found = []
    for tr in BeautifulSoup(index_html, "lxml").select("table.tableFile tr"):
        cells = tr.find_all("td")
        if len(cells) >= 4 and cells[3].get_text(strip=True).upper().startswith("EX-99"):
            link = cells[2].find("a")
            if link and link.get("href"):
                name = link["href"].rsplit("/", 1)[-1]
                found.append(Document(name, cells[3].get_text(strip=True).upper(), base + name))
    return found


def fetch_company(
    client: EdgarClient,
    company: Company,
    data_dir: Path,
    since: str,
    forms: tuple[str, ...] = FORMS,
    refresh: bool = False,
) -> tuple[list[FilingV4], list[str]]:
    cache = data_dir / "raw" / "edgar" / company.ticker
    cik = f"CIK{company.cik:010d}"
    submissions = client.cached_json(
        f"{DATA_URL}/submissions/{cik}.json", cache / "submissions.json", refresh
    )
    facts = client.cached_json(
        f"{DATA_URL}/api/xbrl/companyfacts/{cik}.json", cache / "companyfacts.json", refresh
    )
    pages = [submissions["filings"]["recent"]] + [
        client.cached_json(f"{DATA_URL}/submissions/{page['name']}", cache / page["name"], refresh)
        for page in submissions["filings"].get("files", [])
        if page["filingTo"] >= since
    ]
    primary = {r["accessionNumber"]: r["primaryDocument"] for r in filing_rows(pages, since)}
    filings, problems = resolve(company, filing_rows(pages, since), facts)
    filings = [f for f in filings if f.form in forms]

    for f in filings:
        base = ARCHIVE_URL.format(
            cik=company.cik, accession=f.accession_no.replace("-", ""), document=""
        )
        target = data_dir / "raw" / "filings" / company.ticker / f.folder
        meta_path = target / "metadata.json"
        if meta_path.exists() and not refresh:
            old = json.loads(meta_path.read_text())
            f.documents = [Document(**d) for d in old["documents"]]
            f.retrieved_at = old["retrieved_at"]
            continue
        if f.form == "ER":
            index = client.get(
                INDEX_URL.format(
                    cik=company.cik,
                    accession=f.accession_no.replace("-", ""),
                    accession_dashed=f.accession_no,
                )
            ).decode("utf-8", "replace")
            f.documents = exhibits(index, base)
            if not f.documents:
                problems.append(f"{company.ticker} ER {f.period} ({f.accession_no}): no EX-99")
                continue
        else:
            name = primary[f.accession_no]
            f.documents = [Document(name, f.form, base + name)]
        target.mkdir(parents=True, exist_ok=True)
        for doc in f.documents:
            (target / doc.name).write_bytes(client.get(doc.url))
        f.retrieved_at = datetime.now(UTC).isoformat(timespec="seconds")
        meta_path.write_text(json.dumps(asdict(f), indent=2) + "\n")
        log.info("Downloaded %s %s", company.ticker, f.folder)
    return filings, problems


def fetch(
    companies: list[Company],
    data_dir: Path,
    user_agent: str,
    since: str,
    forms: tuple[str, ...] = FORMS,
    refresh: bool = False,
) -> list[str]:
    client = EdgarClient(user_agent)
    manifest_path = data_dir / "raw" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    problems = []
    for company in companies:
        try:
            filings, found = fetch_company(client, company, data_dir, since, forms, refresh)
        except EdgarError as error:
            problems.append(str(error))
            continue
        problems += found
        kept = [f for f in manifest.get(company.ticker, []) if f["form"] not in forms]
        manifest[company.ticker] = kept + [asdict(f) for f in filings if f.retrieved_at]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return problems
