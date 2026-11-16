from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from era_agent.store import Store

METRICS_FILE = Path(__file__).with_name("metrics.yaml")
MAX_GAP_USD = 1e6
SPANS = {"Q": (85, 95), "H1": (178, 184), "9M": (270, 276), "FY": (360, 368)}


def metrics() -> dict[str, dict]:
    return yaml.safe_load(METRICS_FILE.read_text())


@dataclass
class Quarter:
    ticker: str
    metric: str
    fiscal_year: int
    period: str
    value: float
    unit: str
    derivation: str
    inputs: list[str] = field(default_factory=list)
    filed: str = ""
    restated: bool = False

    @property
    def id(self) -> str:
        key = f"FY{self.fiscal_year}" if self.period == "FY" else f"{self.fiscal_year}{self.period}"
        return f"q:{self.ticker}:{self.metric}:{key}"


def _span(start: str, end: str) -> str | None:
    days = (date.fromisoformat(end) - date.fromisoformat(start)).days
    return next((name for name, (lo, hi) in SPANS.items() if lo <= days <= hi), None)


def restated_periods(store: Store, ticker: str, concepts: list[str]) -> set[tuple]:
    values: dict[tuple, set] = {}
    for row in store.facts(ticker, concepts):
        if row["source"] == "companyfacts":
            values.setdefault((row["concept"], row["start"], row["end"]), set()).add(row["value"])
    return {(start, end) for (_, start, end), v in values.items() if max(v) - min(v) > MAX_GAP_USD}


def recast_years(store: Store, ticker: str, concepts: list[str]) -> set[int]:
    copies: dict[tuple, list[dict]] = {}
    for row in store.facts(ticker, concepts):
        if row["source"] == "companyfacts":
            copies.setdefault((row["concept"], row["start"], row["end"]), []).append(row)
    years = set()
    for (_, _, end), rows in copies.items():
        if max(r["value"] for r in rows) - min(r["value"] for r in rows) > MAX_GAP_USD:
            years.add(int(end[:4]))
            years.add(max(rows, key=lambda r: r["filed"])["fy"] or int(end[:4]))
    return years


def _visible(store: Store, ticker: str, concepts: list[str]) -> dict[tuple, dict]:
    by_concept = {}
    for concept in concepts:
        rows = [r for r in store.facts(ticker, [concept]) if r["source"] == "companyfacts"]
        by_concept[concept] = store.asof.latest(
            rows, key=lambda r: (r["start"], r["end"]), published=lambda r: r["filed"]
        )
    chosen: dict[tuple, dict] = {}
    for year in sorted({int(end[:4]) for latest in by_concept.values() for _, end in latest}):
        for latest in by_concept.values():
            year_facts = {k: r for k, r in latest.items() if int(k[1][:4]) == year}
            if year_facts:
                chosen |= year_facts
                break
    return chosen


def derive(store: Store, ticker: str, metric: str, spec: dict) -> list[Quarter]:
    concepts = spec.get("companies", {}).get(ticker, spec["concepts"])
    facts = _visible(store, ticker, concepts)
    restated = restated_periods(store, ticker, concepts)
    kind = spec["kind"]
    out: list[Quarter] = []

    def q(year, period, value, unit, derivation, rows):
        out.append(
            Quarter(
                ticker,
                metric,
                year,
                period,
                value,
                unit,
                derivation,
                [r["id"] for r in rows],
                max(r["filed"] for r in rows),
                any((r["start"], r["end"]) in restated for r in rows),
            )
        )

    if kind == "instant":
        for (start, end), r in sorted(facts.items(), key=lambda kv: kv[0][1]):
            d = date.fromisoformat(end)
            if start is None and d.month in (3, 6, 9, 12) and d.day >= 28:
                q(d.year, f"Q{d.month // 3}", r["value"], r["unit"], "instant", [r])
        return out

    by_year: dict[int, dict[str, dict]] = {}
    for (start, end), r in facts.items():
        if start is None:
            continue
        span, d = _span(start, end), date.fromisoformat(end)
        if span is None or d.month not in (3, 6, 9, 12):
            continue
        if span == "Q":
            name = f"Q{d.month // 3}"
        elif date.fromisoformat(start) != date(d.year, 1, 1):
            continue
        else:
            name = span
        by_year.setdefault(d.year, {})[name] = r

    for year, p in sorted(by_year.items()):
        unit = next(iter(p.values()))["unit"]
        if "FY" in p:
            q(year, "FY", p["FY"]["value"], unit, "direct", [p["FY"]])
        for n, (ytd, prior) in enumerate(((None, None), ("H1", "Q1"), ("9M", "H1")), start=1):
            name = f"Q{n}"
            if name in p:
                q(year, name, p[name]["value"], unit, "direct", [p[name]])
            elif kind == "flow" and ytd in p and prior in p:
                q(
                    year,
                    name,
                    p[ytd]["value"] - p[prior]["value"],
                    unit,
                    "ytd_diff",
                    [p[ytd], p[prior]],
                )
        if kind == "flow" and "FY" in p and "9M" in p and "Q4" not in p:
            q(
                year,
                "Q4",
                p["FY"]["value"] - p["9M"]["value"],
                unit,
                "fy_minus_9m",
                [p["FY"], p["9M"]],
            )
        elif "Q4" in p:
            q(year, "Q4", p["Q4"]["value"], unit, "direct", [p["Q4"]])
    return out


def checks(
    store: Store, ticker: str, metric: str, spec: dict, quarters: list[Quarter]
) -> list[dict]:
    if spec["kind"] != "flow":
        return []
    concepts = spec.get("companies", {}).get(ticker, spec["concepts"])
    recast = recast_years(store, ticker, concepts)
    rows = []
    facts = _visible(store, ticker, spec.get("companies", {}).get(ticker, spec["concepts"]))
    ytd = {}
    for (start, end), r in facts.items():
        if start and _span(start, end) in ("H1", "9M") and start.endswith("-01-01"):
            ytd[(int(end[:4]), _span(start, end))] = r
    by_key = {(x.fiscal_year, x.period): x for x in quarters}
    for (year, period), x in by_key.items():
        if period in ("Q2", "Q3") and x.derivation == "direct":
            span, prior = ("H1", "Q1") if period == "Q2" else ("9M", "H1")
            if (year, span) in ytd:
                before = ytd.get((year, prior)) if prior == "H1" else by_key.get((year, "Q1"))
                if before is not None:
                    base = before["value"] if isinstance(before, dict) else before.value
                    gap = abs(ytd[(year, span)]["value"] - base - x.value)
                    rows.append(
                        _check(
                            "ytd_consistency",
                            ticker,
                            metric,
                            f"{year}{period}",
                            gap,
                            year in recast,
                        )
                    )
    for year in sorted({y for y, _ in by_key}):
        parts = [by_key.get((year, f"Q{n}")) for n in range(1, 5)]
        if all(parts) and (year, "FY") in by_key:
            gap = abs(sum(p.value for p in parts) - by_key[(year, "FY")].value)
            rows.append(_check("q4_ties", ticker, metric, f"FY{year}", gap, year in recast))
    return rows


def _check(name: str, ticker: str, metric: str, period: str, gap: float, recast: bool) -> dict:
    return {
        "check": name,
        "ticker": ticker,
        "metric": metric,
        "period": period,
        "gap": gap,
        "status": "pass" if gap <= MAX_GAP_USD else ("restated" if recast else "fail"),
    }
