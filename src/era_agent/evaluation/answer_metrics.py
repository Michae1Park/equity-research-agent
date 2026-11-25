from era_agent.corpus import resolve_company
from era_agent.evaluation import gold
from era_agent.generation.verify import numbers_in, unit_scale
from era_agent.numbers import hundredths


def matches(expected: dict, claim_value, claim_unit) -> bool:
    try:
        value = float(str(claim_value).replace(",", "").replace("$", "").replace("%", ""))
    except ValueError:
        return False
    target_scale = unit_scale(expected["unit"]) or 1.0
    claim_scale = unit_scale(claim_unit)
    candidates = [value * claim_scale / target_scale] if claim_scale is not None else [value, value / target_scale]
    target = hundredths(abs(float(expected["value"])))
    return any(hundredths(abs(c)) == target for c in candidates)


def score_numbers(q: dict, answer: dict) -> dict | None:
    expected = gold.expected_values(q)
    if not expected:
        return None
    source = q["sources"][0] if q.get("sources") else {}
    default_company = source.get("company")
    default_year = source.get("fiscal_year") if q.get("answer_type") == "number" else None
    claims = [c for c in answer.get("claims", []) if c.get("value") is not None]
    matched = []
    for value in expected:
        company = value.get("company", default_company)
        year = value.get("fiscal_year", default_year)
        hit = any(
            matches(value, c["value"], c.get("unit"))
            and (year is None or c.get("fiscal_year") in (None, year))
            and (company is None or not c.get("company") or _same_company(c["company"], company))
            for c in claims
        )
        if not hit and not claims:
            hit = len(expected) == 1 and any(
                matches(value, n.value, text_unit(n))
                for n in numbers_in(answer.get("answer", ""))
            )
        matched.append(hit)
    return {"expected": len(expected), "matched": sum(matched), "all": all(matched)}


def _same_company(claimed: str, ticker: str) -> bool:
    company = resolve_company(claimed)
    return company is not None and company.ticker == ticker


def text_unit(number) -> str | None:
    if number.percent:
        return "percent"
    return {1e3: "USD thousands", 1e6: "USD millions", 1e9: "USD billions"}.get(number.scale)
