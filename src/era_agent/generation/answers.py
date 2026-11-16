from era_agent.generation.verify import verify_claim
from era_agent.store import Store


def finalize(raw: dict, store: Store, sec=None, book=None) -> dict:
    claims = []
    for i, claim in enumerate(raw.get("claims") or [], 1):
        if not isinstance(claim, dict):
            continue
        claim = {
            "id": claim.get("id", i),
            "text": str(claim.get("text", "")),
            "value": claim.get("value"),
            "unit": claim.get("unit"),
            "company": claim.get("company"),
            "fiscal_year": claim.get("fiscal_year"),
            "chunk_ids": _ids(claim.get("chunk_ids")),
            "calculation": claim.get("calculation"),
        } | ({"period": claim["period"]} if "period" in claim else {})
        claims.append(verify_claim(claim, store, sec, book))
    return {
        "answer": str(raw.get("answer", "")),
        "claims": claims,
        "table": raw.get("table") if isinstance(raw.get("table"), dict) else None,
        "scope_notice": raw.get("scope_notice"),
        "abstained": bool(raw.get("abstained", False)),
        "unverified": [c["id"] for c in claims if c["status"] == "unverified"],
        "format_error": bool(raw.get("format_error")),
    }


def _ids(value) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(v) for v in value or []] if isinstance(value, list) else []


def unparsed(text: str) -> dict:
    return {
        "answer": text,
        "claims": [],
        "abstained": False,
        "scope_notice": None,
        "table": None,
        "format_error": True,
    }
