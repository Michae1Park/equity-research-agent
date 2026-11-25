from era_agent.evaluation import gold
from era_agent.evaluation.answer_metrics import matches, score_numbers, text_unit
from era_agent.generation.verify import numbers_in


def grade(q: dict, answer: dict, context: list[str], per_source: list[set[str]]) -> dict:
    claims = answer["claims"]
    numeric = [c for c in claims if c["value"] is not None]
    cited = {cid for c in claims for cid in c["chunk_ids"]}
    shown = set(context)
    row = {
        "numbers": stated_numbers(q, answer),
        "claimed": score_numbers(q, answer),
        "points": None,
        "claim_count": len(claims),
        "numeric_claims": len(numeric),
        "grounded": sum(c["status"] in ("verified", "calculated") for c in numeric),
        "cited_unseen": sorted(cited - shown),
        "gold_in_context": _share([bool(shown & ids) for ids in per_source]),
        "gold_cited": _share([bool(cited & ids) for ids in per_source]),
    }
    return row | {"correct": correct(q, answer["abstained"], row)}


def stated_numbers(q: dict, answer: dict) -> dict | None:
    expected = gold.expected_values(q)
    if not expected:
        return None
    table = answer["table"] or {}
    cells = [str(cell) for row in table.get("rows") or [] for cell in row]
    found = numbers_in(" ".join([answer["answer"], *cells]))
    matched = [
        any(matches(value, n.value, text_unit(n)) for n in found) for value in expected
    ]
    return {"expected": len(expected), "matched": sum(matched), "all": all(matched)}


def correct(q: dict, abstained: bool, row: dict) -> bool | None:
    if q.get("should_abstain"):
        return abstained
    if abstained:
        return False
    if row["numbers"] is not None:
        return row["numbers"]["all"]
    return None if row["points"] is None else all(row["points"])


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if not r["should_abstain"]]
    declines = [r for r in rows if r["should_abstain"]]
    scored = [r for r in rows if r["correct"] is not None]
    numbers = [r["numbers"] for r in rows if r["numbers"]]
    claimed = [r["claimed"] for r in rows if r["claimed"]]
    points = [p for r in rows if r["points"] for p in r["points"]]
    numeric = sum(r["numeric_claims"] for r in rows)
    full = [r for r in answerable if r["gold_in_context"] == 1 and r["correct"] is not None]
    partial = [r for r in answerable if r["gold_in_context"] != 1 and r["correct"] is not None]
    return {
        "questions": len(rows),
        "correct": _share([r["correct"] for r in scored]),
        "scored": len(scored),
        "unjudged": [r["id"] for r in rows if r["correct"] is None],
        "by_kind": {
            kind: {"questions": len(group), "correct": _share([r["correct"] for r in group])}
            for kind in dict.fromkeys(r["kind"] for r in rows)
            for group in [[r for r in scored if r["kind"] == kind]]
        },
        "declined_when_should": _share([r["abstained"] for r in declines]),
        "declined_when_answerable": _share([r["abstained"] for r in answerable]),
        "numbers_matched": (
            sum(n["matched"] for n in numbers) / sum(n["expected"] for n in numbers)
            if numbers
            else None
        ),
        "numbers_claimed": (
            sum(n["matched"] for n in claimed) / sum(n["expected"] for n in claimed)
            if claimed
            else None
        ),
        "points_covered": _share(points),
        "numeric_claims": numeric,
        "grounded": sum(r["grounded"] for r in rows) / numeric if numeric else None,
        "cited_unseen": [r["id"] for r in rows if r["cited_unseen"]],
        "gold_cited": _share([r["gold_cited"] for r in answerable if not r["abstained"]]),
        "correct_with_all_gold": _share([r["correct"] for r in full]),
        "correct_with_gold_missing": _share([r["correct"] for r in partial]),
        "with_all_gold": len(full),
        "with_gold_missing": len(partial),
        "format_errors": [r["id"] for r in rows if r["format_error"]],
        "seconds_mean": sum(r["seconds"] for r in rows) / len(rows),
        "output_tokens_mean": sum(r["output_tokens"] for r in rows) / len(rows),
    }


def _share(values: list) -> float | None:
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None
