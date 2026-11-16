from dataclasses import asdict, dataclass, field

from era_agent.facts import quarters as xbrl_quarters
from era_agent.research.datapack import Point
from era_agent.research.model.graph import Model, load_template
from era_agent.store import Store

TOLERANCE = 1.0


@dataclass
class Check:
    name: str
    status: str
    tested: int = 0
    failures: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    note: str = ""


def run(model: Model, pack: dict[str, dict[str, Point]], store: Store | None) -> list[Check]:
    _, template = load_template(model.template, model.ticker)
    return [
        balance(model), quarters_sum(model, pack, store),
        *([q4_ties(model, store)] if store is not None else []),
        take_rate(model, pack), segments_sum(model, template.get("sums") or {}),
        no_future_input(model), formula_only(model),
    ]


def exceptions(model: Model, checks: list[Check]) -> list[dict]:
    out = [{"severity": "info", "item": f"default used: {d}"} for d in model.defaults]
    for line in model.lines:
        missing = [c for c in model.quarters
                   if line.kind == "input" and model.cells[(line.name, c)].kind == "missing"]
        if missing:
            out.append({"severity": "info", "item": f"{line.name} missing in {', '.join(missing)}"})
    for c in checks:
        if c.status == "fail":
            out.append({"severity": "error", "item": f"{c.name}: {'; '.join(c.failures)}"})
        if c.skipped:
            out.append({"severity": "info",
                        "item": f"{c.name} skipped {len(c.skipped)}: {', '.join(c.skipped)}"})
    return out


def _reported(model: Model) -> list[str]:
    years = {q[:4] for q in model.forecast}
    return [c for c in model.columns if c not in model.forecast
            and not (c.startswith("FY") and c[2:] in years)]


def _result(name: str, tested: int, failures: list, skipped: list, note: str = "") -> Check:
    status = "fail" if failures else "pass" if tested else "skip"
    return Check(name, status, tested, failures, skipped, note)


def balance(model: Model) -> Check:
    tested, failures, skipped = 0, [], []
    gaps = [g for g in ("balance_gap", "sides_gap") if any(x.name == g for x in model.lines)]
    for column in _reported(model):
        for name in gaps:
            gap = model.value(name, column)
            if gap is None:
                skipped.append(f"{name} {column}")
                continue
            tested += 1
            if abs(gap) > TOLERANCE:
                failures.append(f"{name} {column} off by {gap:,.1f}")
    return _result("balance", tested, failures, skipped)


def quarters_sum(model: Model, pack, store: Store | None = None) -> Check:
    specs = xbrl_quarters.metrics()
    tested, failures, skipped = 0, [], []
    for line in model.lines:
        if line.kind != "input" or line.fy != "sum" or not line.source.startswith("q:"):
            continue
        metric = line.source[2:]
        concepts = specs[metric].get("companies", {}).get(model.ticker, specs[metric]["concepts"])
        recast = xbrl_quarters.recast_years(store, model.ticker, concepts) if store else set()
        for column in (c for c in _reported(model) if c.startswith("FY")):
            if int(column[2:]) in recast:
                skipped.append(f"{line.name} {column} (recast)")
                continue
            reported = pack.get(line.source, {}).get(column)
            summed = model.value(line.name, column)
            if reported is None or summed is None:
                skipped.append(f"{line.name} {column}")
                continue
            tested += 1
            if abs(summed - reported.value) > TOLERANCE:
                failures.append(f"{line.name} {column}: quarters {summed:,.1f} vs reported "
                                f"{reported.value:,.1f}")
    return _result("quarters_sum", tested, failures, skipped)


def q4_ties(model: Model, store: Store) -> Check:
    specs = xbrl_quarters.metrics()
    tested, failures, skipped = 0, [], []
    for line in model.lines:
        if line.kind != "input" or not line.source.startswith("q:"):
            continue
        metric = line.source[2:]
        if specs[metric]["kind"] != "flow":
            continue
        derived = xbrl_quarters.derive(store, model.ticker, metric, specs[metric])
        for result in xbrl_quarters.checks(store, model.ticker, metric, specs[metric], derived):
            if result["check"] != "q4_ties":
                continue
            if result["status"] == "pass":
                tested += 1
            elif result["status"] == "fail":
                tested += 1
                failures.append(f"{metric} {result['period']}: off by {result['gap'] / 1e6:,.1f}")
            else:
                skipped.append(f"{metric} {result['period']} (recast)")
    return _result("q4_ties", tested, failures, skipped)


def take_rate(model: Model, pack) -> Check:
    tested, failures, skipped = 0, [], []
    for line in model.lines:
        if not line.name.startswith("take_rate") or line.name.endswith("_shift"):
            continue
        suffix = line.name[len("take_rate"):]
        top, bottom = model.line(f"revenue{suffix}"), model.line(f"gross_bookings{suffix}")
        for column in model.history:
            rev = pack.get(top.source, {}).get(column)
            gb = pack.get(bottom.source, {}).get(column)
            value = model.value(line.name, column)
            if not rev or not gb or value is None:
                skipped.append(f"{line.name} {column}")
                continue
            tested += 1
            if abs(value - rev.value / gb.value) > 1e-9:
                failures.append(f"{line.name} {column}: {value:.4f} vs {rev.value / gb.value:.4f}")
    return _result("take_rate", tested, failures, skipped)


def segments_sum(model: Model, sums: dict[str, list]) -> Check:
    tested, failures, skipped = 0, [], []
    for total, parts_spec in sums.items():
        options = parts_spec if parts_spec and isinstance(parts_spec[0], list) else [parts_spec]
        for column in model.columns:
            reported = model.value(total, column)
            parts = next((o for o in options
                          if all(model.value(p, column) is not None for p in o)), None)
            if reported is None or parts is None:
                skipped.append(f"{total} {column}")
                continue
            values = [model.value(p, column) for p in parts]
            slack = max(TOLERANCE, 0.5 * len(parts))
            tested += 1
            if abs(sum(values) - reported) > slack:
                failures.append(f"{total} {column}: parts {sum(values):,.1f} vs {reported:,.1f}")
    return _result("segments_sum", tested, failures, skipped)


def no_future_input(model: Model) -> Check:
    tested, failures = 0, []
    for (line, column), cell in model.cells.items():
        if cell.kind == "input":
            tested += 1
            if not cell.filed or cell.filed >= model.as_of:
                failures.append(f"{line} {column} filed {cell.filed}")
    return _result("no_future_input", tested, failures, [])


def formula_only(model: Model) -> Check:
    failures = [f"{line} {column}" for (line, column), cell in model.cells.items()
                if column in model.forecast and cell.kind in ("input", "default")]
    tested = sum(1 for (_, c), cell in model.cells.items()
                 if c in model.forecast and cell.kind == "formula")
    return Check("formula_only", "fail" if failures else "pass", tested, failures,
                 note="" if model.forecast else "no forecast columns")


def to_json(checks: list[Check]) -> list[dict]:
    return [asdict(c) for c in checks]


def wacc_gt_g(valuation: dict, settings: dict) -> Check:
    g = settings["valuation"]["terminal_growth"]
    wacc = valuation["cost_of_capital"]["wacc"]
    failures = [] if wacc > g else [f"WACC {wacc:.2%} <= terminal growth {g:.2%}"]
    return _result("wacc_gt_g", 1, failures, [])


def weights(settings: dict) -> Check:
    failures = []
    for name, w in (("scenario", settings["scenarios"]["weights"]),
                    ("valuation", settings["valuation"]["weights"])):
        if abs(sum(w.values()) - 1) > 1e-9 or any(v < 0 for v in w.values()):
            failures.append(f"{name} weights {w} do not sum to 1")
    return _result("weights", 2, failures, [])


def valuation_checks(valuation: dict, settings: dict) -> list[Check]:
    return [wacc_gt_g(valuation, settings), weights(settings)]
