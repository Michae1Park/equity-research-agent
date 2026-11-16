import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from era_agent.research.forecast import Forecast, forecast, knobs
from era_agent.research.model.graph import Model
from era_agent.store import Store


@dataclass
class Book:
    store: Store
    settings: dict = field(default_factory=knobs)
    forecasts: dict[str, Forecast] = field(default_factory=dict)
    scratch: dict[str, Model] = field(default_factory=dict)
    valuations: dict[str, dict] = field(default_factory=dict)
    earlier: dict[str, "Book"] = field(default_factory=dict)

    def at(self, as_of: str) -> "Book":
        if as_of >= self.as_of:
            raise ValueError(f"{as_of} is not before this book's as-of {self.as_of}")
        if as_of not in self.earlier:
            from era_agent.asof import AsOf

            self.earlier[as_of] = Book(Store(self.store.path, AsOf.of(as_of)), self.settings)
        return self.earlier[as_of]

    @property
    def as_of(self) -> str:
        return self.store.asof.date.isoformat()

    def base(self, ticker: str) -> Forecast:
        if ticker not in self.forecasts:
            defaults = dict(self.settings, proposer=dict(self.settings["proposer"],
                                                         method="defaults"))
            self.forecasts[ticker] = forecast(self.store, ticker, defaults)
        return self.forecasts[ticker]

    def model(self, ticker: str, version: str) -> Model | None:
        if version == "base":
            return self.base(ticker).model
        return self.scratch.get(f"{ticker}:{version}")

    def add_scratch(self, model: Model) -> str:
        n = sum(1 for k in self.scratch if k.startswith(f"{model.ticker}:")) + 1
        version = f"scratch-{n}"
        self.scratch[f"{model.ticker}:{version}"] = model
        return version

    def valuation(self, ticker: str, version: str = "base") -> dict:
        key = f"{ticker}:{version}"
        if key not in self.valuations:
            from era_agent.research.valuation import value

            f = self.base(ticker)
            if version != "base":
                f = Forecast(self.scratch[key], f.pack, f.defaults, f.anchors)
            self.valuations[key] = value(self.store, f, self.settings, self._peer_cache())
        return self.valuations[key]

    def _peer_cache(self) -> dict:
        return {(t, self.as_of): f for t, f in self.forecasts.items()}

    def ref(self, ticker: str, version: str, line: str, period: str) -> str:
        return f"model:{ticker}:{self.as_of}:{version}:{line}.{period}"

    def resolve(self, ticker: str, as_of: str, version: str, key: str) -> dict | None:
        if SAVED_VERSION.fullmatch(version):
            return saved_value(self.store, ticker, as_of, version, key)
        if as_of < self.as_of and version == "base":
            return self.at(as_of).resolve(ticker, as_of, version, key)
        if as_of != self.as_of:
            return None
        if SAVED_VERSION.fullmatch(version):
            return saved_value(self.store, ticker, as_of, version, key)
        if version.startswith("val-"):
            return valuation_field(self.valuation(ticker, version[4:]), key)
        if version.startswith("var-"):
            from era_agent.agent.tools import variance

            found = variance(self, ticker, version[4:]).get("bridge") or {}
            return {"value": found[key], "unit": "USD millions"} if found.get(key) is not None \
                else None
        line, _, period = key.rpartition(".")
        model = self.model(ticker, version)
        if model is None or not any(x.name == line for x in model.lines):
            return None
        value, unit = model.value(line, period), model.line(line).unit
        if value is None:
            return None
        if unit == "fraction":
            return {"value": value * 100, "unit": "percent"}
        return {"value": value, "unit": unit}


SAVED_VERSION = re.compile(r"v\d+")
PERIOD = re.compile(r"\d{4}Q[1-4]|FY\d{4}")


def valuation_field(v: dict, key: str) -> dict | None:
    multiples = v.get("multiples") or v.get("multiples_2027e") or {}
    for where in (v, multiples, (v.get("scenarios") or {}).get("base") or {}):
        if key in where:
            value = where[key]
            break
    else:
        return None
    if key == "rating":
        return {"text": value}
    if not isinstance(value, int | float):
        return None
    if key in ("return", "fcf_yield"):
        return {"value": value * 100, "unit": "percent"}
    unit = "USD" if key in ("price_target", "price", "dcf_pt", "multiple_pt") else (
        "USD millions" if key in ("ev", "market_cap") else "")
    return {"value": value, "unit": unit}


def saved_version(store, ticker: str, as_of: str | None = None,
                  version: str | None = None) -> dict | None:
    rows = store.model_versions(ticker)
    if version:
        rows = [r for r in rows if r["id"] == f"{ticker}:{version}"
                and (as_of is None or r["as_of"] == as_of)]
    elif as_of:
        rows = [r for r in rows if r["as_of"] <= as_of]
    if not rows:
        return None
    row = dict(rows[-1])
    folder = Path(row["path"])
    try:
        row["model"] = json.loads((folder / "model.json").read_text())
        row["valuation"] = json.loads((folder / "valuation.json").read_text())
    except (OSError, ValueError):
        return None
    return row


def saved_line(model: dict, line: str, period: str) -> dict | None:
    unit = next((x["unit"] for x in model["lines"] if x["name"] == line), None)
    cell = model["cells"].get(f"{line}|{period}")
    if unit is None or not cell or cell.get("value") is None:
        return None
    if unit == "fraction":
        return {"value": cell["value"] * 100, "unit": "percent"}
    return {"value": cell["value"], "unit": unit}


def saved_value(store, ticker: str, as_of: str, version: str, key: str) -> dict | None:
    row = saved_version(store, ticker, as_of, version)
    if row is None:
        return None
    line, _, period = key.rpartition(".")
    if line and PERIOD.fullmatch(period):
        return saved_line(row["model"], line, period)
    return valuation_field(row["valuation"], key)
