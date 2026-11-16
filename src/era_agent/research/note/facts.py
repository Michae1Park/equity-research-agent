from dataclasses import asdict, dataclass, field

SCALE_WORDS = {"USD billions": (1e3, "B"), "USD millions": (1.0, "M")}


@dataclass
class Fact:
    id: str
    label: str
    value: float | str
    unit: str
    source: str
    period: str = ""
    filed: str = ""

    def text(self, digits: int | None = None) -> str:
        v = self.value
        if isinstance(v, str):
            return v
        if self.unit == "USD millions":
            sign, v = ("-" if v < 0 else ""), abs(v)
            return (f"{sign}${v / 1e3:,.{1 if digits is None else digits}f}B" if v >= 1e3
                    else f"{sign}${v:,.0f}M")
        if self.unit == "USD":
            return f"-${-v:,.2f}" if v < 0 else f"${v:,.2f}"
        if self.unit == "USD per share":
            return f"${v:,.2f}"
        if self.unit == "percent":
            signed = any(w in self.label for w in ("change", "variance", "return"))
            return f"{v:+.1f}%" if signed else f"{v:.1f}%"
        if self.unit == "points":
            return f"{v:+.1f} pts"
        if self.unit == "weight":
            return f"{v * 100:.0f}%"
        if self.unit == "count":
            return f"{v:,.0f}"
        if self.unit == "x":
            return f"{v:.1f}x"
        if self.unit == "count millions":
            return f"{v / 1e3:,.2f}B" if abs(v) >= 1e3 else f"{v:,.1f}M"
        return f"{v:,.2f}"


@dataclass
class Facts:
    items: list[Fact] = field(default_factory=list)

    def add(self, label: str, value, unit: str, source: str, period: str = "",
            filed: str = "") -> Fact:
        fact = Fact(f"F-{len(self.items) + 1}", label, value, unit, source, period, filed)
        self.items.append(fact)
        return fact

    def get(self, fact_id: str) -> Fact | None:
        return next((f for f in self.items if f.id == fact_id), None)

    def listing(self) -> str:
        return "\n".join(f"{f.id} | {f.label} | {f.text()} | {f.source}" for f in self.items)

    def to_json(self) -> list[dict]:
        return [asdict(f) for f in self.items]
