from dataclasses import dataclass, field


@dataclass(frozen=True)
class FormSpec:
    form: str
    items: tuple[str, ...]
    statement_homes: frozenset[str]
    item_names: dict[str, str] = field(default_factory=dict)


TEN_K = FormSpec(
    form="10-K",
    items=tuple("1 1A 1B 1C 2 3 4 5 6 7 7A 8 9 9A 9B 9C 10 11 12 13 14 15 16".split()),
    statement_homes=frozenset({"8", "15"}),
    item_names={
        "1": "Business",
        "1A": "Risk Factors",
        "7": "Management's Discussion and Analysis",
        "7A": "Market Risk",
        "8": "Financial Statements",
    },
)

TEN_Q = FormSpec(
    form="10-Q",
    items=tuple("I.1 I.2 I.3 I.4 II.1 II.1A II.2 II.3 II.4 II.5 II.6".split()),
    statement_homes=frozenset({"I.1"}),
    item_names={
        "I.1": "Financial Statements",
        "I.2": "Management's Discussion and Analysis",
        "I.3": "Market Risk",
        "I.4": "Controls and Procedures",
        "II.1": "Legal Proceedings",
        "II.1A": "Risk Factors",
        "II.2": "Unregistered Sales of Equity and Use of Proceeds",
        "II.5": "Other Information",
    },
)

RELEASE = FormSpec(
    form="ER",
    items=("EX99", "EX99.2", "EX99.3", "EX99.4"),
    statement_homes=frozenset({"EX99"}),
    item_names={"EX99": "Earnings Release", "EX99.2": "Shareholder Letter"},
)

PROXY = FormSpec(
    form="DEF14A",
    items=("PROXY",),
    statement_homes=frozenset(),
    item_names={"PROXY": "Proxy Statement"},
)

FORMS: dict[str, FormSpec] = {spec.form: spec for spec in (TEN_K, TEN_Q, RELEASE, PROXY)}


def form_spec(form: str) -> FormSpec:
    if form not in FORMS:
        raise KeyError(f"no FormSpec for {form!r}; known: {sorted(FORMS)}")
    return FORMS[form]
