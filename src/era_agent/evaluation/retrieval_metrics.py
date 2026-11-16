from dataclasses import dataclass

from era_agent.evaluation import gold as gold_set
from era_agent.store import Store


@dataclass
class GoldChunks:
    per_source: list[set[str]]

    @property
    def all(self) -> set[str]:
        return set().union(*self.per_source) if self.per_source else set()


def gold_chunks(store: Store, q: dict) -> GoldChunks:
    sources = q.get("sources", [])
    resolved = gold_set.resolve_evidence(store, q)
    per_source = [set(ids) for ids in resolved[: len(sources)]]
    for alt, ids in zip(q.get("acceptable_sources", []), resolved[len(sources) :], strict=True):
        per_source[alt["restates"]] |= set(ids)
    return GoldChunks(per_source)


def recall_at(retrieved: list[str], gold: GoldChunks, k: int) -> float:
    top = set(retrieved[:k])
    return sum(bool(top & ids) for ids in gold.per_source) / len(gold.per_source)


def first_gold_rank(retrieved: list[str], gold: GoldChunks) -> int | None:
    relevant = gold.all
    return next((rank for rank, cid in enumerate(retrieved, 1) if cid in relevant), None)


def reciprocal_rank(retrieved: list[str], gold: GoldChunks) -> float:
    rank = first_gold_rank(retrieved, gold)
    return 1 / rank if rank else 0.0
