from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, timedelta


def _day(value: date | str) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])


@dataclass(frozen=True)
class AsOf:
    date: date

    @classmethod
    def of(cls, value: date | str) -> "AsOf":
        return cls(_day(value))

    @classmethod
    def today(cls) -> "AsOf":
        return cls(date.today())

    @classmethod
    def after(cls, value: date | str) -> "AsOf":
        return cls(_day(value) + timedelta(days=1))

    def sees(self, published: date | str | None) -> bool:
        return published is not None and _day(published) < self.date

    def sql(self, alias: str, column: str = "filed") -> tuple[str, list[str]]:
        return f"{alias}.{column} < ?", [self.date.isoformat()]

    def latest[T, K](
        self, rows: Iterable[T], key: Callable[[T], K], published: Callable[[T], str]
    ) -> dict[K, T]:
        best: dict[K, T] = {}
        for row in rows:
            when = published(row)
            if self.sees(when) and (
                key(row) not in best or _day(when) > _day(published(best[key(row)]))
            ):
                best[key(row)] = row
        return best

    def __str__(self) -> str:
        return self.date.isoformat()
