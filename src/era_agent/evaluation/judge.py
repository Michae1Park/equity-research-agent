from dataclasses import dataclass

from era_agent.models import Model, parse_json
from era_agent.prompts import PROMPTS

PROMPT = PROMPTS / "judge.md"


@dataclass
class Judge:
    model: Model

    def points_covered(self, question: str, points: list[str], answer: str) -> list[bool]:
        prompt = (
            PROMPT.read_text()
            .strip()
            .replace("{question}", question)
            .replace("{points}", "\n".join(f"{i}. {p}" for i, p in enumerate(points, 1)))
            .replace("{answer}", answer)
        )
        response = self.model.complete([{"role": "user", "content": prompt}], max_tokens=2000)
        covered = parse_json(response.text).get("covered", [])
        if len(covered) != len(points):
            raise ValueError(f"judge returned {len(covered)} verdicts, expected {len(points)}")
        return [bool(v) for v in covered]
