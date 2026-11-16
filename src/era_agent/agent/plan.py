import json
import time
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher

from era_agent.agent.loop import JSON_ONLY, Agent, Step, coverage_table
from era_agent.generation.answers import finalize, unparsed
from era_agent.models import Model, parse_json
from era_agent.prompts import PROMPTS

PLAN_PROMPT = PROMPTS / "plan.md"
COMBINE_PROMPT = PROMPTS / "combine.md"
MODES = ("ask", "research")
USABLE = ("verified", "calculated", "cited")
NOT_FROM_SUB = "not from a sub-answer"
REPLAN = (
    "Here is what each sub-question found. If a part of the original question is still "
    "unanswered, or a finding needs a follow-up, add at most {max_follow} follow-up "
    "sub-questions; otherwise add none. Never repeat a sub-question.\n\n{findings}\n\n"
    'Reply with only a JSON object: {{"sub_questions": [...]}} (an empty list for none).'
)
DUPLICATE_SIMILARITY = 0.9


@dataclass
class SubRun:
    id: str
    question: str
    mode: str
    why: str = ""
    follow_up: bool = False
    answer: dict | None = None
    stop: str = ""
    calls: int = 0
    seconds: float = 0.0
    steps: list[Step] = field(default_factory=list)


@dataclass
class Investigation:
    question: str
    answer: dict
    sub_runs: list[SubRun] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    fallback: bool = False
    stop: str = ""
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Investigator:
    model: Model
    agents: dict[str, Agent]
    max_sub: int = 5
    max_follow: int = 2
    max_tokens: int = 4000
    combine_tokens: int = 16000
    combine_output: int = 8000

    def __post_init__(self):
        toolbox = self.agents["research"].toolbox
        self.toolbox = toolbox
        coverage = coverage_table(toolbox.store)
        as_of = toolbox.asof.date.isoformat()
        self.plan_system = (PLAN_PROMPT.read_text().replace("{as_of}", as_of)
                            .replace("{max_sub}", str(self.max_sub))
                            .replace("{coverage}", coverage))
        self.combine_system = COMBINE_PROMPT.read_text().replace("{as_of}", as_of)

    def run(self, question: str) -> Investigation:
        start = time.perf_counter()
        inv = Investigation(question, {})
        messages = [{"role": "user", "content": f"Question: {question}"}]
        planned = self._json(self.plan_system, messages, inv)
        subs = self._subs(planned, question, self.max_sub, [], inv) if planned else []
        if not subs:
            inv.log.append("plan unparseable or empty: one Research run")
            run = self.agents["research"].run(question)
            inv.answer, inv.fallback, inv.stop = run.answer, True, "fallback"
            inv.sub_runs = [SubRun("s1", question, "research", "fallback", False, run.answer,
                                   run.stop, len(run.tool_calls), run.seconds, run.steps)]
            inv.seconds = round(time.perf_counter() - start, 3)
            return inv
        for sub in subs:
            self._run(sub, inv)

        messages += [{"role": "assistant", "content": json.dumps(planned)},
                     {"role": "user", "content": REPLAN.format(
                         max_follow=self.max_follow, findings=self._findings(inv))}]
        replanned = self._json(self.plan_system, messages, inv, retry=False)
        follow = self._subs(replanned, question, self.max_follow, inv.sub_runs, inv,
                            follow_up=True) if replanned else []
        if replanned is None:
            inv.log.append("replan unparseable: no follow-ups")
        for sub in follow:
            self._run(sub, inv)

        inv.answer, inv.stop = self._combine(question, inv)
        inv.seconds = round(time.perf_counter() - start, 3)
        return inv


    def _json(self, system: str, messages: list[dict], inv: Investigation,
              retry: bool = True, max_tokens: int | None = None) -> dict | None:
        for _ in range(2 if retry else 1):
            started = time.perf_counter()
            response = self.model.complete(messages, system=system,
                                           max_tokens=max_tokens or self.max_tokens)
            inv.steps.append(Step("model", round(time.perf_counter() - started, 3),
                                  "plan", text=response.text,
                                  input_tokens=response.input_tokens,
                                  output_tokens=response.output_tokens))
            try:
                reply = parse_json(response.text)
                if isinstance(reply, dict):
                    return reply
            except ValueError:
                pass
            messages = messages + [{"role": "assistant", "content": response.text},
                                   {"role": "user", "content": JSON_ONLY}]
        return None

    def _subs(self, reply: dict, question: str, limit: int, earlier: list[SubRun],
              inv: Investigation, follow_up: bool = False) -> list[SubRun]:
        items = reply.get("sub_questions") if isinstance(reply, dict) else None
        seen = {_norm(s.question) for s in earlier}
        subs = []
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            text, mode = str(item.get("question") or "").strip(), item.get("mode")
            if not text or mode not in MODES:
                inv.log.append(f"sub-question rejected (mode {mode!r}): {text[:80]}")
                continue
            if SequenceMatcher(None, _norm(text), _norm(question)).ratio() >= DUPLICATE_SIMILARITY:
                inv.log.append(f"sub-question rejected (restates the question): {text[:80]}")
                continue
            if _norm(text) in seen:
                inv.log.append(f"sub-question rejected (repeat): {text[:80]}")
                continue
            seen.add(_norm(text))
            subs.append(SubRun("", text, mode, str(item.get("why") or ""), follow_up))
        if len(subs) > limit:
            inv.log.append(f"{len(subs)} sub-questions cut to {limit}")
            subs = subs[:limit]
        for i, sub in enumerate(subs, len(earlier) + 1):
            sub.id = f"s{i}"
        return subs

    def _run(self, sub: SubRun, inv: Investigation) -> None:
        try:
            run = self.agents[sub.mode].run(sub.question)
        except Exception as e:  # noqa: BLE001  a failed sub-run is a gap, not a failed run
            sub.stop, sub.answer = f"failed: {type(e).__name__}: {e}"[:300], None
            inv.log.append(f"{sub.id} failed: {type(e).__name__}")
        else:
            sub.answer, sub.stop, sub.calls = run.answer, run.stop, len(run.tool_calls)
            sub.seconds, sub.steps = run.seconds, run.steps
        inv.sub_runs.append(sub)

    def _findings(self, inv: Investigation) -> str:
        lines = []
        for sub in inv.sub_runs:
            if sub.answer is None:
                lines.append(f"{sub.id} ({sub.mode}) {sub.question}\n  FAILED: {sub.stop}")
                continue
            statuses = [c["status"] for c in sub.answer["claims"]]
            counts = ", ".join(f"{statuses.count(s)} {s}" for s in dict.fromkeys(statuses))
            lines.append(f"{sub.id} ({sub.mode}) {sub.question}\n  answer: "
                         f"{sub.answer['answer'][:600]}\n  claims: {counts or 'none'}"
                         + ("; abstained" if sub.answer.get("abstained") else ""))
        return "\n".join(lines)


    def _combine(self, question: str, inv: Investigation) -> tuple[dict, str]:
        payload = self._payload(inv)
        messages = [{"role": "user", "content": f"Question: {question}\n\nSub-answers:\n"
                     + json.dumps(payload, ensure_ascii=False)}]
        reply = self._json(self.combine_system, messages, inv, max_tokens=self.combine_output)
        if reply is None or "answer" not in reply:
            text = inv.steps[-1].text if inv.steps else ""
            return finalize(unparsed(text), self.toolbox.store), "unparsed"
        answer = finalize(reply, self.toolbox.store, self.toolbox.sec, self.toolbox.book)
        allowed = {i for sub in inv.sub_runs if sub.answer for c in sub.answer["claims"]
                   if c["status"] in USABLE for i in c["chunk_ids"]}
        for claim in answer["claims"]:
            new = [i for i in claim["chunk_ids"] if i not in allowed]
            if new:
                claim["status"] = "unverified"
                claim["status_reason"] = f"{NOT_FROM_SUB}: {new}"
        answer["unverified"] = [c["id"] for c in answer["claims"] if c["status"] == "unverified"]
        answer["assumptions"] = reply.get("assumptions") or []
        answer["work"] = str(reply.get("work") or "")
        return answer, "answered"

    def _payload(self, inv: Investigation) -> list[dict]:
        def claim(c: dict, text: bool) -> dict:
            keep = ("id", "value", "unit", "company", "period", "chunk_ids", "status")
            return {k: c.get(k) for k in keep + (("text",) if text else ())
                    if c.get(k) is not None}

        def build(text: bool, answer_chars: int) -> list[dict]:
            out = []
            for sub in inv.sub_runs:
                item = {"id": sub.id, "mode": sub.mode, "question": sub.question}
                if sub.answer is None:
                    out.append(item | {"failed": sub.stop})
                    continue
                out.append(item | {
                    "answer": sub.answer["answer"][:answer_chars],
                    "abstained": sub.answer.get("abstained", False),
                    "claims": [claim(c, text) for c in sub.answer["claims"]]})
            return out

        cap = self.combine_tokens * 3
        for text, chars in ((True, 2000), (False, 2000), (False, 600), (False, 200)):
            payload = build(text, chars)
            if len(json.dumps(payload, ensure_ascii=False)) <= cap:
                return payload
        inv.log.append("combine input over its cap after cutting")
        return payload


def _norm(text: str) -> str:
    return " ".join(text.lower().replace("?", " ").split())


def answer_text(inv: Investigation) -> str:
    lines = [f"Question: {inv.question}", ""]
    if inv.fallback:
        lines.append("Plan failed: answered as one Research run.")
    for sub in inv.sub_runs:
        tag = " (follow-up)" if sub.follow_up else ""
        lines.append(f"{sub.id} [{sub.mode}]{tag} {sub.question}")
        lines.append(f"   {sub.answer['answer'] if sub.answer else sub.stop}")
    lines += ["", f"Answer: {inv.answer.get('answer', '')}"]
    for c in inv.answer.get("claims", []):
        lines.append(f"  [{c['id']}] {c['status']}: {c.get('text', '')} {c['chunk_ids']}")
    return "\n".join(lines)
