import json

from era_agent.prompts import PROMPTS

STRATEGIES = ("auto", "append", "rewrite", "none")
REWRITE_PROMPT = PROMPTS / "rewrite.md"
REWRITE_CHARS_PER_ANSWER = 2000
REWRITE_HISTORY_CHARS = 60000
CHARS_PER_TOKEN = 3


def answer_text(answer: dict | None) -> str:
    if not answer:
        return ""
    parts = [answer.get("answer") or ""]
    if table := answer.get("table"):
        parts.append(table if isinstance(table, str) else json.dumps(table, ensure_ascii=False))
    if assumptions := answer.get("assumptions"):
        parts.append("Assumptions: " + "; ".join(map(str, assumptions)))
    return "\n\n".join(p for p in parts if p)


def turns(messages: list[dict], limit: int | None = None) -> list[dict]:
    out, question = [], None
    for m in messages:
        if m["role"] == "user":
            question = m["text"]
        elif question is not None:
            if text := answer_text(m.get("answer")):
                out.append({"question": m.get("rewritten") or question, "answer": text})
            question = None
    return out if limit is None else out[-limit:] if limit else []


def tokens(history: list[dict]) -> int:
    return sum(len(t["question"]) + len(t["answer"]) + 20 for t in history) // CHARS_PER_TOKEN


def resolve(strategy: str, history: list[dict], budget: int) -> str:
    if not history or strategy == "none":
        return "none"
    if strategy == "auto":
        return "append" if tokens(history) <= budget else "rewrite"
    return strategy


def as_messages(history: list[dict]) -> list[dict]:
    out = []
    for t in history:
        out += [{"role": "user", "content": f"Question: {t['question']}"},
                {"role": "assistant", "content": t["answer"]}]
    return out


def as_context(history: list[dict], question: str) -> str:
    lines = ["Conversation so far (oldest first):"]
    for t in history:
        lines += [f"User: {t['question']}", f"Analyst: {t['answer']}", ""]
    return "\n".join(lines) + f"\nFollow-up question (answer this one): {question}"


def rewrite(model, history: list[dict], question: str) -> str:
    blocks, size = [], 0
    for t in reversed(history):
        block = f"User: {t['question']}\nAnalyst: {t['answer'][:REWRITE_CHARS_PER_ANSWER]}"
        if blocks and size + len(block) > REWRITE_HISTORY_CHARS:
            break
        blocks.insert(0, block)
        size += len(block)
    shown = "\n\n".join(blocks)
    prompt = REWRITE_PROMPT.read_text().replace("{history}", shown).replace("{question}",
                                                                             question)
    response = model.complete([{"role": "user", "content": prompt}], max_tokens=300)
    lines = (response.text or "").strip().splitlines()
    text = lines[0].strip().strip('"').strip() if lines else ""
    return text or question
