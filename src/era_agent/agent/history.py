"""Multi-turn chat: earlier turns as context for a follow-up question.

Strategies (config `chat.history`, or per request):
  auto     append while the earlier turns fit `chat.append_budget` tokens; past it, rewrite
  append   the earlier turns go into the agent's conversation as user/assistant messages
           (Ask, Research) or as a "Conversation so far" block in the question (Investigate)
  rewrite  one model call rewrites the follow-up into a standalone question; the agent runs
           on it with no history
  none     each question stands alone (the single-turn behaviour)
"""

import json

from era_agent.prompts import PROMPTS

STRATEGIES = ("auto", "append", "rewrite", "none")
REWRITE_PROMPT = PROMPTS / "rewrite.md"
REWRITE_ANSWER_CHARS = 2000  # an earlier answer as the rewrite call sees it
REWRITE_CHARS = 60000  # the most recent turns the rewrite call sees, at most
CHARS_PER_TOKEN = 3  # a low estimate (as the agent loop's), so the history is not undercounted


def answer_text(answer: dict | None) -> str:
    """An answer as the model reads it back: the prose, its table and assumptions."""
    if not answer:
        return ""
    parts = [answer.get("answer") or ""]
    if table := answer.get("table"):
        parts.append(table if isinstance(table, str) else json.dumps(table, ensure_ascii=False))
    if assumptions := answer.get("assumptions"):
        parts.append("Assumptions: " + "; ".join(map(str, assumptions)))
    return "\n\n".join(p for p in parts if p)


def turns(messages: list[dict], limit: int | None = None) -> list[dict]:
    """The last `limit` answered turns of a saved chat (None: all) → [{question, answer}]. A
    question whose run failed or was cancelled is skipped."""
    out, question = [], None
    for m in messages:
        if m["role"] == "user":
            question = m["text"]
        elif question is not None:
            if text := answer_text(m.get("answer")):  # a rewrite stands in for the follow-up
                out.append({"question": m.get("rewritten") or question, "answer": text})
            question = None
    return out if limit is None else out[-limit:] if limit else []


def tokens(history: list[dict]) -> int:
    """Estimated tokens the turns add to the agent's input when appended."""
    return sum(len(t["question"]) + len(t["answer"]) + 20 for t in history) // CHARS_PER_TOKEN


def resolve(strategy: str, history: list[dict], budget: int) -> str:
    """The strategy a follow-up runs with: auto → append while the history fits `budget`
    tokens, else rewrite; none when there is no history."""
    if not history or strategy == "none":
        return "none"
    if strategy == "auto":
        return "append" if tokens(history) <= budget else "rewrite"
    return strategy


def as_messages(history: list[dict]) -> list[dict]:
    """Earlier turns as the agent loop's messages, oldest first."""
    out = []
    for t in history:
        out += [{"role": "user", "content": f"Question: {t['question']}"},
                {"role": "assistant", "content": t["answer"]}]
    return out


def as_context(history: list[dict], question: str) -> str:
    """Earlier turns and the follow-up as one question text (for the Investigate planner)."""
    lines = ["Conversation so far (oldest first):"]
    for t in history:
        lines += [f"User: {t['question']}", f"Analyst: {t['answer']}", ""]
    return "\n".join(lines) + f"\nFollow-up question (answer this one): {question}"


def rewrite(model, history: list[dict], question: str) -> str:
    """The follow-up as a standalone question; the question unchanged if the call fails."""
    blocks, size = [], 0
    for t in reversed(history):  # the most recent turns, as many as fit
        block = f"User: {t['question']}\nAnalyst: {t['answer'][:REWRITE_ANSWER_CHARS]}"
        if blocks and size + len(block) > REWRITE_CHARS:
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
