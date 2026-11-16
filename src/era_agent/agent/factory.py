import threading
from dataclasses import replace
from pathlib import Path

from era_agent.agent.loop import Agent
from era_agent.agent.tools import Toolbox
from era_agent.asof import AsOf
from era_agent.models import get_model
from era_agent.research.book import Book
from era_agent.retrieval.search import SubQuestionRetriever
from era_agent.store import Store


def agent_factory(db: Path, retrieval: dict, generation: dict, agent_config: dict):
    base = SubQuestionRetriever.from_settings(Store(db), retrieval)
    model = get_model(generation["model"], base_url=generation["base_url"], thinking=False)
    lock, search_lock = threading.Lock(), threading.Lock()
    books: dict[str, Book] = {}

    def agent(as_of: str, tool_set: str = "research", session: Book | None = None) -> Agent:
        store = Store(db, AsOf.of(as_of))
        with lock:
            book = books.setdefault(as_of, Book(store))
        session = session or Book(store, book.settings, book.forecasts, {}, {}, book.earlier)
        search = replace(base, decomposer=replace(base.decomposer, store=store),
                         hybrid=replace(base.hybrid, store=store))
        toolbox = Toolbox(store, search, None, tool_set, session)
        call = toolbox.call

        def locked(name, arguments):
            if name == "search_filings":
                with search_lock:
                    return call(name, arguments)
            with lock:
                return call(name, arguments)

        toolbox.call = locked
        return Agent(model, toolbox, max_calls=agent_config["max_calls"],
                     max_turns=agent_config["max_turns"], max_tokens=generation["max_tokens"],
                     result_chars=agent_config["result_chars"],
                     max_context=agent_config["max_context"])

    return agent
