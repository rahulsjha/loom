"""LoomMemoryService: ADK's memory service over a fake Loom agent — what each
method sends, abstention, and what an ADK Runner's model sees through
preload_memory and load_memory."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from google.adk.agents import LlmAgent
from google.adk.memory import BaseMemoryService
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import load_memory, preload_memory
from google.genai import types
from loom import SyncAgent
from loom.integrations.google_adk import LoomMemoryService
from loom.memory import MemorySearchHit, MemorySearchResult
from pydantic import Field


class _FakeMemory:
    def __init__(self, hits: list[MemorySearchHit] | None = None, abstain: bool = False) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self._hits = hits or []
        self._abstain = abstain

    async def search(self, query: str, **kw: Any) -> MemorySearchResult:
        self.calls.append(("search", (query,), kw))
        result = MemorySearchResult(self._hits)
        result.abstain = self._abstain
        return result

    async def set(self, key: str, value: str, **kw: Any) -> dict[str, Any]:
        self.calls.append(("set", (key, value), kw))
        return {}

    async def set_from_messages(self, messages: list[dict[str, object]], **kw: Any) -> dict:
        self.calls.append(("set_from_messages", (messages,), kw))
        return {}

    async def delete(self, key: str, **kw: Any) -> bool:
        self.calls.append(("delete", (key,), kw))
        return True


class _Agent:
    def __init__(self, memory: _FakeMemory) -> None:
        self.memory = memory


class _ScriptedLlm(BaseLlm):
    """Answers each model call with the next scripted content; keeps the requests."""

    model: str = "scripted"
    script: list[types.Content] = Field(default_factory=list)
    requests: list[LlmRequest] = Field(default_factory=list)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.requests.append(llm_request)
        yield LlmResponse(content=self.script.pop(0))


def _hit(key: str, text: str, **kw: Any) -> MemorySearchHit:
    return MemorySearchHit(
        memory_key=key, memory_type="semantic", content_excerpt=text, similarity=0.9, **kw
    )


def _model(text: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part(text=text)])


def _service(fake: _FakeMemory, **kw: Any) -> LoomMemoryService:
    return LoomMemoryService(_Agent(fake), **kw)  # type: ignore[arg-type]


async def _run(
    service: LoomMemoryService, llm: _ScriptedLlm, *texts: str, **agent_kw: Any
) -> InMemorySessionService:
    sessions = InMemorySessionService()
    runner = Runner(
        app_name="travel",
        agent=LlmAgent(name="assistant", model=llm, **agent_kw),
        session_service=sessions,
        memory_service=service,
    )
    session = await sessions.create_session(app_name="travel", user_id="alice")
    for text in texts:
        message = types.Content(role="user", parts=[types.Part(text=text)])
        async for _ in runner.run_async(
            user_id="alice", session_id=session.id, new_message=message
        ):
            pass
    return sessions


def _prompt(request: LlmRequest) -> str:
    return " ".join(p.text or "" for c in request.contents for p in c.parts or [])


def test_it_is_an_adk_memory_service() -> None:
    assert issubclass(LoomMemoryService, BaseMemoryService)


def test_it_refuses_a_sync_agent_and_a_top_k_out_of_bounds() -> None:
    with pytest.raises(TypeError, match="ADK awaits"):
        LoomMemoryService(SyncAgent(org="o", namespace="n", base_url="http://127.0.0.1:9"))
    for top_k in (0, 501):
        with pytest.raises(ValueError, match="top_k"):
            _service(_FakeMemory(), top_k=top_k)


def test_preload_memory_puts_the_matches_in_the_prompt() -> None:
    fake = _FakeMemory([_hit("adk/1", "Alice prefers aisle seats", valid_at="2026-11-01")])
    llm = _ScriptedLlm(script=[_model("An aisle seat.")])
    asyncio.run(_run(_service(fake), llm, "Which seat for alice?", tools=[preload_memory]))
    (search,) = [c for c in fake.calls if c[0] == "search"]
    assert search[1] == ("Which seat for alice?",)
    assert search[2] == {
        "top_k": 10,
        "user_id": "alice",
        "agent_id": "travel",
        "namespace_scope": "this",
        "bodies": "full",
        "explain": False,
    }
    prompt = _prompt(llm.requests[0])
    assert "Alice prefers aisle seats" in prompt and "Time: 2026-11-01" in prompt


def test_the_model_reads_load_memory_results() -> None:
    fake = _FakeMemory([_hit("adk/1", "Alice prefers aisle seats")])
    call = types.Content(
        role="model",
        parts=[
            types.Part(function_call=types.FunctionCall(name="load_memory", args={"query": "seat"}))
        ],
    )
    llm = _ScriptedLlm(script=[call, _model("An aisle seat.")])
    asyncio.run(_run(_service(fake), llm, "Which seat?", tools=[load_memory]))
    (response,) = [
        p.function_response
        for c in llm.requests[1].contents
        for p in c.parts or []
        if p.function_response
    ]
    (memory,) = response.response["result"].memories
    assert memory.id == "adk/1"
    assert memory.content.parts[0].text == "Alice prefers aisle seats"


def test_search_leaves_out_rows_below_the_floor_and_an_abstention() -> None:
    hits = [_hit("adk/1", "kept"), _hit("adk/2", "weak", below_threshold=True)]
    found = asyncio.run(
        _service(_FakeMemory(hits)).search_memory(app_name="a", user_id="u", query="q")
    )
    assert [m.id for m in found.memories] == ["adk/1"]
    everything = _service(_FakeMemory(hits), honour_abstain=False)
    found = asyncio.run(everything.search_memory(app_name="a", user_id="u", query="q"))
    assert [m.id for m in found.memories] == ["adk/1", "adk/2"]
    abstained = _service(_FakeMemory(hits, abstain=True))
    assert not asyncio.run(abstained.search_memory(app_name="a", user_id="u", query="q")).memories


def test_search_cuts_to_top_k_and_skips_an_empty_query() -> None:
    fake = _FakeMemory([_hit(f"adk/{i}", f"fact {i}") for i in range(4)])
    found = asyncio.run(_service(fake, top_k=2).search_memory(app_name="a", user_id="u", query="q"))
    assert [m.id for m in found.memories] == ["adk/0", "adk/1"]
    assert not asyncio.run(
        _service(fake).search_memory(app_name="a", user_id="u", query="  ")
    ).memories
    assert len(fake.calls) == 1


def test_a_saved_session_goes_to_extraction() -> None:
    fake = _FakeMemory()

    async def save(callback_context: Any) -> None:
        await callback_context.add_session_to_memory()

    llm = _ScriptedLlm(script=[_model("Noted.")])
    sessions = asyncio.run(
        _run(_service(fake), llm, "I prefer aisle seats", after_agent_callback=save)
    )
    (extract,) = [c for c in fake.calls if c[0] == "set_from_messages"]
    (session,) = asyncio.run(sessions.list_sessions(app_name="travel", user_id="alice")).sessions
    messages = extract[1][0]
    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", "I prefer aisle seats"),
        ("assistant", "Noted."),
    ]
    assert all(str(m["timestamp"]).endswith("+00:00") for m in messages)
    assert extract[2] == {
        "user_id": "alice",
        "session_id": session.id,
        "agent_id": "travel",
        "defer": True,
    }


def test_extraction_leaves_out_thoughts_tool_calls_and_rewound_turns() -> None:
    from google.adk.events import Event, EventActions
    from google.adk.sessions import Session

    def event(inv: str, author: str, *parts: types.Part, **kw: Any) -> Event:
        return Event(
            invocation_id=inv, author=author, content=types.Content(parts=list(parts)), **kw
        )

    session = Session(
        id="s1",
        app_name="travel",
        user_id="alice",
        events=[
            event("i1", "user", types.Part(text="I live in Lisbon")),
            event(
                "i1",
                "assistant",
                types.Part(text="planning", thought=True),
                types.Part(function_call=types.FunctionCall(name="f", args={})),
                types.Part(text="Lisbon it is"),
            ),
            event("i2", "user", types.Part(text="Actually Porto")),
            Event(
                invocation_id="i3",
                author="user",
                actions=EventActions(rewind_before_invocation_id="i2"),
            ),
        ],
    )
    fake = _FakeMemory()
    asyncio.run(_service(fake).add_session_to_memory(session))
    (extract,) = fake.calls
    assert [m["content"] for m in extract[1][0]] == ["I live in Lisbon", "Lisbon it is"]


def test_events_without_text_send_nothing() -> None:
    from google.adk.events import Event

    fake = _FakeMemory()
    silent = Event(author="user", content=types.Content(parts=[types.Part(text=" ")]))
    asyncio.run(
        _service(fake, defer=False).add_events_to_memory(app_name="a", user_id="u", events=[silent])
    )
    assert fake.calls == []


def test_add_memory_writes_content_keyed_memories() -> None:
    fake = _FakeMemory()
    entries = [
        MemoryEntry(
            content=_model("Alice prefers aisle seats"), timestamp="2026-11-01T09:00:00+00:00"
        ),
        MemoryEntry(content=_model("  Alice prefers aisle seats ")),
        MemoryEntry(content=types.Content(parts=[types.Part(text="hidden", thought=True)])),
    ]
    asyncio.run(_service(fake).add_memory(app_name="travel", user_id="alice", memories=entries))
    (first, again) = fake.calls
    assert first[1][0] == again[1][0] and first[1][0].startswith("adk/")
    assert first[1][1] == "Alice prefers aisle seats"
    assert first[2]["user_id"] == "alice" and first[2]["agent_id"] == "travel"
    assert first[2]["valid_at"].isoformat() == "2026-11-01T09:00:00+00:00"
    assert again[2]["valid_at"] is None


def test_delete_memory_deletes_the_key_and_needs_one() -> None:
    fake = _FakeMemory()
    service = _service(fake)
    asyncio.run(service.delete_memory(app_name="travel", user_id="alice", memory_id="adk/1"))
    assert fake.calls == [("delete", ("adk/1",), {"user_id": "alice"})]
    with pytest.raises(ValueError, match="memory_id"):
        asyncio.run(service.delete_memory(app_name="travel", user_id="alice"))


def test_an_entry_id_is_the_key_delete_memory_deletes() -> None:
    hits = [
        _hit("adk/1", "a fact"),
        _hit("chunk:s1:0", "a turn", source="chunk", parent_memory_key="adk/1"),
        _hit("summary:s1", "a session", source="summary"),
    ]
    found = asyncio.run(
        _service(_FakeMemory(hits)).search_memory(app_name="a", user_id="u", query="q")
    )
    assert [m.id for m in found.memories] == ["adk/1", "adk/1", None]
    assert [m.custom_metadata["source"] for m in found.memories] == ["fact", "chunk", "summary"]
