"""LoomMemoryService against a running Loom: a session an ADK Runner saves is
extracted, and preload_memory puts it in a later session's prompt for the same
app and user only; a memory, extracted or explicit, is deleted by the id a search
returned."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest

from tests.integration._eventually import eventually

# An SDK extra, outside the server's environment: install google-adk to run these.
pytest.importorskip("google.adk")

from google.adk.agents import LlmAgent
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import preload_memory
from google.genai import types
from loom.integrations.google_adk import LoomMemoryService
from pydantic import Field


class _EchoLlm(BaseLlm):
    """Answers "Noted." to every call and keeps the requests."""

    model: str = "echo"
    requests: list[LlmRequest] = Field(default_factory=list)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.requests.append(llm_request)
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="Noted.")]))


async def _save(callback_context: Any) -> None:
    await callback_context.add_session_to_memory()


async def _turn(service: LoomMemoryService, app: str, user: str, text: str) -> str:
    """One turn in a new session; returns the text of the model's prompt."""
    llm = _EchoLlm()
    sessions = InMemorySessionService()
    runner = Runner(
        app_name=app,
        agent=LlmAgent(
            name="assistant", model=llm, tools=[preload_memory], after_agent_callback=_save
        ),
        session_service=sessions,
        memory_service=service,
    )
    session = await sessions.create_session(app_name=app, user_id=user)
    message = types.Content(role="user", parts=[types.Part(text=text)])
    async for _ in runner.run_async(user_id=user, session_id=session.id, new_message=message):
        pass
    return " ".join(p.text or "" for c in llm.requests[0].contents for p in c.parts or [])


async def test_a_saved_session_reaches_a_later_prompt(agent) -> None:
    # defer=False: extraction inline, since the test server runs no background workers
    service = LoomMemoryService(agent, honour_abstain=False, defer=False)
    await _turn(service, "travel", "alice", "I fly to Lisbon every month.")

    async def recalled() -> bool:
        prompt = await _turn(service, "travel", "alice", "Where does Alice fly every month?")
        return "lisbon" in prompt.lower()

    # A write is searchable once its embedding lands; eventually raises on timeout.
    await eventually(recalled, description="'lisbon' in the next session's prompt")
    for app, user in (("travel", "bob"), ("billing", "alice")):
        found = await service.search_memory(app_name=app, user_id=user, query="Lisbon")
        assert not found.memories, (app, user)
    # The extracted facts carry the session; the ids a search returns still delete them.
    found = await service.search_memory(app_name="travel", user_id="alice", query="Lisbon")
    facts = {m.id for m in found.memories if m.custom_metadata["source"] == "fact"}
    assert facts
    for memory_id in facts:
        await service.delete_memory(app_name="travel", user_id="alice", memory_id=memory_id)
    found = await service.search_memory(app_name="travel", user_id="alice", query="Lisbon")
    assert not {m.id for m in found.memories if m.custom_metadata["source"] == "fact"} & facts


async def test_add_memory_then_delete_by_id(agent) -> None:
    service = LoomMemoryService(agent, honour_abstain=False)
    entry = MemoryEntry(content=types.Content(parts=[types.Part(text="Alice prefers aisle seats")]))
    await service.add_memory(app_name="travel", user_id="alice", memories=[entry])

    async def ids() -> list[str | None]:
        found = await service.search_memory(app_name="travel", user_id="alice", query="aisle seat")
        return [m.id for m in found.memories]

    await eventually(ids, description="the memory found")
    (memory_id,) = await ids()
    await service.delete_memory(app_name="travel", user_id="alice", memory_id=memory_id)
    assert await ids() == []
