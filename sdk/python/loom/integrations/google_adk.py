"""Loom as a Google ADK memory service.

``LoomMemoryService`` implements ``google.adk.memory.BaseMemoryService``, the
service a ``Runner`` takes as ``memory_service=``. ADK's memory tools read it:
``preload_memory`` searches with the user's message before each model call,
``load_memory`` lets the model search. A callback saves each turn::

    from google.adk.agents import LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.tools import preload_memory
    from loom import Agent
    from loom.integrations.google_adk import LoomMemoryService


    async def save_turn(ctx):
        turn = [e for e in ctx.session.events if e.invocation_id == ctx.invocation_id]
        await ctx.add_events_to_memory(events=turn)


    async with Agent(org="acme", namespace="support") as loom_agent:
        runner = Runner(
            app_name="support",
            agent=LlmAgent(
                name="assistant",
                model="gemini-flash-latest",
                tools=[preload_memory],
                after_agent_callback=save_turn,
            ),
            session_service=InMemorySessionService(),
            memory_service=LoomMemoryService(loom_agent),
        )

ADK scopes memory by app and user: the app name becomes the Loom ``agent`` of
each write and narrows each search, and the ADK user is the Loom ``user_id``.
ADK's memory spans sessions, so rows carry no session. The namespace and
isolation come from the Loom agent. Install with
``pip install 'clickhouse-loom[google-adk]'``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

try:
    from google.adk.events import Event, get_active_events
    from google.adk.memory import BaseMemoryService
    from google.adk.memory.base_memory_service import SearchMemoryResponse
    from google.adk.memory.memory_entry import MemoryEntry
    from google.adk.sessions import Session
    from google.genai import types
except ImportError as e:  # the framework is an optional extra
    raise ImportError(
        "loom.integrations.google_adk needs google-adk: pip install 'clickhouse-loom[google-adk]'"
    ) from e

from loom.sync import SyncAgent

if TYPE_CHECKING:
    from loom.agent import Agent
    from loom.memory import MemorySearchHit


class LoomMemoryService(BaseMemoryService):
    """ADK ``BaseMemoryService`` over a Loom ``Agent``.

    ``add_session_to_memory`` and ``add_events_to_memory`` send the text of the
    events to ``memory.set_from_messages``, where Loom extracts the facts and
    resolves duplicates and conflicts; a session added again is extracted
    again, so a callback that runs every turn should add only that turn's
    events. Events a rewind took back are left out. ``add_memory`` writes each
    entry with ``memory.set`` under a key derived from its text
    (``<key_prefix>/<first 20 hex of sha256>``), its ``timestamp`` as the
    memory's ``valid_at`` when it is ISO 8601. ``search_memory`` runs ``memory.search`` of the
    namespace it writes to; with ``honour_abstain`` it leaves out memories
    below Loom's floor and returns nothing when Loom abstains. A returned
    entry's ``id`` is the Loom key ``delete_memory`` deletes, in the app and
    user given: the memory's, or for a verbatim chunk the memory it was
    written under. An entry Loom cannot delete by key, such as a session
    summary, has no ``id``; Loom's erasure removes it.
    ``custom_metadata`` is not kept. ADK awaits its memory service, so the
    service takes a ``loom.Agent``.
    """

    def __init__(
        self,
        agent: Agent,
        *,
        top_k: int = 10,
        honour_abstain: bool = True,
        defer: bool = True,
        key_prefix: str = "adk",
    ) -> None:
        if isinstance(agent, SyncAgent):
            raise TypeError("LoomMemoryService needs a loom.Agent: ADK awaits its memory service")
        if not 1 <= top_k <= 500:  # memory.search's bounds
            raise ValueError("top_k must be between 1 and 500")
        self._agent = agent
        self._top_k = top_k
        self._honour_abstain = honour_abstain
        self._defer = defer
        if not key_prefix.strip("/"):
            raise ValueError("key_prefix must not be empty")
        self._key_prefix = key_prefix.strip("/")

    async def add_session_to_memory(self, session: Session) -> None:
        await self._extract(session.app_name, session.user_id, get_active_events(session.events))

    async def add_events_to_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        events: Sequence[Event],
        session_id: str | None = None,
        custom_metadata: Mapping[str, object] | None = None,
    ) -> None:
        await self._extract(app_name, user_id, events)

    async def add_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        memories: Sequence[MemoryEntry],
        custom_metadata: Mapping[str, object] | None = None,
    ) -> None:
        for memory in memories:
            text = _text(memory.content)
            if not text:
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
            await self._agent.memory.set(
                f"{self._key_prefix}/{digest}",
                text,
                user_id=user_id,
                agent_id=app_name,
                valid_at=_iso_time(memory.timestamp),
            )

    async def search_memory(
        self, *, app_name: str, user_id: str, query: str
    ) -> SearchMemoryResponse:
        if not query.strip():
            return SearchMemoryResponse()
        result = await self._agent.memory.search(
            query,
            top_k=self._top_k,
            user_id=user_id,
            agent_id=app_name,
            # Read the namespace this service writes to, whatever the deployment's default.
            namespace_scope="this",
            # Unpinned, a large top_k serves excerpts (above 15 by default)
            # and drops each hit's explanation.
            bodies="full",
            explain=False,
        )
        hits = list(result)
        if self._honour_abstain:
            if result.abstain:
                return SearchMemoryResponse()
            # A response that does not abstain can still carry rows below the floor.
            hits = [hit for hit in hits if not hit.below_threshold]
        return SearchMemoryResponse(
            memories=[
                MemoryEntry(
                    id=_deletable_key(hit),
                    content=types.Content(
                        role="user", parts=[types.Part(text=hit.content_excerpt)]
                    ),
                    custom_metadata={"similarity": hit.similarity, "source": hit.source},
                    timestamp=hit.valid_at or None,
                )
                # Derived rows may come back above the top_k slice: cut after filtering.
                for hit in hits[: self._top_k]
            ]
        )

    async def delete_memory(
        self, *, app_name: str, user_id: str, memory_id: str | None = None
    ) -> None:
        if not memory_id:
            raise ValueError("delete_memory needs the memory_id a search returned")
        # memory.delete falls back to the key under any agent, which could be another
        # app's row. Read the version this app's row holds, then delete that version
        # under the exact identity; a row of another app is never touched.
        row = await self._agent.memory.get(memory_id, user_id=user_id, agent_id=app_name)
        if row is not None:
            await self._agent.call(
                "memory.delete_version",
                {"memory_key": memory_id, "version": row["version"]},
                user_id=user_id,
                agent_id=app_name,
            )

    async def _extract(self, app_name: str, user_id: str, events: Sequence[Event]) -> None:
        messages: list[dict[str, object]] = []
        for event in events:
            text = _text(event.content)
            if text:
                messages.append(
                    {
                        "role": "user" if event.author == "user" else "assistant",
                        "content": text,
                        "timestamp": datetime.fromtimestamp(event.timestamp, UTC).isoformat(),
                    }
                )
        if messages:
            await self._agent.memory.set_from_messages(
                messages,
                user_id=user_id,
                agent_id=app_name,
                defer=self._defer,
            )


def _deletable_key(hit: MemorySearchHit) -> str | None:
    if hit.source == "fact":
        return hit.memory_key
    return hit.parent_memory_key or None


def _iso_time(text: str | None) -> datetime | None:
    """An ISO 8601 timestamp; ADK allows any text, which is not a ``valid_at``."""
    try:
        return datetime.fromisoformat(text) if text else None
    except ValueError:
        return None


def _text(content: types.Content | None) -> str:
    """The text of a content, without the model's thoughts or tool calls."""
    if content is None or not content.parts:
        return ""
    return " ".join(p.text for p in content.parts if p.text and not p.thought).strip()
