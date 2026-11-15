"""memory.redact reaches a shared copy of the text it scrubs: the loom.shared versions
promoted from the key whose value is that text, and their search rows. CH-free."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

import server.memory.store as memory_repo
from server.core import namespace_write_marks
from server.core.hashing import sha256_value
from server.memory.app import curation
from server.memory.domain.excerpt import build_excerpt
from server.memory.domain.promotion_source import promotion_source
from server.memory.store.history import HistoryTarget

SECRET = {"note": "the vault code is 4417"}
SOURCE = promotion_source("acme", "shop", "a", "-", "-", "k")


def _target(value: Any = SECRET) -> HistoryTarget:
    return HistoryTarget(
        namespace="shop",
        agent="a",
        user_id="-",
        session_id="-",
        memory_key="k",
        version=7,
        org="acme",
        memory_id="33333333-3333-3333-3333-333333333333",
        content_sha256=sha256_value(value),
    )


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {"read": [], "scrubs": [], "bumps": []}
    copies = [
        {"memory_key": "k", "value": json.dumps(SECRET)},
        {"memory_key": "renamed", "value": json.dumps(SECRET)},
        {"memory_key": "k", "value": json.dumps({"note": "another text"})},
    ]

    async def shared_copies(org: str, namespace: str, source: str) -> list[dict[str, Any]]:
        seen["read"].append((org, namespace, source))
        return copies

    async def run_scrubs(scrubs: list[Any], *, apply: bool) -> dict[str, int]:
        seen["scrubs"].extend(scrubs)
        return {}

    async def bump(org: str, namespace: str, plane: str) -> None:
        seen["bumps"].append((org, namespace, plane))

    monkeypatch.setattr(memory_repo, "shared_copies", shared_copies)
    monkeypatch.setattr(memory_repo, "run_scrubs", run_scrubs)
    monkeypatch.setattr(namespace_write_marks, "bump", bump)
    return seen


async def test_the_copies_holding_the_text_are_scrubbed(calls: dict[str, Any]) -> None:
    keys = await curation._scrub_shared_copies("acme", _target(), None)
    assert keys == ["k", "renamed"]
    assert calls["read"] == [("acme", "shop", SOURCE)]
    purge, scrub = calls["scrubs"]
    # The search rows first, by the excerpt a promotion embeds.
    assert purge.table == "loom.memory_embeddings"
    assert purge.params["ex"] == [build_excerpt(json.dumps(SECRET), memory_key="k")]
    assert purge.params["keys"] == ["k", "renamed"]
    assert scrub.table == "loom.shared"
    assert scrub.params["values"] == [json.dumps(SECRET)]
    assert scrub.params["src"] == SOURCE
    assert calls["bumps"] == [("acme", "shop", "memory")]


async def test_a_live_head_holding_the_text_keeps_its_copies(calls: dict[str, Any]) -> None:
    head = SimpleNamespace(content_sha256=sha256_value(SECRET))
    assert await curation._scrub_shared_copies("acme", _target(), head) == []  # type: ignore[arg-type]
    assert calls == {"read": [], "scrubs": [], "bumps": []}


async def test_no_copy_of_the_text_rewrites_nothing(calls: dict[str, Any]) -> None:
    assert (
        await curation._scrub_shared_copies("acme", _target({"note": "never promoted"}), None) == []
    )
    assert calls["scrubs"] == [] and calls["bumps"] == []


def test_the_scrub_covers_the_shared_rows_text_and_only_the_copies() -> None:
    purge, scrub = memory_repo.shared_copy_scrubs(
        "acme", "shop", SOURCE, [("k", '"x"')], ["x (#k)"]
    )
    assert purge.update_sql().startswith("DELETE FROM loom.memory_embeddings")
    assert "scope = 'shared'" in purge.where and "agent = {a:String}" in purge.where
    assert purge.params["a"] == "_shared"
    assert scrub.update_sql().startswith(
        """ALTER TABLE loom.shared UPDATE value = '"[REDACTED]"'"""
    )
    assert "has(promoted_from, {src:String})" in scrub.where
    assert "value IN {values:Array(String)}" in scrub.where
