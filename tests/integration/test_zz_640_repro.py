"""Fork-only repro for ClickHouse/loom#640 (never part of a PR).

A key's first memory.set answers at its commit. When a second write to the key commits
before the first write's flush reads the heads, the first revision's search document is
stale and its delivery is handed to the delivery worker, which this job runs off. Each
iteration prints whether the org's deliveries settled within 15 s:
OLD = set, set, wait  (the pattern the three tests had)
NEW = set, wait, set, wait  (the fixed pattern)
Bursts of other first writes drain the per-process flush token bucket, so the first
write's flush is paced behind the second write's commit.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest

import httpx
import os

from tests.integration._resources import unique_org


@pytest.fixture
async def pg_conn(control_plane_url: str) -> Any:
    import asyncpg

    from server.core import pg

    dsn = os.environ.get("LOOM_PG_DSN") or pytest.skip("LOOM_PG_DSN not set")
    conn = await asyncpg.connect(dsn)
    await pg.init_connection(conn)
    try:
        yield conn
    finally:
        await conn.close()


@pytest.fixture
def org() -> str:
    return unique_org("pgrepro")


@pytest.fixture
def call(control_plane_url: str, mint_test_token: Any, org: str) -> Any:
    token = mint_test_token(org=org, namespaces=["*"], roles=["writer", "admin", "org_admin"])

    async def _call(op: str, body: dict[str, Any]) -> httpx.Response:
        async with httpx.AsyncClient(base_url=control_plane_url, timeout=120) as c:
            return await c.post(
                f"/v1/{op}", json={"org": org, **body}, headers={"Authorization": f"Bearer {token}"}
            )

    return _call


def _set(ns: str, key: str, value: str) -> dict[str, Any]:
    return {"namespace": ns, "agent": "a", "memory_key": key, "memory_type": "semantic", "value": value}


pytestmark = pytest.mark.asyncio


async def _settled_within(pg_conn: Any, org: str, seconds: float) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + seconds
    while loop.time() < deadline:
        owed = await pg_conn.fetchval(
            "SELECT count(*) FROM loom.memory_deliveries WHERE org = $1 AND NOT settled", org
        )
        if owed == 0:
            return True
        await asyncio.sleep(0.1)
    return False


@pytest.mark.parametrize("pattern", ["OLD", "NEW"])
@pytest.mark.parametrize("i", range(8))
async def test_overtake(call: Any, pg_conn: Any, org: str, pattern: str, i: int) -> None:
    ns = f"pg-{uuid.uuid4().hex[:8]}"
    burst = [call("memory.set", _set(f"{ns}-b{j}", "k", f"burst {j}")) for j in range(16)]
    first, *_ = await asyncio.gather(
        call("memory.set", _set(ns, "k", "the token is hunter2")), *burst
    )
    assert first.status_code == 200, first.text
    if pattern == "NEW":
        assert await _settled_within(pg_conn, org, 15), "NEW: first write never published"
    second = await call("memory.set", _set(ns, "k", "the token was rotated"))
    assert second.status_code == 200, second.text
    ok = await _settled_within(pg_conn, org, 15)
    stale = await pg_conn.fetchval(
        "SELECT count(*) FROM loom.memory_deliveries WHERE org = $1 AND NOT settled", org
    )
    print(f"REPRO640 pattern={pattern} i={i} settled={int(ok)} owed={stale}")
