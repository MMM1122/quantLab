"""Background jobs with persisted state (PENDING/RUNNING/COMPLETED/FAILED).

A small thread pool is enough for a single-user local app; no external queue. Exceptions are
recorded on the job row (never swallowed). Jobs left RUNNING by a crashed process are marked
FAILED on startup so the UI never shows a phantom running job.
"""

from __future__ import annotations

import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from quantlab.context import AppContext
from quantlab.core.errors import NotFoundError
from quantlab.core.ids import new_id, validate_uuid
from quantlab.domain.models import RunStatus
from quantlab.storage.db import dumps, loads, utcnow


class JobRunner:
    def __init__(self, ctx: AppContext, max_workers: int = 2):
        self.ctx = ctx
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="quantlab-job")
        self._lock = threading.Lock()
        with ctx.db.transaction() as con:
            con.execute(
                "UPDATE jobs SET status=?, error=?, updated_at=? WHERE status IN ('PENDING','RUNNING')",
                (RunStatus.FAILED.value, "interrupted: process restarted before completion", utcnow()),
            )

    def submit(self, kind: str, params: dict[str, Any], fn: Callable[[], str | None]) -> str:
        jid = new_id()
        now = utcnow()
        self.ctx.db.execute(
            "INSERT INTO jobs(id, kind, status, params, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (jid, kind, RunStatus.PENDING.value, dumps(params), now, now),
        )
        self.pool.submit(self._run, jid, fn)
        return jid

    def _set(self, jid: str, status: RunStatus, result_ref: str | None = None, error: str | None = None) -> None:
        self.ctx.db.execute(
            "UPDATE jobs SET status=?, result_ref=COALESCE(?, result_ref), error=?, updated_at=? WHERE id=?",
            (status.value, result_ref, error, utcnow(), jid),
        )

    def _run(self, jid: str, fn: Callable[[], str | None]) -> None:
        self._set(jid, RunStatus.RUNNING)
        try:
            ref = fn()
            self._set(jid, RunStatus.COMPLETED, result_ref=ref)
        except Exception as exc:
            tb = traceback.format_exc(limit=5)
            self._set(jid, RunStatus.FAILED, error=f"{type(exc).__name__}: {exc}\n{tb}")

    def get(self, job_id: str) -> dict[str, Any]:
        jid = validate_uuid(job_id, "job_id")
        r = self.ctx.db.query_one("SELECT * FROM jobs WHERE id=?", (jid,))
        if r is None:
            raise NotFoundError(f"job {jid} not found")
        r["params"] = loads(r["params"])
        return r

    def wait(self, job_id: str, timeout: float = 120.0) -> dict[str, Any]:
        t0 = time.time()
        while time.time() - t0 < timeout:
            j = self.get(job_id)
            if j["status"] in (RunStatus.COMPLETED.value, RunStatus.FAILED.value):
                return j
            time.sleep(0.05)
        raise TimeoutError(f"job {job_id} did not finish within {timeout}s")
