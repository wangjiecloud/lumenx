"""Central registry for in-process background generation tasks.

LumenX kicks off long-running work (asset image generation, motion reference
videos, storyboard video tasks) through FastAPI ``BackgroundTasks``. Those run
in the same process, but until now there was no single place to see what is
running or to ask it to stop.

This registry is intentionally cooperative:

* ``checkpoint(task_id)`` is called at phase boundaries inside a worker. It
  raises :class:`TaskStopped` when the user requested a stop, and blocks while
  a pause is requested (returning once resumed or stopped).
* A blocking provider call cannot be interrupted mid-flight; the pause/stop
  therefore takes effect at the next checkpoint. The UI wording reflects this.

Records live in memory only — these are transient jobs, not persisted
artifacts, so a process restart naturally clears them.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict, List, Optional


TERMINAL_STATUSES = {"completed", "failed", "canceled"}


class TaskStopped(Exception):
    """Raised from a checkpoint when a background task should stop."""


class BackgroundTaskManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tasks: Dict[str, Dict[str, Any]] = {}
        # One event per task. `set` means "not paused" (runnable). Pausing
        # clears it; resuming or stopping sets it so a waiting worker wakes up.
        self._run_events: Dict[str, threading.Event] = {}
        self._max_finished = 100

    # ── lifecycle ──────────────────────────────────────────────

    def register(
        self,
        task_id: Optional[str] = None,
        *,
        kind: str,
        title: str,
        project_id: Optional[str] = None,
        series_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        asset_type: Optional[str] = None,
        frame_id: Optional[str] = None,
        message: Optional[str] = None,
    ) -> str:
        with self._lock:
            tid = task_id or str(uuid.uuid4())
            now = time.time()
            self._tasks[tid] = {
                "id": tid,
                "kind": kind,
                "title": title,
                "status": "running",
                "progress": 0,
                "message": message,
                "error": None,
                "project_id": project_id,
                "series_id": series_id,
                "asset_id": asset_id,
                "asset_type": asset_type,
                "frame_id": frame_id,
                "pause_requested": False,
                "stop_requested": False,
                "created_at": now,
                "updated_at": now,
                "finished_at": None,
            }
            event = self._run_events.get(tid)
            if event is None:
                event = threading.Event()
                self._run_events[tid] = event
            event.set()
            self._trim()
            return tid

    def update(self, task_id: str, **fields: Any) -> None:
        with self._lock:
            rec = self._tasks.get(task_id)
            if not rec:
                return
            for key, value in fields.items():
                if key in rec:
                    rec[key] = value
            rec["updated_at"] = time.time()

    def finish(self, task_id: str, status: str, *, error: Optional[str] = None) -> None:
        with self._lock:
            rec = self._tasks.get(task_id)
            if not rec:
                return
            rec["status"] = status
            rec["error"] = error
            rec["finished_at"] = time.time()
            rec["updated_at"] = rec["finished_at"]
            rec["pause_requested"] = False
            if status == "completed":
                rec["progress"] = 100
            event = self._run_events.get(task_id)
            if event is not None:
                event.set()

    # ── control ────────────────────────────────────────────────

    def request_pause(self, task_id: str) -> bool:
        with self._lock:
            rec = self._tasks.get(task_id)
            if not rec or rec["status"] in TERMINAL_STATUSES:
                return False
            rec["pause_requested"] = True
            rec["message"] = "暂停将在当前步骤完成后生效"
            rec["updated_at"] = time.time()
            event = self._run_events.get(task_id)
            if event is not None:
                event.clear()
            return True

    def request_resume(self, task_id: str) -> bool:
        with self._lock:
            rec = self._tasks.get(task_id)
            if not rec or rec["status"] in TERMINAL_STATUSES:
                return False
            rec["pause_requested"] = False
            if rec["status"] == "paused":
                rec["status"] = "running"
            rec["message"] = None
            rec["updated_at"] = time.time()
            event = self._run_events.get(task_id)
            if event is not None:
                event.set()
            return True

    def request_stop(self, task_id: str) -> bool:
        with self._lock:
            rec = self._tasks.get(task_id)
            if not rec or rec["status"] in TERMINAL_STATUSES:
                return False
            rec["stop_requested"] = True
            rec["pause_requested"] = False
            rec["message"] = "正在停止…"
            rec["updated_at"] = time.time()
            event = self._run_events.get(task_id)
            if event is not None:
                event.set()
            return True

    def checkpoint(self, task_id: str) -> None:
        """Block while paused; raise TaskStopped when a stop was requested."""
        while True:
            with self._lock:
                rec = self._tasks.get(task_id)
                if not rec:
                    return
                if rec["stop_requested"]:
                    raise TaskStopped(task_id)
                if not rec["pause_requested"]:
                    if rec["status"] == "paused":
                        rec["status"] = "running"
                    return
                rec["status"] = "paused"
                rec["updated_at"] = time.time()
                event = self._run_events.get(task_id)
            # Wait outside the lock so control calls can set the event.
            if event is not None:
                event.wait(timeout=0.5)

    # ── queries ────────────────────────────────────────────────

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            rec = self._tasks.get(task_id)
            return dict(rec) if rec else None

    def list_tasks(self, active_only: bool = False) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._tasks.values())
            if active_only:
                items = [t for t in items if t["status"] not in TERMINAL_STATUSES]
            items.sort(key=lambda t: t.get("created_at", 0), reverse=True)
            return [dict(t) for t in items]

    def _trim(self) -> None:
        """Keep only the most recent N finished tasks to bound memory."""
        finished = [t for t in self._tasks.values() if t["status"] in TERMINAL_STATUSES]
        if len(finished) <= self._max_finished:
            return
        finished.sort(key=lambda t: t.get("finished_at") or 0)
        for rec in finished[: len(finished) - self._max_finished]:
            self._tasks.pop(rec["id"], None)
            self._run_events.pop(rec["id"], None)
