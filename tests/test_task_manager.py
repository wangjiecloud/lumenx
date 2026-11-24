"""Tests for the in-process background task registry."""

import threading
import time

import pytest

from src.apps.comic_gen.task_manager import BackgroundTaskManager, TaskStopped


def test_register_and_list():
    tm = BackgroundTaskManager()
    tid = tm.register(kind="asset_generation", title="T", project_id="p1")
    tasks = tm.list_tasks()
    assert len(tasks) == 1
    assert tasks[0]["id"] == tid
    assert tasks[0]["status"] == "running"


def test_finish_marks_terminal():
    tm = BackgroundTaskManager()
    tid = tm.register(kind="video", title="T")
    tm.finish(tid, "completed")
    rec = tm.get(tid)
    assert rec["status"] == "completed"
    assert rec["progress"] == 100
    assert tm.list_tasks(active_only=True) == []


def test_stop_raises_at_checkpoint():
    tm = BackgroundTaskManager()
    tid = tm.register(kind="video", title="T")
    tm.request_stop(tid)
    with pytest.raises(TaskStopped):
        tm.checkpoint(tid)


def test_pause_blocks_until_resume():
    tm = BackgroundTaskManager()
    tid = tm.register(kind="video", title="T")
    tm.request_pause(tid)
    done = threading.Event()

    def worker():
        tm.checkpoint(tid)
        done.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    time.sleep(0.2)
    assert not done.is_set()
    assert tm.get(tid)["status"] == "paused"

    tm.request_resume(tid)
    thread.join(timeout=2)
    assert done.is_set()
    assert tm.get(tid)["status"] == "running"


def test_resume_unblocks_waiting_paused_worker():
    tm = BackgroundTaskManager()
    tid = tm.register(kind="video", title="T")
    tm.request_pause(tid)
    result = {}

    def worker():
        try:
            tm.checkpoint(tid)
            result["ok"] = True
        except TaskStopped:
            result["stopped"] = True

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    time.sleep(0.1)
    tm.request_stop(tid)
    thread.join(timeout=2)
    assert result.get("stopped") is True
