import time

import pytest

from simple_local import dblog
from simple_local.config import MySQLLog


class FakeCursor:
    def __init__(self, executed):
        self.executed = executed

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))


class FakeConn:
    def __init__(self, executed):
        self.executed = executed
        self.closed = False

    def cursor(self):
        return FakeCursor(self.executed)

    def close(self):
        self.closed = True


class FakePyMySQL:
    def __init__(self, fail_connects=0):
        self.executed = []
        self.connects = 0
        self.fail_connects = fail_connects

    def connect(self, **kwargs):
        self.connects += 1
        if self.connects <= self.fail_connects:
            raise ConnectionError("db down")
        return FakeConn(self.executed)


def wait_for(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


RECORD = {
    "endpoint": "chat",
    "model": "base",
    "status": 200,
    "duration_ms": 12,
    "prompt_tokens": 3,
    "completion_tokens": 5,
    "error": None,
    "request": "{}",
    "response": "{}",
}


def test_creates_table_and_inserts(monkeypatch):
    fake = FakePyMySQL()
    monkeypatch.setattr(dblog, "pymysql", fake)
    sink = dblog.MySQLLogSink(MySQLLog(table="request_log"))
    sink.log(RECORD)
    assert wait_for(lambda: len(fake.executed) == 2)
    ddl, insert = fake.executed
    assert "CREATE TABLE IF NOT EXISTS request_log" in ddl[0]
    assert insert[1] == ("chat", "base", 200, 12, 3, 5, None, "{}", "{}")
    sink.stop()


def test_db_down_never_blocks_and_recovers(monkeypatch):
    fake = FakePyMySQL(fail_connects=2)
    monkeypatch.setattr(dblog, "pymysql", fake)
    monkeypatch.setattr(dblog, "RETRY_BACKOFF_MAX", 0.05)
    sink = dblog.MySQLLogSink(MySQLLog())
    sink.log(RECORD)  # returns immediately even though connects fail
    assert wait_for(lambda: any("INSERT" in sql for sql, _ in fake.executed))
    assert fake.connects == 3  # two failures, then recovery
    sink.stop()


def test_stop_while_db_down_drops_and_exits(monkeypatch):
    fake = FakePyMySQL(fail_connects=10**6)
    monkeypatch.setattr(dblog, "pymysql", fake)
    sink = dblog.MySQLLogSink(MySQLLog())
    sink.log(RECORD)
    assert wait_for(lambda: fake.connects >= 1)
    sink.stop()
    assert not sink._thread.is_alive()
    assert sink.dropped >= 1


def test_clip():
    assert dblog.clip(None) is None
    assert dblog.clip("short") == "short"
    clipped = dblog.clip("x" * (dblog.MAX_BODY_CHARS + 10))
    assert len(clipped) < dblog.MAX_BODY_CHARS + 20
    assert clipped.endswith("…[truncated]")
