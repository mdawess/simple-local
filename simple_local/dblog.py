import logging
import queue
import threading

import pymysql

from .config import MySQLLog

log = logging.getLogger("simple_local.dblog")

MAX_BODY_CHARS = 60_000
RETRY_BACKOFF_MAX = 30.0

DDL = """CREATE TABLE IF NOT EXISTS {table} (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  endpoint VARCHAR(64) NOT NULL,
  model VARCHAR(255) NULL,
  status SMALLINT NULL,
  duration_ms INT NULL,
  prompt_tokens INT NULL,
  completion_tokens INT NULL,
  error TEXT NULL,
  request MEDIUMTEXT NULL,
  response MEDIUMTEXT NULL,
  KEY idx_model_created (model, created_at),
  KEY idx_created (created_at)
)"""

INSERT = (
    "INSERT INTO {table} (endpoint, model, status, duration_ms, prompt_tokens, "
    "completion_tokens, error, request, response) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
)

FIELDS = (
    "endpoint",
    "model",
    "status",
    "duration_ms",
    "prompt_tokens",
    "completion_tokens",
    "error",
    "request",
    "response",
)


def clip(text: str | None) -> str | None:
    if text is not None and len(text) > MAX_BODY_CHARS:
        return text[:MAX_BODY_CHARS] + "…[truncated]"
    return text


class MySQLLogSink:
    """Fire-and-forget request logging: `log()` never blocks and never raises.
    A worker thread inserts queued records, reconnecting with backoff; when the
    database is unreachable the queue absorbs bursts and overflow is dropped
    (counted) so serving is never affected."""

    def __init__(self, cfg: MySQLLog):
        self.cfg = cfg
        self.queue: queue.Queue = queue.Queue(maxsize=2000)
        self.dropped = 0
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name="mysql-log", daemon=True)
        self._thread.start()
        log.info(
            "logging requests to mysql://%s:%s/%s.%s",
            cfg.host, cfg.port, cfg.database, cfg.table,
        )

    def log(self, record: dict) -> None:
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1
            if self.dropped % 100 == 1:
                log.warning("log queue full; dropped %d records so far", self.dropped)

    def stop(self) -> None:
        self._stopped.set()
        self._thread.join(timeout=5)

    def _connect(self):
        options = {}
        if self.cfg.ssl_ca:
            options["ssl_ca"] = self.cfg.ssl_ca
        elif self.cfg.ssl:
            # Encrypt without pinning a CA — pymysql treats a dict as "use TLS".
            options["ssl"] = {"check_hostname": False}
        conn = pymysql.connect(
            host=self.cfg.host,
            port=self.cfg.port,
            user=self.cfg.user,
            password=self.cfg.password,
            database=self.cfg.database,
            autocommit=True,
            connect_timeout=5,
            **options,
        )
        with conn.cursor() as cur:
            cur.execute(DDL.format(table=self.cfg.table))
        return conn

    def _run(self) -> None:
        conn = None
        backoff = 1.0
        insert = INSERT.format(table=self.cfg.table)
        while True:
            try:
                record = self.queue.get(timeout=0.5)
            except queue.Empty:
                if self._stopped.is_set():
                    break
                continue
            while True:
                try:
                    if conn is None:
                        conn = self._connect()
                        backoff = 1.0
                    with conn.cursor() as cur:
                        cur.execute(insert, tuple(record.get(f) for f in FIELDS))
                    break
                except Exception as e:
                    if conn is not None:
                        try:
                            conn.close()
                        except Exception:
                            pass
                        conn = None
                    log.warning("mysql insert failed (%s); retrying in %.0fs", e, backoff)
                    if self._stopped.wait(backoff):
                        self.dropped += 1 + self.queue.qsize()
                        return
                    backoff = min(backoff * 2, RETRY_BACKOFF_MAX)
        if conn is not None:
            conn.close()
