"""Persistent exact-count cache. Failure is never a corpus count of zero."""
from concurrent.futures import Future
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid


class QueryFailure(RuntimeError):
    pass


class SQLiteCache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS counts (key TEXT PRIMARY KEY, payload TEXT NOT NULL, created REAL NOT NULL)")
        self.db.commit()

    def get(self, key):
        with self.lock:
            row = self.db.execute("SELECT payload FROM counts WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, payload):
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO counts VALUES (?,?,?)",
                            (key, json.dumps(payload, sort_keys=True), time.time()))

    def snapshot(self, destination):
        """SQLite online backup includes WAL; never copy a live database file."""
        destination = Path(destination)
        if destination.resolve() == self.path.resolve():
            raise ValueError("Backup must differ from live database")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".part")
        try:
            with self.lock:
                target = sqlite3.connect(temporary)
                try:
                    self.db.backup(target)
                finally:
                    target.close()
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def restore_if_empty(self, source):
        """Run before workers start. Refuse to overwrite a populated local cache."""
        source = Path(source)
        if not source.exists():
            return False
        if source.resolve() == self.path.resolve():
            raise ValueError("Backup and local cache must differ")
        with self.lock:
            if self.db.execute("SELECT COUNT(*) FROM counts").fetchone()[0]:
                return False
            incoming = sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True)
            try:
                if incoming.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError("Invalid cache snapshot")
                incoming.backup(self.db)
            finally:
                incoming.close()
        return True

    def close(self):
        with self.lock:
            self.db.close()


class HTTPTransport:
    def __init__(self, timeout=10.0, attempts=3):
        self.timeout, self.attempts = timeout, attempts
        self.local = threading.local()
        self.sessions = []
        self.lock = threading.Lock()
        self.access_denial = None

    def __call__(self, payload):
        import requests
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
            with self.lock:
                self.sessions.append(self.local.session)
        last_error = None
        for attempt in range(self.attempts):
            # A denial applies to this transport's remaining work too. Requests
            # already in flight may finish; queued work must not keep retrying.
            with self.lock:
                if self.access_denial is not None:
                    raise QueryFailure(self.access_denial)
            retry_after = 0.0
            try:
                response = self.local.session.post("https://api.infini-gram.io/",
                                                   json=payload, timeout=self.timeout)
                if response.status_code in (401, 403):
                    headers = {key: response.headers[key][:80] for key in
                               ("Content-Type", "Retry-After", "Server", "CF-Ray", "X-Request-ID")
                               if key in response.headers}
                    body = " ".join(response.text[:1000].split())[:200]
                    message = (f"Infini-gram HTTP {response.status_code}: access denied; "
                               f"further HTTP requests stopped for this process. "
                               f"headers={json.dumps(headers)} body={body!r}")
                    with self.lock:
                        self.access_denial = message
                    raise QueryFailure(message)
                if response.status_code == 429:
                    try:
                        retry_after = min(float(response.headers.get("Retry-After", 1)), 10)
                    except ValueError:
                        retry_after = 1
                response.raise_for_status()
                result = response.json()
                if "error" in result:
                    raise QueryFailure(str(result["error"]))
                return result
            except (requests.RequestException, ValueError, QueryFailure) as exc:
                with self.lock:
                    if self.access_denial is not None:
                        raise QueryFailure(self.access_denial) from exc
                last_error = exc
                if attempt + 1 < self.attempts:
                    time.sleep(max(retry_after, min(0.25 * 2 ** attempt, 4)))
        raise QueryFailure(f"Infini-gram failed after {self.attempts} attempts: {last_error}")

    def close(self):
        for session in self.sessions:
            session.close()


class InfiniGramClient:
    def __init__(self, cache, index, tokenizer_fingerprint, transport=None):
        self.cache, self.index = cache, index
        self.tokenizer_fingerprint = tokenizer_fingerprint
        self.transport = transport or HTTPTransport()
        self.lock = threading.Lock()
        self.inflight = {}
        self.stats = dict(hits=0, misses=0, shared=0, failures=0, request_seconds=0.0)

    def count(self, token_ids):
        payload = {"index": self.index, "query_type": "count", "query_ids": list(token_ids)}
        identity = {"schema": "exact-count-v2", "tokenizer": self.tokenizer_fingerprint, **payload}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        with self.lock:
            cached = self.cache.get(key)
            if cached is not None:
                self.stats["hits"] += 1
                return cached["count"]
            if key in self.inflight:
                future, owner = self.inflight[key], False
                self.stats["shared"] += 1
            else:
                future, owner = Future(), True
                self.inflight[key] = future
                self.stats["misses"] += 1
        if not owner:
            return future.result()
        try:
            started = time.perf_counter()
            try:
                result = self.transport(payload)
            finally:
                with self.lock:
                    self.stats["request_seconds"] += time.perf_counter() - started
            if result.get("error"):
                raise QueryFailure(str(result["error"]))
            count = result.get("count")
            if type(count) is not int or count < 0 or result.get("approx", True) is not False:
                raise QueryFailure(f"Exact count required: count={count!r}, approx={result.get('approx')!r}; "
                                   f"response keys={sorted(result)}")
            self.cache.put(key, {"count": count, "approx": False})
            future.set_result(count)
            return count
        except Exception as exc:
            with self.lock:
                self.stats["failures"] += 1
            error = exc if isinstance(exc, QueryFailure) else QueryFailure(str(exc))
            future.set_exception(error)
            raise error
        finally:
            with self.lock:
                self.inflight.pop(key, None)

    def close(self):
        if hasattr(self.transport, "close"):
            self.transport.close()
        self.cache.close()
