import asyncio
import logging
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from telegram.config import settings
from telegram.detector import DetectionResult
from telegram.metrics import metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalCacheRecord:
    cache_key: str
    media_type: str
    file_id: str
    nsfw: bool
    confidence: float
    label: str
    sticker_set_name: str = ""
    emoji: str = ""
    first_detected_at: Optional[float] = None
    last_seen_at: Optional[float] = None
    total_hits: int = 0
    expires_at: Optional[float] = None

    @property
    def is_expired(self) -> bool:
        return self.expires_at is not None and self.expires_at <= time.time()

    @property
    def is_sticker(self) -> bool:
        return "sticker" in self.media_type


class LruTtlCache:
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self._items: OrderedDict[str, tuple[float, LocalCacheRecord]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[LocalCacheRecord]:
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if not item:
                return None
            expires_at, record = item
            if expires_at <= now or record.is_expired:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return record

    def set(self, key: str, record: LocalCacheRecord, ttl_seconds: float) -> None:
        expires_at = time.monotonic() + max(1.0, ttl_seconds)
        with self._lock:
            self._items[key] = (expires_at, record)
            self._items.move_to_end(key)
            while len(self._items) > self.max_size:
                self._items.popitem(last=False)

    def size(self) -> int:
        self._purge_expired()
        with self._lock:
            return len(self._items)

    def _purge_expired(self) -> None:
        now = time.monotonic()
        with self._lock:
            expired = [key for key, (expires_at, record) in self._items.items() if expires_at <= now or record.is_expired]
            for key in expired:
                self._items.pop(key, None)


class LocalNsfwCache:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._memory = LruTtlCache(settings.hot_cache_max_size)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        self._writer_queue: Optional[asyncio.Queue[Optional[tuple[str, tuple[Any, ...]]]]] = None
        self._writer_task: Optional[asyncio.Task] = None
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        await asyncio.to_thread(self._open_and_init)
        self._writer_queue = asyncio.Queue(maxsize=settings.local_cache_write_queue_size)
        self._writer_task = asyncio.create_task(self._writer_loop(), name="local-cache-writer")
        self._started = True
        logger.info("Local NSFW cache ready at %s", self.db_path)

    async def stop(self) -> None:
        if self._writer_queue and self._writer_task:
            try:
                await self._writer_queue.put(None)
                await asyncio.wait_for(self._writer_task, timeout=10)
            except Exception:
                logger.exception("Local cache writer did not stop cleanly")
        await asyncio.to_thread(self._close)
        self._started = False

    async def lookup(self, cache_key: str) -> Optional[LocalCacheRecord]:
        if not cache_key:
            return None

        record = self._memory.get(cache_key)
        if record:
            metrics.inc("local_cache_memory_hits")
            if record.nsfw:
                self.touch_async(record)
            return record

        try:
            record = await asyncio.to_thread(self._lookup_db, cache_key)
        except Exception:
            metrics.inc("local_cache_errors")
            logger.exception("Local cache lookup failed for key=%s", cache_key)
            return None

        if not record:
            metrics.inc("local_cache_misses")
            return None

        if record.is_expired:
            self.delete_async(cache_key)
            metrics.inc("local_cache_misses")
            return None

        metrics.inc("local_cache_disk_hits")
        self._remember_hot(record)
        self.touch_async(record)
        return record

    async def store_detection(
        self,
        cache_key: str,
        file_id: str,
        media_type: str,
        result: DetectionResult,
        sticker_set_name: str = "",
        emoji: str = "",
    ) -> None:
        if not cache_key or result.status not in {"nsfw", "safe"}:
            return
        if result.status == "safe" and settings.clean_media_cache_ttl_seconds <= 0:
            return

        now = time.time()
        is_nsfw = result.status == "nsfw"
        expires_at = None if is_nsfw else now + settings.clean_media_cache_ttl_seconds
        record = LocalCacheRecord(
            cache_key=cache_key,
            media_type=media_type,
            file_id=file_id,
            nsfw=is_nsfw,
            confidence=float(result.confidence or 0.0),
            label=result.label,
            sticker_set_name=sticker_set_name or "",
            emoji=emoji or "",
            first_detected_at=now if is_nsfw else None,
            last_seen_at=now,
            total_hits=1,
            expires_at=expires_at,
        )
        self._remember_hot(record)

        if is_nsfw:
            await asyncio.to_thread(self._upsert_record, record)
        else:
            self._enqueue_write(self._upsert_sql(), self._record_params(record))

    def touch_async(self, record: LocalCacheRecord) -> None:
        self._enqueue_write(
            """
            UPDATE nsfw_cache
            SET last_seen_at = ?, total_hits = total_hits + 1
            WHERE cache_key = ?
            """,
            (time.time(), record.cache_key),
        )

    def delete_async(self, cache_key: str) -> None:
        self._enqueue_write("DELETE FROM nsfw_cache WHERE cache_key = ?", (cache_key,))

    async def stats(self) -> Dict[str, int]:
        try:
            db_stats = await asyncio.to_thread(self._stats_db)
        except Exception:
            metrics.inc("local_cache_errors")
            logger.exception("Local cache stats failed")
            db_stats = {
                "nsfw_stickers": 0,
                "nsfw_media": 0,
                "clean_temporary": 0,
                "total_rows": 0,
            }
        db_stats["db_size_bytes"] = self._db_size_bytes()
        db_stats["hot_memory_size"] = self._memory.size()
        db_stats["write_queue_size"] = self._writer_queue.qsize() if self._writer_queue else 0
        return db_stats

    def _open_and_init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            self.db_path,
            timeout=settings.local_cache_sqlite_timeout_seconds,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn = conn
            self._execute("PRAGMA journal_mode=WAL")
            self._execute("PRAGMA synchronous=NORMAL")
            self._execute("PRAGMA temp_store=MEMORY")
            self._execute(f"PRAGMA busy_timeout={int(settings.local_cache_sqlite_timeout_seconds * 1000)}")
            self._execute(
                """
                CREATE TABLE IF NOT EXISTS nsfw_cache (
                    cache_key TEXT PRIMARY KEY,
                    media_type TEXT NOT NULL,
                    file_id TEXT,
                    nsfw INTEGER NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0,
                    label TEXT,
                    sticker_set_name TEXT,
                    emoji TEXT,
                    first_detected_at REAL,
                    last_seen_at REAL NOT NULL,
                    total_hits INTEGER NOT NULL DEFAULT 0,
                    expires_at REAL
                )
                """
            )
            self._execute("CREATE INDEX IF NOT EXISTS idx_nsfw_cache_type_nsfw ON nsfw_cache(media_type, nsfw)")
            self._execute("CREATE INDEX IF NOT EXISTS idx_nsfw_cache_expires ON nsfw_cache(expires_at)")
            self._execute("CREATE INDEX IF NOT EXISTS idx_nsfw_cache_file_id ON nsfw_cache(file_id)")
            self._execute("DELETE FROM nsfw_cache WHERE nsfw = 0 AND expires_at IS NOT NULL AND expires_at <= ?", (time.time(),))

    async def _writer_loop(self) -> None:
        assert self._writer_queue is not None
        while True:
            item = await self._writer_queue.get()
            try:
                if item is None:
                    return
                sql, params = item
                await asyncio.to_thread(self._execute, sql, params)
            except Exception:
                metrics.inc("local_cache_errors")
                logger.exception("Local cache background write failed")
            finally:
                self._writer_queue.task_done()

    def _close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    def _lookup_db(self, cache_key: str) -> Optional[LocalCacheRecord]:
        row = self._fetchone(
            """
            SELECT cache_key, media_type, file_id, nsfw, confidence, label,
                   sticker_set_name, emoji, first_detected_at, last_seen_at,
                   total_hits, expires_at
            FROM nsfw_cache
            WHERE cache_key = ?
            """,
            (cache_key,),
        )
        return self._row_to_record(row) if row else None

    def _stats_db(self) -> Dict[str, int]:
        now = time.time()
        return {
            "nsfw_stickers": self._count("nsfw = 1 AND media_type LIKE '%sticker%'"),
            "nsfw_media": self._count("nsfw = 1 AND media_type NOT LIKE '%sticker%'"),
            "clean_temporary": self._count("nsfw = 0 AND (expires_at IS NULL OR expires_at > ?)", (now,)),
            "total_rows": self._count("1 = 1"),
        }

    def _count(self, where: str, params: tuple[Any, ...] = ()) -> int:
        row = self._fetchone(f"SELECT COUNT(*) AS count FROM nsfw_cache WHERE {where}", params)
        return int(row["count"] if row else 0)

    def _upsert_record(self, record: LocalCacheRecord) -> None:
        self._execute(self._upsert_sql(), self._record_params(record))

    def _upsert_sql(self) -> str:
        return """
            INSERT INTO nsfw_cache (
                cache_key, media_type, file_id, nsfw, confidence, label,
                sticker_set_name, emoji, first_detected_at, last_seen_at,
                total_hits, expires_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                media_type = excluded.media_type,
                file_id = COALESCE(NULLIF(excluded.file_id, ''), nsfw_cache.file_id),
                nsfw = excluded.nsfw,
                confidence = excluded.confidence,
                label = excluded.label,
                sticker_set_name = COALESCE(NULLIF(excluded.sticker_set_name, ''), nsfw_cache.sticker_set_name),
                emoji = COALESCE(NULLIF(excluded.emoji, ''), nsfw_cache.emoji),
                first_detected_at = COALESCE(nsfw_cache.first_detected_at, excluded.first_detected_at),
                last_seen_at = excluded.last_seen_at,
                total_hits = nsfw_cache.total_hits + 1,
                expires_at = excluded.expires_at
        """

    def _record_params(self, record: LocalCacheRecord) -> tuple[Any, ...]:
        return (
            record.cache_key,
            record.media_type,
            record.file_id,
            int(record.nsfw),
            record.confidence,
            record.label,
            record.sticker_set_name,
            record.emoji,
            record.first_detected_at,
            record.last_seen_at or time.time(),
            record.total_hits,
            record.expires_at,
        )

    def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self._lock:
            if not self._conn:
                raise RuntimeError("local cache is not started")
            self._conn.execute(sql, params)

    def _fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> Optional[sqlite3.Row]:
        with self._lock:
            if not self._conn:
                raise RuntimeError("local cache is not started")
            return self._conn.execute(sql, params).fetchone()

    def _enqueue_write(self, sql: str, params: tuple[Any, ...]) -> None:
        if not self._writer_queue:
            return
        try:
            self._writer_queue.put_nowait((sql, params))
        except asyncio.QueueFull:
            metrics.inc("local_cache_write_drops")
            logger.warning("Local cache write queue full; dropped cache update")

    def _remember_hot(self, record: LocalCacheRecord) -> None:
        ttl = settings.hot_cache_ttl_seconds if record.nsfw else settings.clean_media_cache_ttl_seconds
        if ttl > 0:
            self._memory.set(record.cache_key, record, ttl)

    def _db_size_bytes(self) -> int:
        total = 0
        for suffix in ("", "-wal", "-shm"):
            path = Path(f"{self.db_path}{suffix}")
            if path.exists():
                try:
                    total += os.path.getsize(path)
                except OSError:
                    pass
        return total

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> LocalCacheRecord:
        return LocalCacheRecord(
            cache_key=str(row["cache_key"]),
            media_type=str(row["media_type"]),
            file_id=str(row["file_id"] or ""),
            nsfw=bool(row["nsfw"]),
            confidence=float(row["confidence"] or 0.0),
            label=str(row["label"] or ""),
            sticker_set_name=str(row["sticker_set_name"] or ""),
            emoji=str(row["emoji"] or ""),
            first_detected_at=row["first_detected_at"],
            last_seen_at=row["last_seen_at"],
            total_hits=int(row["total_hits"] or 0),
            expires_at=row["expires_at"],
        )


local_nsfw_cache = LocalNsfwCache(settings.local_cache_db)
