import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def _int(name: str, default: int, minimum: Optional[int] = None) -> int:
    raw = _str(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logging.warning("Invalid integer for %s=%r; using %s", name, raw, default)
        return default
    if minimum is not None and value < minimum:
        logging.warning("%s=%s is below %s; using %s", name, value, minimum, default)
        return default
    return value


def _float(name: str, default: float, minimum: Optional[float] = None) -> float:
    raw = _str(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logging.warning("Invalid float for %s=%r; using %s", name, raw, default)
        return default
    if minimum is not None and value < minimum:
        logging.warning("%s=%s is below %s; using %s", name, value, minimum, default)
        return default
    return value


def _bool(name: str, default: bool = False) -> bool:
    raw = _str(name)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on", "enable", "enabled"}


def _default_cache_db() -> Path:
    configured = _str("LOCAL_CACHE_DB")
    if configured:
        return Path(configured).expanduser().resolve()

    if os.name == "nt":
        return Path("./data/nsfw_cache.sqlite").resolve()
    return Path("/tmp/nsfw_cache.sqlite").resolve()


def _default_temp_dir() -> Path:
    configured = _str("TEMP_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(tempfile.gettempdir()).resolve()


@dataclass(frozen=True)
class Settings:
    api_id: int
    api_hash: str
    bot_token: str
    owner_id: int
    mongo_uri: str
    database_name: str
    mongo_timeout_ms: int
    local_cache_db: Path
    hot_cache_max_size: int
    hot_cache_ttl_seconds: float
    clean_media_cache_ttl_seconds: float
    local_cache_write_queue_size: int
    local_cache_sqlite_timeout_seconds: float

    model_name: str
    nsfw_threshold: float
    torch_device: str
    torch_num_threads: int
    online_fallback_enabled: bool
    online_fallback_provider: str
    online_fallback_url: str
    online_fallback_timeout_seconds: float
    online_fallback_max_size_mb: int
    online_fallback_nsfw_threshold: float
    online_fallback_min_local_nsfw_score: float
    online_fallback_fast_mode: bool

    queue_max_size: int
    worker_count: int
    inference_workers: int
    db_tracking_concurrency: int
    chat_track_interval_seconds: float

    max_image_size_mb: int
    max_video_size_mb: int
    max_document_size_mb: int
    download_timeout_seconds: float
    processing_timeout_seconds: float
    queued_job_max_age_seconds: float
    max_video_frames: int
    video_frame_interval_seconds: float
    duplicate_pending_limit: int

    rate_limit_window_seconds: float
    per_user_rate_limit: int
    per_chat_rate_limit: int
    global_rate_limit: int
    moderation_notice_cooldown_seconds: float

    broadcast_delay_seconds: float
    broadcast_max_flood_wait_seconds: int

    session_name: str
    pyrogram_workers: int
    pyrogram_sleep_threshold_seconds: int
    temp_dir: Path
    log_level: str

    @property
    def max_image_size_bytes(self) -> int:
        return self.max_image_size_mb * 1024 * 1024

    @property
    def max_video_size_bytes(self) -> int:
        return self.max_video_size_mb * 1024 * 1024

    @property
    def max_document_size_bytes(self) -> int:
        return self.max_document_size_mb * 1024 * 1024

    @property
    def online_fallback_max_size_bytes(self) -> int:
        return self.online_fallback_max_size_mb * 1024 * 1024


settings = Settings(
    api_id=_int("API_ID", 0, 0),
    api_hash=_str("API_HASH"),
    bot_token=_str("BOT_TOKEN"),
    owner_id=_int("OWNER_ID", 0, 0),
    mongo_uri=_str("MONGO_URI", "mongodb://localhost:27017"),
    database_name=_str("DATABASE_NAME", "nsfw"),
    mongo_timeout_ms=_int("MONGO_TIMEOUT_MS", 5000, 500),
    local_cache_db=_default_cache_db(),
    hot_cache_max_size=_int("HOT_CACHE_MAX_SIZE", 10000, 1),
    hot_cache_ttl_seconds=_float("HOT_CACHE_TTL_SECONDS", 86400.0, 1.0),
    clean_media_cache_ttl_seconds=_float("CLEAN_MEDIA_CACHE_TTL_SECONDS", 3600.0, 0.0),
    local_cache_write_queue_size=_int("LOCAL_CACHE_WRITE_QUEUE_SIZE", 10000, 1),
    local_cache_sqlite_timeout_seconds=_float("LOCAL_CACHE_SQLITE_TIMEOUT_SECONDS", 5.0, 0.1),
    model_name=_str("NSFW_MODEL_NAME", "Falconsai/nsfw_image_detection"),
    nsfw_threshold=_float("NSFW_THRESHOLD", 0.85, 0.0),
    torch_device=_str("TORCH_DEVICE", "auto").lower(),
    torch_num_threads=_int("TORCH_NUM_THREADS", 0, 0),
    online_fallback_enabled=_bool("ONLINE_FALLBACK_ENABLED", False),
    online_fallback_provider=_str("ONLINE_FALLBACK_PROVIDER", "naas").lower(),
    online_fallback_url=_str("ONLINE_FALLBACK_URL", "https://nsfw-categorize.it/api/upload"),
    online_fallback_timeout_seconds=_float("ONLINE_FALLBACK_TIMEOUT_SECONDS", 20.0, 1.0),
    online_fallback_max_size_mb=_int("ONLINE_FALLBACK_MAX_SIZE_MB", 15, 1),
    online_fallback_nsfw_threshold=_float("ONLINE_FALLBACK_NSFW_THRESHOLD", 0.85, 0.0),
    online_fallback_min_local_nsfw_score=_float("ONLINE_FALLBACK_MIN_LOCAL_NSFW_SCORE", 0.35, 0.0),
    online_fallback_fast_mode=_bool("ONLINE_FALLBACK_FAST_MODE", True),
    queue_max_size=_int("QUEUE_MAX_SIZE", 500, 1),
    worker_count=_int("WORKER_COUNT", 4, 1),
    inference_workers=_int("INFERENCE_WORKERS", 1, 1),
    db_tracking_concurrency=_int("DB_TRACKING_CONCURRENCY", 4, 1),
    chat_track_interval_seconds=_float("CHAT_TRACK_INTERVAL_SECONDS", 60.0, 0.0),
    max_image_size_mb=_int("MAX_IMAGE_SIZE_MB", 12, 1),
    max_video_size_mb=_int("MAX_VIDEO_SIZE_MB", 80, 1),
    max_document_size_mb=_int("MAX_DOCUMENT_SIZE_MB", 25, 1),
    download_timeout_seconds=_float("DOWNLOAD_TIMEOUT_SECONDS", 45.0, 1.0),
    processing_timeout_seconds=_float("PROCESSING_TIMEOUT_SECONDS", 60.0, 1.0),
    queued_job_max_age_seconds=_float("QUEUED_JOB_MAX_AGE_SECONDS", 180.0, 1.0),
    max_video_frames=_int("MAX_VIDEO_FRAMES", 8, 1),
    video_frame_interval_seconds=_float("VIDEO_FRAME_INTERVAL_SECONDS", 4.0, 0.1),
    duplicate_pending_limit=_int("DUPLICATE_PENDING_LIMIT", 75, 1),
    rate_limit_window_seconds=_float("RATE_LIMIT_WINDOW_SECONDS", 30.0, 1.0),
    per_user_rate_limit=_int("PER_USER_RATE_LIMIT", 12, 0),
    per_chat_rate_limit=_int("PER_CHAT_RATE_LIMIT", 60, 0),
    global_rate_limit=_int("GLOBAL_RATE_LIMIT", 300, 0),
    moderation_notice_cooldown_seconds=_float("MODERATION_NOTICE_COOLDOWN_SECONDS", 10.0, 0.0),
    broadcast_delay_seconds=_float("BROADCAST_DELAY_SECONDS", 0.25, 0.0),
    broadcast_max_flood_wait_seconds=_int("BROADCAST_MAX_FLOOD_WAIT_SECONDS", 60, 0),
    session_name=_str("SESSION_NAME", "antinsfw"),
    pyrogram_workers=_int("PYROGRAM_WORKERS", 32, 1),
    pyrogram_sleep_threshold_seconds=_int("PYROGRAM_SLEEP_THRESHOLD_SECONDS", 30, 0),
    temp_dir=_default_temp_dir(),
    log_level=_str("LOG_LEVEL", "INFO").upper(),
)


def configure_logging() -> None:
    level = getattr(logging, settings.log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def require_bot_settings() -> None:
    missing = []
    if not settings.bot_token:
        missing.append("BOT_TOKEN")
    if not settings.api_id:
        missing.append("API_ID")
    if not settings.api_hash:
        missing.append("API_HASH")
    if not settings.owner_id:
        missing.append("OWNER_ID")
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {joined}")
