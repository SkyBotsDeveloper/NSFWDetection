import asyncio
import contextlib
import hashlib
import logging
import mimetypes
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pyrogram import filters
from pyrogram.errors import FloodWait, RPCError
from pyrogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message

from telegram.bot import client
from telegram.config import settings
from telegram.db import (
    mark_chat_active,
    mark_chat_inactive,
    normalize_chat_type,
    track_chat_from_message,
)
from telegram.detector import DetectionResult, detector
from telegram.local_cache import LocalCacheRecord, local_nsfw_cache
from telegram.metrics import metrics
from telegram.online_fallback import detect_with_online_fallback
from telegram.rate_limit import SlidingWindowRateLimiter

logger = logging.getLogger(__name__)

MODERATED_CHAT_TYPES = {"group", "supergroup", "channel"}
SUPPORTED_IMAGE_MIME_PREFIX = "image/"
SUPPORTED_VIDEO_MIME_PREFIX = "video/"
UNSUPPORTED_STICKER_MIME_TYPES = {"application/x-tgsticker"}
HELP_TEXT = """NSFW moderation bot help

Minimum group setup:
- Add me to your group/supergroup.
- Make me admin.
- Enable only this required admin power: Delete messages.

Recommended:
- Send messages, only if you want me to post a short removal notice.

Not required:
- Ban users.
- Add new admins.
- Change group info.
- Pin messages.
- Manage video chats.

Detected media:
- Photos.
- Static stickers.
- Video stickers.
- Videos.
- GIFs/animations.
- Image/video/GIF documents.

Skipped safely:
- TGS/Lottie animated stickers.
- Unsupported documents.
- Files above configured size limits.
- Failed downloads or failed model processing.

Failed processing is never treated as NSFW."""


@dataclass(frozen=True)
class MediaInfo:
    media: Any
    media_type: str
    detector_kind: str
    file_id: str
    cache_key: str
    file_size: int
    is_document: bool = False
    extension: str = ".bin"
    unsupported_reason: str = ""
    sticker_set_name: str = ""
    emoji: str = ""


@dataclass(frozen=True)
class PendingMessageRef:
    chat_id: int
    message_id: int
    chat_type: str


@dataclass(frozen=True)
class MediaJob:
    media: Any
    media_type: str
    detector_kind: str
    file_id: str
    cache_key: str
    dedupe_key: str
    file_size: int
    extension: str
    ref: PendingMessageRef
    sticker_set_name: str = ""
    emoji: str = ""
    created_at: float = field(default_factory=time.monotonic)


media_queue: asyncio.Queue[MediaJob] = asyncio.Queue(maxsize=settings.queue_max_size)
_worker_tasks: List[asyncio.Task] = []
_active_worker_count = 0
_executor: Optional[ThreadPoolExecutor] = None
_inflight: Dict[str, List[PendingMessageRef]] = {}
_inflight_lock = asyncio.Lock()
_track_semaphore: Optional[asyncio.Semaphore] = None
_last_chat_track: Dict[int, float] = {}
_last_notice_at: Dict[int, float] = {}

_user_limiter = SlidingWindowRateLimiter(settings.per_user_rate_limit, settings.rate_limit_window_seconds)
_chat_limiter = SlidingWindowRateLimiter(settings.per_chat_rate_limit, settings.rate_limit_window_seconds)
_global_limiter = SlidingWindowRateLimiter(settings.global_rate_limit, settings.rate_limit_window_seconds)


def get_runtime_stats() -> Dict[str, int]:
    return metrics.snapshot(queue_size=media_queue.qsize())


async def start_runtime() -> None:
    global _executor, _track_semaphore
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_temp_dirs()
    await local_nsfw_cache.start()
    _track_semaphore = asyncio.Semaphore(settings.db_tracking_concurrency)
    _executor = ThreadPoolExecutor(max_workers=settings.inference_workers, thread_name_prefix="nsfw")
    for index in range(settings.worker_count):
        _worker_tasks.append(asyncio.create_task(_worker_loop(index), name=f"media-worker-{index}"))
    logger.info(
        "Started media runtime: workers=%s inference_workers=%s queue_max=%s",
        settings.worker_count,
        settings.inference_workers,
        settings.queue_max_size,
    )


async def stop_runtime() -> None:
    for task in _worker_tasks:
        task.cancel()
    if _worker_tasks:
        await asyncio.gather(*_worker_tasks, return_exceptions=True)
    _worker_tasks.clear()
    if _executor:
        _executor.shutdown(wait=False, cancel_futures=True)
    await local_nsfw_cache.stop()
    logger.info("Media runtime stopped")


@client.on_message(filters.all, group=-100)
async def track_chat_handler(_, message: Message) -> None:
    _schedule_chat_tracking(message)


@client.on_message(filters.private & filters.text, group=-90)
async def private_text_logger(_, message: Message) -> None:
    text = message.text or message.caption or ""
    if text.startswith("/"):
        logger.info(
            "private command received chat_id=%s user_id=%s text=%s",
            getattr(message.chat, "id", None),
            getattr(getattr(message, "from_user", None), "id", None),
            text.split(maxsplit=1)[0],
        )


@client.on_message(filters.regex(r"^/start(?:@\w+)?(?:\s|$)") & filters.text, group=1)
async def start(_, message: Message) -> None:
    logger.info(
        "start command handler chat_id=%s user_id=%s",
        getattr(message.chat, "id", None),
        getattr(getattr(message, "from_user", None), "id", None),
    )
    _schedule_chat_tracking(message, force=True)
    buttons = [
        [
            InlineKeyboardButton("Support Chat", url="https://t.me/VivaanSupport"),
            InlineKeyboardButton("News Channel", url="https://t.me/VivaanUpdates"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(buttons)
    await _safe_reply(
        message,
        "Hello. Send media here or add me as an admin with delete-message permission in a group, and I will moderate NSFW media.\n\nUse /help to see the minimum permissions I need.",
        reply_markup=reply_markup,
    )


@client.on_message(filters.regex(r"^/help(?:@\w+)?(?:\s|$)") & filters.text, group=1)
async def help_command(_, message: Message) -> None:
    logger.info(
        "help command handler chat_id=%s user_id=%s",
        getattr(message.chat, "id", None),
        getattr(getattr(message, "from_user", None), "id", None),
    )
    _schedule_chat_tracking(message, force=True)
    await _safe_reply(message, HELP_TEXT)


@client.on_message(
    filters.photo | filters.sticker | filters.animation | filters.video | filters.document,
    group=0,
)
async def media_handler(_, message: Message) -> None:
    try:
        await _handle_media_message(message)
    except Exception:
        metrics.inc("errors")
        logger.exception("Unhandled media handler failure")


if hasattr(client, "on_chat_member_updated"):

    @client.on_chat_member_updated()
    async def chat_member_updated_handler(_, update: Any) -> None:
        try:
            await _handle_chat_member_update(update)
        except Exception:
            metrics.inc("errors")
            logger.exception("Failed to handle chat member update")


async def _handle_media_message(message: Message) -> None:
    chat = getattr(message, "chat", None)
    if not chat:
        metrics.inc("skipped_messages")
        return

    media_info = _extract_media_info(message)
    if media_info.unsupported_reason:
        metrics.inc("skipped_messages")
        logger.info(
            "media skipped chat_id=%s message_id=%s reason=%s",
            chat.id,
            message.id,
            media_info.unsupported_reason,
        )
        return

    if _is_too_large(media_info):
        metrics.inc("skipped_messages")
        logger.info(
            "media skipped chat_id=%s message_id=%s media=%s size=%s reason=too_large",
            chat.id,
            message.id,
            media_info.media_type,
            media_info.file_size,
        )
        return

    ref = PendingMessageRef(
        chat_id=chat.id,
        message_id=message.id,
        chat_type=normalize_chat_type(chat.type),
    )

    cached = await local_nsfw_cache.lookup(media_info.cache_key) if media_info.cache_key else None
    if cached:
        if cached.nsfw:
            metrics.inc("processed_messages")
            metrics.inc("nsfw_detected")
            await _moderate_refs([ref], cached.confidence, cached.label or "local_cache")
            logger.info(
                "local cache hit chat_id=%s message_id=%s media=%s confidence=%.4f action=moderated",
                chat.id,
                message.id,
                media_info.media_type,
                cached.confidence,
            )
        else:
            metrics.inc("skipped_messages")
            logger.info(
                "local clean cache hit chat_id=%s message_id=%s media=%s action=skipped",
                chat.id,
                message.id,
                media_info.media_type,
            )
        return

    dedupe_key = media_info.cache_key or f"message:{chat.id}:{message.id}"
    if await _attach_to_inflight(dedupe_key, ref):
        metrics.inc("duplicate_messages")
        return

    if not _passes_rate_limits(message):
        await _remove_inflight(dedupe_key)
        metrics.inc("skipped_messages")
        logger.info(
            "media skipped chat_id=%s message_id=%s media=%s reason=rate_limited",
            chat.id,
            message.id,
            media_info.media_type,
        )
        return

    job = MediaJob(
        media=media_info.media,
        media_type=media_info.media_type,
        detector_kind=media_info.detector_kind,
        file_id=media_info.file_id,
        cache_key=media_info.cache_key,
        dedupe_key=dedupe_key,
        file_size=media_info.file_size,
        extension=media_info.extension,
        ref=ref,
        sticker_set_name=media_info.sticker_set_name,
        emoji=media_info.emoji,
    )

    try:
        media_queue.put_nowait(job)
        metrics.inc("queued_messages")
    except asyncio.QueueFull:
        await _remove_inflight(dedupe_key)
        metrics.inc("skipped_messages")
        logger.warning("media queue full; skipped chat_id=%s message_id=%s", chat.id, message.id)


async def _worker_loop(index: int) -> None:
    global _active_worker_count
    while True:
        job = await media_queue.get()
        _active_worker_count += 1
        metrics.set_active_workers(_active_worker_count)
        try:
            await _process_job(job)
        except asyncio.CancelledError:
            raise
        except Exception:
            metrics.inc("errors")
            logger.exception("Worker %s failed processing job key=%s", index, job.dedupe_key)
            await _remove_inflight(job.dedupe_key)
        finally:
            _active_worker_count -= 1
            metrics.set_active_workers(_active_worker_count)
            media_queue.task_done()


async def _process_job(job: MediaJob) -> None:
    age = time.monotonic() - job.created_at
    if age > settings.queued_job_max_age_seconds:
        metrics.inc("skipped_messages")
        await _remove_inflight(job.dedupe_key)
        logger.info("media skipped key=%s reason=queue_age age=%.2f", job.cache_key, age)
        return

    result = DetectionResult("error", reason="not_processed")
    downloaded_path: Optional[Path] = None
    cache_key = job.cache_key

    temp_dir = Path(tempfile.mkdtemp(prefix="nsfw_", dir=settings.temp_dir))
    try:
        try:
            destination = temp_dir / f"media{job.extension}"
            downloaded = await asyncio.wait_for(
                client.download_media(job.media, file_name=str(destination)),
                timeout=settings.download_timeout_seconds,
            )
            if not downloaded:
                result = DetectionResult("error", reason="download_returned_empty_path")
            else:
                downloaded_path = Path(downloaded)
                if downloaded_path.exists() and downloaded_path.stat().st_size > _size_limit_for_job(job):
                    result = DetectionResult("skipped", reason="downloaded_file_too_large")
                else:
                    if not cache_key:
                        cache_key = await _file_fingerprint(downloaded_path)
                        cached = await local_nsfw_cache.lookup(cache_key)
                        if cached:
                            result = _result_from_cache(cached)
                        else:
                            result = await _detect_with_timeout(downloaded_path, job.detector_kind)
                    else:
                        result = await _detect_with_timeout(downloaded_path, job.detector_kind)
                    fallback_result = await detect_with_online_fallback(
                        downloaded_path,
                        job.detector_kind,
                        result,
                    )
                    if fallback_result is not None:
                        logger.info(
                            "online fallback result key=%s media=%s status=%s confidence=%.4f label=%s",
                            cache_key or job.dedupe_key,
                            job.media_type,
                            fallback_result.status,
                            fallback_result.confidence,
                            fallback_result.label,
                        )
                        result = fallback_result
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    except asyncio.TimeoutError:
        result = DetectionResult("error", reason="timeout")
    except Exception as exc:
        metrics.inc("errors")
        logger.exception("media processing failed key=%s", job.cache_key)
        result = DetectionResult("error", reason=str(exc))
    finally:
        refs = await _pop_inflight(job.dedupe_key)

    action = "none"
    if result.status == "nsfw":
        metrics.inc("processed_messages", max(1, len(refs)))
        metrics.inc("nsfw_detected", max(1, len(refs)))
        await local_nsfw_cache.store_detection(
            cache_key,
            job.file_id,
            job.media_type,
            result,
            sticker_set_name=job.sticker_set_name,
            emoji=job.emoji,
        )
        await _moderate_refs(refs, result.confidence, result.label)
        action = "moderated"
    elif result.status == "safe":
        metrics.inc("processed_messages", max(1, len(refs)))
        await local_nsfw_cache.store_detection(
            cache_key,
            job.file_id,
            media_type=job.media_type,
            result=result,
            sticker_set_name=job.sticker_set_name,
            emoji=job.emoji,
        )
    elif result.status == "skipped":
        metrics.inc("skipped_messages", max(1, len(refs)))
    else:
        metrics.inc("errors")

    logger.info(
        "detection result key=%s media=%s status=%s confidence=%.4f label=%s frames=%s action=%s reason=%s path=%s",
        cache_key or job.dedupe_key,
        job.media_type,
        result.status,
        result.confidence,
        result.label,
        result.frames_checked,
        action,
        result.reason,
        downloaded_path,
    )


async def _detect_with_timeout(path: Path, detector_kind: str) -> DetectionResult:
    if _executor is None:
        return DetectionResult("error", reason="runtime_not_started")
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(_executor, detector.detect_file, path, detector_kind),
        timeout=settings.processing_timeout_seconds,
    )


async def _file_fingerprint(path: Path) -> str:
    return await asyncio.to_thread(_hash_file, path)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _result_from_cache(record: LocalCacheRecord) -> DetectionResult:
    return DetectionResult(
        "nsfw" if record.nsfw else "safe",
        confidence=record.confidence,
        label=record.label or "local_cache",
        reason="local_cache",
        frames_checked=0,
    )


def _extract_media_info(message: Message) -> MediaInfo:
    if message.photo:
        return _build_media_info(message.photo, "photo", "image", ".jpg")

    if message.sticker:
        sticker = message.sticker
        mime_type = getattr(sticker, "mime_type", "") or ""
        if mime_type in UNSUPPORTED_STICKER_MIME_TYPES or getattr(sticker, "is_animated", False):
            return _unsupported(sticker, "sticker", "animated_tgs_sticker")
        if getattr(sticker, "is_video", False) or mime_type == "video/webm":
            return _build_media_info(sticker, "video_sticker", "video", ".webm")
        if mime_type.startswith(SUPPORTED_IMAGE_MIME_PREFIX) or mime_type in {"", "image/webp"}:
            return _build_media_info(sticker, "sticker", "image", ".webp")
        return _unsupported(sticker, "sticker", f"unsupported_sticker_mime:{mime_type}")

    if message.animation:
        return _build_media_info(message.animation, "animation", "video", ".mp4")

    if message.video:
        return _build_media_info(message.video, "video", "video", ".mp4")

    if message.document:
        document = message.document
        mime_type = getattr(document, "mime_type", "") or mimetypes.guess_type(getattr(document, "file_name", "") or "")[0] or ""
        extension = _extension_from_document(document, mime_type)
        if mime_type.startswith(SUPPORTED_IMAGE_MIME_PREFIX) and mime_type != "image/gif":
            return _build_media_info(document, "document_image", "image", extension, is_document=True)
        if mime_type == "image/gif":
            return _build_media_info(document, "document_gif", "gif", extension, is_document=True)
        if mime_type.startswith(SUPPORTED_VIDEO_MIME_PREFIX):
            return _build_media_info(document, "document_video", "video", extension, is_document=True)
        return _unsupported(document, "document", f"unsupported_document_mime:{mime_type or 'unknown'}")

    return _unsupported(None, "unknown", "no_supported_media")


def _build_media_info(
    media: Any,
    media_type: str,
    detector_kind: str,
    extension: str,
    is_document: bool = False,
) -> MediaInfo:
    file_id = getattr(media, "file_id", "") or ""
    unique_id = getattr(media, "file_unique_id", "") or file_id
    return MediaInfo(
        media=media,
        media_type=media_type,
        detector_kind=detector_kind,
        file_id=file_id,
        cache_key=unique_id,
        file_size=int(getattr(media, "file_size", 0) or 0),
        is_document=is_document,
        extension=extension or ".bin",
        sticker_set_name=getattr(media, "set_name", "") or "",
        emoji=getattr(media, "emoji", "") or "",
    )


def _unsupported(media: Any, media_type: str, reason: str) -> MediaInfo:
    return MediaInfo(
        media=media,
        media_type=media_type,
        detector_kind="unsupported",
        file_id=getattr(media, "file_id", "") if media else "",
        cache_key=getattr(media, "file_unique_id", "") if media else "",
        file_size=int(getattr(media, "file_size", 0) or 0) if media else 0,
        unsupported_reason=reason,
    )


def _extension_from_document(document: Any, mime_type: str) -> str:
    file_name = getattr(document, "file_name", "") or ""
    suffix = Path(file_name).suffix.lower()
    if suffix:
        return suffix
    guessed = mimetypes.guess_extension(mime_type or "")
    return guessed or ".bin"


def _is_too_large(media_info: MediaInfo) -> bool:
    if media_info.file_size <= 0:
        return False
    limit = settings.max_image_size_bytes if media_info.detector_kind == "image" else settings.max_video_size_bytes
    if media_info.is_document:
        limit = min(limit, settings.max_document_size_bytes)
    return media_info.file_size > limit


def _size_limit_for_job(job: MediaJob) -> int:
    limit = settings.max_image_size_bytes if job.detector_kind == "image" else settings.max_video_size_bytes
    if job.media_type.startswith("document_"):
        limit = min(limit, settings.max_document_size_bytes)
    return limit


async def _attach_to_inflight(cache_key: str, ref: PendingMessageRef) -> bool:
    async with _inflight_lock:
        refs = _inflight.get(cache_key)
        if refs is None:
            _inflight[cache_key] = [ref]
            return False
        if len(refs) < settings.duplicate_pending_limit:
            refs.append(ref)
        return True


async def _remove_inflight(cache_key: str) -> None:
    async with _inflight_lock:
        _inflight.pop(cache_key, None)


async def _pop_inflight(cache_key: str) -> List[PendingMessageRef]:
    async with _inflight_lock:
        return _inflight.pop(cache_key, [])


def _passes_rate_limits(message: Message) -> bool:
    chat = getattr(message, "chat", None)
    user = getattr(message, "from_user", None)
    chat_id = getattr(chat, "id", None)
    user_id = getattr(user, "id", None)

    if not _global_limiter.allow("global"):
        metrics.inc("global_rate_limited")
        return False
    if chat_id is not None and not _chat_limiter.allow(chat_id):
        metrics.inc("chat_rate_limited")
        return False
    if user_id is not None and not _user_limiter.allow(user_id):
        metrics.inc("user_rate_limited")
        return False
    return True


async def _moderate_refs(refs: List[PendingMessageRef], confidence: float, label: str) -> None:
    if not refs:
        return

    grouped: Dict[int, List[PendingMessageRef]] = {}
    for ref in refs:
        grouped.setdefault(ref.chat_id, []).append(ref)

    for chat_id, chat_refs in grouped.items():
        chat_type = chat_refs[0].chat_type
        message_ids = [ref.message_id for ref in chat_refs]
        if chat_type in MODERATED_CHAT_TYPES:
            await _safe_delete_messages(chat_id, message_ids)
            await _safe_group_notice(chat_id, confidence, label)
        else:
            for message_id in message_ids:
                await _safe_send_private_warning(chat_id, message_id, confidence)


async def _safe_delete_messages(chat_id: int, message_ids: List[int]) -> None:
    try:
        await client.delete_messages(chat_id, message_ids)
    except FloodWait as exc:
        await _short_flood_sleep(exc)
        with contextlib.suppress(Exception):
            await client.delete_messages(chat_id, message_ids)
    except RPCError as exc:
        metrics.inc("delete_errors")
        logger.warning("Failed to delete messages chat_id=%s ids=%s error=%s", chat_id, message_ids, exc)
    except Exception as exc:
        metrics.inc("delete_errors")
        logger.warning("Unexpected delete failure chat_id=%s ids=%s error=%s", chat_id, message_ids, exc)


async def _safe_group_notice(chat_id: int, confidence: float, label: str) -> None:
    now = time.monotonic()
    last_notice = _last_notice_at.get(chat_id, 0.0)
    if now - last_notice < settings.moderation_notice_cooldown_seconds:
        return
    _last_notice_at[chat_id] = now
    text = f"NSFW media removed. Confidence: {confidence:.2%}."
    try:
        await client.send_message(chat_id, text, disable_notification=True)
    except FloodWait as exc:
        await _short_flood_sleep(exc)
    except RPCError as exc:
        metrics.inc("notice_errors")
        logger.warning("Failed to send moderation notice chat_id=%s label=%s error=%s", chat_id, label, exc)
    except Exception as exc:
        metrics.inc("notice_errors")
        logger.warning("Unexpected notice failure chat_id=%s label=%s error=%s", chat_id, label, exc)


async def _safe_send_private_warning(chat_id: int, message_id: int, confidence: float) -> None:
    try:
        await client.send_message(
            chat_id,
            f"NSFW media detected. Confidence: {confidence:.2%}.",
            reply_to_message_id=message_id,
        )
    except FloodWait as exc:
        await _short_flood_sleep(exc)
    except RPCError as exc:
        metrics.inc("notice_errors")
        logger.warning("Failed to send private warning chat_id=%s error=%s", chat_id, exc)
    except Exception as exc:
        metrics.inc("notice_errors")
        logger.warning("Unexpected private warning failure chat_id=%s error=%s", chat_id, exc)


async def _short_flood_sleep(exc: FloodWait) -> None:
    wait_seconds = int(getattr(exc, "value", 0) or 0)
    if 0 < wait_seconds <= 5:
        await asyncio.sleep(wait_seconds)


async def _safe_reply(message: Message, text: str, **kwargs: Any) -> None:
    try:
        await message.reply_text(text, **kwargs)
        logger.info(
            "reply sent chat_id=%s message_id=%s",
            getattr(message.chat, "id", None),
            getattr(message, "id", None),
        )
    except RPCError as exc:
        logger.warning("Failed to reply chat_id=%s error=%s", getattr(message.chat, "id", None), exc)
    except Exception as exc:
        logger.warning("Unexpected reply failure error=%s", exc)


def _schedule_chat_tracking(message: Message, force: bool = False) -> None:
    chat = getattr(message, "chat", None)
    if not chat:
        return

    now = time.monotonic()
    chat_id = getattr(chat, "id", None)
    if chat_id is None:
        return

    if not force and now - _last_chat_track.get(chat_id, 0.0) < settings.chat_track_interval_seconds:
        return
    _last_chat_track[chat_id] = now

    async def runner() -> None:
        if _track_semaphore is None:
            return
        async with _track_semaphore:
            await track_chat_from_message(message)

    task = asyncio.create_task(runner())
    task.add_done_callback(_log_background_task_failure)


def _log_background_task_failure(task: asyncio.Task) -> None:
    with contextlib.suppress(asyncio.CancelledError):
        exc = task.exception()
        if exc:
            metrics.inc("errors")
            logger.error("Background task failed", exc_info=(type(exc), exc, exc.__traceback__))


async def _handle_chat_member_update(update: Any) -> None:
    chat = getattr(update, "chat", None)
    if not chat:
        return

    new_member = getattr(update, "new_chat_member", None)
    old_member = getattr(update, "old_chat_member", None)
    member = new_member or old_member
    user = getattr(member, "user", None)

    is_self = bool(getattr(user, "is_self", False))
    client_me = getattr(client, "me", None)
    if not is_self and getattr(user, "id", None) != getattr(client_me, "id", None):
        return

    status = str(getattr(new_member, "status", "")).lower()
    chat_type = normalize_chat_type(getattr(chat, "type", "unknown"))
    if any(value in status for value in ("left", "kicked", "banned")):
        await mark_chat_inactive(getattr(chat, "id"), reason=f"chat_member_status:{status}")
    else:
        await mark_chat_active(
            getattr(chat, "id"),
            chat_type,
            title=getattr(chat, "title", None),
            username=getattr(chat, "username", None),
        )


def _cleanup_stale_temp_dirs() -> None:
    temp_root = settings.temp_dir.resolve()
    try:
        for path in temp_root.glob("nsfw_*"):
            resolved = path.resolve()
            if not resolved.is_dir() or resolved.parent != temp_root:
                continue
            shutil.rmtree(resolved, ignore_errors=True)
    except Exception as exc:
        logger.warning("Failed to clean stale temp media files: %s", exc)
