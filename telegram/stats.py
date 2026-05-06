import asyncio
import logging
from typing import Any, Dict, Optional

from pyrogram import filters
from pyrogram.errors import FloodWait, RPCError
from pyrogram.types import Message

from telegram.antinsfw import get_runtime_stats
from telegram.bot import client
from telegram.config import settings
from telegram.db import get_active_broadcast_targets, get_chat_counts, mark_chat_inactive
from telegram.local_cache import local_nsfw_cache
from telegram.metrics import metrics

logger = logging.getLogger(__name__)
_broadcast_lock = asyncio.Lock()


@client.on_message(filters.command("stats"), group=1)
async def stats(_, message: Message) -> None:
    if not _is_owner(message):
        return

    counts = await get_chat_counts()
    runtime = get_runtime_stats()
    await _safe_reply_text(message, _format_stats(counts, runtime))


@client.on_message(filters.command("broadcast"), group=1)
async def broadcast(app, message: Message) -> None:
    if not _is_owner(message):
        return

    text = _command_payload(message)
    replied = message.reply_to_message
    if not text and not replied:
        await _safe_reply_text(message, "Usage: /broadcast <message> or reply to a message with /broadcast")
        return

    if _broadcast_lock.locked():
        await _safe_reply_text(message, "A broadcast is already running.")
        return

    async with _broadcast_lock:
        counts = await get_chat_counts()
        targets = await get_active_broadcast_targets()
        status_message = await _safe_reply_text(message, f"Broadcast started. Targets: {len(targets)}")
        report = await _run_broadcast(app, targets, text, replied)
        skipped_inactive = max(0, counts.get("known_chats", 0) - len(targets))
        report["skipped_inactive"] = skipped_inactive
        final_text = (
            "**Broadcast report**\n\n"
            f"Total targets: {report['total']}\n"
            f"Sent: {report['sent']}\n"
            f"Failed: {report['failed']}\n"
            f"Skipped/inactive: {report['skipped_inactive']}"
        )
        if status_message:
            try:
                await status_message.edit_text(final_text)
                return
            except Exception:
                pass
        await _safe_reply_text(message, final_text)


@client.on_message(filters.command("cache_stats"), group=1)
async def cache_stats(_, message: Message) -> None:
    if not _is_owner(message):
        return

    stats = await local_nsfw_cache.stats()
    await _safe_reply_text(message, _format_cache_stats(stats))


def _is_owner(message: Message) -> bool:
    user = getattr(message, "from_user", None)
    return bool(settings.owner_id and user and getattr(user, "id", None) == settings.owner_id)


def _command_payload(message: Message) -> str:
    text = message.text or message.caption or ""
    parts = text.split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


async def _safe_reply_text(message: Message, text: str) -> Optional[Message]:
    try:
        return await message.reply_text(text)
    except RPCError as exc:
        logger.warning("Failed to send owner command reply chat_id=%s error=%s", getattr(message.chat, "id", None), exc)
    except Exception as exc:
        logger.warning("Unexpected owner reply failure error=%s", exc)
    return None


def _format_stats(counts: Dict[str, int], runtime: Dict[str, int]) -> str:
    return (
        "**Stats**\n\n"
        f"Active groups: {counts.get('active_groups', 0)}\n"
        f"Active channels: {counts.get('active_channels', 0)}\n"
        f"Private users: {counts.get('private_users', 0)}\n"
        f"Total known chats: {counts.get('known_chats', 0)}\n"
        f"Inactive/removed chats: {counts.get('inactive_chats', 0)}\n"
        f"Bot uptime: {_format_duration(runtime.get('uptime_seconds', 0))}\n"
        f"Queue size: {runtime.get('queue_size', 0)}\n"
        f"Active workers: {runtime.get('active_workers', 0)}\n"
        f"Processed messages: {runtime.get('processed_messages', 0)}\n"
        f"Skipped messages: {runtime.get('skipped_messages', 0)}\n"
        f"NSFW detected: {runtime.get('nsfw_detected', 0)}\n"
        f"Errors: {runtime.get('errors', 0)}"
    )


def _format_cache_stats(stats: Dict[str, int]) -> str:
    return (
        "**Local cache stats**\n\n"
        f"Cached NSFW stickers: {stats.get('nsfw_stickers', 0)}\n"
        f"Cached NSFW media: {stats.get('nsfw_media', 0)}\n"
        f"Temporary clean cache: {stats.get('clean_temporary', 0)}\n"
        f"Total cache rows: {stats.get('total_rows', 0)}\n"
        f"Cache DB size: {_format_bytes(stats.get('db_size_bytes', 0))}\n"
        f"Hot memory cache size: {stats.get('hot_memory_size', 0)}\n"
        f"Pending cache writes: {stats.get('write_queue_size', 0)}"
    )


def _format_bytes(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} GB"


def _format_duration(seconds: int) -> str:
    days, remainder = divmod(int(seconds), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


async def _run_broadcast(
    app: Any,
    targets: list[Dict[str, Any]],
    text: str,
    replied: Optional[Message],
) -> Dict[str, int]:
    report = {"total": len(targets), "sent": 0, "failed": 0}
    for target in targets:
        chat_id = target.get("chat_id")
        if chat_id is None:
            report["failed"] += 1
            continue

        sent = await _send_broadcast_to_chat(app, int(chat_id), text, replied)
        if sent:
            report["sent"] += 1
        else:
            report["failed"] += 1

        if settings.broadcast_delay_seconds > 0:
            await asyncio.sleep(settings.broadcast_delay_seconds)

    metrics.inc("broadcasts_completed")
    return report


async def _send_broadcast_to_chat(
    app: Any,
    chat_id: int,
    text: str,
    replied: Optional[Message],
) -> bool:
    try:
        await _copy_or_send(app, chat_id, text, replied)
        return True
    except FloodWait as exc:
        wait_seconds = int(getattr(exc, "value", 0) or 0)
        if wait_seconds <= settings.broadcast_max_flood_wait_seconds:
            await asyncio.sleep(wait_seconds + 1)
            try:
                await _copy_or_send(app, chat_id, text, replied)
                return True
            except Exception as retry_exc:
                await _handle_broadcast_failure(chat_id, retry_exc)
                return False
        await _handle_broadcast_failure(chat_id, exc)
        return False
    except Exception as exc:
        await _handle_broadcast_failure(chat_id, exc)
        return False


async def _copy_or_send(
    app: Any,
    chat_id: int,
    text: str,
    replied: Optional[Message],
) -> None:
    if replied:
        await replied.copy(chat_id)
    else:
        await app.send_message(chat_id, text, disable_web_page_preview=True)


async def _handle_broadcast_failure(chat_id: int, exc: Exception) -> None:
    logger.warning("Broadcast failed chat_id=%s error=%s", chat_id, exc)
    if _means_chat_unreachable(exc):
        await mark_chat_inactive(chat_id, reason=f"broadcast_failed:{exc.__class__.__name__}")


def _means_chat_unreachable(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    text = str(exc).lower()
    if isinstance(exc, RPCError):
        haystack = f"{name} {text}"
    else:
        haystack = text
    return any(
        token in haystack
        for token in (
            "blocked",
            "forbidden",
            "peer_id_invalid",
            "write_forbidden",
            "chat_write_forbidden",
            "chat_admin_required",
            "channel_private",
            "not a member",
            "kicked",
            "deactivated",
        )
    )
