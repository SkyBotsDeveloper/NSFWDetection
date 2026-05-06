import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import motor.motor_asyncio
from pymongo import ASCENDING
from pymongo.errors import PyMongoError

from telegram.config import settings

logger = logging.getLogger(__name__)

client = motor.motor_asyncio.AsyncIOMotorClient(
    settings.mongo_uri,
    serverSelectionTimeoutMS=settings.mongo_timeout_ms,
    connectTimeoutMS=settings.mongo_timeout_ms,
    socketTimeoutMS=max(settings.mongo_timeout_ms, 15000),
)
db = client[settings.database_name]

userdb = db.users
chatdb = db.chats


def _now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_chat_type(chat_type: Any) -> str:
    value = getattr(chat_type, "value", chat_type)
    return str(value).replace("ChatType.", "").lower()


async def init_db() -> None:
    try:
        await userdb.create_index([("user_id", ASCENDING)], unique=True)
        await chatdb.create_index([("chat_id", ASCENDING)], unique=True)
        await chatdb.create_index([("active", ASCENDING), ("chat_type", ASCENDING)])
    except PyMongoError:
        logger.exception("MongoDB index initialization failed")
    except Exception:
        logger.exception("Unexpected DB initialization failure")


async def add_user(
    user_id: int,
    username: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
) -> bool:
    now = _now()
    try:
        user_id = int(user_id)
        await userdb.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "user_id": user_id,
                    "username": username or "None",
                    "first_name": first_name,
                    "last_name": last_name,
                    "last_seen": now,
                },
                "$setOnInsert": {"first_seen": now},
            },
            upsert=True,
        )
        return True
    except PyMongoError:
        logger.exception("Failed to upsert user_id=%s", user_id)
    except Exception:
        logger.exception("Unexpected user upsert failure for user_id=%s", user_id)
    return False


async def upsert_chat(
    chat_id: int,
    chat_type: str,
    title: Optional[str] = None,
    username: Optional[str] = None,
    active: bool = True,
) -> bool:
    now = _now()
    try:
        chat_id = int(chat_id)
        update: Dict[str, Any] = {
            "$set": {
                "chat_id": chat_id,
                "chat_type": chat_type,
                "title": title,
                "username": username,
                "active": bool(active),
                "last_seen": now,
            },
            "$setOnInsert": {"first_seen": now},
        }
        if active:
            update["$unset"] = {"inactive_reason": "", "inactive_at": ""}

        await chatdb.update_one({"chat_id": chat_id}, update, upsert=True)
        return True
    except PyMongoError:
        logger.exception("Failed to upsert chat_id=%s", chat_id)
    except Exception:
        logger.exception("Unexpected chat upsert failure for chat_id=%s", chat_id)
    return False


async def track_chat_from_message(message: Any) -> bool:
    chat = getattr(message, "chat", None)
    if not chat:
        return False

    chat_type = normalize_chat_type(getattr(chat, "type", "unknown"))
    title = getattr(chat, "title", None)
    username = getattr(chat, "username", None)

    if chat_type == "private":
        user = getattr(message, "from_user", None)
        if user:
            username = getattr(user, "username", None) or username
            title = " ".join(
                part
                for part in [getattr(user, "first_name", None), getattr(user, "last_name", None)]
                if part
            ) or username
            await add_user(
                getattr(user, "id"),
                username=username,
                first_name=getattr(user, "first_name", None),
                last_name=getattr(user, "last_name", None),
            )

    return await upsert_chat(
        chat_id=getattr(chat, "id"),
        chat_type=chat_type,
        title=title,
        username=username,
        active=True,
    )


async def mark_chat_inactive(chat_id: int, reason: str = "unreachable") -> bool:
    now = _now()
    try:
        result = await chatdb.update_one(
            {"chat_id": int(chat_id)},
            {
                "$set": {
                    "active": False,
                    "inactive_reason": reason,
                    "inactive_at": now,
                    "last_seen": now,
                }
            },
        )
        return result.modified_count > 0
    except PyMongoError:
        logger.exception("Failed to mark chat_id=%s inactive", chat_id)
    except Exception:
        logger.exception("Unexpected inactive update failure for chat_id=%s", chat_id)
    return False


async def mark_chat_active(
    chat_id: int,
    chat_type: str,
    title: Optional[str] = None,
    username: Optional[str] = None,
) -> bool:
    return await upsert_chat(chat_id, chat_type, title=title, username=username, active=True)


async def get_chat_counts() -> Dict[str, int]:
    try:
        active = {"active": True}
        counts = {
            "active_groups": await chatdb.count_documents(
                {**active, "chat_type": {"$in": ["group", "supergroup"]}}
            ),
            "active_channels": await chatdb.count_documents({**active, "chat_type": "channel"}),
            "private_users": await chatdb.count_documents({**active, "chat_type": "private"}),
            "known_chats": await chatdb.count_documents({}),
            "inactive_chats": await chatdb.count_documents({"active": False}),
            "known_users": await userdb.count_documents({}),
        }
        return counts
    except PyMongoError:
        logger.exception("Failed to load chat counts")
    except Exception:
        logger.exception("Unexpected count failure")
    return {
        "active_groups": 0,
        "active_channels": 0,
        "private_users": 0,
        "known_chats": 0,
        "inactive_chats": 0,
        "known_users": 0,
    }


async def get_active_broadcast_targets() -> List[Dict[str, Any]]:
    try:
        cursor = chatdb.find(
            {
                "active": True,
                "chat_type": {"$in": ["group", "supergroup", "channel", "private"]},
            },
            {"_id": 0, "chat_id": 1, "chat_type": 1, "title": 1, "username": 1},
        )
        return [doc async for doc in cursor]
    except PyMongoError:
        logger.exception("Failed to load broadcast targets")
    except Exception:
        logger.exception("Unexpected broadcast target failure")
    return []


async def add_chat(chat_id: int) -> bool:
    return await upsert_chat(chat_id, "supergroup", active=True)
