import logging
import os

from pyrogram import Client

from telegram.config import configure_logging, require_bot_settings, settings

logger = logging.getLogger(__name__)


def _install_uvloop() -> None:
    if os.name == "nt":
        return
    try:
        import uvloop

        uvloop.install()
    except Exception as exc:
        logger.warning("uvloop is unavailable; using the default asyncio loop: %s", exc)


configure_logging()
_install_uvloop()
require_bot_settings()

client = Client(
    settings.session_name,
    api_id=settings.api_id,
    api_hash=settings.api_hash,
    bot_token=settings.bot_token,
    workers=settings.pyrogram_workers,
    sleep_threshold=settings.pyrogram_sleep_threshold_seconds,
)
