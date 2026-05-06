import asyncio
import importlib
import logging
import os

from pyrogram import idle

from telegram.antinsfw import start_runtime, stop_runtime
from telegram.bot import client
from telegram.db import init_db

logger = logging.getLogger(__name__)


async def main() -> None:
    importlib.import_module("telegram.antinsfw")
    importlib.import_module("telegram.stats")

    await init_db()
    await start_runtime()

    started = False
    try:
        if os.getenv("STARTUP_CHECK_ONLY") == "1":
            logger.info("Startup check completed")
            return

        await client.start()
        started = True
        me = await client.get_me()
        logger.info("Bot started as @%s (%s)", me.username, me.id)
        await idle()
    finally:
        await stop_runtime()
        if started:
            await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
