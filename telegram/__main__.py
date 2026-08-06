import asyncio
import importlib
import logging
import os

try:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
except RuntimeError:
    pass

from pyrogram import idle

logger = logging.getLogger(__name__)


async def main() -> None:
    antinsfw = importlib.import_module("telegram.antinsfw")
    importlib.import_module("telegram.stats")

    from telegram.bot import client
    from telegram.db import init_db

    await init_db()
    await antinsfw.start_runtime()

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
        if started:
            try:
                await client.stop()
            except RuntimeError as exc:
                logger.warning("Pyrogram shutdown warning: %s", exc)
        await antinsfw.stop_runtime()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
