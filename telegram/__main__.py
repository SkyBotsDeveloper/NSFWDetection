import asyncio
import importlib
from telegram import client
from uvloop import install
from pyrogram import idle
import logging

loop = asyncio.get_event_loop()


imported_module = importlib.import_module("telegram.antinsfw")
imported_module = importlib.import_module("telegram.stats")
imported_module = importlib.import_module("telegram.db")

async def gae():
    install()
    await client.start()
    await idle()
    await client.stop()

if __name__ == "__main__":
    logging.info("Bot Started! Powered By @VivaanNetwork")
    loop.run_until_complete(gae())
