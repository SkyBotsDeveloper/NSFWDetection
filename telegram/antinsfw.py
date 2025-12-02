import cv2
import os
import logging
import time
from collections import defaultdict

from PIL import Image
import torch
from telegram import client
from pyrogram import filters
from telegram.db import is_nsfw, add_chat, add_user, add_nsfw, remove_nsfw
from transformers import AutoModelForImageClassification, ViTImageProcessor
from pyrogram.enums import ChatType
from pyrogram.types import InlineKeyboardButton, InlineKeyboardMarkup

# Load model once, at startup
model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
processor = ViTImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")

logger = logging.getLogger(__name__)

# -------- anti-spam state (in-memory) --------

# Last time we sent a "NSFW detected" warning per chat
nsfw_warning_cache = {}  # chat_id -> last_warning_timestamp

# Track how many NSFW messages each user sent in each chat recently
# (chat_id, user_id) -> {"count": int, "first_ts": float}
user_nsfw_count = defaultdict(lambda: {"count": 0, "first_ts": 0.0})

# Config for spam handling
MAX_VIOLATIONS_WINDOW = 300  # seconds = 5 minutes
MAX_VIOLATIONS = 5          # how many NSFW msgs in that window to be considered spammy
WARNING_COOLDOWN = 10       # seconds between warning messages in the same chat


@client.on_message(filters.photo | filters.sticker | filters.animation | filters.video)
async def getimage(client, event):
    """
    Main handler for photos, stickers, animations and videos.
    - Photos + static stickers are treated as images.
    - Animations, video stickers and videos are treated as videos and passed to videoShit().
    """

    file_path = None
    file_id = None

    # Photo (normal images)
    if event.photo:
        file_id = event.photo.file_id

        # Skip if we already know it's NSFW
        if await is_nsfw(file_id):
            await send_msg(event)
            return

        try:
            # download_media returns the full file path
            file_path = await client.download_media(event.photo)
        except Exception as e:
            logger.error("Failed to download image. Error: %s", e)
            return

    # Stickers
    elif event.sticker:
        file_id = event.sticker.file_id

        if await is_nsfw(file_id):
            await send_msg(event)
            return

        # Animated/video sticker
        if event.sticker.mime_type == "video/webm":
            try:
                file_path = await client.download_media(event.sticker)
            except Exception as e:
                logger.error("Failed to download animated sticker. Error: %s", e)
                return

            # Handle like a video and then stop
            await videoShit(event, file_path, file_id)
            return

        # Static sticker (treated like an image)
        else:
            try:
                file_path = await client.download_media(event.sticker)
            except Exception as e:
                logger.error("Failed to download sticker. Error: %s", e)
                return

    # GIF / animation
    elif event.animation:
        file_id = event.animation.file_id

        if await is_nsfw(file_id):
            await send_msg(event)
            return

        try:
            file_path = await client.download_media(event.animation)
        except Exception as e:
            logger.error("Failed to download GIF. Error: %s", e)
            return

        await videoShit(event, file_path, file_id)
        return

    # Normal video
    elif event.video:
        file_id = event.video.file_id

        if await is_nsfw(file_id):
            await send_msg(event)
            return

        try:
            file_path = await client.download_media(event.video)
        except Exception as e:
            logger.error("Failed to download video. Error: %s", e)
            return

        await videoShit(event, file_path, file_id)
        return

    else:
        # Not something we care about
        return

    # From here on, we expect an image file at file_path
    if not file_path or not os.path.exists(file_path):
        logger.error("Downloaded image file does not exist: %s", file_path)
        return

    try:
        img = Image.open(file_path).convert("RGB")
    except Exception as e:
        logger.error("Failed to open image %s: %s", file_path, e)
        return

    # Run NSFW model
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_label = logits.argmax(-1).item()

    # 1 means nsfw for this model
    if predicted_label:
        await add_nsfw(file_id)
        await send_msg(event)  # this handles spam logic + warnings
    else:
        await remove_nsfw(file_id)

    # Optional: delete downloaded image to save space
    try:
        os.remove(file_path)
    except OSError:
        pass


@client.on_message(filters.command("start"))
async def start(_, event):
    buttons = [[
        InlineKeyboardButton("Support Chat", url="t.me/VivaanSupport"),
        InlineKeyboardButton("News Channel", url="t.me/VivaanUpdates")
    ]]
    reply_markup = InlineKeyboardMarkup(buttons)
    await event.reply_text(
        "Hello, I am a bot that detects NSFW (Not Safe for Work) images. "
        "Send me an image to check if it is NSFW or not. In groups, just make me "
        "an admin with delete message rights and I will delete all NSFW images sent by anyone.",
        reply_markup=reply_markup
    )
    if event.from_user and event.from_user.username:
        await add_user(event.from_user.id, event.from_user.username)
    elif event.from_user:
        await add_user(event.from_user.id, "None")


async def send_msg(event):
    """
    Send a message when NSFW content is detected.
    In groups: delete the message, then send a warning (rate-limited).
    In private: reply with a simple notice.

    Also tracks how many NSFW messages a user sent recently.
    """

    chat_id = event.chat.id
    user_id = event.from_user.id if event.from_user else None
    now = time.time()

    # ---- update per-user NSFW counter ----
    if user_id is not None:
        key = (chat_id, user_id)
        data = user_nsfw_count[key]

        # If outside the window, reset
        if now - data["first_ts"] > MAX_VIOLATIONS_WINDOW or data["first_ts"] == 0:
            data["first_ts"] = now
            data["count"] = 1
        else:
            data["count"] += 1

    # Always try to delete the offending message in groups
    if event.chat.type == ChatType.SUPERGROUP:
        try:
            await event.delete()
        except Exception:
            pass

        # Rate-limit warning messages to avoid spam in the group
        last_warning = nsfw_warning_cache.get(chat_id, 0)
        if now - last_warning >= WARNING_COOLDOWN:
            base_text = "NSFW content detected and removed."

            # If user is repeatedly sending NSFW content, mention it once in a while
            extra_text = ""
            if user_id is not None:
                data = user_nsfw_count[(chat_id, user_id)]
                if data["count"] >= MAX_VIOLATIONS:
                    # Basic extra warning about spammer
                    name = event.from_user.first_name if event.from_user else "A user"
                    extra_text = (
                        f"\n{name} has sent {data['count']} NSFW messages "
                        f"in the last few minutes. Please consider taking action."
                    )

            try:
                await client.send_message(chat_id, base_text + extra_text)
            except Exception:
                pass

            nsfw_warning_cache[chat_id] = now

        # Track chat in DB
        await add_chat(chat_id)

    else:
        # Private chat: just reply, no spam problem here
        await event.reply("NSFW Image.")


def capture_screenshot(path):
    """
    Capture one frame every 10 seconds from the video.
    Returns a list of image file paths.
    """
    vidObj = cv2.VideoCapture(path)
    fps = vidObj.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25  # fallback if FPS can't be read

    frames_to_skip = int(fps * 10)

    count = 0
    success = True
    saved_image_names = []

    while success:
        success, image = vidObj.read()
        if not success:
            break

        if frames_to_skip > 0 and count % frames_to_skip == 0:
            image_name = f"frame_{count // frames_to_skip}.png"
            cv2.imwrite(image_name, image)
            saved_image_names.append(image_name)

        count += 1

    vidObj.release()
    return saved_image_names


async def videoShit(event, video_path, file_id):
    """
    Process video/gif/animated sticker:
    - Take screenshots every 10 seconds.
    - Run NSFW model on each frame.
    - If any frame is NSFW, treat the whole file as NSFW.
    """

    # Double-check cache
    if await is_nsfw(file_id):
        await send_msg(event)
        return

    # Take frames from video
    image_names = capture_screenshot(video_path)

    if not image_names:
        logger.error("No frames captured from video: %s", video_path)
        return

    is_video_nsfw = False

    for img_path in image_names:
        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error("Failed to open frame %s: %s", img_path, e)
            continue

        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits

        predicted_label = logits.argmax(-1).item()

        if predicted_label:
            is_video_nsfw = True
            break

    if is_video_nsfw:
        await add_nsfw(file_id)
        await send_msg(event)
    else:
        await remove_nsfw(file_id)

    # Cleanup frames and video file
    for img_path in image_names:
        try:
            os.remove(img_path)
        except OSError:
            pass

    try:
        os.remove(video_path)
    except OSError:
        pass
