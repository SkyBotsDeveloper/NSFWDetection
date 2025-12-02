import cv2
import os
import logging
import time
from collections import defaultdict

from PIL import Image
import torch
import clip  # CLIP fallback model
from telegram import client
from pyrogram import filters
from telegram.db import is_nsfw, add_chat, add_user, add_nsfw, remove_nsfw
from transformers import AutoModelForImageClassification, ViTImageProcessor
from pyrogram.enums import ChatType
from pyrogram.types import InlineKeyboardButton, InlineKeyboardMarkup

# ---------------------- DEVICE SETUP ----------------------

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# ---------------------- MAIN NSFW MODEL ----------------------

model = AutoModelForImageClassification.from_pretrained(
    "Falconsai/nsfw_image_detection"
).to(device)
processor = ViTImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")

# ---------------------- CLIP FALLBACK MODEL ----------------------

clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
clip_model.eval()

CLIP_NSFWTEXT = [
    "a pornographic image",
    "a nude person",
    "a naked woman",
    "a naked man",
    "sexual content",
    "explicit content",
    "adult content",
    "nsfw image",
]

# Pre-tokenize text once and move to device
CLIP_TEXT_TOKENS = clip.tokenize(CLIP_NSFWTEXT).to(device)

# Threshold for CLIP NSFW probability (tune if needed)
CLIP_THRESHOLD = 0.35

# ---------------------- ANTI-SPAM STATE ----------------------

nsfw_warning_cache = {}  # chat_id -> last_warning_timestamp

# (chat_id, user_id) -> {"count": int, "first_ts": float}
user_nsfw_count = defaultdict(lambda: {"count": 0, "first_ts": 0.0})

MAX_VIOLATIONS_WINDOW = 300  # seconds = 5 minutes
MAX_VIOLATIONS = 5           # how many NSFW msgs in that window to be considered spammy
WARNING_COOLDOWN = 10        # seconds between warning messages in the same chat


# ---------------------- HELPERS ----------------------


def get_media_key_for_db(message):
    """
    Use Telegram's file_unique_id as the key for NSFW cache.
    Same sticker / photo / video => same unique id, so reuse NSFW decision.
    """
    if message.photo:
        return message.photo.file_unique_id
    if message.sticker:
        return message.sticker.file_unique_id
    if message.animation:
        return message.animation.file_unique_id
    if message.video:
        return message.video.file_unique_id
    return None


def run_main_model(img: Image.Image) -> bool:
    """
    Run Falconsai model on a PIL image.
    Returns True if NSFW, False otherwise.
    """
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt")
        # Move tensors to device
        for k in inputs:
            inputs[k] = inputs[k].to(device)

        outputs = model(**inputs)
        logits = outputs.logits

    predicted_label = logits.argmax(-1).item()
    return predicted_label == 1  # 1 == nsfw for this model


def fallback_clip_detect(img: Image.Image) -> bool:
    """
    Uses CLIP to judge whether the image is NSFW.
    Returns True if NSFW, else False.
    """
    try:
        image = clip_preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits_per_image, _ = clip_model(image, CLIP_TEXT_TOKENS)
            probs = logits_per_image.softmax(dim=-1)

        if probs.max().item() > CLIP_THRESHOLD:
            return True

        return False

    except Exception as e:
        logger.error(f"CLIP fallback failed: {e}")
        return False


def is_nsfw_image(img: Image.Image) -> bool:
    """
    Combined NSFW decision:
    1) main model
    2) if safe, fallback CLIP
    """
    # Main model
    if run_main_model(img):
        return True

    # Fallback
    if fallback_clip_detect(img):
        return True

    return False


# ---------------------- MAIN HANDLER ----------------------


@client.on_message(filters.photo | filters.sticker | filters.animation | filters.video)
async def getimage(client, event):
    """
    Main handler for photos, stickers, animations and videos.
    - Photos + static stickers are treated as images.
    - Animations, video stickers and videos are treated as videos and passed to videoShit().
    """

    file_path = None
    nsfw_key = get_media_key_for_db(event)

    if nsfw_key is None:
        return

    # Fast path: if DB already knows this media (by unique id) is NSFW, delete immediately
    if await is_nsfw(nsfw_key):
        await send_msg(event)
        return

    # -------- PHOTO --------
    if event.photo:
        try:
            file_path = await client.download_media(event.photo)
        except Exception as e:
            logger.error("Failed to download image. Error: %s", e)
            # Can't scan -> do nothing, don't delete, don't mark NSFW
            return

    # -------- STICKER --------
    elif event.sticker:
        # Video sticker (webm) -> treat like video
        if event.sticker.is_video or event.sticker.mime_type == "video/webm":
            try:
                file_path = await client.download_media(event.sticker)
            except Exception as e:
                logger.error("Failed to download video sticker. Error: %s", e)
                # Can't scan -> ignore (might miss NSFW, but no false delete)
                return

            await videoShit(event, file_path, nsfw_key)
            return

        # Animated .tgs sticker (vector format) -> NOT supported by Pillow
        if event.sticker.is_animated and not event.sticker.is_video:
            logger.info(
                "Skipping animated .tgs sticker (%s); NSFW scan not supported for this format yet.",
                event.sticker.file_unique_id,
            )
            # We just ignore it. No delete, no DB mark, no error spam.
            return

        # Static sticker (webp/png) -> treat like an image
        try:
            file_path = await client.download_media(event.sticker)
        except Exception as e:
            logger.error("Failed to download static sticker. Error: %s", e)
            # Can't scan -> ignore
            return

    # -------- GIF / ANIMATION --------
    elif event.animation:
        try:
            file_path = await client.download_media(event.animation)
        except Exception as e:
            logger.error("Failed to download GIF. Error: %s", e)
            # Can't scan -> ignore
            return

        await videoShit(event, file_path, nsfw_key)
        return

    # -------- VIDEO --------
    elif event.video:
        try:
            file_path = await client.download_media(event.video)
        except Exception as e:
            logger.error("Failed to download video. Error: %s", e)
            # Can't scan -> ignore
            return

        await videoShit(event, file_path, nsfw_key)
        return

    else:
        # Not something we care about
        return

    # ----- IMAGE PROCESSING (photos + static stickers) -----

    if not file_path or not os.path.exists(file_path):
        logger.error("Downloaded image file does not exist: %s", file_path)
        # Can't scan -> ignore
        return

    try:
        img = Image.open(file_path).convert("RGB")
    except Exception as e:
        logger.error("Failed to open image %s: %s", file_path, e)
        # Can't scan -> ignore
        return

    # Combined NSFW decision (main model + CLIP fallback)
    detected_nsfw = is_nsfw_image(img)

    if detected_nsfw:
        await add_nsfw(nsfw_key)   # cache by unique id
        await send_msg(event)      # delete + warn (with anti-spam logic)
    else:
        await remove_nsfw(nsfw_key)

    # Optional: delete downloaded image to save space
    try:
        os.remove(file_path)
    except OSError:
        pass


# ---------------------- START COMMAND ----------------------


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


# ---------------------- SEND MESSAGE / ANTI-SPAM ----------------------


async def send_msg(event):
    """
    Called only when something is CONFIRMED NSFW (by model or DB).
    In groups: delete the message, then send a warning (rate-limited).
    In private: reply with a simple notice.

    Also tracks how many NSFW messages a user sent recently, and clearly
    tells admins who is spamming.
    """

    chat_id = event.chat.id
    user = event.from_user
    user_id = user.id if user else None
    now = time.time()

    # ---- update per-user NSFW counter ----
    if user_id is not None:
        key = (chat_id, user_id)
        data = user_nsfw_count[key]

        # If outside the window, reset
        if data["first_ts"] == 0 or now - data["first_ts"] > MAX_VIOLATIONS_WINDOW:
            data["first_ts"] = now
            data["count"] = 1
        else:
            data["count"] += 1

    # Always try to delete the offending message in groups & supergroups
    if event.chat.type in (ChatType.SUPERGROUP, ChatType.GROUP):
        try:
            await event.delete()
        except Exception:
            pass

        # Rate-limit warning messages to avoid spam in the group
        last_warning = nsfw_warning_cache.get(chat_id, 0)
        if now - last_warning >= WARNING_COOLDOWN:
            base_text = "🚫 NSFW content detected and removed."

            # If user is repeatedly sending NSFW content, mention them clearly
            extra_text = ""
            if user_id is not None:
                data = user_nsfw_count[(chat_id, user_id)]

                if getattr(user, "username", None):
                    user_mention = f"@{user.username}"
                else:
                    user_mention = user.first_name or "this user"

                if data["count"] >= MAX_VIOLATIONS:
                    extra_text = (
                        f"\n⚠️ Admins, please check {user_mention}: "
                        f"they have sent {data['count']} NSFW messages "
                        f"in the last few minutes."
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


# ---------------------- VIDEO / GIF / VIDEO STICKER SUPPORT ----------------------


def capture_screenshot(path, max_frames=8):
    """
    Capture up to max_frames evenly spaced frames from the video.
    Returns a list of image file paths.
    More robust than "every 10 seconds", works better for short webm stickers.
    """
    vidObj = cv2.VideoCapture(path)
    saved_image_names = []

    total_frames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames > 0:
        # Evenly pick up to max_frames across the whole video
        step = max(total_frames // max_frames, 1)
        target_indices = set(i * step for i in range(max_frames) if i * step < total_frames)

        idx = 0
        success, frame = vidObj.read()
        while success:
            if idx in target_indices:
                image_name = f"frame_{idx}.png"
                cv2.imwrite(image_name, frame)
                saved_image_names.append(image_name)
            idx += 1
            success, frame = vidObj.read()
    else:
        # Fallback: capture every ~10 seconds
        fps = vidObj.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25
        frames_to_skip = int(fps * 10)

        count = 0
        success, frame = vidObj.read()
        while success:
            if frames_to_skip > 0 and count % frames_to_skip == 0:
                image_name = f"frame_{count}.png"
                cv2.imwrite(image_name, frame)
                saved_image_names.append(image_name)
            count += 1
            success, frame = vidObj.read()

    vidObj.release()
    return saved_image_names


async def videoShit(event, video_path, nsfw_key):
    """
    Process video/gif/animated sticker:
    - Take multiple screenshots across the video.
    - Run NSFW model + CLIP on each frame.
    - If any frame is NSFW, treat the whole file as NSFW.
    - If we can't process frames at all, we DO NOT delete. (no false positives)
    """

    # Double-check cache
    if await is_nsfw(nsfw_key):
        await send_msg(event)
        return

    # Take frames from video
    image_names = capture_screenshot(video_path)

    if not image_names:
        logger.error("No frames captured from video: %s", video_path)
        # Can't scan this video/sticker -> ignore (might miss NSFW but no false delete)
        try:
            os.remove(video_path)
        except OSError:
            pass
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

        if is_nsfw_image(img):
            is_video_nsfw = True
            break

    if is_video_nsfw:
        await add_nsfw(nsfw_key)
        await send_msg(event)
    else:
        await remove_nsfw(nsfw_key)

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
