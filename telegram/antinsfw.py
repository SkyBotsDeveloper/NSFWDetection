import os
import logging

import cv2
from PIL import Image
import torch

from telegram import client       # your existing pyrogram.Client instance
from pyrogram import filters
from pyrogram.enums import ChatType
from transformers import AutoModelForImageClassification, ViTImageProcessor

logger = logging.getLogger(__name__)

# ------------------ MODEL SETUP ------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"[NSFW BOT] Using device: {device}")

model = AutoModelForImageClassification.from_pretrained(
    "Falconsai/nsfw_image_detection"
).to(device)
processor = ViTImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")

# We assume index 1 in logits is NSFW (this matches earlier usage)
NSFW_INDEX = 1

# Thresholds (tune these if needed)
PHOTO_THRESHOLD = 0.70    # photos, image documents
VIDEO_THRESHOLD = 0.70    # frames from video / gif / video sticker
STICKER_THRESHOLD = 0.90  # static stickers only deleted if VERY likely NSFW


def classify_nsfw_prob(img: Image.Image) -> float:
    """
    Run Falconsai model on a PIL image.
    Returns NSFW probability (0.0 - 1.0).
    """
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt")
        for k in inputs:
            inputs[k] = inputs[k].to(device)

        outputs = model(**inputs)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)
        nsfw_prob = float(probs[NSFW_INDEX].item())
        return nsfw_prob


def is_nsfw_image(img: Image.Image, kind: str) -> bool:
    """
    Decide NSFW based on kind:
    - 'photo' or 'document' -> PHOTO_THRESHOLD
    - 'video'                -> VIDEO_THRESHOLD
    - 'sticker'              -> STICKER_THRESHOLD (very conservative)
    """
    nsfw_prob = classify_nsfw_prob(img)

    if kind == "sticker":
        threshold = STICKER_THRESHOLD
    elif kind == "video":
        threshold = VIDEO_THRESHOLD
    else:
        threshold = PHOTO_THRESHOLD

    logger.debug(
        f"[NSFW BOT] kind={kind}, nsfw_prob={nsfw_prob:.3f}, threshold={threshold:.3f}"
    )

    return nsfw_prob >= threshold


def capture_frames(path: str, max_frames: int = 8) -> list[str]:
    """
    Capture up to max_frames evenly spaced frames from a video/gif.
    Returns a list of image file paths.
    """
    vid = cv2.VideoCapture(path)
    saved: list[str] = []

    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames > 0:
        # Evenly sample frames
        step = max(total_frames // max_frames, 1)
        targets = set(i * step for i in range(max_frames) if i * step < total_frames)

        idx = 0
        success, frame = vid.read()
        while success:
            if idx in targets:
                img_name = f"frame_{idx}.png"
                cv2.imwrite(img_name, frame)
                saved.append(img_name)
            idx += 1
            success, frame = vid.read()
    else:
        # Fallback: every ~10 seconds
        fps = vid.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25
        frames_to_skip = int(fps * 10)

        count = 0
        success, frame = vid.read()
        while success:
            if frames_to_skip > 0 and count % frames_to_skip == 0:
                img_name = f"frame_{count}.png"
                cv2.imwrite(img_name, frame)
                saved.append(img_name)
            count += 1
            success, frame = vid.read()

    vid.release()
    return saved


async def delete_nsfw_message(message):
    """
    What happens when we are SURE the content is NSFW.
    Only deletes NSFW content. No DB, no spam logic.
    """
    if message.chat.type in (ChatType.SUPERGROUP, ChatType.GROUP):
        # In groups, just delete the message and send a simple warning
        try:
            await message.delete()
        except Exception:
            pass
        try:
            await message.chat.send_message("🚫 NSFW content detected and removed.")
        except Exception:
            pass
    else:
        # In private: just reply
        try:
            await message.reply("🚫 NSFW content detected.")
        except Exception:
            pass


# ------------------ MAIN HANDLER ------------------


@client.on_message(
    filters.incoming
    & (
        filters.photo
        | filters.sticker
        | filters.animation
        | filters.video
        | filters.document
    )
)
async def getimage(client, message):
    """
    Handles:
      - photos
      - static stickers
      - video stickers (webm)
      - animations (GIF / MPEG4)
      - videos
      - documents that are images

    Only deletes when detection is confident enough.
    """

    file_path = None
    kind = "photo"  # default kind

    # -------- PHOTOS --------
    if message.photo:
        kind = "photo"
        try:
            file_path = await message.download()
        except Exception as e:
            logger.error("Failed to download photo: %s", e)
            return

    # -------- STICKERS --------
    elif message.sticker:
        # Video sticker (webm) -> treat like video
        if message.sticker.is_video or message.sticker.mime_type == "video/webm":
            kind = "video"
            try:
                file_path = await message.download()
            except Exception as e:
                logger.error("Failed to download video sticker: %s", e)
                return

        # Animated .tgs sticker -> NOT supported by PIL/OpenCV, skip safely
        elif message.sticker.is_animated and not message.sticker.is_video:
            logger.info(
                "Skipping animated .tgs sticker (%s); not supported for NSFW scan yet.",
                message.sticker.file_unique_id,
            )
            return

        # Static sticker (webp/png)
        else:
            kind = "sticker"
            try:
                file_path = await message.download()
            except Exception as e:
                logger.error("Failed to download static sticker: %s", e)
                return

    # -------- ANIMATIONS (GIF, MPEG4) --------
    elif message.animation:
        kind = "video"
        try:
            file_path = await message.download()
        except Exception as e:
            logger.error("Failed to download animation: %s", e)
            return

    # -------- VIDEOS --------
    elif message.video:
        kind = "video"
        try:
            file_path = await message.download()
        except Exception as e:
            logger.error("Failed to download video: %s", e)
            return

    # -------- DOCUMENT IMAGES --------
    elif message.document:
        # Only care about documents that are images
        mime = (message.document.mime_type or "").lower()
        if not mime.startswith("image/"):
            return  # ignore non-image docs

        kind = "photo"
        try:
            file_path = await message.download()
        except Exception as e:
            logger.error("Failed to download image document: %s", e)
            return

    else:
        # Not a type we care about
        return

    # -------- CHECK FILE EXISTS --------
    if not file_path or not os.path.exists(file_path):
        logger.error("Downloaded file does not exist: %s", file_path)
        return

    # -------- PROCESS STATIC IMAGE TYPES --------
    if kind in ("photo", "sticker"):
        try:
            img = Image.open(file_path).convert("RGB")
        except Exception as e:
            logger.error("Failed to open image %s: %s", file_path, e)
            return

        nsfw_flag = is_nsfw_image(img, kind)

        if nsfw_flag:
            await delete_nsfw_message(message)

    # -------- PROCESS VIDEO-LIKE MEDIA --------
    elif kind == "video":
        frame_paths = capture_frames(file_path, max_frames=8)
        if not frame_paths:
            logger.error("No frames captured from video: %s", file_path)
        else:
            nsfw_flag = False

            for fp in frame_paths:
                if not os.path.exists(fp):
                    continue
                try:
                    img = Image.open(fp).convert("RGB")
                except Exception as e:
                    logger.error("Failed to open frame %s: %s", fp, e)
                    continue

                if is_nsfw_image(img, "video"):
                    nsfw_flag = True
                    break

            if nsfw_flag:
                await delete_nsfw_message(message)

            # cleanup extracted frames
            for fp in frame_paths:
                try:
                    os.remove(fp)
                except OSError:
                    pass

    # -------- CLEANUP ORIGINAL FILE --------
    try:
        os.remove(file_path)
    except OSError:
        pass
