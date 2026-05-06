# Telegram NSFW Detection Bot

Production-oriented Telegram moderation bot for detecting and removing NSFW media in groups, channels, and private chats.

The bot uses Pyrogram, MongoDB, and the `Falconsai/nsfw_image_detection` Hugging Face model. Media processing is handled through a bounded async queue with worker limits, file-size limits, download and processing timeouts, duplicate suppression, and temp-file cleanup.

Sticker/media NSFW cache is stored locally on disk in SQLite WAL mode at `./data/nsfw_cache.sqlite` by default. MongoDB is only used for chat/user persistence and broadcast targets.

## Setup

```bash
git clone https://github.com/SkyBotsDeveloper/NSFWDetection
cd NSFWDetection
python3 -m venv venv
source venv/bin/activate
pip install -U pip && pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set at minimum:

```env
BOT_TOKEN=your_bot_token
API_ID=your_api_id
API_HASH=your_api_hash
OWNER_ID=your_telegram_user_id
MONGO_URI=mongodb://localhost:27017
DATABASE_NAME=nsfw
LOCAL_CACHE_DB=./data/nsfw_cache.sqlite
NSFW_THRESHOLD=0.85
```

Start the bot:

```bash
bash start
```

On Windows PowerShell, activate the virtual environment with:

```powershell
.\venv\Scripts\Activate.ps1
```

## VPS Auto Restart

Use `systemd` on a Linux VPS so the bot starts automatically after reboot and restarts if it crashes.

After setup and after `.env` is filled:

```bash
chmod +x start scripts/install_systemd.sh
sudo -E bash scripts/install_systemd.sh
```

This creates and starts `/etc/systemd/system/nsfw-bot.service` for the current repo path. It uses `./venv/bin/python` automatically when the virtual environment exists.

Useful service commands:

```bash
sudo systemctl status nsfw-bot
sudo journalctl -u nsfw-bot -f
sudo systemctl restart nsfw-bot
sudo systemctl stop nsfw-bot
```

To install with a different service name or user:

```bash
sudo env SERVICE_NAME=my-nsfw-bot SERVICE_USER=ubuntu bash scripts/install_systemd.sh
```

After VPS reboot, systemd will start the bot automatically because the installer runs `systemctl enable nsfw-bot`.

## Media Detection Coverage

Detected when within configured size limits:

- Telegram photos.
- Static image stickers such as WebP/image stickers.
- Video stickers such as WebM stickers.
- Telegram videos.
- Telegram animations/GIF-style media.
- Documents that are images.
- Documents that are videos.
- GIF documents.

Skipped safely:

- TGS/Lottie animated stickers (`application/x-tgsticker`) because they need a separate renderer before frame detection.
- Unsupported document MIME types.
- Media larger than `MAX_IMAGE_SIZE_MB`, `MAX_VIDEO_SIZE_MB`, or `MAX_DOCUMENT_SIZE_MB`.
- Media that fails download, decode, or model processing.

Failed processing is never treated as NSFW. The bot logs the error and skips that item.

## Owner Commands

`/stats`

Shows active groups, active channels, private users, known chats, inactive chats, uptime, queue size, active workers, processed messages, skipped messages, NSFW detections, and errors.

`/broadcast <message>`

Sends a text broadcast to every active group, channel, and private chat in MongoDB. You can also reply to a message with `/broadcast` to copy that replied message to all active targets. Broadcasts are rate-limited and mark unreachable chats inactive when Telegram reports blocked, forbidden, removed, or no-permission errors.

`/cache_stats`

Shows local SQLite cache counts for NSFW stickers, NSFW media, temporary clean entries, database size, hot in-memory cache size, and pending cache writes.

## Important Configuration

- `NSFW_THRESHOLD`: Confidence required before taking moderation action. Failed processing is never treated as NSFW.
- `LOCAL_CACHE_DB`: Local SQLite WAL cache path for sticker/media NSFW decisions. Keep this on fast VPS disk.
- `HOT_CACHE_MAX_SIZE`, `HOT_CACHE_TTL_SECONDS`: In-memory hot cache for repeated sticker/media spam.
- `CLEAN_MEDIA_CACHE_TTL_SECONDS`: Short TTL for clean media cache entries. NSFW entries are kept long-term.
- `QUEUE_MAX_SIZE`, `WORKER_COUNT`, `INFERENCE_WORKERS`: Backpressure and concurrency controls.
- `MAX_IMAGE_SIZE_MB`, `MAX_VIDEO_SIZE_MB`, `MAX_DOCUMENT_SIZE_MB`: File-size limits before download or processing.
- `DOWNLOAD_TIMEOUT_SECONDS`, `PROCESSING_TIMEOUT_SECONDS`: Prevent stuck downloads or model/video processing from blocking workers indefinitely.
- `PER_USER_RATE_LIMIT`, `PER_CHAT_RATE_LIMIT`, `GLOBAL_RATE_LIMIT`: Spam protection.
- `BROADCAST_DELAY_SECONDS`: Delay between broadcast sends to reduce Telegram flood risk.

## Local File Test

You can run detection against a local image or video without starting the bot:

```bash
python3 main.py /path/to/media.jpg
```
