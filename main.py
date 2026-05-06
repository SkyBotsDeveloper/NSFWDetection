import argparse
from pathlib import Path

from telegram.detector import detector


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
GIF_EXTENSIONS = {".gif"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NSFW detection on a local media file.")
    parser.add_argument("path", type=Path, help="Image or video path")
    args = parser.parse_args()

    path = args.path.expanduser().resolve()
    if not path.exists():
        print(f"File not found: {path}")
        return 2

    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        media_kind = "image"
    elif suffix in GIF_EXTENSIONS:
        media_kind = "gif"
    elif suffix in VIDEO_EXTENSIONS:
        media_kind = "video"
    else:
        print(f"Unsupported file type: {suffix or 'unknown'}")
        return 2

    result = detector.detect_file(path, media_kind)
    print(
        f"status={result.status} confidence={result.confidence:.4f} "
        f"label={result.label or 'unknown'} frames={result.frames_checked} reason={result.reason or '-'}"
    )
    return 0 if result.status in {"safe", "nsfw"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
