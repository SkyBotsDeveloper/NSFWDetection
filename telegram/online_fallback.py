import logging
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from telegram.config import settings
from telegram.detector import DetectionResult
from telegram.metrics import metrics

logger = logging.getLogger(__name__)


SUPPORTED_FALLBACK_KINDS = {"image", "gif", "video"}


async def detect_with_online_fallback(path: Path, media_kind: str, local_result: DetectionResult) -> Optional[DetectionResult]:
    if not _should_call_fallback(path, media_kind, local_result):
        return None

    provider = settings.online_fallback_provider
    if provider != "naas":
        logger.warning("Unknown ONLINE_FALLBACK_PROVIDER=%s", provider)
        return None

    try:
        metrics.inc("online_fallback_calls")
        return await _call_naas(path)
    except Exception as exc:
        metrics.inc("online_fallback_errors")
        logger.warning("Online NSFW fallback failed path=%s error=%s", path, exc)
        return None


def _should_call_fallback(path: Path, media_kind: str, local_result: DetectionResult) -> bool:
    if not settings.online_fallback_enabled:
        return False
    if media_kind not in SUPPORTED_FALLBACK_KINDS:
        return False
    if local_result.status != "safe":
        return False
    if local_result.confidence < settings.online_fallback_min_local_nsfw_score:
        return False
    try:
        if path.stat().st_size > settings.online_fallback_max_size_bytes:
            metrics.inc("online_fallback_skipped_size")
            return False
    except OSError:
        return False
    return True


async def _call_naas(path: Path) -> Optional[DetectionResult]:
    url = _naas_url()
    with path.open("rb") as file:
        files = {"image": (path.name, file, "application/octet-stream")}
        async with httpx.AsyncClient(timeout=settings.online_fallback_timeout_seconds) as http:
            response = await http.post(url, files=files)
            response.raise_for_status()
            payload = response.json()

    return _parse_naas_response(payload)


def _naas_url() -> str:
    url = settings.online_fallback_url
    if not settings.online_fallback_fast_mode:
        return url

    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.setdefault("fast", "1")
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _parse_naas_response(payload: dict[str, Any]) -> Optional[DetectionResult]:
    status = str(payload.get("status", "")).upper()
    if status == "NOQUOTA":
        metrics.inc("online_fallback_noquota")
        logger.warning("Online NSFW fallback quota exhausted")
        return None
    if status != "OK":
        reason = str(payload.get("reason") or f"online_status:{status or 'unknown'}")
        raise RuntimeError(reason)

    data = payload.get("data") or {}
    confidence = _normalize_confidence(data.get("confidence", 0.0))
    classification = str(data.get("classification") or "").lower()
    nsfw = bool(data.get("nsfw")) or bool(data.get("porn")) or classification in {"nsfw", "porn"}
    nsfw = nsfw and confidence >= settings.online_fallback_nsfw_threshold
    label = f"online:{classification or 'nsfw'}"

    if nsfw:
        metrics.inc("online_fallback_nsfw")
    else:
        metrics.inc("online_fallback_safe")

    return DetectionResult(
        "nsfw" if nsfw else "safe",
        confidence=confidence,
        label=label,
        reason="online_fallback:naas",
        frames_checked=0,
    )


def _normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if confidence > 1.0:
        confidence /= 100.0
    return max(0.0, min(1.0, confidence))
