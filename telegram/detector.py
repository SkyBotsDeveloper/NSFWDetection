import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from telegram.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectionResult:
    status: str
    confidence: float = 0.0
    label: str = ""
    reason: str = ""
    frames_checked: int = 0

    @property
    def is_nsfw(self) -> bool:
        return self.status == "nsfw"


class NsfwDetector:
    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._torch = None
        self._id2label: Dict[int, str] = {}
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    def detect_file(self, path: Path, media_kind: str) -> DetectionResult:
        try:
            if media_kind == "image":
                return self.detect_image(path)
            if media_kind == "gif":
                return self.detect_gif(path)
            if media_kind == "video":
                return self.detect_video(path)
            return DetectionResult("skipped", reason=f"unsupported_media_kind:{media_kind}")
        except ImportError as exc:
            logger.exception("NSFW detector dependency is missing")
            return DetectionResult("error", reason=f"model_dependency_missing:{exc}")
        except Exception as exc:
            logger.exception("NSFW detection failed for %s", path)
            return DetectionResult("error", reason=str(exc))

    def detect_image(self, path: Path) -> DetectionResult:
        from PIL import Image, ImageOps

        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            is_nsfw, confidence, label = self._predict(image)
        return DetectionResult(
            "nsfw" if is_nsfw else "safe",
            confidence=confidence,
            label=label,
            frames_checked=1,
        )

    def detect_gif(self, path: Path) -> DetectionResult:
        from PIL import Image, ImageOps, ImageSequence

        with Image.open(path) as image:
            frame_indexes = self._sample_gif_indexes(getattr(image, "n_frames", 1))
            best_confidence = 0.0
            best_label = ""
            frames_checked = 0

            for index, frame in enumerate(ImageSequence.Iterator(image)):
                if index not in frame_indexes:
                    continue
                frame = ImageOps.exif_transpose(frame).convert("RGB")
                is_nsfw, confidence, label = self._predict(frame)
                frames_checked += 1

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_label = label

                if is_nsfw:
                    return DetectionResult(
                        "nsfw",
                        confidence=confidence,
                        label=label,
                        frames_checked=frames_checked,
                    )

            if frames_checked == 0:
                return DetectionResult("skipped", reason="gif_decode_failed")

            return DetectionResult(
                "safe",
                confidence=best_confidence,
                label=best_label,
                frames_checked=frames_checked,
            )

    def detect_video(self, path: Path) -> DetectionResult:
        import cv2
        from PIL import Image

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return DetectionResult("error", reason="video_open_failed")

        try:
            positions = self._sample_positions(cap)
            if not positions:
                return DetectionResult("skipped", reason="no_video_frames")

            best_confidence = 0.0
            best_label = ""
            frames_checked = 0

            for position in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, position)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                is_nsfw, confidence, label = self._predict(image)
                frames_checked += 1

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_label = label

                if is_nsfw:
                    return DetectionResult(
                        "nsfw",
                        confidence=confidence,
                        label=label,
                        frames_checked=frames_checked,
                    )

            if frames_checked == 0:
                return DetectionResult("skipped", reason="video_decode_failed")

            return DetectionResult(
                "safe",
                confidence=best_confidence,
                label=best_label,
                frames_checked=frames_checked,
            )
        finally:
            cap.release()

    def _sample_positions(self, cap: Any) -> List[int]:
        import cv2

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        max_frames = settings.max_video_frames

        if frame_count <= 0:
            return [0]

        positions = {0, frame_count // 2, frame_count - 1}
        if frame_count <= max_frames:
            return self._cap_positions(sorted(positions), frame_count, max_frames)

        if max_frames > 3:
            positions.update(self._evenly_spaced_indexes(frame_count, max_frames))
        if fps > 0:
            step = max(1, int(fps * settings.video_frame_interval_seconds))
            positions.update(range(0, frame_count, step))

        return self._cap_positions(sorted(positions), frame_count, max_frames)

    def _sample_gif_indexes(self, frame_count: int) -> set[int]:
        frame_count = max(1, int(frame_count or 1))
        max_frames = min(settings.max_video_frames, frame_count)
        if max_frames == 1:
            return {frame_count // 2}
        if max_frames == 2:
            return {0, frame_count - 1}

        positions = {0, frame_count // 2, frame_count - 1}
        if max_frames > 3:
            positions.update(self._evenly_spaced_indexes(frame_count, max_frames))
        return set(self._cap_positions(sorted(positions), frame_count, max_frames))

    def _evenly_spaced_indexes(self, frame_count: int, max_frames: int) -> List[int]:
        frame_count = max(1, int(frame_count or 1))
        max_frames = min(max(1, max_frames), frame_count)
        if max_frames == 1:
            return [frame_count // 2]
        stride = (frame_count - 1) / float(max_frames - 1)
        return [int(round(i * stride)) for i in range(max_frames)]

    def _cap_positions(self, positions: List[int], frame_count: int, max_frames: int) -> List[int]:
        bounded = sorted(set(max(0, min(frame_count - 1, pos)) for pos in positions))
        if len(bounded) <= max_frames:
            return bounded
        selected = self._evenly_spaced_indexes(len(bounded), max_frames)
        return [bounded[index] for index in selected]

    def _predict(self, image: Any) -> Tuple[bool, float, str]:
        self._ensure_loaded()
        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None

        with self._predict_lock:
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {
                key: value.to(self._device)
                for key, value in inputs.items()
                if hasattr(value, "to")
            }
            with self._torch.inference_mode():
                outputs = self._model(**inputs)
                probabilities = self._torch.softmax(outputs.logits, dim=-1)[0].detach().cpu()

        scores = [
            (self._id2label.get(index, str(index)), float(probabilities[index]))
            for index in range(len(probabilities))
        ]
        nsfw_score = 0.0
        nsfw_labels = []
        for label, score in scores:
            if self._is_nsfw_label(label):
                nsfw_score += score
                nsfw_labels.append(label)

        top_label, top_score = max(scores, key=lambda item: item[1])
        if not nsfw_labels and len(scores) == 2 and self._id2label.get(0) in {"normal", "safe", "sfw"}:
            nsfw_labels = [self._id2label.get(1, "nsfw")]
            nsfw_score = scores[1][1]

        if nsfw_labels:
            confidence = nsfw_score
            label = "+".join(nsfw_labels)
        else:
            confidence = top_score
            label = top_label

        return confidence >= settings.nsfw_threshold and bool(nsfw_labels), confidence, label

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        with self._load_lock:
            if self._model is not None and self._processor is not None:
                return

            import torch
            from transformers import AutoModelForImageClassification, ViTImageProcessor

            if settings.torch_num_threads:
                torch.set_num_threads(settings.torch_num_threads)

            self._torch = torch
            self._device = self._select_device()
            logger.info("Loading NSFW model %s on %s", settings.model_name, self._device)
            processor = ViTImageProcessor.from_pretrained(settings.model_name)
            model = AutoModelForImageClassification.from_pretrained(settings.model_name)
            model.eval()
            model.to(self._device)

            self._processor = processor
            self._model = model
            self._id2label = {
                int(key): str(value).lower()
                for key, value in getattr(model.config, "id2label", {}).items()
            }
            logger.info("NSFW model loaded with labels: %s", self._id2label)

    def _select_device(self) -> str:
        requested = settings.torch_device
        if requested == "auto":
            return "cuda" if self._torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not self._torch.cuda.is_available():
            logger.warning("TORCH_DEVICE=%s requested but CUDA is unavailable; using CPU", requested)
            return "cpu"
        return requested

    @staticmethod
    def _is_nsfw_label(label: str) -> bool:
        normalized = label.lower().replace("-", "_").replace(" ", "_")
        safe_labels = {"safe", "sfw", "normal", "neutral", "not_nsfw"}
        if normalized in safe_labels:
            return False
        return any(
            token in normalized
            for token in ("nsfw", "porn", "hentai", "sexy", "explicit", "nude", "sexual")
        )


detector = NsfwDetector()
