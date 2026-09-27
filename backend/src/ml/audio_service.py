import os
import re
import subprocess
from dataclasses import dataclass
from typing import Optional

import whisper


WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
MIN_TRANSCRIPT_CONFIDENCE = float(os.getenv("WHISPER_MIN_CONFIDENCE", "0.3"))
FALLBACK_TEXT = "Audio unintelligible or not provided."

# Common Whisper-quantization misreads on base.en/base when phonemes match but the
# beam search lands on a shorter, higher-prior token. Keys are lowercase words or
# word tuples; values are the replacement strings. All substitutions are case-aware.
_WHISPER_TEXT_FIXES = [
    (re.compile(r"\b(e\.?g\.?|eg)\b(?=\s+(rash|skin|spot|lesion|bump|patch|area|mark|blister))", re.IGNORECASE), "itchy"),
    (re.compile(r"\bitche\b"), "itchy"),
    (re.compile(r"\bitchi\b"), "itchy"),
    (re.compile(r"\bmy saw (throat|chest|arm|leg|knee|back|head|ear|eye|stomach|tooth|tongue|shoulder|wrist|ankle|neck)\b", re.IGNORECASE), r"my sore \1"),
    (re.compile(r"\bthroat ?a ?che\b", re.IGNORECASE), "sore throat"),
    (re.compile(r"\bshort ?(?:ness)? ?(?:of)? ?breath\b", re.IGNORECASE), "shortness of breath"),
    (re.compile(r"\bdizzy ?ness\b"), "dizziness"),
    (re.compile(r"\bno ?zia\b", re.IGNORECASE), "nausea"),
    (re.compile(r"\bno ?see ?ah\b", re.IGNORECASE), "nausea"),
    (re.compile(r"\bno ?see\b", re.IGNORECASE), "nausea"),
    (re.compile(r"\bvom ?it ?ting\b"), "vomiting"),
    (re.compile(r"\bvo ?mit ?ting\b", re.IGNORECASE), "vomiting"),
    (re.compile(r"^\s*(?:uh|um|erm|er)[,.!?\s]*", re.IGNORECASE), ""),
    (re.compile(r"\s+([.,!?])"), r"\1"),
]


@dataclass
class TranscriptionResult:
    text: str
    avg_confidence: Optional[float]
    fallback_used: bool = False


class AudioTranscriptionService:
    """Singleton Whisper transcriber. One GPU model load per Celery worker process."""

    _instance: Optional["AudioTranscriptionService"] = None

    def __init__(self, model_name: str = WHISPER_MODEL_NAME):
        # Load on construct
        self.model_name = model_name
        self._model = whisper.load_model(model_name)

    @classmethod
    def get_instance(cls) -> "AudioTranscriptionService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _has_ffmpeg() -> bool:
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    @staticmethod
    def _avg_segment_confidence(result_segments: list) -> Optional[float]:
        if not result_segments:
            return None
        total = 0.0
        count = 0
        for seg in result_segments:
            # Whisper exposes no_logprob_sum + compression_ratio etc. Derive a
            # confidence proxy from `avg_logprob` (nats) -> exp().
            logprob = seg.get("avg_logprob")
            if logprob is None:
                continue
            total += float(logprob)
            count += 1
        if count == 0:
            return None
        return float(total / count)

    # ------------------------------------------------------------------ api
    def transcribe(self, audio_path: Optional[str]) -> TranscriptionResult:
        """Transcribe audio with Edge Case 2 handling.

        - Missing file / empty string / no audio provided -> FALLBACK_TEXT
        - ffmpeg missing -> FALLBACK_TEXT + fallback_used=True
        - Whisper returns empty string or low confidence (< 0.3) -> FALLBACK_TEXT
        """
        if not audio_path or not isinstance(audio_path, str) or not os.path.exists(audio_path):
            return TranscriptionResult(text=FALLBACK_TEXT, avg_confidence=None, fallback_used=True)

        if not self._has_ffmpeg():
            return TranscriptionResult(text=FALLBACK_TEXT, avg_confidence=None, fallback_used=True)

        try:
            raw = self._model.transcribe(
                audio_path,
                language="english",
                task="transcribe",
                fp16=False,  # safe default for MPS/CPU-only workers
            )
        except Exception as e:
            print(f"[AudioService] Whisper failed: {type(e).__name__}: {e}")
            return TranscriptionResult(text=FALLBACK_TEXT, avg_confidence=None, fallback_used=True)

        text = (raw.get("text") or "").strip()
        segments = raw.get("segments") or []
        avg_logprob = self._avg_segment_confidence(segments)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Apply common Whisper quantization corrections. Keep this ordered list
        # conservative — only correct high-precision patterns that cannot
        # accidentally rewrite legitimate words.
        for pattern, replacement in _WHISPER_TEXT_FIXES:
            try:
                new_text = pattern.sub(replacement, text)
            except (re.error, TypeError):
                continue
            text = new_text

        # Once more collapse whitespace after regex substitutions
        text = re.sub(r"\s+", " ", text).strip()
        if text and text[-1] not in ".!?":
            text = text + "."

        # Edge Case 2: empty transcript
        if not text:
            return TranscriptionResult(text=FALLBACK_TEXT, avg_confidence=avg_logprob, fallback_used=True)

        # Edge Case 2: low-confidence output (derived from avg_logprob)
        if avg_logprob is not None:
            # Typical Whisper avg_logprob is negative; use exp to map to [0,1]
            confidence = 2.718281828459045 ** avg_logprob
            if confidence < MIN_TRANSCRIPT_CONFIDENCE:
                print(
                    f"[AudioService] Low-confidence transcript (conf={confidence:.3f}), "
                    f"using fallback. Raw: {text[:100]!r}"
                )
                return TranscriptionResult(
                    text=FALLBACK_TEXT, avg_confidence=avg_logprob, fallback_used=True
                )

        return TranscriptionResult(text=text, avg_confidence=avg_logprob, fallback_used=False)
