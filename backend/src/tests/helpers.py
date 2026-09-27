"""Synthetic file generators shared across the T1-T12 tests.

These produce real, valid (or deliberately malformed) bytes in-memory so
tests never depend on external datasets for structural/mechanism checks.
Tests that need real clinical content (T3's attribute F1, T6's full
HAM10000-sourced adversarial set, T10's accent corpus) are scoped
separately and documented as needing that real data.
"""

from io import BytesIO

from PIL import Image


def make_jpeg_bytes(size=(64, 64), color=(200, 120, 100)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def make_png_bytes(size=(64, 64), color=(90, 180, 90)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def make_wav_bytes() -> bytes:
    """A minimal valid (silent) WAV file. Content is irrelevant for tests
    that mock AudioTranscriptionService; this just needs to be accepted as
    a well-formed audio/wav upload."""
    import struct

    sample_rate = 16000
    n_samples = 1600  # 0.1s of silence
    data = b"\x00\x00" * n_samples
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    fmt = b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    data_chunk = b"data" + struct.pack("<I", len(data)) + data
    return header + fmt + data_chunk
