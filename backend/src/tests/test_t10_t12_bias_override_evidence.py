"""T10, T11, T12 (Appendix C)."""

import os

import pytest

from tests.conftest import run_task_eagerly, get_submission
from tests.helpers import make_jpeg_bytes, make_wav_bytes

# --------------------------------------------------------------------- T10
# Gate D3 / test T10 needs a real native-vs-non-native accented audio corpus
# with human reference transcripts (PPR Appendix A: Common Voice, CC0-1.0,
# filtered by self-reported accent metadata). No such corpus is checked into
# the repo -- fabricating a differential number here would violate the PPR's
# own instruction ("do not carry forward a figure you cannot re-derive").
# This test is a runnable scaffold: point ACCENT_CORPUS_DIR at a directory
# laid out as
#   <dir>/native/<id>.wav + <dir>/native/<id>.txt   (reference transcript)
#   <dir>/non_native/<id>.wav + <dir>/non_native/<id>.txt
# and it will compute the real differential against gate D3's <= 25pp
# threshold. Until that corpus exists, it skips with an explicit reason
# rather than asserting a number nobody measured.
ACCENT_CORPUS_DIR = os.getenv("ACCENT_CORPUS_DIR", "")


def _word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    # Standard Levenshtein distance over words.
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(ref_words)][len(hyp_words)] / max(1, len(ref_words))


@pytest.mark.skipif(
    not ACCENT_CORPUS_DIR or not os.path.isdir(ACCENT_CORPUS_DIR),
    reason=(
        "T10 requires a real accent-stratified audio corpus with reference "
        "transcripts (PPR Appendix A). Set ACCENT_CORPUS_DIR to run this for "
        "real; see the docstring above for the expected layout."
    ),
)
def test_t10_accent_differential_bias_check():
    """Gate D3: non-native minus native transcription error <= 25 pp."""
    import whisper

    model = whisper.load_model(os.getenv("WHISPER_MODEL", "base"))

    def _avg_wer(subdir: str) -> float:
        folder = os.path.join(ACCENT_CORPUS_DIR, subdir)
        rates = []
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith(".wav"):
                continue
            stem = fname[: -len(".wav")]
            ref_path = os.path.join(folder, f"{stem}.txt")
            if not os.path.exists(ref_path):
                continue
            with open(ref_path) as f:
                reference = f.read().strip()
            result = model.transcribe(os.path.join(folder, fname), language="english")
            rates.append(_word_error_rate(reference, result["text"]))
        assert rates, f"No paired .wav/.txt files found under {folder}"
        return sum(rates) / len(rates)

    native_wer = _avg_wer("native")
    non_native_wer = _avg_wer("non_native")
    differential_pp = (non_native_wer - native_wer) * 100
    assert differential_pp <= 25, (
        f"Accent differential {differential_pp:.1f}pp exceeds gate D3's 25pp threshold"
    )


# --------------------------------------------------------------------- T11
def test_t11_override_requires_mandatory_comment(client, patient):
    """CESS ceiling <= 0: under-length comment rejected; valid comment persisted."""
    resp = client.post(
        "/api/v1/triage/submit",
        data={"patient_id": patient["id"]},
        files={"audio": ("recording.wav", make_wav_bytes(), "audio/wav")},
    )
    job_id = resp.json()["job_id"]

    short_resp = client.patch(
        f"/api/v1/triage/review/{job_id}",
        json={"action": "override", "override_severity": "severe", "clinician_notes": "too short"[:9]},
    )
    assert short_resp.status_code in (400, 422), short_resp.text

    valid_resp = client.patch(
        f"/api/v1/triage/review/{job_id}",
        json={
            "action": "override",
            "override_severity": "severe",
            "clinician_notes": "Clinician disagrees with severity, escalating for review.",
        },
    )
    assert valid_resp.status_code == 200, valid_resp.text
    body = valid_resp.json()
    assert body["clinician_override_severity"] == "severe"
    assert body["clinician_notes"].startswith("Clinician disagrees")


# --------------------------------------------------------------------- T12
def test_t12_raw_evidence_cells_all_render(client, patient, mock_pipeline_services, db_session):
    """CESS ceiling <= 1: all four raw-evidence cells non-empty after a
    completed submission with both audio and image provided."""
    resp = client.post(
        "/api/v1/triage/submit",
        data={"patient_id": patient["id"]},
        files={
            "audio": ("recording.wav", make_wav_bytes(), "audio/wav"),
            "image": ("photo.jpg", make_jpeg_bytes(), "image/jpeg"),
        },
    )
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]
    run_task_eagerly(job_id)

    status_resp = client.get(f"/api/v1/triage/status/{job_id}")
    body = status_resp.json()
    assert body["audio_file_path"], "audio_file_path cell empty"
    assert body["image_file_path"], "image_file_path cell empty"
    assert body["raw_transcript"], "raw_transcript cell empty"
    assert body["vision_analysis"], "vision_analysis cell empty"
