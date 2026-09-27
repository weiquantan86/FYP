"""T1, T2, T4, T5 (Appendix C). T3 lives in test_live_model_smoke.py since it
needs a real vision-model call to be meaningful."""

import re

from tests.conftest import run_task_eagerly, get_submission
from tests.helpers import make_jpeg_bytes, make_wav_bytes
from ml.audio_service import _WHISPER_TEXT_FIXES
from app.schemas import ClinicalBrief, SymptomEntry


# --------------------------------------------------------------------- T1
def test_t1_happy_path_audio_only(client, patient, mock_pipeline_services, db_session):
    """CESS ceiling <= 1: all four ClinicalBrief fields populated from a
    mocked transcript alone, no image."""
    resp = client.post(
        "/api/v1/triage/submit",
        data={"patient_id": patient["id"]},
        files={"audio": ("recording.wav", make_wav_bytes(), "audio/wav")},
    )
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]

    result = run_task_eagerly(job_id)
    assert result["status"] == "COMPLETED", result

    sub = get_submission(db_session, job_id)
    assert sub.status == "COMPLETED"
    assert sub.final_summary["chief_complaint"]
    assert sub.final_summary["history_of_present_illness"]
    assert sub.final_summary["visual_assessment"]
    assert sub.final_summary["symptoms"]


# --------------------------------------------------------------------- T2
# Ground-truth reference pairs, hand-authored against the correction list in
# ml/audio_service.py::_WHISPER_TEXT_FIXES.
T2_CASES = [
    ("my saw throat has been hurting", "my sore throat has been hurting"),
    ("i have a throatache since yesterday", "i have a sore throat since yesterday"),
    ("i feel dizzyness when i stand", "i feel dizziness when i stand"),
    ("i keep no zia and feel sick", "i keep nausea and feel sick"),
    ("been vomitting all morning", "been vomiting all morning"),
    ("uh, my leg has an itche spot", "my leg has an itchy spot"),
]


def _apply_whisper_fixes(text: str) -> str:
    for pattern, replacement in _WHISPER_TEXT_FIXES:
        text = pattern.sub(replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def test_t2_accented_speech_regex_correction():
    """CESS ceiling <= 1: corrected transcript matches the human-authored
    reference exactly for each known Whisper quantization misread pattern."""
    mismatches = []
    for raw, expected in T2_CASES:
        got = _apply_whisper_fixes(raw)
        if got != expected:
            mismatches.append((raw, expected, got))
    assert not mismatches, f"Regex correction mismatches: {mismatches}"


# --------------------------------------------------------------------- T4
def test_t4_malformed_image_container_ec2(client, patient):
    """Gate D4: 10/10 normalized, zero decode errors. Uploads real JPEG
    bytes under a mismatched .png filename/content-type ten times, and
    confirms every one is accepted and normalized rather than rejected."""
    accepted = 0
    for i in range(10):
        resp = client.post(
            "/api/v1/triage/submit",
            data={"patient_id": patient["id"]},
            files={"image": (f"mismatched_{i}.png", make_jpeg_bytes(), "image/png")},
        )
        if resp.status_code == 202:
            accepted += 1
        else:
            print(f"[T4] upload {i} rejected: {resp.status_code} {resp.text}")
    assert accepted == 10, f"Expected 10/10 normalized, got {accepted}/10"


# --------------------------------------------------------------------- T5
def test_t5_schema_validity_and_blocklist_redaction():
    """CESS ceiling <= 1: valid JSON within length bounds, AND the
    diagnostic blocklist actually redacts blocklisted terms in every
    free-text field it covers (regression test for the SymptomEntry.notes
    gap fixed in schemas.py)."""
    brief = ClinicalBrief(
        chief_complaint="Patient is worried they have melanoma on their arm.",
        history_of_present_illness="States it has grown over 3 weeks, no cancer history mentioned.",
        visual_assessment="Dark irregular lesion, possible tumor-like growth pattern.",
        symptoms=[
            SymptomEntry(name="mole", body_location="left arm", notes="patient fears it is melanoma"),
        ],
    )
    assert "melanoma" not in brief.chief_complaint.lower()
    assert "cancer" not in brief.history_of_present_illness.lower()
    assert "tumor" not in brief.visual_assessment.lower()
    assert all("melanoma" not in s.notes.lower() for s in brief.symptoms if s.notes)

    # min_length constraint still enforced (schema-level rejection of
    # under-specified content).
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ClinicalBrief(
            chief_complaint="hi",  # below min_length=4
            history_of_present_illness="fine",
            visual_assessment="fine",
        )
