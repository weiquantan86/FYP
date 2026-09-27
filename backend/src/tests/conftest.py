"""Shared fixtures for the T1-T12 integration test suite (PPR Appendix C).

Points DATABASE_URL at a dedicated `triage_test_db` (created alongside the
real `triage_db` on the same Postgres container) and UPLOAD_DIR at a temp
directory, BEFORE any app module is imported, so these tests never touch
production data or the real uploads/ folder.

Model calls (Whisper / Ollama) are mocked by default via the
`mock_pipeline_services` fixture, per Table 5's "mocked model server"
requirement. A small number of tests (T3, T6) deliberately call the real,
already-running Ollama server for a genuine live spot-check; those are
marked `live_model` and documented as such at the point of use.
"""

import os
import sys
import uuid

os.environ["DATABASE_URL"] = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://triage_user:triage_password@localhost:5432/triage_test_db",
)
os.environ["UPLOAD_DIR"] = os.getenv("TEST_UPLOAD_DIR", "/tmp/triage_test_uploads")
os.makedirs(os.environ["UPLOAD_DIR"], exist_ok=True)

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_TESTS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import pytest
from fastapi.testclient import TestClient

import main as main_module  # noqa: E402
import tasks as tasks_module  # noqa: E402
import celery_worker  # noqa: E402
from app.database import Base, engine, SessionLocal  # noqa: E402
from app.models import TriageSubmission  # noqa: E402
from ml.audio_service import TranscriptionResult  # noqa: E402
from ml.vision_service import VisionResult  # noqa: E402
from ml.llm_synthesizer import SynthesisResult  # noqa: E402
from app.schemas import ClinicalBrief, SymptomEntry, ReportedDuration  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _schema():
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture()
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client():
    return TestClient(main_module.app)


@pytest.fixture()
def patient(client):
    resp = client.post(
        "/users",
        json={
            "first_name": "Test",
            "last_name": f"Patient-{uuid.uuid4().hex[:8]}",
            "dob": "1990-01-01",
            "role": "patient",
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def make_stub_brief(**overrides) -> ClinicalBrief:
    defaults = dict(
        chief_complaint="Patient reports an itchy rash on the lower leg for several days.",
        history_of_present_illness="Patient states the rash appeared gradually and has been itchy since onset.",
        visual_assessment="Erythematous patch with irregular borders on the lower leg, no visible bleeding.",
        symptoms=[SymptomEntry(name="itchy rash", body_location="lower leg")],
        reported_duration=ReportedDuration(raw_text="a few days", days=4, unit="days"),
        patient_concerns=[],
    )
    defaults.update(overrides)
    return ClinicalBrief(**defaults)


@pytest.fixture()
def mock_pipeline_services(monkeypatch):
    """Patches the three model services tasks.py calls, so the Celery task
    body can be exercised directly (no live Whisper/Ollama, no worker
    process) while still running every line of real orchestration code."""

    class _StubAudio:
        def transcribe(self, path):
            return TranscriptionResult(
                text="Patient reports an itchy rash on the lower leg for four days.",
                avg_confidence=-0.1,
                fallback_used=False,
            )

    class _StubVision:
        def analyze(self, path):
            return VisionResult(
                text="Erythematous patch with irregular borders, no bleeding observed.",
                source="vision_model",
                model="stub-vision",
            )

    class _StubLLM:
        def synthesize(self, transcript, vision_analysis, max_retries=None):
            return SynthesisResult(
                brief=make_stub_brief(),
                attempts=1,
                guardrail_passed=True,
            )

    monkeypatch.setattr(
        tasks_module.AudioTranscriptionService, "get_instance", classmethod(lambda cls: _StubAudio())
    )
    monkeypatch.setattr(
        tasks_module.VisionAnalysisService, "get_instance", classmethod(lambda cls: _StubVision())
    )
    monkeypatch.setattr(
        tasks_module.LLMSynthesizer, "get_instance", classmethod(lambda cls: _StubLLM())
    )
    return {
        "audio": _StubAudio,
        "vision": _StubVision,
        "llm": _StubLLM,
    }


def run_task_eagerly(job_id: str):
    """Invoke the Celery task body directly, in-process, bypassing the
    broker/worker (no worker process is running in the test environment)."""
    return tasks_module.process_triage(job_id)


def get_submission(db_session, job_id: str) -> TriageSubmission:
    return (
        db_session.query(TriageSubmission)
        .filter(TriageSubmission.id == uuid.UUID(job_id))
        .first()
    )
