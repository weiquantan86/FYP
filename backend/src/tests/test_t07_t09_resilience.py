"""T7 (EC3), T8 (EC4), T9 (Appendix C)."""

import uuid
from datetime import datetime, timedelta

from tests.conftest import get_submission
from tests.helpers import make_wav_bytes
import celery_worker
import tasks as tasks_module
from app.models import TriageSubmission


# --------------------------------------------------------------------- T7
def test_t7_model_server_killed_mid_inference_ec3(client, patient, db_session):
    """CESS ceiling <= 4: row not lost, error banner persisted, retriable.

    Simulates the worker process dying mid-inference by invoking the
    production on_failure handler directly against a PROCESSING row -- this
    is exactly what Celery calls when a task's retries are exhausted or the
    worker is killed and acks_late redelivers past task_time_limit.
    """
    resp = client.post(
        "/api/v1/triage/submit",
        data={"patient_id": patient["id"]},
        files={"audio": ("recording.wav", make_wav_bytes(), "audio/wav")},
    )
    job_id = resp.json()["job_id"]

    sub = get_submission(db_session, job_id)
    sub.status = "PROCESSING"
    db_session.commit()

    handler = tasks_module.TriageTask()
    handler.on_failure(
        exc=RuntimeError("simulated worker death"),
        task_id="fake-task-id",
        args=[job_id],
        kwargs={},
        einfo=None,
    )

    db_session.expire_all()
    sub = get_submission(db_session, job_id)
    assert sub is not None, "Row was lost -- acks_late/on_failure contract violated"
    assert sub.status == "FAILED"
    assert sub.error_message and "simulated worker death" in sub.error_message
    # Retriable: status isn't a terminal COMPLETED/GUARDRAIL_ABORTED state,
    # so a resubmitted task would still run process_triage() to completion.
    assert sub.status not in ("COMPLETED", "GUARDRAIL_ABORTED")


# --------------------------------------------------------------------- T8
def test_t8_broker_unavailable_ec4(client, patient, db_session, monkeypatch):
    """CESS ceiling <= 1: 503 returned, rollback executed, zero orphan rows."""
    before_count = (
        db_session.query(TriageSubmission)
        .filter(TriageSubmission.patient_id == uuid.UUID(patient["id"]))
        .count()
    )

    def _raise_broker_down(*args, **kwargs):
        raise ConnectionError("simulated broker unavailable")

    monkeypatch.setattr(celery_worker.celery_app, "send_task", _raise_broker_down)

    resp = client.post(
        "/api/v1/triage/submit",
        data={"patient_id": patient["id"]},
        files={"audio": ("recording.wav", make_wav_bytes(), "audio/wav")},
    )
    assert resp.status_code == 503, resp.text

    db_session.expire_all()
    after_count = (
        db_session.query(TriageSubmission)
        .filter(TriageSubmission.patient_id == uuid.UUID(patient["id"]))
        .count()
    )
    assert after_count == before_count, "Orphan row left behind after broker failure"


# --------------------------------------------------------------------- T9
def test_t9_mark_reviewed_writes_timestamp(client, patient, db_session):
    """CESS ceiling <= 0: server-side timestamp within tolerance; status
    transitions PENDING -> REVIEWED with no severity override."""
    resp = client.post(
        "/api/v1/triage/submit",
        data={"patient_id": patient["id"]},
        files={"audio": ("recording.wav", make_wav_bytes(), "audio/wav")},
    )
    job_id = resp.json()["job_id"]

    before = datetime.utcnow()
    review_resp = client.patch(
        f"/api/v1/triage/review/{job_id}",
        json={"action": "review"},
    )
    after = datetime.utcnow()

    assert review_resp.status_code == 200, review_resp.text
    body = review_resp.json()
    assert body["status"] == "REVIEWED"
    reviewed_at = datetime.fromisoformat(body["clinician_reviewed_at"])
    assert before - timedelta(seconds=5) <= reviewed_at <= after + timedelta(seconds=5)
    assert body["clinician_override_severity"] is None
