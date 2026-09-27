import os
import sys
import uuid
from datetime import datetime

from celery import Task

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

from celery_worker import celery_app  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models import TriageSubmission  # noqa: E402
from app.schemas import ClinicalBrief  # noqa: E402
from ml.audio_service import AudioTranscriptionService  # noqa: E402
from ml.vision_service import VisionAnalysisService  # noqa: E402
from ml.llm_synthesizer import GuardrailAborted, LLMSynthesizer  # noqa: E402


class TriageTask(Task):
    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        job_id = args[0] if args else None
        if not job_id:
            return
        try:
            job_uuid = uuid.UUID(job_id)
        except (ValueError, TypeError, AttributeError):
            return
        db = SessionLocal()
        try:
            sub = db.query(TriageSubmission).filter(TriageSubmission.id == job_uuid).first()
            if sub and sub.status not in ("COMPLETED",):
                sub.status = "FAILED"
                sub.error_message = (
                    f"Permanent worker failure (retries exhausted): "
                    f"{type(exc).__name__}: {exc}"
                )
                sub.updated_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()


def _edge_case5_fallback_summary(raw_transcript: str, vision_analysis: str, reason: str) -> dict:
    """Edge Case 5 (adversarial diagnostic request / stubborn medical
    hallucinations). Abort LLM synthesis, write raw data to JSONB with a
    system note banner for clinicians."""
    return {
        "_fallback_reason": (
            "[AI Synthesis Aborted due to Safety Guardrail - Raw Data Provided]"
        ),
        "_guardrail_error": reason,
        "chief_complaint": None,
        "history_of_present_illness": None,
        "visual_assessment": None,
        "raw_transcript": raw_transcript,
        "raw_vision_analysis": vision_analysis,
        "symptoms": [],
        "patient_concerns": [],
        "reported_duration": {"raw_text": None, "days": None, "unit": None},
        "requires_manual_intake": True,
        "synthesizer_version": ClinicalBrief.model_fields["synthesizer_version"].default,
    }


@celery_app.task(
    base=TriageTask,
    bind=True,
    name="tasks.process_triage",
    autoretry_for=(RuntimeError, ConnectionError, OSError),
    retry_backoff=60,
    retry_backoff_max=300,
    max_retries=3,
    acks_late=True,
)
def process_triage(self, job_id: str) -> dict:
    try:
        job_uuid = uuid.UUID(job_id)
    except (ValueError, TypeError, AttributeError):
        return {"job_id": job_id, "error": "invalid_uuid"}

    db = SessionLocal()
    try:
        sub = db.query(TriageSubmission).filter(TriageSubmission.id == job_uuid).first()
        if not sub:
            return {"job_id": job_id, "error": "submission_not_found"}
        if sub.status == "COMPLETED":
            return {"job_id": job_id, "status": sub.status}

        # -------------------------------------------------------------- 1.
        sub.status = "PROCESSING"
        sub.error_message = None
        sub.updated_at = datetime.utcnow()
        db.commit()

        # -------------------------------------------------------------- 2.
        # EC1 (mic denied) fallback: /api/submit may have pre-populated
        # raw_transcript with the patient's typed text when no audio file
        # was recorded. Only run Whisper when there is actually an audio
        # file to transcribe, so that manual text isn't clobbered by the
        # "audio not provided" fallback string.
        if sub.audio_file_path:
            audio_svc = AudioTranscriptionService.get_instance()
            transcript_result = audio_svc.transcribe(sub.audio_file_path)
            sub.raw_transcript = transcript_result.text
            db.commit()
        elif not sub.raw_transcript:
            audio_svc = AudioTranscriptionService.get_instance()
            sub.raw_transcript = audio_svc.transcribe(None).text
            db.commit()

        # -------------------------------------------------------------- 3.
        vision_svc = VisionAnalysisService.get_instance()
        vision_result = vision_svc.analyze(sub.image_file_path)
        sub.vision_analysis = vision_result.text
        db.commit()

        # -------------------------------------------------------------- 4.
        llm = LLMSynthesizer.get_instance()
        try:
            synthesis = llm.synthesize(sub.raw_transcript, sub.vision_analysis)
            brief = synthesis.brief
            sub.final_summary = brief.model_dump(mode="json")
            sub.status = "COMPLETED"
            sub.error_message = None
            sub.updated_at = datetime.utcnow()
            db.commit()
            return {
                "job_id": job_id,
                "status": sub.status,
                "attempts": synthesis.attempts,
                "guardrail_passed": synthesis.guardrail_passed,
                "summary": sub.final_summary,
            }
        except GuardrailAborted as e:
            # Edge Case 5: Guardrail validation loop exhausted -> raw fallback
            reason = str(e)
            sub.final_summary = _edge_case5_fallback_summary(
                raw_transcript=sub.raw_transcript or "",
                vision_analysis=sub.vision_analysis or "",
                reason=reason,
            )
            sub.status = "GUARDRAIL_ABORTED"
            sub.error_message = (
                "[AI Synthesis Aborted due to Safety Guardrail - "
                "Raw Data Provided]"
            )
            sub.updated_at = datetime.utcnow()
            db.commit()
            return {
                "job_id": job_id,
                "status": sub.status,
                "guardrail_aborted": True,
                "reason": reason,
                "summary": sub.final_summary,
            }

    except RuntimeError as e:
        msg = str(e).lower()
        if "cuda" in msg or "out of memory" in msg or "oom" in msg:
            try:
                db.rollback()
            except Exception:
                pass
            raise self.retry(countdown=60, exc=e, max_retries=3)
        raise
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        raise
    finally:
        db.close()
