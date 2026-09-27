import os
import uuid
from datetime import datetime
from io import BytesIO
from typing import Optional, Tuple

from dotenv import load_dotenv
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import ValidationError
from PIL import Image, ImageFile, UnidentifiedImageError
from sqlalchemy.orm import Session

# Mirror the same hardening in vision_service so server/gateway agree on decodes.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# AVIF / HEIC support is provided by pillow-heif (optional). Register the openers
# when the package is available so that Safari / iOS photos uploaded as AVIF with
# a .png extension are still decoded correctly.
try:
    import pillow_heif  # type: ignore[import-not-found]
    pillow_heif.register_heif_opener()
    _HAS_HEIF: bool = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_HEIF = False

from app.database import get_db
from app.models import TriageSubmission, User
from app.schemas import (
    JobAccepted,
    TriageQueueResponse,
    TriageQueueItem,
    TriageStatus,
    ReviewSubmissionRequest,
)

load_dotenv()

router = APIRouter(prefix="/api/v1/triage", tags=["triage"])

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

SUPPORTED_IMAGE_MIMES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "image/avif",       # Safari default for "take photo" uploads
    "image/heic",       # iOS camera native format
    "image/heif",       # HEIF container used by HEIC/AVIF
    "image/x-heic",
}
SUPPORTED_AUDIO_MIMES = {
    "audio/webm", "audio/wav", "audio/wave", "audio/x-wav", "audio/ogg",
    "audio/mpeg", "audio/mp3", "audio/mp4", "audio/x-m4a", "audio/aac", "audio/flac",
}


def _mime_prefix(content_type: Optional[str]) -> str:
    """Strip charset / codec parameters so "audio/webm;codecs=opus" matches "audio/webm"."""
    if not content_type:
        return ""
    return content_type.split(";", 1)[0].strip().lower()



def _safe_save(file: UploadFile, suffix: str) -> str:
    safe_name = f"{uuid.uuid4()}{suffix}"
    dest = os.path.join(UPLOAD_DIR, safe_name)
    with open(dest, "wb") as f:
        f.write(file.file.read())
    return dest


def _decode_uploaded_image_bytes(raw: bytes) -> Optional[Image.Image]:
    """Open an in-memory image with header-sniffing retries across Pillow formats.
    Returns PIL Image (RGB) on success, or None if no plugin can decode the bytes.
    """
    if not raw or len(raw) < 16:
        return None
    attempts_list = [
        ("auto",   None),
        ("JPEG",   "JPEG"),
        ("PNG",    "PNG"),
        ("WEBP",   "WEBP"),
        ("BMP",    "BMP"),
        ("GIF",    "GIF"),
        ("TIFF",   "TIFF"),
    ]
    if _HAS_HEIF:
        # pillow-heif registers a single "HEIF" plugin key that covers both HEIC and
        # AVIF container types. Requesting the literal non-registered "AVIF" raises
        # KeyError in Pillow's registry, so we only add HEIF to the explicit list.
        attempts_list.append(("HEIF", "HEIF"))
    attempts: Tuple[Tuple[str, Optional[str]], ...] = tuple(attempts_list)
    last: Optional[str] = None
    for (_label, fmt) in attempts:
        try:
            buf = BytesIO(raw)
            if fmt is None:
                img = Image.open(buf)
            else:
                img = Image.open(buf, formats=[fmt])
            img.load()
            return img.convert("RGB")
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            continue

    # pillow-heif direct API fallback. The registered Pillow plugin handles
    # .heic/.heif files but AVIF containers (ftypavif) are NOT matched by the
    # plugin on pillow-heif 0.16+. Bypass the plugin by using open_heif()
    # directly, which parses HEIC/HEIF/AVIF uniformly.
    if _HAS_HEIF:
        try:
            import pillow_heif  # type: ignore[import-not-found]
            # Try by in-memory BytesIO first (we already have the bytes)
            try:
                hf = pillow_heif.open_heif(BytesIO(raw))
                if hasattr(hf, "to_pillow"):
                    return hf.to_pillow().convert("RGB")
            except Exception:
                # write to a real tempfile, try open_heif by path
                import tempfile
                ext = ".heif"
                head = bytes(raw[:16]).lower()
                if b"ftypavif" in head or b"avis" in head:
                    ext = ".avif"
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
                    tf.write(raw)
                    tmp_path = tf.name
                try:
                    hf2 = pillow_heif.open_heif(tmp_path)
                    if hasattr(hf2, "to_pillow"):
                        return hf2.to_pillow().convert("RGB")
                finally:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
        except Exception as e:
            last = f"pillow_heif.open_heif: {type(e).__name__}: {e}"
    return None


def _save_and_normalize_image(file: UploadFile, orig_suffix: str) -> Tuple[str, Optional[str]]:
    """Save an uploaded image to UPLOAD_DIR as a canonical JPEG regardless of
    the client-provided format / extension claim. This is Edge Case 5 hardened
    against UnidentifiedImageError caused by mismatched extension headers
    (JPEG bytes uploaded with .png extension, truncated files, etc.).

    Returns (absolute_path, error_detail_or_None).
    """
    try:
        raw = file.file.read()
    except Exception as e:
        return "", f"Could not read upload bytes. ({type(e).__name__})"

    img = _decode_uploaded_image_bytes(raw)
    if img is None:
        # Last-ditch: write the raw bytes to disk and try Image.open on the file
        # path (a few Pillow plugins work better via real files than BytesIO).
        fallback_suffix = orig_suffix.lower() or ".img"
        fallback_name = f"{uuid.uuid4()}{fallback_suffix}"
        fallback_path = os.path.join(UPLOAD_DIR, fallback_name)
        with open(fallback_path, "wb") as f:
            f.write(raw)
        try:
            with Image.open(fallback_path) as im2:
                im2.load()
                img = im2.convert("RGB")
        except Exception as e:
            try:
                os.remove(fallback_path)
            except OSError:
                pass
            return "", f"Uploaded image is corrupted or unsupported. ({type(e).__name__})"

    # Always write a .jpg on disk. If the client claimed .png / .webp / .bmp /
    # etc., we overwrite the file with a standards-compliant JPEG so the
    # clinician dashboard preview, the vision service, and any downstream
    # consumer never see the original (possibly mismatched) bytes.
    canonical_name = f"{uuid.uuid4()}.jpg"
    canonical_path = os.path.join(UPLOAD_DIR, canonical_name)
    try:
        img.save(canonical_path, format="JPEG", quality=92, optimize=True)
    except Exception as e:
        return "", f"Could not save normalized image. ({type(e).__name__})"
    return canonical_path, None


@router.post(
    "/submit",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobAccepted,
    responses={
        400: {"description": "Missing patient_id or invalid input"},
        404: {"description": "Patient not found"},
        415: {"description": "Unsupported media type for audio/image"},
        503: {"description": "Task queue (Redis) unavailable"},
    },
)
def submit_triage(
    patient_id: str = Form(...),
    audio: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    manual_transcript: Optional[str] = Form(
        None,
        description=(
            "Edge Case 1 (microphone permission denied) fallback: the patient's "
            "typed symptom description, used verbatim as raw_transcript instead "
            "of running Whisper on an audio file."
        ),
    ),
    db: Session = Depends(get_db),
):
    """Accepts patient audio + image, writes PENDING row, enqueues Celery task.

    Returns 202 Accepted with {job_id}.
    """
    if not patient_id:
        raise HTTPException(status_code=400, detail="patient_id is required")
    try:
        p_uuid = uuid.UUID(patient_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="patient_id is not a valid UUID")

    patient = db.query(User).filter(User.id == p_uuid).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    manual_transcript = (manual_transcript or "").strip() or None
    if audio is None and image is None and manual_transcript is None:
        raise HTTPException(
            status_code=400,
            detail="At least one of audio, manual_transcript, or image must be provided",
        )

    # Validate and persist audio
    audio_path: Optional[str] = None
    if audio:
        mime = _mime_prefix(audio.content_type)
        if mime and mime not in SUPPORTED_AUDIO_MIMES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=(
                    "One of the uploaded files is not supported. "
                    f"(Unsupported audio type: {audio.content_type})"
                ),
            )
        _, ext = os.path.splitext(audio.filename or "audio.webm")
        ext = ext.lower() or ".webm"
        audio_path = _safe_save(audio, ext)
        audio.file.seek(0)

    # Validate and persist image (Edge Case 5: normalize + canonical JPEG save)
    image_path: Optional[str] = None
    if image:
        mime = _mime_prefix(image.content_type)
        if mime and mime not in SUPPORTED_IMAGE_MIMES:
            if audio_path:
                try:
                    os.remove(audio_path)
                except OSError:
                    pass
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=(
                    "One of the uploaded files is not supported. "
                    f"(Unsupported image type: {image.content_type})"
                ),
            )
        _, ext = os.path.splitext(image.filename or "image.jpg")
        ext = ext.lower() or ".jpg"
        image_path, err = _save_and_normalize_image(image, ext)
        if err is not None:
            if audio_path:
                try:
                    os.remove(audio_path)
                except OSError:
                    pass
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=err,
            )

    # Insert PENDING row. Persist paths as absolute so Celery worker CWD never
    # affects file existence check or the /uploads static mount resolution.
    job_id = uuid.uuid4()
    submission = TriageSubmission(
        id=job_id,
        patient_id=patient.id,
        audio_file_path=os.path.abspath(audio_path) if audio_path else None,
        image_file_path=os.path.abspath(image_path) if image_path else None,
        # EC1 fallback: pre-populate raw_transcript directly so the worker
        # skips Whisper for this submission (see tasks.py step 2).
        raw_transcript=manual_transcript,
        status="PENDING",
    )
    db.add(submission)
    db.commit()
    db.refresh(submission)

    # Enqueue Celery task. Edge Case 4: if Redis is down -> 503
    try:
        from celery_worker import celery_app  # lazy import to avoid circulars
        celery_app.send_task("tasks.process_triage", args=[str(job_id)])
    except Exception as e:  # redis.ConnectionError / kombu exceptions
        # Best-effort revert file artifacts
        for p in (audio_path, image_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        db.delete(submission)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Systems currently under heavy load. Please try again in 5 minutes.",
        ) from e

    return JobAccepted(job_id=job_id, status="PENDING")


@router.get("/status/{job_id}", response_model=TriageStatus)
def get_status(job_id: str, db: Session = Depends(get_db)):
    try:
        s_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="job_id is not a valid UUID")

    sub = db.query(TriageSubmission).filter(TriageSubmission.id == s_uuid).first()
    if not sub:
        raise HTTPException(status_code=404, detail="Job not found")

    return TriageStatus(
        job_id=sub.id,
        status=sub.status,
        data=sub.final_summary,
        error=sub.error_message,
        created_at=sub.created_at,
        updated_at=sub.updated_at,
        audio_file_path=sub.audio_file_path,
        image_file_path=sub.image_file_path,
        raw_transcript=sub.raw_transcript,
        vision_analysis=sub.vision_analysis,
        clinician_reviewed_at=sub.clinician_reviewed_at,
        clinician_notes=sub.clinician_notes,
        clinician_override_severity=sub.clinician_override_severity,
    )


@router.get("/queue", response_model=TriageQueueResponse)
def get_queue(
    statuses: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """Fetches recent COMPLETED/FAILED/GUARDRAIL_ABORTED/REVIEWED/ESCALATED/PROCESSING submissions.

    Use ?statuses=COMPLETED,FAILED to filter (comma-separated). Defaults to
    COMPLETED + GUARDRAIL_ABORTED + REVIEWED + ESCALATED + FAILED as per the spec
    (the clinician queue must show quarantined / already-reviewed rows too).
    """
    if statuses:
        allowed = {s.strip().upper() for s in statuses.split(",") if s.strip()}
    else:
        allowed = {"COMPLETED", "GUARDRAIL_ABORTED", "REVIEWED", "ESCALATED", "FAILED"}

    q = (
        db.query(TriageSubmission)
        .filter(TriageSubmission.status.in_(allowed))
        .order_by(TriageSubmission.created_at.desc())
        .limit(limit)
    )

    items: list[TriageQueueItem] = []
    for sub in q.all():
        patient = db.query(User).filter(User.id == sub.patient_id).first()
        if not patient:
            continue
        items.append(
            TriageQueueItem(
                id=sub.id,
                patient_id=patient.id,
                patient_name=f"{patient.first_name} {patient.last_name}",
                patient_dob=patient.dob,
                status=sub.status,
                error_message=sub.error_message,
                created_at=sub.created_at or datetime.utcnow(),
                updated_at=sub.updated_at or sub.created_at or datetime.utcnow(),
                final_summary=sub.final_summary,
                audio_file_path=sub.audio_file_path,
                image_file_path=sub.image_file_path,
                raw_transcript=sub.raw_transcript,
                vision_analysis=sub.vision_analysis,
                clinician_reviewed_at=sub.clinician_reviewed_at,
                clinician_notes=sub.clinician_notes,
                clinician_override_severity=sub.clinician_override_severity,
            )
        )

    return TriageQueueResponse(items=items, total=len(items))


# ======================================================== HITL CLINICIAN REVIEW
@router.patch(
    "/review/{job_id}",
    status_code=status.HTTP_200_OK,
    response_model=TriageStatus,
    responses={
        400: {"description": "clinician_notes too short (override/escalate actions require >= 10 chars)"},
        404: {"description": "Job not found"},
        422: {"description": "Invalid review payload schema"},
    },
)
def review_submission(
    job_id: str,
    payload: ReviewSubmissionRequest = Body(...),
    db: Session = Depends(get_db),
):
    """Clinician audit action. Per report L488-492:

    - ALWAYS writes clinician_reviewed_at = current UTC clock_time.
    - If override_severity is set OR action == 'override' or 'escalate':
        the server-side Pydantic model_validator enforces clinician_notes
        len >= 10 chars. A 422 is returned if this fails, with a clean error
        string.
    - action == 'review'     -> status REVIEWED, no severity change
    - action == 'override'   -> status REVIEWED, write clinician_override_severity
    - action == 'escalate'   -> status ESCALATED, override_severity forced to 'severe'
                                (clinician_notes mandatory >=10 chars)
    """
    try:
        s_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="job_id is not a valid UUID")

    sub = db.query(TriageSubmission).filter(TriageSubmission.id == s_uuid).first()
    if not sub:
        raise HTTPException(status_code=404, detail="Job not found")

    # The schema validator already enforces len(notes) >= 10 when an override/
    # escalate action is requested, but re-raise any ValidationError cleanly
    # as HTTP 400 so the frontend can show the exact constraint text.
    try:
        ReviewSubmissionRequest.model_validate(payload.model_dump())
    except ValidationError as e:
        first_msg = "; ".join(err.get("msg", str(e)) for err in e.errors())
        raise HTTPException(status_code=400, detail=first_msg)

    # Clock timestamp: written for ALL three action types per report L491.
    clock = datetime.utcnow()
    sub.clinician_reviewed_at = clock
    if payload.clinician_notes is not None:
        notes_stripped = payload.clinician_notes.strip()
        if notes_stripped:
            sub.clinician_notes = notes_stripped

    action = (payload.action or "review").lower()
    if action == "escalate":
        # Escalate = explicit Severity = SEVERE impression, plus mandatory note
        # already enforced by ReviewSubmissionRequest._mandatory_comment_if_overridden
        sub.status = "ESCALATED"
        sub.clinician_override_severity = "severe"
    elif action == "override" or payload.override_severity is not None:
        sub.status = "REVIEWED"
        if payload.override_severity is not None:
            sub.clinician_override_severity = payload.override_severity
    else:
        # Plain Mark Reviewed: no override, just audit timestamp + optional note
        sub.status = "REVIEWED"

    sub.updated_at = clock
    db.commit()
    db.refresh(sub)

    return TriageStatus(
        job_id=sub.id,
        status=sub.status,
        data=sub.final_summary,
        error=sub.error_message,
        created_at=sub.created_at,
        updated_at=sub.updated_at,
        audio_file_path=sub.audio_file_path,
        image_file_path=sub.image_file_path,
        raw_transcript=sub.raw_transcript,
        vision_analysis=sub.vision_analysis,
        clinician_reviewed_at=sub.clinician_reviewed_at,
        clinician_notes=sub.clinician_notes,
        clinician_override_severity=sub.clinician_override_severity,
    )
