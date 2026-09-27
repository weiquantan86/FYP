import uuid
from datetime import datetime, date
from enum import Enum
from typing import Any, Optional, List, Literal, ClassVar
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------- Users
class UserCreate(BaseModel):
    first_name: str
    last_name: str
    dob: date
    email: Optional[str] = None
    role: str = "patient"


class UserResponse(BaseModel):
    id: uuid.UUID
    first_name: str
    last_name: str
    dob: date
    email: Optional[str]
    role: str
    created_at: datetime

    class Config:
        from_attributes = True


# ------------------------------------------------------------ Triage Status
class JobAccepted(BaseModel):
    job_id: uuid.UUID
    status: str = "PENDING"
    message: str = "Triage job accepted and queued for processing."


# ------------------------------------------------------------------ HITL Review
class ReviewSubmissionRequest(BaseModel):
    """Clinician audit action payload. Report L491-492 requires a mandatory
    >=10-character comment whenever override_severity is set."""

    clinician_notes: Optional[str] = Field(
        None,
        description="Free-text clinician comment. Required (len >= 10 chars) if override_severity is non-null.",
    )
    override_severity: Optional[Literal["none", "mild", "moderate", "severe", "unknown"]] = Field(
        None,
        description="If set, overrides the LLM-derived severity impression; clinician_notes is then mandatory.",
    )
    action: Optional[Literal["review", "escalate", "override"]] = Field(
        "review",
        description="'review' = Mark Reviewed only, 'override' = Mark Reviewed + override_severity, 'escalate' = Mark Reviewed + severity='severe'",
    )

    @model_validator(mode="after")
    def _mandatory_comment_if_overridden(self):
        if self.override_severity is not None or self.action == "override" or self.action == "escalate":
            stripped = (self.clinician_notes or "").strip()
            if len(stripped) < 10:
                raise ValueError(
                    "clinician_notes must be at least 10 characters when "
                    "override_severity is set or action=override/escalate"
                )
        return self


class TriageStatus(BaseModel):
    job_id: uuid.UUID
    status: str
    data: Optional[Any] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    audio_file_path: Optional[str] = None
    image_file_path: Optional[str] = None
    raw_transcript: Optional[str] = None
    vision_analysis: Optional[str] = None
    # ----- HITL audit fields, surfaced to clinician dashboard -----
    clinician_reviewed_at: Optional[datetime] = None
    clinician_notes: Optional[str] = None
    clinician_override_severity: Optional[str] = None


class TriageQueueItem(BaseModel):
    id: uuid.UUID
    patient_id: uuid.UUID
    patient_name: str
    patient_dob: date
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    final_summary: Optional[Any] = None
    audio_file_path: Optional[str] = None
    image_file_path: Optional[str] = None
    raw_transcript: Optional[str] = None
    vision_analysis: Optional[str] = None
    # ----- HITL audit fields surfaced to queue -----
    clinician_reviewed_at: Optional[datetime] = None
    clinician_notes: Optional[str] = None
    clinician_override_severity: Optional[str] = None


class TriageQueueResponse(BaseModel):
    items: List[TriageQueueItem]
    total: int


# ============================================================ CLINICAL BRIEF
# These Pydantic models are used for Instructor/Outlines constrained decoding
# to physically prevent the LLM from emitting unstructured or diagnostic text.

# Shared across ClinicalBrief and SymptomEntry so every free-text field that
# reaches the clinician dashboard's structured column is redacted the same
# way, regardless of which model instantiates it.
_DIAGNOSTIC_BLOCKLIST: set = {
    "melanoma", "cancer", "carcinoma", "tumor", "tumour", "malignancy",
    "sarcoma", "lymphoma", "leukemia", "neoplasm", "metastasis",
    "diabetes", "hypertension", "pneumonia", "bronchitis", "asthma",
    "fracture", "broken bone", "infection", "sepsis", "bacterial",
    "viral", "hiv", "aids", "hepatitis", "tuberculosis", "stroke",
    "tia", "heart attack", "myocardial infarction", "angina",
    "aneurysm", "embolism", "thrombosis", "arthritis", "osteoarthritis",
    "rheumatoid", "lupus", "eczema", "psoriasis", "dermatitis",
    "shingles", "herpes", "impetigo", "cellulitis", "abscess",
    "ulcer", "gangrene", "necrosis", "glaucoma", "cataract",
    "kidney disease", "renal failure", "liver disease", "cirrhosis",
    "crohn", "ulcerative colitis", "ibs", "celiac",
    "parkinson", "alzheimer", "dementia", "epilepsy", "seizure",
    "concussion", "aneurysm",
    "stage 1", "stage 2", "stage 3", "stage 4",
    "icd-10", "icd10",
}


def _redact_diagnostic_terms(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    lowered = v.lower()
    hits = [w for w in _DIAGNOSTIC_BLOCKLIST if w in lowered]
    if not hits:
        return v
    sanitized = v
    for w in hits:
        sanitized = sanitized.replace(w, "***REDACTED***")
        sanitized = sanitized.replace(w.capitalize(), "***REDACTED***")
        sanitized = sanitized.replace(w.upper(), "***REDACTED***")
    return sanitized


class Severity(str, Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    UNKNOWN = "unknown"


class SymptomEntry(BaseModel):
    """A single reported symptom, strictly symptoms, no diagnoses."""
    name: str = Field(
        ...,
        min_length=2,
        description=(
            "Short plain-language name of the patient's reported symptom "
            "(e.g. 'headache', 'itchy rash', 'knee pain'). MUST NOT be a "
            "diagnosis, disease name, or condition name. MUST NOT include "
            "words like 'melanoma', 'cancer', 'diabetes', 'infection', "
            "'fracture', 'pneumonia', 'tumor' or any ICD-derived label."
        ),
    )
    severity: Severity = Severity.UNKNOWN
    body_location: Optional[str] = Field(
        None,
        description="Optional plain-language body location (e.g. 'left lower abdomen').",
    )
    notes: Optional[str] = Field(
        None,
        description="Free-text elaboration of the symptom as the patient described it.",
    )

    @field_validator("notes", "body_location", mode="before")
    @classmethod
    def _strip_blocklist_from_notes(cls, v: Any) -> Any:
        # Both fields are rendered verbatim on the clinician dashboard's
        # structured column, so they must go through the same redaction as
        # the ClinicalBrief text fields rather than being trusted as-is.
        return _redact_diagnostic_terms(v)


class ReportedDuration(BaseModel):
    raw_text: Optional[str] = Field(
        None,
        description="The duration as the patient said it, left as raw string for safety.",
    )
    days: Optional[int] = Field(
        None, ge=0,
        description="Optional integer duration parsed in days. Leave null if unclear.",
    )
    unit: Optional[Literal["hours", "days", "weeks", "months", "years"]] = None


class ClinicalBrief(BaseModel):
    """Clinician-facing summary produced by the constrained LLM.

    IMPORTANT SAFETY CONTRACT: This schema intentionally has NO field for a
    diagnosis, differential, or prognosis. Every free-text field is
    validated against a diagnostic keyword blocklist via field_validators.
    """

    chief_complaint: str = Field(
        ...,
        min_length=4,
        description=(
            "One-sentence lay-language summary of the main concern AS THE "
            "PATIENT STATED IT. Use wording like 'patient reports X' or "
            "'patient expresses concern about Y'. NEVER assert a diagnosis or "
            "named condition. NEVER use words like 'infection', 'tumor', "
            "'melanoma', 'diabetes', 'pneumonia', 'fracture', 'cancer'."
        ),
    )
    history_of_present_illness: str = Field(
        ...,
        description=(
            "Free-text narrative strictly quoting or paraphrasing the "
            "patient's description of onset, progression, relieving/worsening "
            "factors, associated symptoms, and prior self-treatment. "
            "NO physician interpretation or named diagnoses."
        ),
    )
    visual_assessment: str = Field(
        ...,
        description=(
            "Plain-language restatement of the Vision analysis, describing "
            "only observable features (colour, shape, texture, distribution, "
            "location, size, swelling, lesion type). NEVER a diagnosis, "
            "disease name, or condition."
        ),
    )
    symptoms: List[SymptomEntry] = Field(
        default_factory=list,
        description="Structured, enumerated symptoms (names only, no diagnoses).",
    )
    reported_duration: ReportedDuration = Field(
        default_factory=ReportedDuration,
    )

    @field_validator("reported_duration", mode="before")
    @classmethod
    def _default_duration_if_null(cls, v: Any) -> Any:
        # default_factory only fires when the key is OMITTED. Instructor's
        # constrained decoding sometimes has the model emit an explicit
        # `"reported_duration": null` when duration wasn't stated, which
        # otherwise raises a ValidationError unrelated to clinical safety
        # and burns one of the guardrail's limited retry attempts.
        return v if v is not None else ReportedDuration()
    patient_concerns: List[str] = Field(
        default_factory=list,
        description=(
            "Populate this ONLY if the transcript contains an explicit "
            "hedge phrase where the patient names a specific condition or "
            "diagnosis they fear it might be -- e.g. 'I think it might "
            "be...', 'I'm worried it's...', 'could it be...', 'I read "
            "online that it might be...'. Quote or closely paraphrase that "
            "exact statement. Describing a symptom on its own (e.g. 'I "
            "have a rash', 'it hurts', 'it's getting worse') is NOT a "
            "concern and must NOT be placed here -- that belongs in "
            "chief_complaint or history_of_present_illness instead. If the "
            "transcript contains no named-condition fear, this list MUST "
            "be an empty list. Do not invent, assume, or infer a worry the "
            "patient did not state, and do not reuse wording from these "
            "instructions. These are raw patient statements, never the "
            "system's own assessment."
        ),
    )
    synthesizer_version: str = Field(
        default="instructor-ollama-v1",
        description="Pipeline version identifier, do not modify.",
    )
    safety_flag: Optional[str] = Field(
        None,
        description=(
            "Reserved for the semantic guardrail evaluator. Populated with a "
            "reason string if the output needed correction or fallback."
        ),
    )

    # ------------------------------------------------------------------ safety
    _DIAGNOSTIC_BLOCKLIST: ClassVar[set] = _DIAGNOSTIC_BLOCKLIST

    @field_validator("chief_complaint", "history_of_present_illness", "visual_assessment", mode="before")
    @classmethod
    def _strip_blocklist_from_strings(cls, v: Any) -> Any:
        return _redact_diagnostic_terms(v)

    @model_validator(mode="after")
    def _symptom_names_blocklist(self):
        cleaned: list[SymptomEntry] = []
        for s in self.symptoms:
            nm = (s.name or "").lower()
            if any(w in nm for w in self._DIAGNOSTIC_BLOCKLIST):
                # The offending wording is redacted before it is stashed in
                # notes — SymptomEntry.notes also self-redacts on
                # construction, but do it explicitly here too since we are
                # deliberately relocating diagnostic text that must never
                # surface unredacted, from any source.
                original = _redact_diagnostic_terms(s.name)
                cleaned.append(
                    SymptomEntry(
                        name="patient-reported concern",
                        severity=s.severity,
                        body_location=s.body_location,
                        notes=original + (" " + s.notes if s.notes else ""),
                    )
                )
            else:
                cleaned.append(s)
        self.symptoms = cleaned
        return self
