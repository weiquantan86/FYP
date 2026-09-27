import uuid
from datetime import datetime, date
from sqlalchemy import Column, String, Date, DateTime, ForeignKey, Text, Index
from sqlalchemy.dialects.postgresql import UUID as PgUUID, JSONB
from sqlalchemy.orm import relationship
from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    dob = Column(Date, nullable=False)
    email = Column(String, unique=True, nullable=True)
    role = Column(String, default="patient")  # patient, clinician, admin
    created_at = Column(DateTime, default=datetime.utcnow)

    triage_submissions = relationship(
        "TriageSubmission",
        back_populates="patient",
        cascade="all, delete-orphan",
    )


class Clinician(Base):
    __tablename__ = "clinicians"

    id = Column(PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    role = Column(String, nullable=False)  # Triage Nurse, General Practitioner, ...
    created_at = Column(DateTime, default=datetime.utcnow)


class TriageSubmission(Base):
    __tablename__ = "triage_submissions"

    id = Column(PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    patient_id = Column(
        PgUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    audio_file_path = Column(String, nullable=True)
    image_file_path = Column(String, nullable=True)
    raw_transcript = Column(Text, nullable=True)
    vision_analysis = Column(Text, nullable=True)
    final_summary = Column(JSONB, nullable=True)
    status = Column(String, default="PENDING", index=True)  # PENDING, PROCESSING, COMPLETED, GUARDRAIL_ABORTED, REVIEWED, ESCALATED, FAILED
    error_message = Column(Text, nullable=True)
    # ----- HITL (human-in-the-loop) clinician audit columns -----
    # Written when the clinician presses "Mark Reviewed" on the dashboard.
    # The report L488-492 explicitly requires these three columns with a
    # mandatory >=10-char comment validator on any override_severity action.
    clinician_reviewed_at = Column(DateTime, nullable=True)
    clinician_notes = Column(Text, nullable=True)
    clinician_override_severity = Column(String, nullable=True)  # none / mild / moderate / severe / unknown
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    patient = relationship("User", back_populates="triage_submissions")


Index("ix_triage_submissions_status_created", TriageSubmission.status, TriageSubmission.created_at.desc())
