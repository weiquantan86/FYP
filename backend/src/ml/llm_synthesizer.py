"""Constrained LLM clinical synthesizer.

Architecture:
    1. Primary pass: Instructor library wrapping Ollama /chat with a
       response_model=ClinicalBrief. Instructor uses Pydantic to physically
       force JSON structural output (schema-constrained decoding).
    2. Secondary semantic evaluator: fast same-model LLM pass which
       audits the produced ClinicalBrief for any diagnostic claim that
       leaked past the schema. If found, we raise ValidationError and
       retry the primary pass up to MAX_RETRIES.
    3. MAX_RETRIES exhausted -> raise GuardrailAborted, which the caller
       converts to the Edge Case 5 raw-data fallback JSON.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from pydantic import ValidationError

load_dotenv()

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_CURRENT_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from app.schemas import ClinicalBrief  # noqa: E402

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "180"))  # seconds

SYSTEM_SCRIBE_PROMPT = """You are a clinical scribe. Your ONLY job is to turn
patient-reported information into a structured summary.
You are NOT a diagnostician. You must NEVER:
  - Provide a diagnosis, differential diagnosis, or prognosis
  - Name specific diseases, conditions, infections, tumours, fractures,
    or any ICD-derived label (including words like infection, pneumonia,
    melanoma, diabetes, cancer, tumour, etc.)
  - Suggest treatment, medication, or procedures
You MUST:
  - Paraphrase the patient's words exactly as they described them. Use
    'patient reports X' or 'patient expresses concern about Y'.
  - For patient-concerned diagnoses they THINK they have: write those as
    patient_concerns verbatim, not as facts.
  - If the transcript or vision analysis is a generic placeholder, still
    populate the fields faithfully rather than inventing content.
"""


SYSTEM_GUARDRAIL_PROMPT = """You are a clinical safety auditor scoring
against the Clinical Error Severity Scale (CESS). The two levels you are
screening for, quoted verbatim from that scale, are:

  Severity 4 - "Potential for inappropriate treatment"
    Example: naming a medication or a dose.
  Severity 5 - "Potential for serious and imminent harm"
    Example: any diagnostic statement, e.g. naming a malignancy from a
    photograph, or saying no doctor is needed.

Read the structured clinical brief below. You must respond ONLY with the
word 'PASS' if the brief contains NO Severity-4 or Severity-5 content as
defined above - that is, NO diagnostic assertions, NO named diseases, NO
differentials, NO named medications or doses, and NO prognostic statements.
If you detect ANY such content phrased as a fact (not merely listed as a
patient's own concern in patient_concerns), respond ONLY with:
FAIL: <one-sentence reason>
Do not output anything else. Do not wrap in JSON.
"""


class GuardrailAborted(RuntimeError):
    """Raised when MAX_RETRIES is exhausted and Edge Case 5 fallback is needed."""


@dataclass
class SynthesisResult:
    brief: ClinicalBrief
    attempts: int
    guardrail_passed: bool
    abort_reason: Optional[str] = None


class LLMSynthesizer:
    """Singleton wrapper around Instructor + Ollama with guardrail loop."""

    _instance: Optional["LLMSynthesizer"] = None

    def __init__(
        self,
        ollama_base_url: str = OLLAMA_BASE_URL,
        ollama_model: str = OLLAMA_MODEL,
        max_retries: int = MAX_RETRIES,
        timeout: float = LLM_TIMEOUT,
    ):
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_model = ollama_model
        self.max_retries = max(1, max_retries)
        self.timeout = timeout

        # Lazy-load Instructor client (import failures must not block booting
        # a worker that only needs audio/vision stages).
        self._instructor_client = None

    @classmethod
    def get_instance(cls) -> "LLMSynthesizer":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------ utilities
    def _get_instructor_client(self):
        if self._instructor_client is not None:
            return self._instructor_client
        try:
            import instructor
            from openai import OpenAI
        except Exception as e:  # pragma: no cover - import guard
            raise RuntimeError(
                "Instructor or openai package not installed. Cannot run "
                "constrained synthesis. Install requirements.txt."
            ) from e

        # Instructor supports Ollama via an OpenAI-compatible adapter pointed
        # at Ollama's /v1 endpoint.
        client = instructor.from_openai(
            OpenAI(
                base_url=f"{self.ollama_base_url}/v1",
                api_key="ollama",  # value is ignored by Ollama but required
                timeout=self.timeout,
            ),
            mode=instructor.Mode.JSON,
        )
        self._instructor_client = client
        return client

    @staticmethod
    def _build_user_prompt(
        transcript: str,
        vision: str,
        retry_feedback: Optional[list[str]] = None,
    ) -> str:
        feedback_block = ""
        if retry_feedback:
            joined = "\n".join(f"- {reason}" for reason in retry_feedback)
            feedback_block = (
                "\n\nA SAFETY AUDITOR REJECTED YOUR PREVIOUS ATTEMPT(S) FOR:\n"
                f"{joined}\n"
                "Rewrite the brief so that none of the flagged wording or "
                "diagnostic assertions remain. Move any patient-stated worry "
                "into patient_concerns instead of stating it as fact.\n"
            )
        return (
            "Below is the raw patient input to summarize.\n\n"
            "AUDIO TRANSCRIPT (Whisper output, may contain placeholder text):\n"
            "-----------------------------------------------------------------\n"
            f"{(transcript or '').strip()}\n"
            "-----------------------------------------------------------------\n\n"
            "VISION ANALYSIS (VLM output, may contain placeholder text):\n"
            "-----------------------------------------------------------------\n"
            f"{(vision or '').strip()}\n"
            "-----------------------------------------------------------------\n"
            f"{feedback_block}\n"
            "Populate the ClinicalBrief schema using only the text above. "
            "If the patient states they THINK they have a named disease, "
            "put that exact wording in patient_concerns as their worry, "
            "NEVER in chief_complaint, symptom names, or history as a fact."
        )

    # ------------------------------------------------------------ auditor
    def _semantic_guardrail(self, brief: ClinicalBrief) -> tuple[bool, str]:
        """Secondary evaluator pass. Returns (passed, reason_string)."""
        import httpx

        brief_json = brief.model_dump_json(indent=2)
        prompt = (
            f"{SYSTEM_GUARDRAIL_PROMPT}\n\n"
            f"CLINICAL BRIEF TO AUDIT:\n```\n{brief_json}\n```"
        )

        try:
            with httpx.Client(timeout=max(60.0, self.timeout / 2)) as client:
                resp = client.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.3, "num_ctx": 8192},
                    },
                )
                resp.raise_for_status()
                text = (resp.json().get("response") or "").strip()
        except Exception as e:
            # Fail closed: an unreachable auditor is NOT a pass. The
            # Pydantic structural + blocklist validators catch known
            # phrasing, but they are not a substitute for the independent
            # audit this pass exists to provide, so treat the outage as a
            # rejection — it will retry, and quarantine via GuardrailAborted
            # if retries are exhausted, exactly like a real FAIL verdict.
            print(f"[LLMSynthesizer] Guardrail evaluator unavailable: {e}")
            return False, f"guardrail_unavailable: {type(e).__name__}: {e}"

        if text.upper().startswith("PASS"):
            return True, ""
        # Anything else -> FAIL. Extract a one-line reason.
        reason = re.sub(r"\s+", " ", text).strip()
        if not reason:
            reason = "semantic evaluator returned FAIL without reason"
        return False, reason

    # ------------------------------------------------------------ primary
    def synthesize(
        self,
        transcript: str,
        vision_analysis: str,
        max_retries: Optional[int] = None,
    ) -> SynthesisResult:
        """Primary pass (Instructor constrained) + secondary guardrail loop.

        Raises GuardrailAborted when retries are exhausted so the caller
        can apply Edge Case 5 fallback JSON.
        """
        max_retries = max_retries if max_retries is not None else self.max_retries
        client = self._get_instructor_client()
        last_error: Optional[str] = None
        retry_feedback: list[str] = []

        for attempt in range(1, max_retries + 1):
            print(f"[LLMSynthesizer] synthesis attempt {attempt}/{max_retries}")
            # ---- 1. Constrained pass -------------------------------------
            # Rebuilt every attempt so a prior auditor rejection (or
            # validation failure) actually steers the retry, instead of
            # resending an identical prompt the model already failed on.
            user_prompt = self._build_user_prompt(
                transcript or "", vision_analysis or "", retry_feedback
            )
            try:
                brief: ClinicalBrief = client.chat.completions.create(
                    model=self.ollama_model,
                    response_model=ClinicalBrief,
                    messages=[
                        {"role": "system", "content": SYSTEM_SCRIBE_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_retries=1,
                )
            except Exception as e:
                last_error = f"Instructor generation error: {type(e).__name__}: {e}"
                print(f"[LLMSynthesizer] {last_error}")
                retry_feedback.append(f"Generation/schema error: {last_error}")
                # Pydantic ValidationError raised by Instructor is a retriable
                # JSON-shape failure. RuntimeError / httpx / Ollama down is also
                # retriable.
                continue

            # ---- 2. Apply Pydantic safety blocklist (re-validation) ------
            try:
                # Re-instantiate through model_validate so field_validators
                # and model_validators run again on the Instructor output.
                data = brief.model_dump()
                brief = ClinicalBrief.model_validate(data)
            except ValidationError as e:
                last_error = f"Pydantic post-validation error: {e}"
                print(f"[LLMSynthesizer] {last_error}")
                retry_feedback.append(last_error)
                continue

            # ---- 3. Semantic evaluator -----------------------------------
            passed, reason = self._semantic_guardrail(brief)
            if passed:
                if reason:
                    brief.safety_flag = (brief.safety_flag or "") + f" | {reason}"
                return SynthesisResult(
                    brief=brief,
                    attempts=attempt,
                    guardrail_passed=True,
                )
            last_error = f"Guardrail rejected: {reason}"
            print(f"[LLMSynthesizer] {last_error}")
            brief.safety_flag = reason
            retry_feedback.append(reason)
            # Fall through and retry with the feedback now folded into the
            # next attempt's prompt.

        # ---- Edge Case 5: exhausted -------------------------------------
        raise GuardrailAborted(
            f"Safety guardrail and/or Pydantic validation failed after "
            f"{max_retries} attempts. Last error: {last_error}"
        )


# ----------------------------------------------------------------- CLI test
if __name__ == "__main__":  # pragma: no cover
    # Test 2 from the spec: transcript that explicitly says "stage 4 melanoma"
    # Expected output strips diagnosis from structured fields, preserves only
    # as patient_concern.
    synth = LLMSynthesizer.get_instance()
    try:
        out = synth.synthesize(
            transcript=(
                "I have this weird mole on my back that's growing and bleeding. "
                "I think I have stage 4 melanoma. It's been there for 3 weeks."
            ),
            vision_analysis=(
                "Close up view of a pigmented skin lesion, dark brown with "
                "irregular borders on a patient's upper back."
            ),
        )
        print(out.brief.model_dump_json(indent=2))
    except GuardrailAborted as e:
        print(f"ABORTED (Edge Case 5 fallback needed): {e}")
