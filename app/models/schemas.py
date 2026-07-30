from typing import Annotated, Literal, Optional, Sequence, TypedDict
import operator

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class MedicinePrescription(BaseModel):
    clinical_reasoning_scratchpad: str = Field(
        description="Your internal reasoning step. You MUST explicitly state: 1. The affected body part (e.g., Eyes, Oral Mucosa, Muscle, Stomach). 2. The strictly required Route of Administration (e.g., Ophthalmic eye drops, Oral mucosal gel, Topical skin cream, Oral tablet)."
    )
    search_keyword: str = Field(
        description="The exact standard NZ commercial brand name of the single recommended medicine. CRITICAL: This brand MUST perfectly match the Route of Administration identified in your scratchpad. (1-2 words maximum). It MUST be a proper noun product name. CRITICAL: Strictly NEVER output descriptive phrases, category terms, or punctuation."
    )
    clinical_advice: str = Field(
        description="Friendly, empathetic, and clinically professional explanation, proper usage instructions, and safety warnings for the patient based in New Zealand."
    )


class TriageDecision(BaseModel):
    status: Literal["INQUIRING", "MINOR", "MODERATE", "URGENT"] = Field(
        description=(
            "The triage routing status. Follow these strict clinical rules:\n"
            "- 'INQUIRING': MUST select this if the user describes vague, broad, or newly introduced symptoms where duration or severity is unknown.\n"
            "- 'MINOR': Select this ONLY if the symptom is already highly specific and clearly minor, or if the user explicitly requests medication recommendations.\n"
            "- 'MODERATE': Select this if the user explicitly asks to find a clinic, GP, doctor, or routine check-ups.\n"
            "- 'URGENT': Select this if there are red-flag fatal symptoms."
        )
    )
    reply_or_reason: str = Field(
        description="Follow-up questions if INQUIRING."
    )


# ===========================
# Structured Conversation Memory
# ===========================
class PatientContext(BaseModel):
    primary_symptom: Optional[str] = None
    duration: Optional[str] = None
    severity: Optional[str] = None
    location: Optional[str] = None
    red_flags: list[str] = Field(default_factory=list)


class PatientState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

    decision: Optional[TriageDecision]

    # Structured short-term memory
    patient_context: PatientContext