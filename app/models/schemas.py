from typing import Annotated, Literal, Optional, Sequence, TypedDict
import operator

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class MedicineRecommendation(BaseModel):
    """Structured medicine result backed by one retrieved FAISS record."""

    recommendation_status: Literal[
        "RECOMMENDED",
        "INFORMATION_ONLY",
        "NO_RELIABLE_MATCH",
        "BLOCKED",
    ] = Field(description="Whether the available evidence supports the requested response.")
    source_medicine_name: str = Field(
        description="Exact Medicine value from the selected record; internal validation only."
    )
    generic_ingredient: str = Field(
        description="One generic ingredient copied from the selected record's Composition."
    )
    short_reason: str = Field(description="One short evidence-based reason or general use.")
    key_caution: str = Field(description="One short, important general caution.")
    pharmacy_search_query: str = Field(
        description="Must exactly equal generic_ingredient when a result is supported."
    )


class TriageDecision(BaseModel):
    status: Literal["INQUIRING", "MINOR", "MODERATE", "URGENT"] = Field(
        description=(
            "The triage routing status. Follow these strict clinical rules:\n"
            "- 'INQUIRING': Select this only when a material triage fact is genuinely missing. Ask only for missing facts.\n"
            "- 'MINOR': Select this for non-urgent mild symptoms when primary symptom, duration, and severity are known and there are no red flags. The exact cause need not be known.\n"
            "- 'MODERATE': Select this if the user explicitly asks to find a clinic, GP, doctor, or routine check-ups.\n"
            "- 'URGENT': Select this if there are red-flag fatal symptoms."
        )
    )
    reply_or_reason: str = Field(
        description="Follow-up questions if INQUIRING."
    )


class ToolRoute(BaseModel):
    """The non-emergency user intent selected after clinical triage."""

    intent: Literal[
        "SYMPTOM_TRIAGE",
        "MEDICINE_INFO",
        "FIND_GP",
        "FIND_URGENT_CARE",
        "GENERAL_MEDICAL",
    ] = Field(description="The tool-routing intent for the current user message.")
    reason: str = Field(
        description="A brief explanation grounded in the current message and patient context."
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

    # Tool intent selected only after the urgent-triage safety gate.
    tool_route: Optional[ToolRoute]

    # Structured short-term memory
    patient_context: PatientContext
