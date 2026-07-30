from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.config.settings import llm
from app.models.schemas import (
    PatientContext,
    PatientState,
    TriageDecision,
)
from app.prompts.memory_prompt import MEMORY_PROMPT
from app.prompts.triage_prompt import TRIAGE_PROMPT
from app.tools.google_maps_tool import find_local_gp, find_urgent_care
from app.tools.medicine_tool import get_medicine_link


def build_case_summary(patient_context: PatientContext) -> str:
    """Format the existing conversation memory for medicine retrieval."""
    red_flags = ", ".join(patient_context.red_flags) if patient_context.red_flags else "none"

    return (
        f"Primary symptom: {patient_context.primary_symptom or 'unknown'}\n"
        f"Duration: {patient_context.duration or 'unknown'}\n"
        f"Severity: {patient_context.severity or 'unknown'}\n"
        f"Location: {patient_context.location or 'unknown'}\n"
        f"Red flags: {red_flags}"
    )


def memory_node(state: PatientState):

    current_context = state.get("patient_context")

    if current_context is None:
        current_context = PatientContext()

    memory_llm = llm.with_structured_output(PatientContext)

    updated_context = memory_llm.invoke(
        [
            SystemMessage(content=MEMORY_PROMPT),
            SystemMessage(
                content=f"""
Current patient context:

{current_context.model_dump_json(indent=2)}

Update the context only when the user provides new information.
Keep previously known facts unless the user explicitly changes them.
"""
            ),
            *state["messages"],
        ]
    )

    return {
        "patient_context": updated_context
    }



def triage_node(state: PatientState):

    triage_llm = llm.with_structured_output(TriageDecision)

    decision = triage_llm.invoke(
        [
            SystemMessage(content=TRIAGE_PROMPT),
            SystemMessage(
                content=f"""
Known patient information:

{state.get("patient_context", PatientContext()).model_dump_json(indent=2)}
"""
            ),
            *state["messages"],
        ]
    )

    return {
        "decision": decision
    }


def action_node(state: PatientState):

    decision = state["decision"]
    user_msg = state["messages"][-1].content

    location = (
        state["patient_context"].location
        if state["patient_context"].location
        else (
            "Newmarket"
            if "newmarket" in user_msg.lower()
            else "Auckland"
        )
    )

    if decision.status == "INQUIRING":
        return {
            "messages": [
                AIMessage(content=decision.reply_or_reason)
            ]
        }

    elif decision.status == "MINOR":
        case_summary = build_case_summary(
            state.get("patient_context", PatientContext())
        )

        return {
            "messages": [
                AIMessage(content=get_medicine_link.invoke(case_summary))
            ]
        }

    elif decision.status == "MODERATE":
        return {
            "messages": [
                AIMessage(content=find_local_gp.invoke(location))
            ]
        }

    elif decision.status == "URGENT":
        return {
            "messages": [
                AIMessage(content=find_urgent_care.invoke(location))
            ]
        }



workflow = StateGraph(PatientState)
workflow.add_node("memory", memory_node)

workflow.add_node("triage", triage_node)

workflow.add_node("action", action_node)

workflow.set_entry_point("memory")

workflow.add_edge("memory", "triage")

workflow.add_edge("triage", "action")

workflow.add_edge("action", END)
app = workflow.compile()
