from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.config.settings import llm
from app.models.schemas import (
    PatientContext,
    PatientState,
    TriageDecision,
    ToolRoute,
)
from app.prompts.memory_prompt import MEMORY_PROMPT
from app.prompts.triage_prompt import TRIAGE_PROMPT
from app.prompts.tool_router_prompt import TOOL_ROUTER_PROMPT
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


def tool_router_node(state: PatientState):
    """Classify non-emergency requests into the available tool paths."""
    router_llm = llm.with_structured_output(ToolRoute)
    user_msg = state["messages"][-1].content
    patient_context = state.get("patient_context", PatientContext())

    route = router_llm.invoke(
        [
            SystemMessage(content=TOOL_ROUTER_PROMPT),
            SystemMessage(
                content=f"""
PATIENT CONTEXT:
{patient_context.model_dump_json(indent=2)}

CURRENT USER MESSAGE:
{user_msg}
"""
            ),
        ]
    )

    return {"tool_route": route}


def get_location(state: PatientState) -> str:
    """Use known location, with the existing Auckland fallback behaviour."""
    patient_context = state.get("patient_context", PatientContext())
    user_msg = state["messages"][-1].content

    if patient_context.location:
        return patient_context.location
    if "newmarket" in user_msg.lower():
        return "Newmarket"
    return "Auckland"


def route_after_triage(state: PatientState) -> str:
    """Keep emergency triage ahead of every intent and tool decision."""
    decision = state.get("decision")
    if decision and decision.status == "URGENT":
        return "urgent_care"
    return "tool_router"


def route_after_tool_router(state: PatientState) -> str:
    """Map the structured tool intent to its graph node."""
    route = state.get("tool_route")
    intent = route.intent if route else "GENERAL_MEDICAL"

    return {
        "SYMPTOM_TRIAGE": "action",
        "MEDICINE_INFO": "medicine_info",
        "FIND_GP": "gp",
        "FIND_URGENT_CARE": "urgent_care",
        "GENERAL_MEDICAL": "general_medical",
    }.get(intent, "general_medical")


def action_node(state: PatientState):

    decision = state["decision"]
    location = get_location(state)

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

    return {
        "messages": [
            AIMessage(content="Please describe your symptoms in a little more detail.")
        ]
    }


def gp_node(state: PatientState):
    return {"messages": [AIMessage(content=find_local_gp.invoke(get_location(state)))]}


def urgent_care_node(state: PatientState):
    return {"messages": [AIMessage(content=find_urgent_care.invoke(get_location(state)))]}


def medicine_info_node(state: PatientState):
    """Phase-one safe fallback until medicine-information RAG is added."""
    return {
        "messages": [
            AIMessage(
                content=(
                    "I can help route medicine questions, but detailed medicine "
                    "information is not available in this version yet. Please check "
                    "the medicine label or ask a New Zealand pharmacist. If you have "
                    "severe symptoms, symptoms that are worsening, or red-flag symptoms, "
                    "seek urgent medical help or call 111."
                )
            )
        ]
    }


def general_medical_node(state: PatientState):
    """Phase-one non-diagnostic fallback for questions without a matching tool."""
    return {
        "messages": [
            AIMessage(
                content=(
                    "I can provide general health guidance, but I cannot diagnose. "
                    "Please share any symptoms, how long they have been present, and "
                    "how severe they are so I can help with triage. If symptoms are "
                    "severe, worsening, or include red flags, seek medical help urgently "
                    "or call 111."
                )
            )
        ]
    }



workflow = StateGraph(PatientState)
workflow.add_node("memory", memory_node)

workflow.add_node("triage", triage_node)
workflow.add_node("tool_router", tool_router_node)
workflow.add_node("action", action_node)
workflow.add_node("gp", gp_node)
workflow.add_node("urgent_care", urgent_care_node)
workflow.add_node("medicine_info", medicine_info_node)
workflow.add_node("general_medical", general_medical_node)

workflow.set_entry_point("memory")

workflow.add_edge("memory", "triage")

workflow.add_conditional_edges(
    "triage",
    route_after_triage,
    {
        "urgent_care": "urgent_care",
        "tool_router": "tool_router",
    },
)

workflow.add_conditional_edges(
    "tool_router",
    route_after_tool_router,
    {
        "action": "action",
        "medicine_info": "medicine_info",
        "gp": "gp",
        "urgent_care": "urgent_care",
        "general_medical": "general_medical",
    },
)
workflow.add_edge("action", END)
workflow.add_edge("gp", END)
workflow.add_edge("urgent_care", END)
workflow.add_edge("medicine_info", END)
workflow.add_edge("general_medical", END)
app = workflow.compile()
