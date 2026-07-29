from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.config.settings import llm
from app.models.schemas import PatientState, TriageDecision
from app.prompts.triage_prompt import TRIAGE_PROMPT
from app.tools.google_maps_tool import find_local_gp, find_urgent_care
from app.tools.medicine_tool import get_medicine_link


def triage_node(state: PatientState):
    system_prompt = SystemMessage(content=TRIAGE_PROMPT)
    triage_llm = llm.with_structured_output(TriageDecision)
    decision = triage_llm.invoke([system_prompt] + list(state["messages"]))
    return {"decision": decision}


def action_node(state: PatientState):
    decision = state["decision"]
    user_msg = state["messages"][-1].content
    location = "Newmarket" if "newmarket" in user_msg.lower() else "Auckland"

    if decision.status == "INQUIRING":
        return {"messages": [AIMessage(content=decision.reply_or_reason)]}
    elif decision.status == "MINOR":
        return {"messages": [AIMessage(content=get_medicine_link.invoke(user_msg))]}
    elif decision.status == "MODERATE":
        return {"messages": [AIMessage(content=find_local_gp.invoke(location))]}
    elif decision.status == "URGENT":
        return {"messages": [AIMessage(content=find_urgent_care.invoke(location))]}


workflow = StateGraph(PatientState)
workflow.add_node("triage", triage_node)
workflow.add_node("action_node", action_node)
workflow.set_entry_point("triage")
workflow.add_edge("triage", "action_node")
workflow.add_edge("action_node", END)
app = workflow.compile()
