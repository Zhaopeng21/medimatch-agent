from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from app.config.settings import llm
from app.rag.vector_store import vector_db


GENERAL_MEDICINE_DISCLAIMER = (
    "\n\n*This answer is general information from the project's medicine "
    "database only. It does not replace advice from a doctor or pharmacist. "
    "Read the product label, and consult a health professional if symptoms "
    "persist, worsen, or you are unsure.*"
)

SAFETY_FALLBACK = (
    "I cannot safely recommend a medicine from the available information. "
    "Please speak with a pharmacist or GP for advice."
)

# These are intentionally conservative safety signals, not a medicine catalogue.
# Any match prevents a recommendation rather than trying to classify the product.
HIGH_RISK_TERMS = {
    "antibiotic",
    "amoxicillin",
    "amoxicillin with clavulanic acid",
    "augmentin",
    "azithromycin",
    "ciprofloxacin",
    "doxycycline",
    "flucloxacillin",
    "penicillin",
    "prescription",
    "controlled drug",
    "insulin",
    "warfarin",
    "opioid",
    "morphine",
    "codeine",
    "diazepam",
    "antiviral",
    "antipsychotic",
}
INJECTION_TERMS = {" injection", " inject", " injectable", " vial", " ampoule", " syringe"}


def retrieve_medicine_context(query: str, k: int = 3) -> str:
    """Retrieve only the existing local medicine database as supporting context."""
    if not vector_db:
        return ""

    try:
        docs = vector_db.similarity_search(query, k=k)
        return "\n".join(doc.page_content for doc in docs)
    except Exception:
        return ""


def contains_high_risk_medicine_terms(*texts: str) -> bool:
    combined = " ".join(text.lower() for text in texts if text)
    return any(term in combined for term in HIGH_RISK_TERMS | INJECTION_TERMS)


def has_complete_minor_context(case_summary: str) -> bool:
    """Require the minimum structured facts before a symptom-based suggestion."""
    required_markers = ("Primary symptom: unknown", "Duration: unknown", "Severity: unknown")
    return not any(marker.lower() in case_summary.lower() for marker in required_markers)


def has_red_flags(case_summary: str) -> bool:
    """Treat any recorded red flag as a hard stop for medicine recommendations."""
    for line in case_summary.splitlines():
        if line.lower().startswith("red flags:"):
            return line.split(":", 1)[1].strip().lower() not in {"", "none", "unknown"}
    return False


def is_medicine_recommendation_request(user_question: str) -> bool:
    question = user_question.lower()
    return any(
        phrase in question
        for phrase in (
            "what medicine can i take",
            "what can i take",
            "what medication can i take",
            "recommend a medicine",
            "recommend medication",
        )
    )


def information_needed_reply() -> str:
    return (
        "Before I can give general medicine information for your symptoms, please "
        "tell me the main symptom, how long it has been present, and how severe it is."
        + GENERAL_MEDICINE_DISCLAIMER
    )


def safety_fallback_reply() -> str:
    return SAFETY_FALLBACK + GENERAL_MEDICINE_DISCLAIMER


def generate_medicine_answer(
    *,
    user_question: str,
    case_summary: str,
    triage_status: str,
    retrieved_context: str,
    allow_recommendation: bool,
) -> str:
    """Ask the model to summarise retrieved data without adding treatment claims."""
    task = (
        "Answer the user's medicine question using ONLY the retrieved medicine "
        "database context. State general uses, commonly listed side effects, and "
        "general precautions only when supported by that context. Do not diagnose, "
        "give a dose, alter a prescription, recommend antibiotics, or claim that a "
        "medicine is available or approved in New Zealand. "
    )
    if allow_recommendation:
        task += (
            "The patient has been triaged as a non-urgent minor case with complete "
            "context. You may mention at most one simple non-prescription medicine "
            "category or active ingredient only if the retrieved context clearly "
            "supports it. If uncertain, say that a pharmacist should advise. "
        )
    else:
        task += "Do not recommend a medicine; provide general information only. "

    answer = llm.invoke(
        [
            SystemMessage(content=task),
            HumanMessage(
                content=(
                    f"USER QUESTION:\n{user_question}\n\n"
                    f"TRIAGE STATUS: {triage_status}\n\n"
                    f"STRUCTURED PATIENT CONTEXT:\n{case_summary}\n\n"
                    f"RETRIEVED DATABASE CONTEXT:\n{retrieved_context}"
                )
            ),
        ]
    )
    return answer.content.strip() + GENERAL_MEDICINE_DISCLAIMER


@tool
def get_medicine_link(case_summary: str) -> str:
    """Give a conservative, symptom-driven medicine reference for minor cases."""
    if not has_complete_minor_context(case_summary):
        return information_needed_reply()
    if has_red_flags(case_summary):
        return safety_fallback_reply()

    retrieved_context = retrieve_medicine_context(case_summary, k=2)
    if not retrieved_context or contains_high_risk_medicine_terms(retrieved_context):
        return safety_fallback_reply()

    try:
        return generate_medicine_answer(
            user_question="What general non-prescription medicine information may help this minor symptom?",
            case_summary=case_summary,
            triage_status="MINOR",
            retrieved_context=retrieved_context,
            allow_recommendation=True,
        )
    except Exception:
        return safety_fallback_reply()


@tool
def get_medicine_information(
    user_question: str,
    case_summary: str,
    triage_status: str,
) -> str:
    """Answer a non-urgent medicine question from the local RAG database only."""
    if triage_status == "URGENT":
        return (
            "Because urgent symptoms have been identified, please seek urgent medical "
            "help now rather than relying on medicine information."
        )

    if contains_high_risk_medicine_terms(user_question):
        return safety_fallback_reply()

    recommendation_requested = is_medicine_recommendation_request(user_question)
    if recommendation_requested and not has_complete_minor_context(case_summary):
        return information_needed_reply()

    retrieved_context = retrieve_medicine_context(user_question, k=3)
    if not retrieved_context or contains_high_risk_medicine_terms(retrieved_context):
        return safety_fallback_reply()

    try:
        return generate_medicine_answer(
            user_question=user_question,
            case_summary=case_summary,
            triage_status=triage_status,
            retrieved_context=retrieved_context,
            allow_recommendation=(
                recommendation_requested
                and triage_status == "MINOR"
                and has_complete_minor_context(case_summary)
                and not has_red_flags(case_summary)
            ),
        )
    except Exception:
        return safety_fallback_reply()
