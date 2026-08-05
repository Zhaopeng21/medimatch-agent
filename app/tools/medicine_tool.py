"""Medicine recommendation and information orchestration.

Runtime knowledge comes only from the local FAISS index.  The tool deliberately
keeps retrieval, model selection, and presentation as separate small concerns.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from app.config.settings import llm
from app.models.schemas import MedicineRecommendation
from app.rag.medicine_retriever import (
    MedicineCandidate,
    retrieve_medicine_candidates,
    retrieve_named_medicine_candidates,
)
from app.tools.medicine_presenter import (
    GENERAL_MEDICINE_DISCLAIMER,
    build_pharmacy_link,
    render_medicine_recommendation,
)


SAFETY_FALLBACK = (
    "I could not find a reliable treatment match in the available medicine database, "
    "so I cannot recommend a medicine. Please ask a pharmacist or GP."
)
BLOCKED_MEDICINE_TERMS = ("antibiotic", "amoxicillin", "augmentin", "injection", "injectable")


def _has_red_flags(case_summary: str) -> bool:
    return "red flags:" in case_summary.lower() and "red flags: none" not in case_summary.lower()


def _has_complete_context(case_summary: str) -> bool:
    return all(f"{field}: unknown" not in case_summary.lower() for field in ("primary symptom", "duration", "severity"))


def _is_blocked(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in BLOCKED_MEDICINE_TERMS)


def _composition_ingredients(composition: str) -> set[str]:
    """Extract the explicitly listed active ingredients from one FAISS record."""
    return {
        part.split("(", 1)[0].strip().casefold()
        for part in composition.split("+")
        if part.split("(", 1)[0].strip()
    }


def _is_recommendation_request(question: str) -> bool:
    lowered = question.lower()
    return any(phrase in lowered for phrase in ("what medicine can i take", "what can i take", "recommend a medicine", "recommend medication"))


def _symptom_query(case_summary: str) -> str:
    """Keep FAISS focused on the structured primary symptom, not every case detail."""
    for line in case_summary.splitlines():
        if line.lower().startswith("primary symptom:"):
            symptom = line.split(":", 1)[1].strip()
            if symptom and symptom.lower() != "unknown":
                return f"Medicine for {symptom}"
    return case_summary


def _candidate_is_in_scope(candidate: MedicineCandidate, case_summary: str) -> bool:
    """Keep a general symptom from being upgraded to a more specific condition."""
    primary_symptom = _symptom_query(case_summary).casefold()
    treats = candidate.treats.casefold()
    return not ("migraine" in treats and "migraine" not in primary_symptom)


def _fallback() -> str:
    return f"{SAFETY_FALLBACK}\n\n{GENERAL_MEDICINE_DISCLAIMER}"


def _information_needed() -> str:
    return (
        "Please tell me the main symptom, how long it has been present, and how severe "
        f"it is before I provide a medicine recommendation.\n\n{GENERAL_MEDICINE_DISCLAIMER}"
    )


def _select_recommendation(
    *,
    user_question: str,
    case_summary: str,
    candidates: list[MedicineCandidate],
    status: str,
) -> MedicineRecommendation | None:
    if not candidates:
        return None
    candidate_text = "\n\n".join(candidate.as_prompt_text() for candidate in candidates)
    prompt = (
        "Use only the candidate records. Choose one record only if its Treats field "
        "directly supports the user's symptom or named-medicine question. Never use a "
        "medicine merely because its name is similar or because a symptom appears in "
        "its listed side effects. Do not diagnose, give doses, recommend antibiotics, "
        "or recommend injections. source_medicine_name must exactly equal the selected "
        "candidate Medicine value. generic_ingredient must exactly equal the selected "
        "candidate's Composition field, which is the verified NZ search ingredient. "
        "candidate's Composition, and pharmacy_search_query must exactly equal it. "
        "Do not select a migraine-specific record unless the patient context explicitly "
        "mentions migraine. Return the requested status only when supported; "
        "otherwise return NO_RELIABLE_MATCH with empty fields. Keep each text field short."
    )
    try:
        result = llm.with_structured_output(MedicineRecommendation).invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(
                    content=(
                        f"USER QUESTION:\n{user_question}\n\n"
                        f"PATIENT CONTEXT:\n{case_summary}\n\n"
                        f"CANDIDATES:\n{candidate_text}"
                    )
                ),
            ]
        )
    except Exception:
        return None

    source_names = {candidate.medicine_name for candidate in candidates}
    fields_present = all(
        (
            result.generic_ingredient.strip(),
            result.short_reason.strip(),
            result.key_caution.strip(),
            result.pharmacy_search_query.strip(),
        )
    )
    if result.recommendation_status != status or result.source_medicine_name not in source_names:
        return None
    selected = next(
        candidate for candidate in candidates if candidate.medicine_name == result.source_medicine_name
    )
    generic = result.generic_ingredient.strip()
    if not _candidate_is_in_scope(selected, case_summary):
        return None
    if generic.casefold() not in _composition_ingredients(selected.composition):
        return None
    if result.pharmacy_search_query.strip() != generic:
        return None
    if _is_blocked(f"{selected.medicine_name} {selected.composition} {generic}"):
        return None
    return result if fields_present else None


@tool
def get_medicine_link(case_summary: str) -> str:
    """Recommend one general medicine option for a complete MINOR patient context."""
    if not _has_complete_context(case_summary) or _has_red_flags(case_summary):
        return _fallback()
    candidates = [
        candidate
        for candidate in retrieve_medicine_candidates(_symptom_query(case_summary))
        if _candidate_is_in_scope(candidate, case_summary)
        and candidate.composition.strip()
        and not _is_blocked(f"{candidate.medicine_name} {candidate.composition}")
    ]
    result = _select_recommendation(
        user_question="Recommend one general medicine option for this minor symptom.",
        case_summary=case_summary,
        candidates=candidates,
        status="RECOMMENDED",
    )
    return render_medicine_recommendation(result, include_link=True) if result else _fallback()


@tool
def get_medicine_information(user_question: str, case_summary: str, triage_status: str) -> str:
    """Answer a FAISS-supported medicine question or a complete MINOR request."""
    if triage_status == "URGENT" or _has_red_flags(case_summary) or _is_blocked(user_question):
        return _fallback()
    if _is_recommendation_request(user_question):
        if not _has_complete_context(case_summary):
            return _information_needed()
        if triage_status != "MINOR":
            return _fallback()
        candidates = [
            candidate
            for candidate in retrieve_medicine_candidates(_symptom_query(case_summary))
            if _candidate_is_in_scope(candidate, case_summary)
            and candidate.composition.strip()
            and not _is_blocked(f"{candidate.medicine_name} {candidate.composition}")
        ]
        status, include_link = "RECOMMENDED", True
    else:
        candidates = retrieve_named_medicine_candidates(user_question)
        status, include_link = "INFORMATION_ONLY", any(
            word in user_question.lower() for word in ("buy", "purchase", "view", "chemist warehouse")
        )
    result = _select_recommendation(
        user_question=user_question,
        case_summary=case_summary,
        candidates=candidates,
        status=status,
    )
    return render_medicine_recommendation(result, include_link=include_link) if result else _fallback()


__all__ = ["get_medicine_link", "get_medicine_information", "build_pharmacy_link"]
