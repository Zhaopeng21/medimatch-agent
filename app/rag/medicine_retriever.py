"""Small FAISS-only retrieval adapter for medicine candidates."""

from dataclasses import dataclass

from app.rag.vector_store import vector_db


@dataclass(frozen=True)
class MedicineCandidate:
    medicine_name: str
    treats: str
    composition: str
    side_effects: str

    def as_prompt_text(self) -> str:
        return (
            f"Medicine: {self.medicine_name}\n"
            f"Treats: {self.treats}\n"
            f"Composition: {self.composition or 'Not listed'}\n"
            f"Listed side effects: {self.side_effects or 'Not listed'}"
        )


def _candidate_from_document(document) -> MedicineCandidate | None:
    metadata = document.metadata
    name = str(metadata.get("medicine_name", "")).strip()
    treats = str(metadata.get("treats", "")).strip()
    if not name or not treats:
        return None
    return MedicineCandidate(
        medicine_name=name,
        treats=treats,
        composition=str(metadata.get("composition", "")).strip(),
        side_effects=str(metadata.get("side_effects", "")).strip(),
    )


def retrieve_medicine_candidates(query: str, limit: int = 5) -> list[MedicineCandidate]:
    """Return a small set of candidate records from the existing FAISS index."""
    if not vector_db:
        return []
    try:
        documents = vector_db.similarity_search(query, k=limit)
    except Exception:
        return []
    return [candidate for document in documents if (candidate := _candidate_from_document(document))]


def retrieve_named_medicine_candidates(question: str) -> list[MedicineCandidate]:
    """Require the queried brand token to occur in a returned FAISS candidate."""
    question_words = set(question.lower().replace("?", " ").replace(",", " ").split())
    return [
        candidate
        for candidate in retrieve_medicine_candidates(question)
        if candidate.medicine_name.lower().split()[0] in question_words
    ]
