"""Deterministic presentation for the medicine tool."""

from urllib.parse import quote

from app.models.schemas import MedicineRecommendation


GENERAL_MEDICINE_DISCLAIMER = (
    "*General information only; this does not replace advice from a doctor or "
    "pharmacist. Read the product label, and seek professional advice if symptoms "
    "persist, worsen, or you are unsure.*"
)


def build_pharmacy_link(search_query: str) -> str:
    return f"https://www.chemistwarehouse.co.nz/search?searchtext={quote(search_query.strip())}"


def render_medicine_recommendation(result: MedicineRecommendation, *, include_link: bool) -> str:
    lines = [
        f"**Medicine:** {result.generic_ingredient.strip()}",
        f"**Why:** {result.short_reason.strip()}",
        f"**Key caution:** {result.key_caution.strip()}",
    ]
    if include_link:
        lines.append(
            f"[Search at Chemist Warehouse NZ]({build_pharmacy_link(result.pharmacy_search_query)})"
        )
    lines.append(GENERAL_MEDICINE_DISCLAIMER)
    return "\n\n".join(lines)
