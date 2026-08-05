import unittest
from unittest.mock import patch

from app.models.schemas import MedicineRecommendation
from app.rag.medicine_retriever import MedicineCandidate
from app.tools.medicine_presenter import GENERAL_MEDICINE_DISCLAIMER, build_pharmacy_link
from app.tools.medicine_tool import SAFETY_FALLBACK, get_medicine_link


COMPLETE_CONTEXT = (
    "Primary symptom: mild headache\nDuration: 2 days\nSeverity: mild\n"
    "Location: Auckland\nRed flags: none"
)
HEADACHE_CANDIDATE = MedicineCandidate(
    medicine_name="Paracetamol",
    treats="Pain relief | Treatment of Headache",
    composition="Paracetamol",
    side_effects="Read the NZ product label and ask a pharmacist if unsure.",
)
MIGRAINE_CANDIDATE = MedicineCandidate(
    medicine_name="Rizora 10 Tablet",
    treats="Acute migraine headache",
    composition="Rizatriptan (10mg)",
    side_effects="Dizziness",
)


def recommendation(source="Paracetamol", ingredient="Paracetamol"):
    return MedicineRecommendation(
        recommendation_status="RECOMMENDED",
        source_medicine_name=source,
        generic_ingredient=ingredient,
        short_reason="Listed for treatment of headache.",
        key_caution="Read the product label and ask a pharmacist if unsure.",
        pharmacy_search_query=ingredient,
    )


class MedicineToolTests(unittest.TestCase):
    @patch("app.tools.medicine_tool.llm")
    @patch("app.tools.medicine_tool.retrieve_medicine_candidates", return_value=[HEADACHE_CANDIDATE])
    def test_headache_displays_verified_generic_ingredient_and_link(self, mock_retrieve, mock_llm):
        mock_llm.with_structured_output.return_value.invoke.return_value = recommendation()

        answer = get_medicine_link.invoke(COMPLETE_CONTEXT)

        mock_retrieve.assert_called_once_with("Medicine for mild headache")
        self.assertIn("**Medicine:** Paracetamol", answer)
        self.assertIn("Search at Chemist Warehouse NZ", answer)
        self.assertIn("searchtext=Paracetamol", answer)
        self.assertNotIn("Rizora", answer)
        self.assertIn(GENERAL_MEDICINE_DISCLAIMER, answer)

    @patch("app.tools.medicine_tool.llm")
    @patch("app.tools.medicine_tool.retrieve_medicine_candidates", return_value=[HEADACHE_CANDIDATE])
    def test_ingredient_outside_selected_composition_falls_back(self, _retrieve, mock_llm):
        mock_llm.with_structured_output.return_value.invoke.return_value = recommendation(
            ingredient="Ibuprofen"
        )

        answer = get_medicine_link.invoke(COMPLETE_CONTEXT)

        self.assertIn(SAFETY_FALLBACK, answer)
        self.assertNotIn("chemistwarehouse.co.nz", answer)

    @patch("app.tools.medicine_tool.llm")
    @patch("app.tools.medicine_tool.retrieve_medicine_candidates", return_value=[MIGRAINE_CANDIDATE])
    def test_mild_headache_cannot_select_migraine_only_candidate(self, mock_retrieve, mock_llm):
        answer = get_medicine_link.invoke(COMPLETE_CONTEXT)

        mock_llm.with_structured_output.assert_not_called()
        mock_retrieve.assert_called_once()
        self.assertIn(SAFETY_FALLBACK, answer)

    @patch("app.tools.medicine_tool.llm")
    @patch(
        "app.tools.medicine_tool.retrieve_medicine_candidates",
        return_value=[MedicineCandidate("Headache Tablet", "Treatment of Headache", "", "Nausea")],
    )
    def test_missing_composition_falls_back(self, _retrieve, mock_llm):
        answer = get_medicine_link.invoke(COMPLETE_CONTEXT)

        mock_llm.with_structured_output.assert_not_called()
        self.assertIn(SAFETY_FALLBACK, answer)

    @patch("app.tools.medicine_tool.retrieve_medicine_candidates")
    def test_red_flags_do_not_enter_medicine_path(self, mock_retrieve):
        answer = get_medicine_link.invoke(COMPLETE_CONTEXT.replace("none", "chest pain"))

        mock_retrieve.assert_not_called()
        self.assertIn(SAFETY_FALLBACK, answer)

    def test_pharmacy_link_is_deterministic(self):
        self.assertEqual(
            build_pharmacy_link("Paracetamol"),
            "https://www.chemistwarehouse.co.nz/search?searchtext=Paracetamol",
        )


if __name__ == "__main__":
    unittest.main()
