import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from app.prompts.tool_router_prompt import TOOL_ROUTER_PROMPT
from app.tools.medicine_tool import (
    GENERAL_MEDICINE_DISCLAIMER,
    SAFETY_FALLBACK,
    get_medicine_information,
    has_complete_minor_context,
    has_red_flags,
)


COMPLETE_CONTEXT = (
    "Primary symptom: sore throat\nDuration: 2 days\nSeverity: mild\n"
    "Location: Auckland\nRed flags: none"
)


class MedicineToolTests(unittest.TestCase):
    def test_complete_minor_context_is_required_for_recommendation(self):
        self.assertTrue(has_complete_minor_context(COMPLETE_CONTEXT))
        self.assertFalse(has_complete_minor_context("Primary symptom: unknown\nDuration: unknown"))

    def test_recorded_red_flags_block_recommendation(self):
        self.assertTrue(has_red_flags(COMPLETE_CONTEXT.replace("Red flags: none", "Red flags: chest pain")))

    @patch("app.tools.medicine_tool.retrieve_medicine_context")
    def test_known_medicine_side_effect_question_uses_retrieved_context(self, mock_retrieve):
        mock_retrieve.return_value = "Medicine: Panadol. Treats: pain. Side effects: nausea."
        with patch("app.tools.medicine_tool.llm") as mock_llm:
            mock_llm.invoke.return_value = AIMessage(content="Nausea may occur.")
            answer = get_medicine_information.invoke(
                {
                    "user_question": "What side effects can Panadol have?",
                    "case_summary": COMPLETE_CONTEXT,
                    "triage_status": "INQUIRING",
                }
            )

        self.assertIn("Nausea may occur.", answer)
        self.assertIn(GENERAL_MEDICINE_DISCLAIMER, answer)

    @patch("app.tools.medicine_tool.retrieve_medicine_context")
    def test_known_medicine_precautions_question_uses_retrieved_context(self, mock_retrieve):
        mock_retrieve.return_value = "Medicine: Ibuprofen. Treats: pain. Side effects: stomach upset."
        with patch("app.tools.medicine_tool.llm") as mock_llm:
            mock_llm.invoke.return_value = AIMessage(content="Check the label.")
            answer = get_medicine_information.invoke(
                {
                    "user_question": "What precautions apply to Ibuprofen?",
                    "case_summary": COMPLETE_CONTEXT,
                    "triage_status": "INQUIRING",
                }
            )

        self.assertIn("Check the label.", answer)

    def test_recommendation_question_with_missing_context_asks_follow_up(self):
        answer = get_medicine_information.invoke(
            {
                "user_question": "What medicine can I take?",
                "case_summary": "Primary symptom: unknown\nDuration: unknown\nSeverity: unknown",
                "triage_status": "INQUIRING",
            }
        )

        self.assertIn("please tell me the main symptom", answer.lower())

    def test_antibiotic_question_is_not_recommended(self):
        answer = get_medicine_information.invoke(
            {
                "user_question": "Can you recommend Augmentin?",
                "case_summary": COMPLETE_CONTEXT,
                "triage_status": "MINOR",
            }
        )

        self.assertIn(SAFETY_FALLBACK, answer)

    @patch("app.tools.medicine_tool.retrieve_medicine_context", return_value="")
    def test_unknown_medicine_returns_safe_fallback(self, _mock_retrieve):
        answer = get_medicine_information.invoke(
            {
                "user_question": "What are the side effects of madeupmed?",
                "case_summary": COMPLETE_CONTEXT,
                "triage_status": "INQUIRING",
            }
        )

        self.assertIn(SAFETY_FALLBACK, answer)

    def test_router_prompt_routes_symptom_medicine_request_to_medicine_info(self):
        self.assertIn("asking what medicine they can take", TOOL_ROUTER_PROMPT)


if __name__ == "__main__":
    unittest.main()
