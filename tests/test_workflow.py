import unittest

from langchain_core.messages import HumanMessage

from app.agent.workflow import route_after_tool_router, route_after_triage
from app.models.schemas import PatientContext, ToolRoute, TriageDecision


class WorkflowRoutingTests(unittest.TestCase):
    def make_state(
        self, status: str, intent: str | None = None, message: str = "test message"
    ):
        state = {
            "messages": [HumanMessage(content=message)],
            "patient_context": PatientContext(),
            "decision": TriageDecision(status=status, reply_or_reason="test"),
        }
        if intent:
            state["tool_route"] = ToolRoute(intent=intent, reason="test")
        return state

    def test_red_flag_symptoms_override_medicine_request(self):
        state = self.make_state(
            "URGENT",
            "MEDICINE_INFO",
            "I have chest pain and trouble breathing. Can I take Panadol?",
        )
        state["patient_context"] = PatientContext(
            primary_symptom="chest pain",
            red_flags=["trouble breathing"],
        )
        self.assertEqual(route_after_triage(state), "urgent_care")

    def test_gp_query_routes_to_gp_tool(self):
        state = self.make_state("INQUIRING", "FIND_GP")
        self.assertEqual(route_after_triage(state), "tool_router")
        self.assertEqual(route_after_tool_router(state), "gp")

    def test_urgent_care_query_routes_to_urgent_care_tool(self):
        state = self.make_state("INQUIRING", "FIND_URGENT_CARE")
        self.assertEqual(route_after_tool_router(state), "urgent_care")

    def test_medicine_query_routes_to_safe_phase_one_fallback(self):
        state = self.make_state("INQUIRING", "MEDICINE_INFO")
        self.assertEqual(route_after_tool_router(state), "medicine_info")

    def test_new_or_follow_up_symptoms_route_to_existing_action(self):
        state = self.make_state("INQUIRING", "SYMPTOM_TRIAGE")
        self.assertEqual(route_after_tool_router(state), "action")


if __name__ == "__main__":
    unittest.main()
