import unittest
from app.router.intent_router import IntentRouter
from app.modules.contract_analysis import ContractAnalysisModule
from app.modules.case_law_search import CaseLawSearchModule
from app.modules.case_status import CaseStatusModule, MockECourtsProvider
from app.modules.document_drafting import DocumentDraftingModule
from app.modules.legal_explainer import LegalExplainerModule


class TestIntentRouter(unittest.TestCase):
    def setUp(self):
        self.router = IntentRouter()

    def test_route_contract_query(self):
        res = self.router.route_query("Review this contract for one-sided indemnity and termination risks")
        self.assertEqual(res.module, "contract_analysis")
        self.assertGreaterEqual(res.confidence, 0.6)

    def test_route_case_search_query(self):
        res = self.router.route_query("Find Supreme Court precedent on breach of NDA in software contracts")
        self.assertEqual(res.module, "case_law_search")

    def test_route_case_status_query(self):
        res = self.router.route_query("Check case status for CNR DLHC010012342023 next hearing date")
        self.assertEqual(res.module, "case_status")
        self.assertEqual(res.extracted_entities.get("cnr_number"), "DLHC010012342023")

    def test_route_drafting_query(self):
        res = self.router.route_query("Draft a mutual NDA template for software project")
        self.assertEqual(res.module, "document_drafting")

    def test_route_explainer_query(self):
        res = self.router.route_query("What does anticipatory bail mean under Section 438 CrPC?")
        self.assertEqual(res.module, "legal_explainer")
        self.assertEqual(res.extracted_entities.get("section"), "438")

    def test_route_general_fallback(self):
        res = self.router.route_query("Hello can you help me with a legal topic?")
        self.assertEqual(res.module, "general_qa")


class TestContractAnalysisModule(unittest.TestCase):
    def setUp(self):
        self.module = ContractAnalysisModule()

    def test_process_and_index_contract(self):
        contract_text = (
            "MUTUAL NON-DISCLOSURE AGREEMENT\n\n"
            "1. INDEMNITY: Contractor shall solely indemnify Client against all claims.\n"
            "2. AUTO-RENEWAL: Automatically renews for 1 year unless written notice is given.\n"
            "3. LIABILITY: Aggregate liability shall not exceed Rs 50,000."
        )
        report = self.module.process_and_index_contract("c_101", contract_text, "Test NDA")
        self.assertEqual(report.contract_title, "Test NDA")
        self.assertGreaterEqual(len(report.risks_found), 1)

    def test_answer_contract_query(self):
        res = self.module.answer_contract_query("c_101", "What is the liability cap?")
        self.assertIn("answer", res)
        self.assertIn("hallucination_audit", res)
        self.assertIn("telemetry", res)


class TestCaseLawSearchModule(unittest.TestCase):
    def setUp(self):
        self.module = CaseLawSearchModule()

    def test_search_precedents(self):
        res = self.module.search_precedents("breach of NDA software trade secret injunction")
        self.assertIn("summary_answer", res)
        self.assertGreaterEqual(len(res["precedents"]), 1)
        self.assertIn("ratio_decidendi", res["precedents"][0])


class TestCaseStatusModule(unittest.TestCase):
    def setUp(self):
        self.module = CaseStatusModule()

    def test_lookup_existing_case(self):
        res = self.module.lookup_case_status("DLHC010012342023")
        self.assertTrue(res["status_found"])
        self.assertEqual(res["record"]["cnr_number"], "DLHC010012342023")
        self.assertIn("summary", res)

    def test_lookup_missing_case_anti_hallucination(self):
        res = self.module.lookup_case_status("NONEXISTENT_CNR_99999")
        self.assertFalse(res["status_found"])
        self.assertIn("RECORD NOT FOUND", res["message"])
        self.assertIn("strictly avoids generating fake", res["summary"])


class TestDocumentDraftingModule(unittest.TestCase):
    def setUp(self):
        self.module = DocumentDraftingModule()

    def test_get_available_templates(self):
        templates = self.module.get_available_templates()
        self.assertGreaterEqual(len(templates), 2)
        template_ids = [t["template_id"] for t in templates]
        self.assertIn("nda_mutual", template_ids)

    def test_validate_and_generate_draft(self):
        inputs = {
            "disclosing_party": "Acme Corp Ltd",
            "receiving_party": "Beta Solutions Pvt Ltd",
            "effective_date": "18th August 2026",
            "purpose": "evaluation of software partnership",
            "jurisdiction_city": "New Delhi"
        }
        res = self.module.generate_draft("nda_mutual", inputs)
        self.assertTrue(res["success"])
        self.assertIn("Acme Corp Ltd", res["rendered_text"])
        self.assertIn("Beta Solutions Pvt Ltd", res["rendered_text"])


class TestLegalExplainerModule(unittest.TestCase):
    def setUp(self):
        self.module = LegalExplainerModule()

    def test_explain_legal_concept(self):
        res = self.module.explain_legal_concept("What does anticipatory bail mean?")
        self.assertIn("explanation", res)
        self.assertIn("hallucination_audit", res)
        self.assertIn("telemetry", res)


if __name__ == "__main__":
    unittest.main()
