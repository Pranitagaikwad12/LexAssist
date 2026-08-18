import unittest
from fastapi.testclient import TestClient
from main import app


class TestFastAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_read_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("modules", data)

    def test_route_endpoint(self):
        response = self.client.post("/api/route", json={"query": "Find precedents on NDA breach"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["module"], "case_law_search")

    def test_ask_unified_endpoint(self):
        response = self.client.post("/api/ask", json={"query": "Check status for CNR DLHC010012342023"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["routed_module"], "case_status")
        self.assertIn("module_response", data)

    def test_contract_analyze_and_query_endpoints(self):
        analyze_resp = self.client.post(
            "/api/contract/analyze",
            data={"contract_name": "Sample NDA", "contract_text": "Clause 1. Unilateral Indemnity. Contractor shall solely indemnify client."}
        )
        self.assertEqual(analyze_resp.status_code, 200)
        adata = analyze_resp.json()
        self.assertIn("contract_id", adata)

        cid = adata["contract_id"]
        query_resp = self.client.post(
            "/api/contract/query",
            params={"contract_id": cid, "query": "Is indemnity unilateral?"}
        )
        self.assertEqual(query_resp.status_code, 200)
        qdata = query_resp.json()
        self.assertIn("answer", qdata)

    def test_cases_search_endpoint(self):
        response = self.client.post("/api/cases/search", params={"query": "trade secrets injunction"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("summary_answer", data)

    def test_status_lookup_endpoint(self):
        response = self.client.post("/api/status/lookup", params={"query": "DLHC010012342023"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["status_found"])

    def test_draft_endpoints(self):
        tpl_resp = self.client.get("/api/draft/templates")
        self.assertEqual(tpl_resp.status_code, 200)

        gen_resp = self.client.post(
            "/api/draft/generate",
            json={
                "template_id": "nda_mutual",
                "inputs": {
                    "disclosing_party": "Party A Ltd",
                    "receiving_party": "Party B Ltd",
                    "effective_date": "2026-08-18",
                    "purpose": "software evaluation",
                    "jurisdiction_city": "Mumbai"
                }
            }
        )
        self.assertEqual(gen_resp.status_code, 200)
        gdata = gen_resp.json()
        self.assertTrue(gdata["success"])

    def test_explainer_query_endpoint(self):
        response = self.client.post("/api/explainer/query", params={"query": "Section 438 CrPC"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("explanation", data)

    def test_telemetry_stats_endpoint(self):
        response = self.client.get("/api/telemetry/stats")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_queries", data)
        self.assertIn("avg_grounding_score", data)


if __name__ == "__main__":
    unittest.main()
