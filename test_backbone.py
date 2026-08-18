import unittest
import os
import shutil
import tempfile
from app.backbone.tokenizer import BPEClauseTokenizer
from app.backbone.vector_store import VectorStoreManager
from app.backbone.context_packer import ContextWindowPacker
from app.backbone.auditor import SelfConsistencyAuditor
from app.backbone.cost_proxy import ModelRouter


class TestBPEClauseTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = BPEClauseTokenizer()

    def test_count_tokens(self):
        text = "Confidentiality clause under Indian Contract Act."
        count = self.tokenizer.count_tokens(text)
        self.assertIsInstance(count, int)
        self.assertGreater(count, 0)

    def test_extract_clauses(self):
        contract_text = (
            "SECTION 1. DEFINITIONS\n"
            "This section contains definitions for confidentiality.\n\n"
            "CLAUSE 4.1 INDEMNIFICATION\n"
            "The contractor shall indemnify the client against losses.\n\n"
            "GOVERNING LAW\n"
            "Governed by the laws of India."
        )
        clauses = self.tokenizer.extract_clauses(contract_text)
        self.assertGreaterEqual(len(clauses), 2)
        self.assertIn("clause_id", clauses[0])
        self.assertIn("title", clauses[0])

    def test_chunk_by_token_budget(self):
        text = "Paragraph 1: " + "legal text " * 50 + "\n\nParagraph 2: " + "more legal text " * 50
        chunks = self.tokenizer.chunk_by_token_budget(text, max_tokens=100)
        self.assertGreaterEqual(len(chunks), 1)
        for chunk in chunks:
            self.assertIn("chunk_id", chunk)
            self.assertIn("text", chunk)


class TestVectorStoreManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.vs = VectorStoreManager(persist_directory=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_and_query_documents(self):
        docs = [
            "Party A agrees to indemnify Party B for breach of NDA.",
            "Arbitration shall be held in New Delhi under the Arbitration Act.",
            "Termination requires 30 days written notice."
        ]
        metas = [
            {"clause_id": "c1", "contract_id": "contract_1"},
            {"clause_id": "c2", "contract_id": "contract_1"},
            {"clause_id": "c3", "contract_id": "contract_1"}
        ]
        ids = ["doc1", "doc2", "doc3"]

        success = self.vs.add_documents("test_collection", docs, metas, ids)
        self.assertTrue(success)

        results = self.vs.query("test_collection", "indemnify breach", n_results=2)
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("document", results[0])
        self.assertIn("similarity_score", results[0])


class TestContextWindowPacker(unittest.TestCase):
    def setUp(self):
        self.packer = ContextWindowPacker()

    def test_pack_context(self):
        chunks = [
            {
                "id": "doc1",
                "document": "First high priority clause regarding indemnity.",
                "metadata": {"clause_id": "Clause 4.1", "title": "Indemnity"},
                "similarity_score": 0.95
            },
            {
                "id": "doc2",
                "document": "Second clause regarding dispute resolution in courts.",
                "metadata": {"clause_id": "Clause 12.1", "title": "Jurisdiction"},
                "similarity_score": 0.80
            }
        ]
        packed = self.packer.pack_context(chunks, token_budget=1000)
        self.assertIn("formatted_context", packed)
        self.assertEqual(packed["packed_chunks_count"], 2)
        self.assertEqual(packed["dropped_chunks_count"], 0)


class TestSelfConsistencyAuditor(unittest.TestCase):
    def setUp(self):
        self.auditor = SelfConsistencyAuditor()

    def test_audit_grounded_response(self):
        ref_chunks = [
            {
                "id": "c1",
                "document": "The Contractor shall indemnify the Client against all third party claims [Clause 4.1].",
                "metadata": {"citation_id": "Clause 4.1"}
            }
        ]
        answer = "The Contractor shall indemnify the Client against all third party claims [Clause 4.1]."
        res = self.auditor.audit(answer, ref_chunks, "Who indemnifies the Client?")

        self.assertGreaterEqual(res.grounding_score, 0.7)
        self.assertIn("High", res.confidence_level)
        self.assertFalse(res.is_hallucination_detected)
        self.assertIn("Clause 4.1", res.citations_verified)

    def test_audit_empty_response(self):
        res = self.auditor.audit("", [], "Test query")
        self.assertEqual(res.grounding_score, 0.0)
        self.assertTrue(res.is_hallucination_detected)


class TestModelRouter(unittest.TestCase):
    def setUp(self):
        self.router = ModelRouter()

    def test_evaluate_complexity(self):
        score_simple = self.router.evaluate_complexity("What is the hearing date?", "case_status")
        score_complex = self.router.evaluate_complexity(
            "Analyze unilateral indemnity, liability caps, and auto-renewal traps in this breach of NDA contract.",
            "contract_analysis"
        )
        self.assertGreater(score_complex, score_simple)

    def test_route_and_execute(self):
        res = self.router.route_and_execute("What is anticipatory bail under CrPC?", task_type="legal_explainer")
        self.assertIsNotNone(res.content)
        self.assertIn(res.telemetry.model_tier, ["Lightweight", "Flagship"])
        self.assertGreaterEqual(res.telemetry.total_tokens, 1)


if __name__ == "__main__":
    unittest.main()
