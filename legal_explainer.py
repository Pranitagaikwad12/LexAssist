import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from app.backbone.tokenizer import BPEClauseTokenizer
from app.backbone.vector_store import VectorStoreManager
from app.backbone.context_packer import ContextWindowPacker
from app.backbone.auditor import HallucinationAuditor, SelfConsistencyAuditor
from app.backbone.cost_proxy import CostOptimizationProxy, ModelRouter


class LegalExplainerModule:
    """
    Module 5: Legal Q&A / Statute Concept Explainer.
    Grounds explanations of legal terms, concepts, and statutes against pre-indexed Indian statutory corpus.
    Enforces mandatory section/article citations.
    """
    DEFAULT_STATUTE_CORPUS = [
        {
            "id": "crpc_438_bnss_482",
            "statute": "Code of Criminal Procedure, 1973 / BNSS 2023",
            "section": "Section 438 CrPC / Section 482 BNSS",
            "title": "Direction for grant of bail to person apprehending arrest (Anticipatory Bail)",
            "text": (
                "Section 438 CrPC (now Section 482 BNSS) provides that when any person has reason to believe that he may be arrested "
                "on an accusation of having committed a non-bailable offence, he may apply to the High Court or the Court of Session for a direction "
                "under this section; and that court may, if it thinks fit, direct that in the event of such arrest, he shall be released on bail.\n"
                "Key Factors considered by Court: (i) nature and gravity of allegation, (ii) antecedents of applicant, (iii) possibility of applicant fleeing from justice, "
                "and (iv) whether accusation is made with object of injuring or humiliating applicant."
            )
        },
        {
            "id": "const_art_21",
            "statute": "Constitution of India",
            "section": "Article 21",
            "title": "Protection of life and personal liberty",
            "text": (
                "Article 21 of the Constitution of India provides: 'No person shall be deprived of his life or personal liberty except according to procedure established by law.' "
                "Extensive judicial interpretation by the Supreme Court (Maneka Gandhi v. Union of India) has established that procedure must be fair, just, and reasonable, "
                "encompassing the right to speedy trial, legal aid, privacy, and clean environment."
            )
        },
        {
            "id": "contract_sec_27",
            "statute": "Indian Contract Act, 1872",
            "section": "Section 27",
            "title": "Agreement in restraint of trade, void",
            "text": (
                "Section 27 of the Indian Contract Act, 1872 enacts that every agreement by which any one is restrained from exercising a lawful profession, "
                "trade or business of any kind, is to that extent void.\n"
                "Exception 1: Saving of agreement not to carry on business of which goodwill is sold."
            )
        },
        {
            "id": "ipc_420_bns_318",
            "statute": "Indian Penal Code, 1860 / BNS 2023",
            "section": "Section 420 IPC / Section 318 BNS",
            "title": "Cheating and dishonestly inducing delivery of property",
            "text": (
                "Section 420 IPC deals with cheating and dishonestly inducing delivery of property. Whoever cheats and thereby dishonestly induces the person deceived "
                "to deliver any property, or to make, alter or destroy the whole or any part of a valuable security, shall be punished with imprisonment of either description "
                "for a term which may extend to seven years, and shall also be liable to fine."
            )
        },
        {
            "id": "arbitration_sec_9",
            "statute": "Arbitration and Conciliation Act, 1996",
            "section": "Section 9",
            "title": "Interim measures by Court",
            "text": (
                "Section 9 of the Arbitration and Conciliation Act, 1996 empowers a party to apply to a Court before, during, or after arbitral proceedings "
                "for interim measures of protection including preservation, custody, or sale of goods, securing amount in dispute, or interim injunction."
            )
        }
    ]

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        auditor: Optional[HallucinationAuditor] = None,
        cost_proxy: Optional[CostOptimizationProxy] = None
    ):
        self.tokenizer = BPEClauseTokenizer()
        self.vector_store = vector_store or VectorStoreManager()
        self.packer = ContextWindowPacker(tokenizer=self.tokenizer)
        self.auditor = auditor or SelfConsistencyAuditor()
        self.cost_proxy = cost_proxy or ModelRouter()

        # Seed statute corpus into Chroma collection
        self._seed_statute_corpus()

    def _seed_statute_corpus(self):
        docs = [item["text"] for item in self.DEFAULT_STATUTE_CORPUS]
        metas = [
            {
                "citation_id": f"{item['statute']}, {item['section']}",
                "statute": item["statute"],
                "section": item["section"],
                "title": item["title"]
            }
            for item in self.DEFAULT_STATUTE_CORPUS
        ]
        ids = [item["id"] for item in self.DEFAULT_STATUTE_CORPUS]

        self.vector_store.add_documents(
            collection_name="statutes",
            documents=docs,
            metadatas=metas,
            ids=ids
        )

    def explain_legal_concept(self, user_query: str) -> Dict[str, Any]:
        retrieved = self.vector_store.query(
            collection_name="statutes",
            query_text=user_query,
            n_results=4
        )

        packed = self.packer.pack_context(retrieved, token_budget=4000)

        system_prompt = (
            "You are LexAssist Indian Legal Explainer Specialist.\n"
            "Explain the legal concept or statutory question relying strictly on the retrieved statute sections below.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Every legal definition, ground, or test must cite the specific statute and section in brackets e.g. [Section 438, CrPC] or [Article 21, Constitution of India].\n"
            "2. Break down the explanation into: (a) Legal Definition, (b) Key Statutory Ingredients, (c) Practical Implications.\n"
            "3. Conclude with: '*Legal Information Only — Not Formal Legal Advice.*'"
        )

        full_prompt = f"Statutory & Bare Act Context:\n{packed['formatted_context']}\n\nUser Question: {user_query}"

        proxy_res = self.cost_proxy.route_and_execute(
            prompt=full_prompt,
            task_type="legal_explainer",
            system_prompt=system_prompt
        )

        audit_res = self.auditor.audit(
            generated_answer=proxy_res.content,
            reference_chunks=packed["packed_chunks"],
            user_query=user_query
        )

        return {
            "explanation": proxy_res.content,
            "citations_verified": audit_res.citations_verified,
            "hallucination_audit": audit_res.model_dump(),
            "telemetry": proxy_res.telemetry.model_dump(),
            "statutory_sources": packed["packed_chunks"]
        }
