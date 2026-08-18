import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from app.backbone.tokenizer import BPEClauseTokenizer
from app.backbone.vector_store import VectorStoreManager
from app.backbone.context_packer import ContextWindowPacker
from app.backbone.auditor import HallucinationAuditor, SelfConsistencyAuditor
from app.backbone.cost_proxy import CostOptimizationProxy, ModelRouter


class RiskFlag(BaseModel):
    risk_level: str = Field(..., description="HIGH, MEDIUM, LOW")
    category: str = Field(..., description="Indemnity, Termination, Liability, Auto-Renewal, Jurisdiction, Miscellaneous")
    clause_title: str
    clause_text: str
    issue_description: str
    recommendation: str


class ContractAnalysisReport(BaseModel):
    contract_title: str
    total_clauses: int
    overall_risk_score: str = Field(..., description="HIGH RISK, MODERATE RISK, LOW RISK")
    risks_found: List[RiskFlag]
    clause_summary: List[Dict[str, Any]]


class ContractAnalysisModule:
    """
    Module 1: Contract Analysis & Risk Auditor.
    Extracts clauses, indexes vectors, scans for contractual traps, and answers user questions with mandatory clause citations.
    """
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

    def process_and_index_contract(
        self, contract_id: str, contract_text: str, contract_name: str = "Uploaded Contract"
    ) -> ContractAnalysisReport:
        clauses = self.tokenizer.extract_clauses(contract_text)
        
        # Prepare vectors for collection
        docs = []
        metas = []
        ids = []

        for c in clauses:
            doc_id = f"{contract_id}_{c['clause_id']}"
            docs.append(c["text"])
            metas.append({
                "contract_id": contract_id,
                "contract_name": contract_name,
                "clause_id": c["clause_id"],
                "title": c["title"],
                "token_count": c["token_count"]
            })
            ids.append(doc_id)

        self.vector_store.add_documents(
            collection_name="contracts",
            documents=docs,
            metadatas=metas,
            ids=ids
        )

        # Run automated risk scanner
        risks = self._scan_contract_risks(clauses)

        # Compute overall risk score
        high_cnt = sum(1 for r in risks if r.risk_level == "HIGH")
        if high_cnt >= 2:
            overall = "HIGH RISK"
        elif high_cnt == 1 or len(risks) >= 3:
            overall = "MODERATE RISK"
        else:
            overall = "LOW RISK"

        return ContractAnalysisReport(
            contract_title=contract_name,
            total_clauses=len(clauses),
            overall_risk_score=overall,
            risks_found=risks,
            clause_summary=[{"clause_id": c["clause_id"], "title": c["title"], "tokens": c["token_count"]} for c in clauses]
        )

    def answer_contract_query(
        self, contract_id: str, user_query: str
    ) -> Dict[str, Any]:
        """
        RAG Q&A over contract clauses. Mandatory clause citation format e.g. [Clause X.Y].
        Passes through Cost Proxy and Hallucination Auditor.
        """
        retrieved = self.vector_store.query(
            collection_name="contracts",
            query_text=user_query,
            n_results=4,
            where_filter={"contract_id": contract_id}
        )

        # Fallback if no filter matched or empty collection
        if not retrieved:
            retrieved = self.vector_store.query(
                collection_name="contracts",
                query_text=user_query,
                n_results=4
            )

        packed = self.packer.pack_context(retrieved, token_budget=4000)

        system_prompt = (
            "You are LexAssist Contract Analysis Assistant.\n"
            "Answer the user's question relying strictly on the retrieved contract context below.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Every factual statement must cite its source clause in brackets e.g. [Clause 4.1] or [clause_1].\n"
            "2. Do not introduce external facts not present in the contract text.\n"
            "3. Conclude with: '*Legal Information Only — Not Legal Advice.*'"
        )

        full_prompt = f"Contract Context:\n{packed['formatted_context']}\n\nUser Question: {user_query}"

        # Route via Cost Optimization Proxy
        proxy_res = self.cost_proxy.route_and_execute(
            prompt=full_prompt,
            task_type="contract_analysis",
            system_prompt=system_prompt
        )

        # Audit via Hallucination Auditor
        audit_res = self.auditor.audit(
            generated_answer=proxy_res.content,
            reference_chunks=packed["packed_chunks"],
            user_query=user_query
        )

        return {
            "answer": proxy_res.content,
            "citations": audit_res.citations_verified,
            "hallucination_audit": audit_res.model_dump(),
            "telemetry": proxy_res.telemetry.model_dump(),
            "retrieved_clauses": packed["packed_chunks"]
        }

    def _scan_contract_risks(self, clauses: List[Dict[str, Any]]) -> List[RiskFlag]:
        risks = []
        for c in clauses:
            text_lower = c["text"].lower()
            title = c["title"]

            # 1. Unilateral Indemnity
            if "indemnify" in text_lower or "indemnification" in text_lower:
                if "solely" in text_lower or "unilateral" in text_lower or "hold harmless" in text_lower:
                    if "reciprocal" not in text_lower and "mutual" not in text_lower:
                        risks.append(RiskFlag(
                            risk_level="HIGH",
                            category="Indemnity",
                            clause_title=title,
                            clause_text=c["text"][:150] + "...",
                            issue_description="One-sided indemnification obligation detected. The party bears sole liability without mutual protection.",
                            recommendation="Negotiate reciprocal indemnification obligations and carve-outs for gross negligence."
                        ))

            # 2. Auto-Renewal Traps
            if "automatically renew" in text_lower or "auto-renew" in text_lower or "successive period" in text_lower:
                if "written notice" in text_lower:
                    risks.append(RiskFlag(
                        risk_level="MEDIUM",
                        category="Auto-Renewal",
                        clause_title=title,
                        clause_text=c["text"][:150] + "...",
                        issue_description="Automatic renewal clause detected with strict notice window requirements.",
                        recommendation="Set calendar reminders for termination notice cut-off or negotiate explicit opt-in renewal."
                    ))

            # 3. Liability Caps
            if "limitation of liability" in text_lower or "aggregate liability" in text_lower or "cap" in text_lower:
                if "exceed" in text_lower or "fees paid" in text_lower:
                    risks.append(RiskFlag(
                        risk_level="MEDIUM",
                        category="Liability",
                        clause_title=title,
                        clause_text=c["text"][:150] + "...",
                        issue_description="Liability is capped to fees paid in preceding months, which may limit recovery for major breaches.",
                        recommendation="Ensure exceptions to liability caps cover confidentiality breach, IP infringement, and willful misconduct."
                    ))

            # 4. Termination for Convenience
            if "terminate" in text_lower and "convenience" in text_lower:
                if "without cause" in text_lower or "without assigning any reason" in text_lower:
                    risks.append(RiskFlag(
                        risk_level="LOW",
                        category="Termination",
                        clause_title=title,
                        clause_text=c["text"][:150] + "...",
                        issue_description="Unilateral termination for convenience allowed.",
                        recommendation="Verify that notice period (e.g. 60-90 days) provides sufficient runway for business continuity."
                    ))

        return risks
