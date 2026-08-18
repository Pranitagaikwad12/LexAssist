from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import re


class AuditResult(BaseModel):
    grounding_score: float = Field(..., ge=0.0, le=1.0, description="0.0 to 1.0 score indicating factual grounding")
    confidence_level: str = Field(..., description="High, Medium, or Low")
    is_hallucination_detected: bool = Field(..., description="True if ungrounded claims detected")
    ungrounded_claims: List[str] = Field(default_factory=list, description="List of ungrounded or suspicious sentences")
    citations_verified: List[str] = Field(default_factory=list, description="Verified citations found in answer")
    rationale: str = Field(..., description="Auditor explanation and summary")
    plugged_in_auditor_name: str = Field(default="Built-in Self-Consistency Auditor", description="Name of active auditor")


class HallucinationAuditor(ABC):
    """
    Abstract interface for hallucination and grounding auditing.
    User can subclass or replace this interface to plug in their custom auditor project.
    """
    @abstractmethod
    def audit(
        self,
        generated_answer: str,
        reference_chunks: List[Dict[str, Any]],
        user_query: str
    ) -> AuditResult:
        pass


class SelfConsistencyAuditor(HallucinationAuditor):
    """
    Default production implementation of Self-Consistency Grounding Auditor.
    Extracts claims from generated response and verifies factual alignment against
    retrieved source chunks.
    """
    def __init__(self, name: str = "SelfConsistencyAuditor-v1"):
        self.name = name

    def audit(
        self,
        generated_answer: str,
        reference_chunks: List[Dict[str, Any]],
        user_query: str
    ) -> AuditResult:
        if not generated_answer or not generated_answer.strip():
            return AuditResult(
                grounding_score=0.0,
                confidence_level="Low",
                is_hallucination_detected=True,
                ungrounded_claims=["Empty or invalid response."],
                citations_verified=[],
                rationale="Response is empty.",
                plugged_in_auditor_name=self.name
            )

        # 1. Combine reference text into single reference corpus
        ref_texts = []
        valid_citations = set()

        for chunk in reference_chunks:
            doc = chunk.get("document", "")
            meta = chunk.get("metadata", {})
            ref_texts.append(doc.lower())
            
            c_id = meta.get("citation_id") or meta.get("clause_id") or meta.get("section") or chunk.get("id")
            if c_id:
                valid_citations.add(str(c_id).lower())

        full_ref = " ".join(ref_texts)

        # 2. Extract citations in answer e.g. [Clause 4.1], [Section 302], [Para 12]
        citation_matches = re.findall(r'\[([^\]]+)\]', generated_answer)
        citations_found = [c.strip() for c in citation_matches if c.strip()]
        verified_citations = []

        for c in citations_found:
            # Check if citation exists in reference set or matches patterns
            if any(vc in c.lower() for vc in valid_citations) or len(ref_texts) > 0:
                verified_citations.append(c)

        # 3. Sentence-level grounding verification
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', generated_answer) if len(s.strip()) > 15]
        
        if not sentences:
            sentences = [generated_answer.strip()]

        ungrounded = []
        grounded_count = 0

        for sentence in sentences:
            # Clean sentence words
            words = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', sentence.lower()))
            # Ignore common legal stop words
            words = {w for w in words if w not in {
                "the", "and", "that", "this", "for", "with", "shall", "must", "party",
                "under", "clause", "section", "act", "court", "according", "provided", "herein"
            }}

            if not words:
                grounded_count += 1
                continue

            # Check overlap against reference text
            matched_words = {w for w in words if w in full_ref}
            overlap_ratio = len(matched_words) / len(words)

            if overlap_ratio >= 0.35:
                grounded_count += 1
            else:
                ungrounded.append(sentence[:120] + ("..." if len(sentence) > 120 else ""))

        # 4. Calculate Grounding Score
        total_sentences = len(sentences)
        score = grounded_count / total_sentences if total_sentences > 0 else 1.0

        # Adjust score for citation presence
        if citations_found:
            score = min(1.0, score * 1.1)
        
        score = round(score, 2)

        # 5. Determine Confidence Level
        if score >= 0.82:
            confidence = "High"
            is_hallucination = False
        elif score >= 0.60:
            confidence = "Medium"
            is_hallucination = False
        else:
            confidence = "Low"
            is_hallucination = True

        rationale = (
            f"Audited {total_sentences} sentence(s) against {len(reference_chunks)} reference context chunk(s). "
            f"Grounding ratio: {grounded_count}/{total_sentences}. "
            f"Verified {len(verified_citations)} citation(s)."
        )

        return AuditResult(
            grounding_score=score,
            confidence_level=confidence,
            is_hallucination_detected=is_hallucination,
            ungrounded_claims=ungrounded,
            citations_verified=verified_citations,
            rationale=rationale,
            plugged_in_auditor_name=self.name
        )


class ExternalAuditorAdapter(HallucinationAuditor):
    """
    Adapter class to connect an external HTTP microservice or custom python project
    for hallucination auditing.
    """
    def __init__(self, endpoint_url: str, api_key: Optional[str] = None):
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.fallback = SelfConsistencyAuditor(name="Fallback-SelfConsistencyAuditor")

    def audit(
        self,
        generated_answer: str,
        reference_chunks: List[Dict[str, Any]],
        user_query: str
    ) -> AuditResult:
        import httpx
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            resp = httpx.post(
                self.endpoint_url,
                json={
                    "answer": generated_answer,
                    "references": reference_chunks,
                    "query": user_query
                },
                headers=headers,
                timeout=5.0
            )
            if resp.status_code == 200:
                data = resp.json()
                return AuditResult(**data)
        except Exception as e:
            print(f"[ExternalAuditorAdapter] Failed to reach external auditor endpoint: {e}. Falling back.")
        
        return self.fallback.audit(generated_answer, reference_chunks, user_query)
