import re
from typing import Dict, Any, List
from pydantic import BaseModel, Field


class IntentClassification(BaseModel):
    module: str = Field(..., description="contract_analysis, case_law_search, case_status, document_drafting, legal_explainer, general_qa")
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str
    suggested_action: str
    extracted_entities: Dict[str, Any] = Field(default_factory=dict)


class IntentRouter:
    """
    LLM + Pattern matching router for LexAssist.
    Dispatches user query to one of the 5 legal modules or general legal Q&A fallback.
    """

    PATTERNS = {
        "contract_analysis": [
            r'\b(?:contract|agreement|indemnity|liability|clause|termination|auto-renewal|non-compete|governing law|breach of contract)\b',
            r'\b(?:review this|analyze this|risk report|flag risks|one-sided)\b'
        ],
        "case_law_search": [
            r'\b(?:precedent|case law|judgement|judgment|supreme court|high court|kanoon|scc|air|ratio|ruling|bench)\b',
            r'\b(?:find cases|find precedents|legal precedents|cite cases|court decisions)\b'
        ],
        "case_status": [
            r'\b(?:case status|cnr|hearing date|next hearing|ecourts|litigation status|case number|filing number|bench status|order copy)\b',
            r'\b(?:check status|investigation lookup|track case|case stage)\b'
        ],
        "document_drafting": [
            r'\b(?:draft|generate|template|create nda|affidavit|legal notice|reply to notice|draft agreement|power of attorney)\b',
            r'\b(?:prepare document|drafting engine|legal draft)\b'
        ],
        "legal_explainer": [
            r'\b(?:what does|meaning of|define|statute|section|ipc|bns|crpc|bnss|cpc|anticipatory bail|fir|bailable|cognizable|constitution)\b',
            r'\b(?:explain legal|statutory provision|bare act|legal concept)\b'
        ]
    }

    def route_query(self, query: str) -> IntentClassification:
        query_lower = query.lower()

        scores = {
            "contract_analysis": 0,
            "case_law_search": 0,
            "case_status": 0,
            "document_drafting": 0,
            "legal_explainer": 0
        }

        extracted_entities = {}

        # 1. Pattern matching scoring
        for module, regexes in self.PATTERNS.items():
            for regex in regexes:
                matches = re.findall(regex, query_lower)
                scores[module] += len(matches) * 2

        # Entity Extraction hints
        cnr_match = re.search(r'\b[A-Z0-9]{16}\b', query)
        if cnr_match:
            scores["case_status"] += 5
            extracted_entities["cnr_number"] = cnr_match.group(0)

        section_match = re.search(r'section\s+(\d+[A-Z]?)', query_lower)
        if section_match:
            scores["legal_explainer"] += 4
            extracted_entities["section"] = section_match.group(1)

        # 2. Select winning module
        sorted_modules = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_module, top_score = sorted_modules[0]

        if top_score == 0:
            return IntentClassification(
                module="general_qa",
                confidence=0.70,
                explanation="Query does not contain explicit module keywords; routed to General Legal Q&A fallback.",
                suggested_action="Perform grounded legal Q&A using statute context and general legal principles.",
                extracted_entities=extracted_entities
            )

        confidence = min(0.98, round(0.65 + (top_score * 0.08), 2))

        explanations = {
            "contract_analysis": "Detected contract clause review and risk scanner intent.",
            "case_law_search": "Detected precedent and judicial decision search intent.",
            "case_status": "Detected court case status or eCourts investigation lookup intent.",
            "document_drafting": "Detected legal document template drafting intent.",
            "legal_explainer": "Detected legal term definition or statute section explanation intent."
        }

        actions = {
            "contract_analysis": "Ingest contract clauses, rank risks, and present grounded Q&A with clause citations.",
            "case_law_search": "Query Indian Kanoon API, extract precedents, and summarize ratio decidendi with paragraph citations.",
            "case_status": "Query eCourts provider database for exact status record without hallucinating facts.",
            "document_drafting": "Load template, validate required variables, and render draft document.",
            "legal_explainer": "Retrieve bare act statute sections and generate grounded plain-English breakdown with citations."
        }

        return IntentClassification(
            module=top_module,
            confidence=confidence,
            explanation=explanations[top_module],
            suggested_action=actions[top_module],
            extracted_entities=extracted_entities
        )
