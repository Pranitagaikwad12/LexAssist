from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from app.backbone.auditor import HallucinationAuditor, SelfConsistencyAuditor
from app.backbone.cost_proxy import CostOptimizationProxy, ModelRouter


class CaseRecord(BaseModel):
    cnr_number: str
    case_number: str
    case_type: str
    court_name: str
    filing_date: str
    registration_date: str
    first_party: str
    second_party: str
    advocate_first_party: str
    advocate_second_party: str
    coram_bench: str
    stage_of_case: str
    next_hearing_date: str
    last_hearing_date: str
    disposal_status: str = Field(default="PENDING")
    order_summaries: List[Dict[str, str]] = Field(default_factory=list)
    filings_history: List[Dict[str, str]] = Field(default_factory=list)


class CaseStatusProvider(ABC):
    """
    Abstract Interface for Court Case Status & eCourts Lookup.
    Designed to plug cleanly into official eCourts / NJDG API endpoints when available.
    """
    @abstractmethod
    def get_case_status(self, identifier: str) -> Optional[CaseRecord]:
        pass

    @abstractmethod
    def search_cases(self, party_name: Optional[str] = None, advocate_name: Optional[str] = None) -> List[CaseRecord]:
        pass


class MockECourtsProvider(CaseStatusProvider):
    """
    Realistic mock provider reproducing eCourts record schema.
    """
    def __init__(self):
        self.records: Dict[str, CaseRecord] = {
            "DLHC010012342023": CaseRecord(
                cnr_number="DLHC010012342023",
                case_number="W.P.(C) 4820/2023",
                case_type="Writ Petition (Civil)",
                court_name="High Court of Delhi (Court No. 7)",
                filing_date="2023-04-12",
                registration_date="2023-04-15",
                first_party="M/s LexAssist Tech Solutions Pvt. Ltd.",
                second_party="Union of India & Anr.",
                advocate_first_party="Sh. Rajesh Malhotra, Adv.",
                advocate_second_party="Standing Counsel, UOI",
                coram_bench="Hon'ble Mr. Justice Sanjeev Sachdeva",
                stage_of_case="Final Arguments",
                next_hearing_date="2026-09-24",
                last_hearing_date="2026-07-10",
                disposal_status="PENDING",
                order_summaries=[
                    {"date": "2026-07-10", "order_type": "Interim Order", "summary": "Pleadings complete. Listed for final disposal on 24.09.2026."},
                    {"date": "2026-03-15", "order_type": "Direction", "summary": "Counter affidavit filed by Respondent No. 1. Rejoinder permitted within 4 weeks."}
                ],
                filings_history=[
                    {"date": "2026-04-02", "document": "Rejoinder Affidavit"},
                    {"date": "2026-03-10", "document": "Counter Affidavit by UOI"}
                ]
            ),
            "MHOS019988772024": CaseRecord(
                cnr_number="MHOS019988772024",
                case_number="Comm. Arbitration Petition 312/2024",
                case_type="Commercial Arbitration Petition",
                court_name="High Court of Judicature at Bombay",
                filing_date="2024-01-20",
                registration_date="2024-01-22",
                first_party="Apex Infrastructure Corp",
                second_party="State of Maharashtra & Ors.",
                advocate_first_party="M/s Crawford Bayley & Co.",
                advocate_second_party="Government Pleader",
                coram_bench="Hon'ble Mr. Justice G.S. Kulkarni",
                stage_of_case="For Admission / Interim Relief",
                next_hearing_date="2026-10-05",
                last_hearing_date="2026-08-01",
                disposal_status="PENDING",
                order_summaries=[
                    {"date": "2026-08-01", "order_type": "Notice Issued", "summary": "Notice issued under Section 9 of Arbitration Act. Ad-interim protection extended."}
                ],
                filings_history=[
                    {"date": "2024-01-20", "document": "Petition under Section 9 with Exhibits"}
                ]
            ),
            "SLP142052023": CaseRecord(
                cnr_number="SCIN000142052023",
                case_number="Special Leave Petition (Civil) No. 14205/2023",
                case_type="Special Leave Petition",
                court_name="Supreme Court of India (Court No. 3)",
                filing_date="2023-08-05",
                registration_date="2023-08-10",
                first_party="State Bank of India",
                second_party="Religare Finvest Ltd. & Ors.",
                advocate_first_party="Sh. Mukul Rohatgi, Sr. Adv.",
                advocate_second_party="Sh. Kapil Sibal, Sr. Adv.",
                coram_bench="Hon'ble The Chief Justice of India & Hon'ble Mr. Justice J.B. Pardiwala",
                stage_of_case="For Orders / Admission",
                next_hearing_date="2026-09-18",
                last_hearing_date="2026-05-14",
                disposal_status="PENDING",
                order_summaries=[
                    {"date": "2026-05-14", "order_type": "Record of Proceedings", "summary": "Tag along with SLP(C) No. 11002/2023. List on 18.09.2026."}
                ],
                filings_history=[
                    {"date": "2023-08-05", "document": "SLP Petition with Synopsis and List of Dates"}
                ]
            )
        }

    def get_case_status(self, identifier: str) -> Optional[CaseRecord]:
        clean_id = identifier.replace(" ", "").replace(".", "").replace("/", "").upper()
        
        # Direct lookup by CNR
        if identifier in self.records:
            return self.records[identifier]

        # Match by substring or case number
        for cnr, rec in self.records.items():
            if clean_id in cnr.upper() or clean_id in rec.case_number.replace(" ", "").replace(".", "").replace("/", "").upper():
                return rec

        return None

    def search_cases(self, party_name: Optional[str] = None, advocate_name: Optional[str] = None) -> List[CaseRecord]:
        results = []
        for rec in self.records.values():
            if party_name:
                p_lower = party_name.lower()
                if p_lower in rec.first_party.lower() or p_lower in rec.second_party.lower():
                    results.append(rec)
            elif advocate_name:
                a_lower = advocate_name.lower()
                if a_lower in rec.advocate_first_party.lower() or a_lower in rec.advocate_second_party.lower():
                    results.append(rec)
        return results


class CaseStatusModule:
    """
    Module 3: Court Case Status & Investigation Lookup.
    Retrieval and summarization ONLY — strict anti-hallucination policy.
    If record does not exist in provider, explicitly reports "Record Not Found".
    """
    def __init__(
        self,
        provider: Optional[CaseStatusProvider] = None,
        auditor: Optional[HallucinationAuditor] = None,
        cost_proxy: Optional[CostOptimizationProxy] = None
    ):
        self.provider = provider or MockECourtsProvider()
        self.auditor = auditor or SelfConsistencyAuditor()
        self.cost_proxy = cost_proxy or ModelRouter()

    def lookup_case_status(self, query: str) -> Dict[str, Any]:
        record = self.provider.get_case_status(query)

        if not record:
            # Check party search fallback
            search_results = self.provider.search_cases(party_name=query)
            if search_results:
                record = search_results[0]

        if not record:
            return {
                "status_found": False,
                "message": f"RECORD NOT FOUND: No eCourts litigation record matches identifier/party '{query}'.",
                "summary": "No case record was found in the official eCourts provider database. LexAssist strictly avoids generating fake or estimated case status details to prevent hallucination.",
                "hallucination_audit": {
                    "grounding_score": 1.0,
                    "confidence_level": "High",
                    "is_hallucination_detected": False,
                    "ungrounded_claims": [],
                    "citations_verified": ["eCourts Provider Log"],
                    "rationale": "Record missing in provider; anti-hallucination strict fallback triggered."
                }
            }

        # Prepare factual reference summary
        record_dict = record.model_dump()
        ref_chunks = [{
            "id": record.cnr_number,
            "document": str(record_dict),
            "metadata": {"citation_id": f"eCourts Record {record.case_number}"}
        }]

        system_prompt = (
            "You are LexAssist eCourts Case Status Analyst.\n"
            "Summarize the court case status relying STRICTLY on the retrieved eCourts record dict.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. State exact Case Number, CNR, Court, Stage, Next Hearing Date, and Bench.\n"
            "2. DO NOT invent or extrapolate missing hearing dates or orders.\n"
            "3. Conclude with: '*Data verified against eCourts Provider. Legal Information Only.*'"
        )

        prompt = f"Official eCourts Case Record:\n{record_dict}\n\nUser Lookup Query: {query}"

        proxy_res = self.cost_proxy.route_and_execute(
            prompt=prompt,
            task_type="case_status",
            system_prompt=system_prompt
        )

        audit_res = self.auditor.audit(
            generated_answer=proxy_res.content,
            reference_chunks=ref_chunks,
            user_query=query
        )

        return {
            "status_found": True,
            "record": record_dict,
            "summary": proxy_res.content,
            "hallucination_audit": audit_res.model_dump(),
            "telemetry": proxy_res.telemetry.model_dump()
        }
