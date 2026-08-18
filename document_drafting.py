from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import jinja2

from app.backbone.auditor import HallucinationAuditor, SelfConsistencyAuditor
from app.backbone.cost_proxy import CostOptimizationProxy, ModelRouter


class DocumentTemplate(BaseModel):
    template_id: str
    title: str
    version: str
    description: str
    required_fields: List[str]
    template_content: str


class DocumentDraftingModule:
    """
    Module 4: Legal Document Drafting Engine.
    Fills versioned, validated legal document templates from structured user inputs.
    Prevents free-form generative drift and enforces strict variable validation.
    """
    TEMPLATES: Dict[str, DocumentTemplate] = {
        "nda_mutual": DocumentTemplate(
            template_id="nda_mutual",
            title="Mutual Non-Disclosure Agreement (NDA)",
            version="1.2",
            description="Standard mutual NDA governed by Indian Contract Act, 1872 with 3-year confidentiality term.",
            required_fields=["disclosing_party", "receiving_party", "effective_date", "purpose", "jurisdiction_city"],
            template_content="""MUTUAL NON-DISCLOSURE AGREEMENT

This Mutual Non-Disclosure Agreement ("Agreement") is made and entered into on this {{ effective_date }} ("Effective Date") at {{ jurisdiction_city }}, India.

BY AND BETWEEN:

1. {{ disclosing_party }}, having its principal place of business at {{ disclosing_party_address|default("[Disclosing Party Address]") }} (hereinafter referred to as "First Party");

AND

2. {{ receiving_party }}, having its principal place of business at {{ receiving_party_address|default("[Receiving Party Address]") }} (hereinafter referred to as "Second Party").

WHEREAS, the First Party and Second Party (collectively referred to as "Parties") desire to evaluate and engage in discussions concerning {{ purpose }} ("Purpose").

NOW THEREFORE, in consideration of the mutual covenants contained herein, the Parties agree as follows:

1. DEFINITION OF CONFIDENTIAL INFORMATION [Clause 1.1]
"Confidential Information" shall mean all non-public technical, business, financial, operational, or legal information disclosed by either Party to the other Party.

2. OBLIGATIONS OF CONFIDENTIALITY [Clause 2.1]
Each Party agrees to hold all Confidential Information in strict confidence and shall not disclose such information to any third party without prior written consent, for a period of {{ term_years|default("3") }} years from the Effective Date.

3. EXCLUSIONS [Clause 3.1]
Confidential Information shall not include information that is publicly known through no breach of this Agreement, or was independently developed without reference to the Disclosing Party's information.

4. GOVERNING LAW & JURISDICTION [Clause 9.1]
This Agreement shall be governed by and construed in accordance with the laws of India. The courts located at {{ jurisdiction_city }} shall have exclusive jurisdiction over any disputes arising under this Agreement.

IN WITNESS WHEREOF, the Parties have executed this Agreement as of the Effective Date written above.

FIRST PARTY: {{ disclosing_party }}
By: ___________________________
Name: _________________________
Title: ________________________

SECOND PARTY: {{ receiving_party }}
By: ___________________________
Name: _________________________
Title: ________________________
"""
        ),
        "legal_notice_breach": DocumentTemplate(
            template_id="legal_notice_breach",
            title="Legal Notice for Breach of Contract / Non-Payment",
            version="2.0",
            description="Formal statutory demand notice under Indian Contract Act for non-payment or contractual default.",
            required_fields=["sender_name", "recipient_name", "agreement_date", "default_amount", "notice_days", "advocate_name"],
            template_content="""LEGAL NOTICE / DEMAND NOTICE

BY REGISTERED AD / SPEED POST

Date: {{ notice_date|default("TODAY") }}

TO,
{{ recipient_name }}
{{ recipient_address|default("[Recipient Address]") }}

SUBJECT: LEGAL NOTICE FOR BREACH OF AGREEMENT DATED {{ agreement_date }} AND DEMAND FOR PAYMENT OF RS. {{ default_amount }}

Dear Sir/Madam,

Under instructions from and on behalf of my client, {{ sender_name }} ("My Client"), I hereby serve upon you this Legal Notice:

1. That My Client entered into a binding agreement with you on {{ agreement_date }} ("Agreement") for the provision of legal/technical services [Clause 1.1].

2. That pursuant to the Agreement, My Client performed all requisite obligations. However, in gross breach of your contractual commitments [Section 73, Indian Contract Act, 1872], you have failed to clear outstanding dues amounting to Rs. {{ default_amount }}/- (Rupees {{ default_amount }} Only) despite repeated reminders.

3. TAKE NOTICE that you are hereby called upon to pay the entire outstanding amount of Rs. {{ default_amount }}/- along with interest @ 18% per annum within {{ notice_days }} days from the receipt of this notice.

4. FAIL NOT, failing which My Client has given me clear instructions to initiate appropriate civil litigation and legal proceedings before the competent courts at {{ jurisdiction_city|default("New Delhi") }} at your risk, cost, and consequence.

Copy retained in my office for record.

Sincerely,

_______________________________
{{ advocate_name }}, Advocate
High Court / District Court
"""
        )
    }

    def __init__(
        self,
        auditor: Optional[HallucinationAuditor] = None,
        cost_proxy: Optional[CostOptimizationProxy] = None
    ):
        self.auditor = auditor or SelfConsistencyAuditor()
        self.cost_proxy = cost_proxy or ModelRouter()

    def get_available_templates(self) -> List[Dict[str, Any]]:
        return [
            {
                "template_id": t.template_id,
                "title": t.title,
                "version": t.version,
                "description": t.description,
                "required_fields": t.required_fields
            }
            for t in self.TEMPLATES.values()
        ]

    def validate_inputs(self, template_id: str, user_inputs: Dict[str, Any]) -> Dict[str, Any]:
        template = self.TEMPLATES.get(template_id)
        if not template:
            return {"valid": False, "missing_fields": [], "error": f"Template '{template_id}' not found."}

        missing = [f for f in template.required_fields if not user_inputs.get(f)]
        if missing:
            return {
                "valid": False,
                "missing_fields": missing,
                "error": f"Missing required fields for template generation: {', '.join(missing)}"
            }

        return {"valid": True, "missing_fields": [], "error": None}

    def generate_draft(self, template_id: str, user_inputs: Dict[str, Any]) -> Dict[str, Any]:
        val = self.validate_inputs(template_id, user_inputs)
        if not val["valid"]:
            return {
                "success": False,
                "error": val["error"],
                "missing_fields": val["missing_fields"]
            }

        template = self.TEMPLATES[template_id]
        
        # Render Jinja2 template
        j2_env = jinja2.Environment()
        j2_tpl = j2_env.from_string(template.template_content)
        rendered_document = j2_tpl.render(**user_inputs)

        # Audit generated template document
        audit_res = self.auditor.audit(
            generated_answer=rendered_document,
            reference_chunks=[{
                "id": f"Template_{template_id}",
                "document": template.template_content,
                "metadata": {"citation_id": f"{template.title} v{template.version}"}
            }],
            user_query=f"Draft {template.title}"
        )

        return {
            "success": True,
            "template_id": template_id,
            "document_title": template.title,
            "version": template.version,
            "rendered_text": rendered_document,
            "hallucination_audit": audit_res.model_dump(),
            "telemetry": {
                "selected_model": "Structured Template Engine (v" + template.version + ")",
                "model_tier": "Deterministic Template",
                "input_tokens": len(str(user_inputs)) // 4,
                "output_tokens": len(rendered_document) // 4,
                "total_tokens": (len(str(user_inputs)) + len(rendered_document)) // 4,
                "estimated_cost_usd": 0.0,
                "estimated_cost_inr": 0.0,
                "routing_reason": "Zero generative drift deterministic template rendering.",
                "proxy_name": "LexAssist Drafting Engine"
            }
        }
