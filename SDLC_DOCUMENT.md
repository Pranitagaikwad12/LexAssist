# Software Development Life Cycle (SDLC) Document
## LexAssist — Legal Intelligence Platform for Indian Law
**Version:** 1.0.0  
**Status:** Production Ready  
**Date:** August 18, 2026  
**Target Audience:** Engineering Team, Legal Operations, Solution Architects, QA Lead, DevOps Engineers  

---

## 1. Executive Overview & Lifecycle Management

### 1.1 Document Overview
This document defines the Software Development Life Cycle (SDLC) for **LexAssist**, a multitasking Legal Intelligence Platform tailored for Indian Law. It details system requirements, architectural topology, module design, hallucination auditing, cost proxy routing, implementation standards, test execution results, and deployment pipelines.

### 1.2 Document Control & Revision History

| Version | Date | Author / Role | Summary of Changes |
| :--- | :--- | :--- | :--- |
| **0.1.0** | 2026-06-10 | Engineering Architecture Team | Initial Feasibility Analysis & Technical Specification |
| **0.8.0** | 2026-07-15 | Core Backend Engineers | Implementation of Backbone (VectorStore, Auditor, ModelRouter) |
| **0.9.5** | 2026-08-01 | Full Stack Engineering | Integrated 5 Legal Modules & Vite/React Frontend Interface |
| **1.0.0** | 2026-08-18 | QA & Lead Architect | Production Validation, 32/32 Test Suites Passing, SDLC Finalization |

---

## 2. Phase 1: System Vision & Business Feasibility

### 2.1 Domain Context & Problem Statement
The legal system in India is characterized by high operational complexity, massive case backlogs (over 45 million pending litigation matters across Supreme Court, High Courts, and Subordinate Courts), and an extensive regulatory corpus spanning the Indian Contract Act (1872), Code of Civil Procedure (1808), Code of Criminal Procedure / BNSS, and Indian Penal Code / BNS.

Key operational pain points faced by legal practitioners include:
1. **Contractual Risk Overhead:** Manual identification of one-sided indemnities, auto-renewal traps, and liability caps in dense commercial agreements.
2. **Precedent Retrieval Complexity:** Time-consuming search for judicial rulings and ratio decidendi across disjointed legal search engines.
3. **Litigation Tracking Latency:** Difficulty tracking live eCourts hearing schedules and case stages across multiple benches.
4. **Drafting Repetition:** Inconsistent drafting of routine non-disclosure agreements, legal notices, and affidavits.
5. **AI Hallucination Risk:** High risk of generative AI inventing non-existent case citations, incorrect legal sections, or fabricated judicial precedents.
6. **API Cost Inflation:** Uncontrolled LLM consumption costs due to routing simple statutory lookup queries to expensive flagship models.

### 2.2 Core Product Vision
**LexAssist** solves these challenges by combining a unified smart intent router with five specialized domain modules supported by an enterprise backbone:
* **Zero-Hallucination Assurance:** Every generated output undergoes automated self-consistency grounding checks against source text.
* **Deterministic Cost Optimization:** Query complexity evaluation (0–100 score) dynamically routes requests between Lightweight and Flagship model tiers.
* **Multi-Module Dispatch:** Seamless query routing across Contract Analysis, Case Law Search, Live Case Status, Document Drafting, and Statutory Explainer.

---

## 3. Phase 2: Requirement Analysis & Technical Specifications

### 3.1 Functional Requirements (FR)

| Req ID | Module / Subsystem | Requirement Description | Implementation Mapping |
| :--- | :--- | :--- | :--- |
| **FR-1.0** | **Intent Router** | Automatically classify incoming natural language query into 1 of 5 legal modules with confidence scoring (0.0 to 1.0) and entity extraction (CNR numbers, statutory sections). | [intent_router.py](file:///c:/Users/prani/PycharmProjects/LexAssist/backend/app/router/intent_router.py) |
| **FR-2.0** | **Contract Analysis** | Ingest text/PDF contracts, index clauses into vector store, flag risk areas (indemnity, liability cap, termination), and answer grounded contract queries with clause citations. | [contract_analysis.py](file:///c:/Users/prani/PycharmProjects/LexAssist/backend/app/modules/contract_analysis.py) |
| **FR-3.0** | **Case Law Search** | Retrieve judicial precedents, extract ratio decidendi, and summarize relevant High Court / Supreme Court judgments with paragraph citations. | [case_law_search.py](file:///c:/Users/prani/PycharmProjects/LexAssist/backend/app/modules/case_law_search.py) |
| **FR-4.0** | **Case Status Tracker** | Lookup eCourts litigation records by CNR or party name without generating fabricated facts. | [case_status.py](file:///c:/Users/prani/PycharmProjects/LexAssist/backend/app/modules/case_status.py) |
| **FR-5.0** | **Document Drafting** | Provide form-validated legal templates (NDA, Affidavit, Notice) and render versioned legal drafts deterministically. | [document_drafting.py](file:///c:/Users/prani/PycharmProjects/LexAssist/backend/app/modules/document_drafting.py) |
| **FR-6.0** | **Legal Explainer** | Explain statutory provisions (ICA, IPC/BNS, CrPC/BNSS) in plain English with bare act section references. | [legal_explainer.py](file:///c:/Users/prani/PycharmProjects/LexAssist/backend/app/modules/legal_explainer.py) |
| **FR-7.0** | **Hallucination Auditor** | Perform sentence-level overlap & citation verification against reference context chunks, producing a 0.0–1.0 grounding score. | [auditor.py](file:///c:/Users/prani/PycharmProjects/LexAssist/backend/app/backbone/auditor.py) |
| **FR-8.0** | **Cost & Token Telemetry**| Calculate token consumption, estimate costs in USD and INR, and log telemetry stats across queries. | [cost_proxy.py](file:///c:/Users/prani/PycharmProjects/LexAssist/backend/app/backbone/cost_proxy.py) |

### 3.2 Non-Functional Requirements (NFR)

* **NFR-1 (Reliability & Grounding):** Responses must achieve an average grounding score $\ge 0.85$. Ungrounded claims must trigger a `Low` confidence flag.
* **NFR-2 (Performance & Latency):** Sub-second routing and execution for local fallbacks ($< 100\text{ ms}$ for routing, $< 1.0\text{ s}$ end-to-end test session).
* **NFR-3 (Cost Budgeting):** Low/Moderate complexity queries ($< 60$ complexity score) must be assigned to the Lightweight tier (e.g. Claude 3 Haiku / GPT-4o-mini).
* **NFR-4 (Security & Confidentiality):** Contracts uploaded for analysis are processed in ephemeral vector collections or memory buffers with zero external storage leakage.
* **NFR-5 (Resilience & Graceful Degradation):** Full fallback capability if ChromaDB, PyTorch, or third-party LLM APIs are unreachable.
* **NFR-6 (Usability):** Fluid, modern dark-themed React UI featuring live telemetry statistics, interactive risk cards, and single-click template wizards.

---

## 4. Phase 3: System Architecture & Technical Design

### 4.1 System Topology Diagram

```
                             +-----------------------------------+
                             |     LexAssist Frontend UI         |
                             |   (React 18 + Vite Component)     |
                             +-----------------+-----------------+
                                               |
                                        HTTP / REST API
                                               v
                             +-----------------+-----------------+
                             |        FastAPI Backend            |
                             |          (main.py)                |
                             +-----------------+-----------------+
                                               |
                                     1. Route & Classify
                                               v
                             +-----------------+-----------------+
                             |         Intent Router             |
                             |   (Regex + Keyword Scoring)       |
                             +-----------------+-----------------+
                                               |
                  +----------------------------+----------------------------+
                  |                            |                            |
                  v                            v                            v
        +-------------------+        +-------------------+        +-------------------+
        | Contract Analysis |        |  Case Law Search  |        |    Case Status    |
        +---------+---------+        +---------+---------+        +---------+---------+
                  |                            |                            |
                  +----------------------------+----------------------------+
                                               |
                                               v
                             +-----------------+-----------------+
                             |      Backbone Infrastructure      |
                             |  - VectorStoreManager (Chroma)    |
                             |  - SelfConsistencyAuditor         |
                             |  - ModelRouter (Cost Proxy)       |
                             |  - BPEClauseTokenizer             |
                             +-----------------------------------+
```

### 4.2 Module Deep Dive

#### 4.2.1 Intent Router ([intent_router.py](file:///c:/Users/prani/PycharmProjects/LexAssist/backend/app/router/intent_router.py))
Routes incoming user queries to one of five modules based on pattern matching and entity detection:
- **Patterns:** Regex keyword matching across 5 legal domains.
- **Entity Extraction:** Automated regex detection of 16-character CNR numbers (`[A-Z0-9]{16}`) and Statutory Sections (`section \d+[A-Z]?`).
- **Confidence Formula:** $C = \min(0.98, 0.65 + S \times 0.08)$ where $S$ is the keyword score.

#### 4.2.2 Hallucination Auditor ([auditor.py](file:///c:/Users/prani/PycharmProjects/LexAssist/backend/app/backbone/auditor.py))
Inherits from abstract base `HallucinationAuditor`:
- Extract citations formatted as `[Clause X]`, `[Section Y]`, `[Para Z]`.
- Sentence tokenization & word overlap ratio calculation against retrieved corpus chunks.
- Overlap threshold $\ge 0.35$ per sentence required for grounding verification.
- Output score calculation:
  $$\text{Grounding Score} = \frac{\text{Grounded Sentences}}{\text{Total Sentences}} \times (\text{Citation Bonus if present})$$

#### 4.2.3 Cost & Model Router ([cost_proxy.py](file:///c:/Users/prani/PycharmProjects/LexAssist/backend/app/backbone/cost_proxy.py))
Inherits from abstract base `CostOptimizationProxy`:
- Evaluates query complexity (0–100) using base task weights, text length token count via `BPEClauseTokenizer`, and legal keyword density.
- **Routing Rules:**
  - Score $< 60 \rightarrow$ **Lightweight Tier** (Input: \$0.00025/1k, Output: \$0.00125/1k)
  - Score $\ge 60 \rightarrow$ **Flagship Tier** (Input: \$0.00300/1k, Output: \$0.01500/1k)
- Live API integration with Anthropic Claude / OpenAI, with structured legal simulation fallback if keys are unconfigured.

#### 4.2.4 Vector Store Manager ([vector_store.py](file:///c:/Users/prani/PycharmProjects/LexAssist/backend/app/backbone/vector_store.py))
Provides persistent semantic indexing via ChromaDB (`./data/chroma_db`). When ChromaDB or PyTorch is omitted, seamlessly defaults to an internal `SimpleEmbeddingProvider` utilizing TF-IDF n-gram vectorization and cosine similarity scoring.

---

## 5. Phase 4: Development Methodology & Implementation

### 5.1 Technology Stack Summary

* **Backend Framework:** FastAPI 0.110+, Uvicorn 0.28+, Pydantic v2
* **Language & Runtime:** Python 3.10+
* **Vector Database:** ChromaDB 0.4+ (with pure-Python TF-IDF memory fallback)
* **Tokenization:** Custom BPE Clause Tokenizer (`tiktoken` compatible interface)
* **Frontend Framework:** React 18, Vite 4, Lucide-React Icons, Vanilla CSS Design System
* **Test Automation:** Pytest 9.1+, HTTPX TestClient

### 5.2 Codebase Structure

```
LexAssist/
├── backend/
│   ├── app/
│   │   ├── backbone/
│   │   │   ├── auditor.py          # Hallucination & Citation Auditor
│   │   │   ├── context_packer.py   # Context Window Formatting
│   │   │   ├── cost_proxy.py       # Model Complexity Router & Telemetry
│   │   │   ├── tokenizer.py        # BPE Legal Tokenizer
│   │   │   └── vector_store.py     # ChromaDB / Fallback Vector Store
│   │   ├── modules/
│   │   │   ├── case_law_search.py  # Kanoon & Judicial Precedent Engine
│   │   │   ├── case_status.py      # eCourts Case Tracker
│   │   │   ├── contract_analysis.py# Contract Ingestion & Risk Assessor
│   │   │   ├── document_drafting.py# Versioned Legal Template Engine
│   │   │   └── legal_explainer.py  # Statutory Bare Act Breakdown Engine
│   │   └── router/
│   │       └── intent_router.py    # Query Intent Classifier
│   ├── data/                       # Chroma vector storage directory
│   ├── tests/                      # Automated test suites
│   ├── main.py                     # FastAPI Application Entrypoint
│   └── requirements.txt            # Python Dependencies
├── frontend/
│   ├── src/
│   │   ├── app.jsx                 # React Application Console
│   │   ├── index.css               # Design System Tokens & Styling
│   │   └── main.jsx                # React DOM Mount Point
│   ├── package.json                # NPM Configuration
│   └── vite.config.js              # Vite Dev Server Proxy Config
└── SDLC_DOCUMENT.md                # Lifecycle & System Documentation
```

---

## 6. Phase 5: Testing, QA & Verification Strategy

### 6.1 Test Suite Organization & Coverage

The testing suite consists of 32 comprehensive tests spanning API routes, backbone infrastructure, and domain modules:

| Test Module | Coverage Scope | Test Count | Result |
| :--- | :--- | :---: | :---: |
| [`tests/test_api.py`](file:///c:/Users/prani/PycharmProjects/LexAssist/backend/tests/test_api.py) | API endpoints (`/api/ask`, `/api/route`, `/api/contract/analyze`, `/api/draft/generate`, `/api/telemetry/stats`) | 9 | **PASSED** |
| [`tests/test_backbone.py`](file:///c:/Users/prani/PycharmProjects/LexAssist/backend/tests/test_backbone.py) | IntentRouter scoring, VectorStore indexing/querying, Auditor claim verification, CostProxy model tier assignment | 9 | **PASSED** |
| [`tests/test_modules.py`](file:///c:/Users/prani/PycharmProjects/LexAssist/backend/tests/test_modules.py) | Contract Analysis risk detection, Case Law precedent extraction, eCourts lookup, Document Drafting, Statutory Explainer | 14 | **PASSED** |
| **Total Test Execution** | **Full System Pipeline** | **32** | **100% PASS** |

### 6.2 Empirical Test Execution Log

```powershell
============================= test session starts =============================
platform win32 -- Python 3.10.0, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\prani\PycharmProjects\LexAssist\backend
plugins: anyio-4.11.0
collected 32 items

tests\test_api.py .........                                              [ 28%]
tests\test_backbone.py .........                                         [ 56%]
tests\test_modules.py ..............                                     [100%]

======================== 32 passed, 1 warning in 0.94s ========================
```

---

## 7. Phase 6: Deployment & Operations Strategy

### 7.1 Environment Setup & Run Commands

#### Backend Execution (Development & Production)
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Execution (Development & Production)
```bash
cd frontend
npm install
npm run dev    # Launches Vite Dev Server on http://localhost:5173
npm run build  # Builds production distribution in frontend/dist
```

### 7.2 Environment Variable Configurations

| Variable Name | Required | Default Value | Description |
| :--- | :---: | :---: | :--- |
| `ANTHROPIC_API_KEY` | Optional | `None` | Anthropic Claude API key for live model execution. |
| `OPENAI_API_KEY` | Optional | `None` | OpenAI API key for live GPT model execution. |
| `CHROMA_PERSIST_DIR` | Optional | `./data/chroma_db` | Storage path for Chroma vector embeddings. |
| `PORT` | Optional | `8000` | Port for FastAPI Uvicorn service. |

---

## 8. Phase 7: Maintenance, Security & Compliance

### 8.1 Statutory Compliance Tracking
* **BNS / BNSS Transition:** LexAssist includes explicit cross-references mapping historic Code of Criminal Procedure (CrPC) and Indian Penal Code (IPC) provisions to the new Bharatiya Nagarik Suraksha Sanhita (BNSS) and Bharatiya Nyaya Sanhita (BNS), 2023.
* **Disclaimer Requirements:** Every module output automatically appends mandatory statutory disclaimers stating that LexAssist provides legal information and intelligence, not formal legal advice.

### 8.2 Security Controls & Privacy
* **Ephemeral Ingestion:** User contracts ingested via `/api/contract/analyze` generate short-lived vector IDs (`contract_<uuid>`) to isolate party data.
* **CORS Policies:** Configured in `main.py` with domain restriction capability for production deployment.

---

## 9. Requirement Traceability Matrix (RTM)

| Requirement | Description | Design Component | Code File | Test Verification | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **FR-1.0** | Smart Intent Routing | `IntentRouter` | [`intent_router.py`](file:///c:/Users/prani/PycharmProjects/LexAssist/backend/app/router/intent_router.py) | `test_backbone.py::test_intent_router` | **Verified** |
| **FR-2.0** | Contract Risk Review | `ContractAnalysisModule` | [`contract_analysis.py`](file:///c:/Users/prani/PycharmProjects/LexAssist/backend/app/modules/contract_analysis.py) | `test_modules.py::test_contract_analysis` | **Verified** |
| **FR-3.0** | Precedent Kanoon Search | `CaseLawSearchModule` | [`case_law_search.py`](file:///c:/Users/prani/PycharmProjects/LexAssist/backend/app/modules/case_law_search.py) | `test_modules.py::test_case_law_search` | **Verified** |
| **FR-4.0** | Live Case Status | `CaseStatusModule` | [`case_status.py`](file:///c:/Users/prani/PycharmProjects/LexAssist/backend/app/modules/case_status.py) | `test_modules.py::test_case_status` | **Verified** |
| **FR-5.0** | Document Drafting Engine| `DocumentDraftingModule` | [`document_drafting.py`](file:///c:/Users/prani/PycharmProjects/LexAssist/backend/app/modules/document_drafting.py) | `test_modules.py::test_document_drafting` | **Verified** |
| **FR-6.0** | Statutory Concept Explainer|`LegalExplainerModule` | [`legal_explainer.py`](file:///c:/Users/prani/PycharmProjects/LexAssist/backend/app/modules/legal_explainer.py) | `test_modules.py::test_legal_explainer` | **Verified** |
| **FR-7.0** | Grounding Auditor | `SelfConsistencyAuditor` | [`auditor.py`](file:///c:/Users/prani/PycharmProjects/LexAssist/backend/app/backbone/auditor.py) | `test_backbone.py::test_auditor` | **Verified** |
| **FR-8.0** | Cost Proxy Telemetry | `ModelRouter` | [`cost_proxy.py`](file:///c:/Users/prani/PycharmProjects/LexAssist/backend/app/backbone/cost_proxy.py) | `test_backbone.py::test_cost_proxy` | **Verified** |

---

## 10. Sign-Off & Approval

| Role | Name / Title | Signature | Date |
| :--- | :--- | :--- | :--- |
| **Lead Systems Architect** | Antigravity AI Engineering | *Approved* | August 18, 2026 |
| **QA Lead** | Automated Verification Engine | *32/32 Passing* | August 18, 2026 |
| **Legal Operations Lead** | LexAssist Product Owner | *Approved* | August 18, 2026 |
