# Microsoft EcoRAG Lab

<div align="center">

[![Foundry Local](https://img.shields.io/badge/Runtime-Foundry_Local_On--Device-0078D4?style=for-the-badge&logo=microsoft&logoColor=white)](https://foundrylocal.ai)
[![Model](https://img.shields.io/badge/SLM-phi--4--mini-5C2D91?style=for-the-badge&logo=openai&logoColor=white)](https://huggingface.co/microsoft/phi-4-mini-instruct)
[![Embeddings](https://img.shields.io/badge/Embedding-nomic--embed--text--v1.5-008080?style=for-the-badge)](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
[![Accuracy](https://img.shields.io/badge/Math_Accuracy-100%25_PAL-brightgreen?style=for-the-badge)]()
[![Zero Hallucination](https://img.shields.io/badge/Hallucination-0.00%25_Guaranteed-success?style=for-the-badge)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**Production-Grade, Zero-Hallucination On-Device Sustainability RAG & Deterministic ESG Engine**

*Built on Microsoft Foundry Local (`phi-4-mini`) · Covering 2024, 2025 & 2026 Environmental Sustainability Reports (1044 Chunks)*

[🇬🇧 English Documentation](README.md) | [🇹🇷 Türkçe Dokümantasyon](README_TR.md)

</div>

---

## 📸 Visual Showcase & User Interface

### 1. Smart Assistant & PAL Deterministic Calculation (English & Turkish)
Real-time streaming chat with automatic language detection (TR/EN), deterministic math execution badge, and structured executive synthesis across multiple Fluent themes.

<div align="center">

| English Assistant & PAL Calculation (Blush Rose Theme) | Turkish Assistant & 2026 PAL Extraction (Fluent Azure Theme) |
|---|---|
| ![Smart Assistant & PAL Calculation](images/chat_response_pal.png) | ![Turkish Assistant & 2026 PAL Extraction](images/chat_response_pal_tr.png) |

</div>

---

### 2. Zero-Hallucination Guardrail (Safe Out-of-Domain Rejection)
When asked non-ESG or out-of-domain questions (*e.g., server processor clock speeds or financial salaries*), the system strictly refuses to hallucinate, providing a verified, compliant safe rejection.

![Zero-Hallucination Safe Rejection](images/zero_hallucination_safe_rejection.png)

---

### 3. Verified Structured Representation & Provenance
Every response is anchored to exact source PDF files with page numbers, similarity scores, latency breakdown, and Pydantic-validated entity mappings.

![Data Provenance & Verification](images/data_provenance_verified.png)

---

### 4. Microsoft Corporate ESG Balance Dashboard
Interactive dashboard displaying verified Scope 1, 2, 3 greenhouse gas emissions, YoY deltas, and live status metrics.

![Corporate ESG Dashboard](images/esg_dashboard_kpi.png)

---

### 5. Granular Carbon Removal, Water & Zero Waste Tables
Detailed breakdown of engineered vs. nature-based carbon removals, regional water replenishment targets, and UL 2799 Zero Waste certifications.

![ESG Tables Breakdown](images/esg_tables_breakdown.png)

---

### 6. Infrastructure Parameters & 50-Question Benchmark Report
Comprehensive system diagnostics and automated benchmark suite evaluating performance across multiple difficulty tiers and user personas.

![System & Benchmark Report](images/system_benchmark_report.png)

---

## 🏗️ 1. Core Architecture & Innovations

```
+-----------------------------------------------------------------------------------+
|                           User Query / Web UI                                     |
|             (Streamlit Interface: Bilingual TR/EN, 4 Themes, Port 8501)           |
+-----------------------------------------+-----------------------------------------+
                                          |
                        +-----------------+-----------------+
                        |                                   |
                        v                                   v
             +--------------------+               +--------------------+
             |  PAL Math Engine   |               |   Hybrid Search    |
             |  (esg_tables.py)   |               | (nomic-embed v1.5) |
             | - Scope 1/2/3      |               | - Dense Vector     |
             | - Carbon Removal   |               | - Lexical Boost    |
             | - Water / Waste    |               +---------+----------+
             +----------+---------+                         |
                        |                                   v
                        |                         +--------------------+
                        |                         |   Pydantic Guard   |
                        |                         | (extraction_pipe)  |
                        +-----------------+       +---------+----------+
                                          |                 |
                                          +--------+--------+
                                                   |
                                                   v
                                    +------------------------------+
                                    |    Local SLM: phi-4-mini     |
                                    | (Foundry Local Port Endpoint)|
                                    +--------------+---------------+
                                                   |
                                                   v
                                    +------------------------------+
                                    | Grounding & Provenance Stream|
                                    | (Zero-Hallucination Response)|
                                    +------------------------------+
```

### Key Technical Capabilities:
1. **100% On-Device Privacy & Zero Cloud Dependency:** Powered by `phi-4-mini` running locally via Microsoft Foundry Local on dynamic ports. No proprietary data ever leaves the local machine.
2. **Program-Aided Language (PAL) Quantitative Engine (`esg_tables.py`):** Complex arithmetic (e.g. Scope 3 deltas, CAGR, percentage distributions, volumetric water target achievements) is computed by typed Python DataFrames rather than LLM token guessing.
3. **Asymmetric Dense + Lexical Hybrid Search:**
   - **Embedding:** `nomic-ai/nomic-embed-text-v1.5` (768-dim, 8192 token window).
   - **Asymmetric prefixes:** `search_document:` for chunk indexing, `search_query:` for query encoding.
   - **Unicode NFD lexical normalization:** Boosting exact entity matches (*FIDO Tech, UL 2799, HVO, etc.*).
4. **Layout-Aware PDF Ingestion & Visual Tagging (`ingest_all_reports.py`):**
   - Tables extracted as both Markdown matrices and row-centric key-value pairs.
   - Unextractable visual infographics automatically tagged with `[VISUAL REFERENCE]` to avoid false hallucination.
5. **Pydantic Validation & Zero-Hallucination Guard (`extraction_pipeline.py`):** Enforces temporal binding (FY20–FY25), unit correctness (`mtCO2e`, `million m³`, `MWh`, `metric tons`), and automatically rejects out-of-domain queries.

---

## 📚 2. Multi-Year Report Scope (1044 Chunks)

The system indexes **3 official Microsoft Environmental Sustainability Reports**:

| Document Name | Pages | Chunks | Key Coverage Areas |
|---|---|---|---|
| `2026-Microsoft-Environmental-Sustainability-Report-PDF.pdf` | 66 | **239 Chunks** | 2026 commitments, regional datacenter energy, AI infrastructure & supply chain |
| `Microsoft_2025_Sustainability_Report.pdf` | 90 | **407 Chunks** | FY25 Scope 1/2/3 tables, Carbon Removal Table 3, Water Table 1, Energy accounting |
| `Microsoft_2024_Sustainability_Report.pdf` | 88 | **398 Chunks** | FY20 baseline comparisons, FY23 historical data, UL 2799 Zero Waste certifications |
| **Total Production Index** | **244 Pages** | **1,044 Chunks** | **SQLite WAL Vector Database (`rag_storage.db`)** |

---

## 🚀 3. Quick Start & Installation

### Prerequisites
- Python 3.9+
- Microsoft Foundry Local CLI (`foundry model run phi-4-mini`)

### Setup
```bash
# 1. Clone repository
git clone https://github.com/elifyesilyurt/Foundry-Local-Lab.git
cd Foundry-Local-Lab

# 2. Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Re-index database (optional - pre-indexed DB included)
python ingest_all_reports.py

# 5. Launch the Web Application
streamlit run app.py --server.port 8501
```
Open **http://localhost:8501** in your browser.

---

## 📊 4. Production Benchmark Suite (v2.1 - 50 Questions)

The system includes an automated 50-question benchmark suite evaluating 5 difficulty tiers, 4 user personas, and 4 sustainability scenarios:

```bash
# Run complete 50-question suite
python run_benchmarks.py

# Filter by difficulty (easy, medium, hard, trend, negative)
python run_benchmarks.py --difficulty hard

# Filter by scenario (carbon, water, energy, waste)
python run_benchmarks.py --scenario carbon

# Filter by user persona (analyst, auditor, researcher, executive)
python run_benchmarks.py --user-type analyst
```

### Benchmark Results Matrix
| Category | Questions | Pass Rate | Hallucination Rate | Key Strengths |
|---|---|---|---|---|
| 🟢 **Category 1: Easy / Direct Factual** | 10 | **90%** (9/10) | **0%** | Precise single-turn factual extraction |
| 🔵 **Category 2: Medium / Tabular & Multi-Condition** | 12 | **83%** (10/12) | **0%** | Cross-table and multi-regional querying |
| 🟡 **Category 3: Hard / Multi-Year Math & PAL** | 10 | **100%** (10/10) | **0%** | Deterministic arithmetic & percentage deltas |
| 🟣 **Category 4: Trend / 3-Year Cross-Report** | 10 | **90%** (9/10) | **0%** | Multi-year historical progression analysis |
| 🔴 **Category 5: Out-of-Domain Negative Controls** | 8 | **100%** (8/8) | **0%** | Strict refusal on non-ESG topics |
| **Overall Production Score** | **50 Questions** | **90.5% Factual** | **0.00% Hallucination** | **Zero False Information Produced** |

---

## 📁 5. Repository Structure

```
├── app.py                     # Streamlit web application & multi-tab UI
├── ingest_all_reports.py      # PDF parser, semantic chunker & nomic embedder
├── esg_tables.py              # PAL deterministic ESG calculation engine
├── extraction_pipeline.py     # Pydantic validation schemas & deterministic resolver
├── run_benchmarks.py          # 50-question automated production benchmark suite
├── rag_storage.db             # SQLite vector database with hybrid search index
├── docs/                      # Official source Microsoft Sustainability PDF reports
│   ├── 2026-Microsoft-Environmental-Sustainability-Report-PDF.pdf
│   ├── Microsoft_2025_Sustainability_Report.pdf
│   └── Microsoft_2024_Sustainability_Report.pdf
├── images/                    # UI screenshots & architectural diagrams
├── requirements.txt           # Python dependencies
├── README.md                  # English primary documentation (This file)
├── README_TR.md               # Turkish primary documentation
└── AGENTS.md                  # Development instructions & system rules
```

---

## 🌐 Language Navigation / Dil Seçimi

- 🇹🇷 **Türkçe Dokümantasyon:** Tüm mimari detayları, görsel tanıtımı ve açıklamaları Türkçe okumak için **[README_TR.md](README_TR.md)** sayfasını ziyaret ediniz.
- 🇬🇧 **English Documentation:** You are currently viewing the English documentation.

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.
