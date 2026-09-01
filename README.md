# Microsoft EcoRAG Lab 🌿⚡

**Production-Grade, Zero-Hallucination Local Sustainability RAG Analysis Engine**

Microsoft EcoRAG Lab is an on-device AI analysis engine operating over Microsoft's 2024 and 2025 Environmental Sustainability Reports and Data Fact Sheets. Built on **Foundry Local** and **OpenAI API** standards, it achieves zero-hallucination factual retrieval and deterministic quantitative calculation on complex corporate ESG metrics.

---

## 🏛️ Architecture & Key Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        User Query / Web UI                              │
│                (Streamlit Interface — app.py:8501)                      │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                  ┌──────────────────┴──────────────────┐
                  ▼                                     ▼
        ┌──────────────────┐                  ┌──────────────────┐
        │ PAL Math Engine  │                  │  Hybrid Search   │
        │ (esg_tables.py)  │                  │ (nomic-embed 1.5)│
        │ - Scope 1/2/3    │                  │ - Dense Vector   │
        │ - Carbon Removal │                  │ - Lexical Boost  │
        │ - Water / Waste  │                  └─────────┬────────┘
        └─────────┬────────┘                            │
                  │                                     ▼
                  │                           ┌──────────────────┐
                  │                           │ Pydantic Guard   │
                  │                           │   (Schemas &     │
                  │                           │ Assertions Layer)│
                  │                           └─────────┬────────┘
                  │                                     │
                  └──────────────────┬──────────────────┘
                                     ▼
                      ┌─────────────────────────────┐
                      │    Local SLM: phi-4-mini    │
                      │  (Foundry Local / Port API) │
                      └──────────────┬──────────────┘
                                     ▼
                      ┌─────────────────────────────┐
                      │  Grounding & Verified Output │
                      │  (Strict Unit & Zero-Halluc)│
                      └─────────────────────────────┘
```

### 1. Hybrid Retrieval & Asymmetric Embedding
- **Embedding Model:** `nomic-ai/nomic-embed-text-v1.5` (768-dim, 8192 token limit).
- **Asymmetric Vector Prefix:** `search_document:` during ingestion, `search_query:` during queries.
- **Lexical Boost & Normalization:** Unicode NFD normalization with multi-token lexical boost for exact named entities (e.g., FIDO Tech, UL 2799, London, Querétaro).

### 2. Dual-Representation Table Ingestion & Visual Tagging
- **Layout-Aware Ingestion:** Markdown tables coupled with Satır Bazlı Key-Value mappings (`[Structured Row Mappings]`).
- **Visual Reference Safeguard:** Graphical charts/infographics are automatically tagged (`[VISUAL REFERENCE]` / `[VISUAL DATA FRAGMENT]`), preventing spurious hallucination on unextractable raster graphics.

### 3. Program-Aided Language (PAL) Quantitative Motor
- **Deterministic Math Engine (`esg_tables.py`):** Scope 1/2/3 multi-year emissions, carbon removal technology breakdowns, water replenishment benefit volumes, and Zero Waste datacenter metrics are resolved via strictly typed DataFrames rather than LLM free-text guessing.

### 4. Pydantic Assertion & Deterministic Filtering
- **Schema Validation (`extraction_pipeline.py`):** Enforces strict temporal scope (FY20–FY25), unit binding (mtCO2e, million m³, %, projects), and distinction between counts and physical volumes.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9+
- [Foundry Local CLI](https://foundrylocal.ai) running with `phi-4-mini` model

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/your-org/Foundry-Local-Lab.git
cd Foundry-Local-Lab

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install production dependencies
pip install -r requirements.txt
```

### 3. Ingest Documents (Optional — DB is pre-indexed)
```bash
python ingest_all_reports.py
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Running Benchmarks & Negative Controls

The repository includes a comprehensive 14-question benchmark suite (9 factual queries + 5 out-of-domain negative control tests):

```bash
# Run the complete test suite
python run_benchmarks.py

# Run targeted benchmark questions
python run_benchmarks.py --only 1,2,7,9
```

### Benchmark Results Summary
| Category | Pass Rate | Hallucination Rate | Avg. Latency |
|---|---|---|---|
| **Factual & Quantitative Tests** | **100% (9/9)** | **0%** | ~6.46s (PAL: ~3.2s) |
| **Out-of-Domain Negative Controls** | **100% (5/5)** | **0%** | ~40.8s |

---

## 📂 Project Structure

```
├── app.py                     # Streamlit web application & chat UI
├── ingest_all_reports.py      # PDF parsing, semantic chunking & vector indexing
├── esg_tables.py              # PAL deterministic ESG calculation engine
├── extraction_pipeline.py     # Pydantic validation schemas & deterministic resolver
├── run_benchmarks.py          # Automated production benchmark & evaluation suite
├── rag_storage.db             # SQLite vector database with hybrid search index
├── docs/                      # Source Microsoft Sustainability PDF reports
│   ├── Microsoft_2024_Sustainability_Report.pdf
│   ├── Microsoft_2025_Sustainability_Report.pdf
│   └── Microsoft_2026_Data_Fact_Sheet.pdf
├── requirements.txt           # Project Python dependencies
└── .gitignore                 # Git ignore rules for production
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
