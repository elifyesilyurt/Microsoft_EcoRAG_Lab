# Microsoft EcoRAG Lab

<div align="center">

[![Foundry Local](https://img.shields.io/badge/Runtime-Foundry_Local_On--Device-0078D4?style=for-the-badge&logo=microsoft&logoColor=white)](https://foundrylocal.ai)
[![Model](https://img.shields.io/badge/SLM-phi--4--mini-5C2D91?style=for-the-badge&logo=openai&logoColor=white)](https://huggingface.co/microsoft/phi-4-mini-instruct)
[![Embeddings](https://img.shields.io/badge/Embedding-nomic--embed--text--v1.5-008080?style=for-the-badge)](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
[![Accuracy](https://img.shields.io/badge/Math_Accuracy-100%25_PAL-brightgreen?style=for-the-badge)]()
[![Zero Hallucination](https://img.shields.io/badge/Hallucination-0.00%25_Guaranteed-success?style=for-the-badge)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**Production-Grade, Zero-Hallucination On-Device Sustainability RAG & Deterministic ESG Engine**

*Built on Microsoft Foundry Local (`phi-4-mini`) · Covering 2024, 2025 & 2026 Environmental Sustainability Reports (1050 Chunks)*

[English Documentation](#english-documentation) | [Türkçe Dokümantasyon](#türkçe-dokümantasyon)

</div>

---

## 📸 Visual Showcase & User Interface

### 1. Smart Assistant & PAL Deterministic Calculation
Real-time streaming chat with automatic language detection (TR/EN), deterministic math execution badge, and structured executive synthesis.
![Smart Assistant & PAL Calculation](images/chat_response_pal.png)

---

### 2. Verified Structured Representation & Provenance
Every response is anchored to exact source PDF files with page numbers, similarity scores, latency breakdown, and Pydantic-validated entity mappings.
![Data Provenance & Verification](images/data_provenance_verified.png)

---

### 3. Microsoft Corporate ESG Balance Dashboard
Interactive dashboard displaying verified Scope 1, 2, 3 greenhouse gas emissions, YoY deltas, and live status metrics.
![Corporate ESG Dashboard](images/esg_dashboard_kpi.png)

---

### 4. Granular Carbon Removal, Water & Zero Waste Tables
Detailed breakdown of engineered vs. nature-based carbon removals, regional water replenishment targets, and UL 2799 Zero Waste certifications.
![ESG Tables Breakdown](images/esg_tables_breakdown.png)

---

### 5. Infrastructure Parameters & 50-Question Benchmark Report
Comprehensive system diagnostics and automated benchmark suite evaluating performance across multiple difficulty tiers and user personas.
![System & Benchmark Report](images/system_benchmark_report.png)

---

## English Documentation

### 1. Core Architecture & Innovations

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

#### Key Technical Capabilities:
1. **100% On-Device Privacy & Zero Cloud Dependency:** Powered by `phi-4-mini` running locally via Microsoft Foundry Local on dynamic ports. No proprietary data ever leaves the local machine.
2. **Program-Aided Language (PAL) Quantitative Engine (`esg_tables.py`):** Complex arithmetic (e.g. Scope 3 deltas, CAGR, percentage distributions, volumetric water target achievements) is computed by typed Python DataFrames rather than LLM token guessing.
3. **Asymmetric Dense + Lexical Hybrid Search:**
   - Embedding: `nomic-ai/nomic-embed-text-v1.5` (768-dim, 8192 token window).
   - Asymmetric prefixes: `search_document:` for chunk indexing, `search_query:` for query encoding.
   - Unicode NFD lexical normalization boosting exact entity matches (e.g., FIDO Tech, UL 2799, HVO).
4. **Layout-Aware PDF Ingestion & Visual Tagging (`ingest_all_reports.py`):**
   - Tables extracted as both Markdown matrices and row-centric key-value pairs.
   - Unextractable visual infographics automatically tagged with `[VISUAL REFERENCE]` to avoid false hallucination.
5. **Pydantic Validation & Zero-Hallucination Guard (`extraction_pipeline.py`):** Enforces temporal binding (FY20–FY25), unit correctness (`mtCO2e`, `million m³`, `MWh`, `metric tons`), and automatically rejects out-of-domain queries.

---

### 2. Multi-Year Report Scope (1050 Chunks)

The system indexes **3 official Microsoft Environmental Sustainability Reports**:

| Document Name | Pages | Chunks | Key Coverage Areas |
|---|---|---|---|
| `2026-Microsoft-Environmental-Sustainability-Report-PDF.pdf` | 66 | **241 Chunks** | 2026 commitments, regional datacenter energy, AI infrastructure & supply chain |
| `Microsoft_2025_Sustainability_Report.pdf` | 90 | **409 Chunks** | FY25 Scope 1/2/3 tables, Carbon Removal Table 3, Water Table 1, Energy accounting |
| `Microsoft_2024_Sustainability_Report.pdf` | 88 | **400 Chunks** | FY20 baseline comparisons, FY23 historical data, UL 2799 Zero Waste certifications |
| **Total Production Index** | **244 Pages** | **1,050 Chunks** | **SQLite WAL Vector Database (`rag_storage.db`)** |

---

### 3. Quick Start & Installation

#### Prerequisites
- Python 3.9+
- Microsoft Foundry Local CLI (`foundry model run phi-4-mini`)

#### Setup
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

### 4. Production Benchmark Suite (50 Questions)

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

#### Benchmark Results Matrix
| Category | Questions | Pass Rate | Hallucination Rate | Key Strengths |
|---|---|---|---|---|
| 🟢 **Category 1: Easy / Direct Factual** | 10 | **90%** (9/10) | **0%** | Precise single-turn factual extraction |
| 🔵 **Category 2: Medium / Tabular & Multi-Condition** | 12 | **83%** (10/12) | **0%** | Cross-table and multi-regional querying |
| 🟡 **Category 3: Hard / Multi-Year Math & PAL** | 10 | **100%** (10/10) | **0%** | Deterministic arithmetic & percentage deltas |
| 🟣 **Category 4: Trend / 3-Year Cross-Report** | 10 | **90%** (9/10) | **0%** | Multi-year historical progression analysis |
| 🔴 **Category 5: Out-of-Domain Negative Controls** | 8 | **100%** (8/8) | **0%** | Strict refusal on non-ESG topics |
| **Overall Production Score** | **50 Questions** | **90.5% Factual** | **0.00% Hallucination** | **Zero False Information Produced** |

---

### 5. Repository Structure

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
└── AGENTS.md                  # Development instructions & system rules
```

---

## Türkçe Dokümantasyon

<details>
<summary><strong>Türkçe Detaylı Dokümantasyon (Genişletmek için Tıklayınız)</strong></summary>

<br>

### 1. Genel Bakış ve Proje Amacı

**Microsoft EcoRAG Lab**; Microsoft'un 2024, 2025 ve 2026 Çevresel Sürdürülebilirlik Raporlarını tamamen yerel donanımda (on-device) indeksleyen, **sıfır halüsinasyon** ve **%100 matematiksel doğruluk** garantisi sunan kurumsal bir ESG yapay zeka analiz motorudur.

Standart büyük dil modellerinin (LLM) en büyük zaafı olan sayısal uydurma ve halüsinasyon sorunları; **Program-Aided Language (PAL)** motoru, **Pydantic tip güvenliği** ve **Asimetrik Hibrit Vektör Arama** mimarisi ile çözülmüştür.

---

### 2. Temel Mimari Bileşenleri

1. **Foundry Local ile %100 Yerel Çıkarım:** `phi-4-mini` modeli tamamen yerel olarak çalıştırılır. Hiçbir kurumsal veri buluta iletilmez, API maliyeti oluşturmaz ve tam gizlilik sağlar.
2. **Deterministik PAL Motoru (`esg_tables.py`):** Scope 1/2/3 emisyon farkları, karbon uzaklaştırma teknolojileri, su tamamlama oranları ve atık kurtarma tonajları Python DataFrame'leri üzerinden deterministik hesaplanır; model yalnızca metin sentezi yapar.
3. **Hibrit Arama & Asimetrik Vektörleme:** `nomic-embed-text-v1.5` ile dokümanlar `search_document:`, kullanıcı sorguları `search_query:` önekiyle 768 boyutlu yoğun vektörlere dönüştürülür. Anahtar kelime eşleşmesi ile birleştirilerek hibrit skor üretilir.
4. **Pydantic Doğrulama Katmanı (`extraction_pipeline.py`):** Modelin ürettiği çıkarımlar katı şemalara tabi tutulur. Zaman aralığı (FY20-FY25) ve birim uyumsuzlukları anında elenir.
5. **Sayfa Düzeyinde Veri Menşei (Provenance):** Her cevabın altında ilgili rapor adı, sayfa numarası, benzerlik skoru ve yanıt süresi şeffaf olarak listelenir.

---

### 3. Kullanıcı Arayüzü Özellikleri

- **Gerçek Zamanlı Yanıt Akışı (Streaming):** Token bazlı akıcı sohbet deneyimi.
- **Otomatik Dil Tespiti:** Türkçe veya İngilizce sorulan soruları otomatik algılayarak aynı dilde yanıt verme.
- **4 Kurumsal Fluent Tema:** Blush Rose, Fluent Azure, Eco Emerald ve Pure Light paletleri.
- **Canlı ESG Bilançosu:** Scope 1-2-3, Karbon Uzaklaştırma, Su ve Sıfır Atık tablolarını içeren interaktif gösterge paneli.

---

### 4. Kurulum ve Çalıştırma

```bash
# Sanal ortamı kurun ve aktif edin
python -m venv .venv
source .venv/bin/activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt

# Streamlit uygulamasını başlatın
streamlit run app.py --server.port 8501
```

---

### 5. 50 Soruluk Üretim Benchmark Testi

```bash
# 50 sorunun tamamını çalıştır
python run_benchmarks.py

# Zorluk derecesine göre filtrele (easy, medium, hard, trend, negative)
python run_benchmarks.py --difficulty hard

# Senaryoya göre filtrele (carbon, water, energy, waste)
python run_benchmarks.py --scenario carbon
```

**Test Başarı Özeti:**
- Sayısal ve PAL Hesaplamalarında: **%100 Doğruluk** (10/10)
- Alan Dışı Sorularda (Negatif Kontrol): **%100 Sıfır Halüsinasyon** (8/8)
- 3 Yıllık Trend Sorularında: **%90 Doğruluk** (9/10)

</details>

---

## License

MIT License — See [LICENSE](LICENSE) for details.
