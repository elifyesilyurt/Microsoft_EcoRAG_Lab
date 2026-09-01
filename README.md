# Microsoft EcoRAG Lab

**Production-Grade, Zero-Hallucination Local Sustainability RAG Analysis Engine**

Microsoft EcoRAG Lab is an on-device AI analysis engine operating over Microsoft's 2024 and 2025 Environmental Sustainability Reports and Data Fact Sheets. Built on Foundry Local and OpenAI API standards, it achieves zero-hallucination factual retrieval and deterministic quantitative calculation on complex corporate ESG metrics.

---

### Language / Dil

[English Documentation](#english-documentation) | [Türkçe Dokümantasyon (Tıklayınız)](#türkçe-dokümantasyon)

---

## English Documentation

### 1. Architecture & Key Components

```
+-------------------------------------------------------------------------+
|                        User Query / Web UI                              |
|                (Streamlit Interface - app.py:8501)                      |
+------------------------------------+------------------------------------+
                                     |
                  +------------------+------------------+
                  |                                     |
                  v                                     v
        +------------------+                  +------------------+
        | PAL Math Engine  |                  |  Hybrid Search   |
        | (esg_tables.py)  |                  | (nomic-embed 1.5)|
        | - Scope 1/2/3    |                  | - Dense Vector   |
        | - Carbon Removal |                  | - Lexical Boost  |
        | - Water / Waste  |                  +---------+--------+
        +---------+--------+                            |
                  |                                     v
                  |                           +------------------+
                  |                           | Pydantic Guard   |
                  |                           |   (Schemas &     |
                  |                           | Assertions Layer)|
                  |                           +---------+--------+
                  |                                     |
                  +------------------+------------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Local SLM: phi-4-mini    |
                      |  (Foundry Local / Port API) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Grounding & Verified Output |
                      |  (Strict Unit & Zero-Halluc)|
                      +-----------------------------+
```

#### Hybrid Retrieval & Asymmetric Embedding
- **Embedding Model:** `nomic-ai/nomic-embed-text-v1.5` (768-dim, 8192 token context window).
- **Asymmetric Vector Prefix:** `search_document:` during document indexing, `search_query:` during user queries.
- **Lexical Boost & Normalization:** Unicode NFD normalization with multi-token lexical boost for named entities (e.g., FIDO Tech, UL 2799, London, Queretaro).

#### Dual-Representation Table Ingestion & Visual Tagging
- **Layout-Aware Ingestion:** Markdown tables coupled with Row-Centric Key-Value mappings (`[Structured Row Mappings]`).
- **Visual Reference Safeguard:** Graphical charts/infographics are automatically tagged (`[VISUAL REFERENCE]` / `[VISUAL DATA FRAGMENT]`), preventing spurious hallucination on unextractable raster graphics.

#### Program-Aided Language (PAL) Quantitative Motor
- **Deterministic Math Engine (`esg_tables.py`):** Scope 1/2/3 multi-year emissions, carbon removal technology breakdowns, water replenishment benefit volumes, and Zero Waste datacenter metrics are resolved via strictly typed DataFrames rather than SLM free-text approximation.

#### Pydantic Assertion & Deterministic Filtering
- **Schema Validation (`extraction_pipeline.py`):** Enforces strict temporal scope (FY20-FY25), unit binding (mtCO2e, million m3, %, projects), and distinction between counts and physical volumes.

---

### 2. Quick Start

#### Prerequisites
- Python 3.9+
- Foundry Local CLI running with `phi-4-mini` model

#### Installation
```bash
# Clone the repository
git clone https://github.com/elifyesilyurt/Foundry-Local-Lab.git
cd Foundry-Local-Lab

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install production dependencies
pip install -r requirements.txt
```

#### Ingest Documents (Optional - DB is pre-indexed)
```bash
python ingest_all_reports.py
```

#### Run the Streamlit Application
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

---

### 3. Running Benchmarks & Negative Controls

The repository includes an automated 14-question benchmark suite (9 factual queries + 5 out-of-domain negative control tests):

```bash
# Run the complete test suite
python run_benchmarks.py

# Run targeted benchmark questions
python run_benchmarks.py --only 1,2,7,9
```

#### Benchmark Results Summary
| Category | Pass Rate | Hallucination Rate | Avg. Latency |
|---|---|---|---|
| **Factual & Quantitative Tests** | **100% (9/9)** | **0%** | ~6.46s (PAL: ~3.2s) |
| **Out-of-Domain Negative Controls** | **100% (5/5)** | **0%** | ~40.8s |

---

### 4. Project Structure

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

## Türkçe Dokümantasyon

<details>
<summary><strong>Türkçe Dokümantasyonu Görüntülemek İçin Tıklayınız (Genişlet / Daralt)</strong></summary>

<br>

### 1. Genel Bakış ve Mimari

Microsoft EcoRAG Lab; Microsoft'un 2024 ve 2025 Çevresel Sürdürülebilirlik Raporları (ESG) ile Veri Tabloları üzerinde çalışan, yerel, sıfır halüsinasyon hedefli bir sürdürülebilirlik analiz motorudur.

Karmaşık ESG metriklerinde, çok yıllı emisyon tablolarında ve birim eşleşmelerinde tam doğruluğa ulaşmak için hibrit arama (Dense Vector + Lexical Boost), Program-Aided Language (PAL) deterministik hesaplama motoru ve Pydantic tip doğrulama katmanı birlikte çalışır.

#### Temel Mimari Katmanları:
1. **Asimetrik Vektör Arama:** `nomic-ai/nomic-embed-text-v1.5` modeli kullanılarak indekslemede `search_document:`, sorgulama sırasında `search_query:` önekleri uygulanır.
2. **Tablo ve Görsel Ayrıştırma:** Tablolar hem Markdown hem de satır bazlı anahtar-değer formatında indekslenir. Grafik/şema gibi çıkarılamayan raster görseller `[VISUAL REFERENCE]` etiketi ile işaretlenerek modelin uydurma veri üretmesi engellenir.
3. **PAL Deterministik Veri Motoru (`esg_tables.py`):** Karbon emisyonları (Scope 1/2/3), karbon uzaklaştırma teknolojileri, su yenileme projeleri ve Sıfır Atık sertifikasyonları SLM'in serbest tahminine bırakılmadan Python DataFrame'leri üzerinden deterministik olarak çözülür.
4. **Pydantic Doğrulama Katmanı (`extraction_pipeline.py`):** Zaman kapsamı (FY20-FY25), birim bağıntısı (mtCO2e, million m3, adet, %) ve proje sayısı ile fiziksel hacim ayrımı katı kurallarla denetlenir.

---

### 2. Kurulum ve Çalıştırma

#### Gereksinimler
- Python 3.9 veya üzeri
- `phi-4-mini` modelini çalıştıran Foundry Local servisi

#### Adımlar
```bash
# Depoyu klonlayın
git clone https://github.com/elifyesilyurt/Foundry-Local-Lab.git
cd Foundry-Local-Lab

# Sanal ortam oluşturun ve etkinleştirin
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt

# Streamlit uygulamasını başlatın
streamlit run app.py
```
Uygulamaya tarayıcınızdan http://localhost:8501 adresinden erişebilirsiniz.

---

### 3. Otomatik Benchmark ve Değerlendirme

Sistem performansını, birim tutarlılığını ve alan dışı reddetme yeteneğini test etmek için 14 soruluk test paketi mevcuttur:

```bash
# Tüm benchmark testlerini çalıştır
python run_benchmarks.py

# Belirli soruları test et
python run_benchmarks.py --only 1,2,7,9
```

#### Test Sonuçları Özeti
- **Factual & Sayısal Doğruluk:** %100 (9/9 Başarılı)
- **Negatif Kontrol (Alan Dışı Reddetme):** %100 (5/5 Başarılı, 0 Halüsinasyon)
- **Ortalama Yanıt Süresi:** ~6.46 saniye (PAL sorguları: ~3.2 saniye)

</details>

---

## License / Lisans

MIT License - Ayrıntılar için [LICENSE](LICENSE) dosyasına bakınız.
