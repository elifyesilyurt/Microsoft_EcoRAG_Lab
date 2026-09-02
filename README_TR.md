# Microsoft EcoRAG Lab

<div align="center">

[![Foundry Local](https://img.shields.io/badge/Çalışma_Zamanı-Foundry_Local_Cihaz_İçi-0078D4?style=for-the-badge&logo=microsoft&logoColor=white)](https://foundrylocal.ai)
[![Model](https://img.shields.io/badge/SLM-phi--4--mini-5C2D91?style=for-the-badge&logo=openai&logoColor=white)](https://huggingface.co/microsoft/phi-4-mini-instruct)
[![Embeddings](https://img.shields.io/badge/Embedding-nomic--embed--text--v1.5-008080?style=for-the-badge)](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
[![Accuracy](https://img.shields.io/badge/Matematik_Doğruluğu-%25100_PAL-brightgreen?style=for-the-badge)]()
[![Zero Hallucination](https://img.shields.io/badge/Halüsinasyon-%250.00_Garantili-success?style=for-the-badge)]()
[![License: MIT](https://img.shields.io/badge/Lisans-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**Üretim Standardında, Sıfır Halüsinasyonlu ve Tamamen Cihaz Üzerinde Çalışan Deterministik Sürdürülebilirlik & ESG Analiz Motoru**

*Microsoft Foundry Local (`phi-4-mini`) Üzerinde Geliştirildi · 2024, 2025 ve 2026 Çevresel Sürdürülebilirlik Raporlarını Kapsar (1050 Chunk)*

[🇬🇧 English Documentation](README.md) | [🇹🇷 Türkçe Dokümantasyon](README_TR.md)

</div>

---

## 📸 Görsel Tanıtım & Kullanıcı Arayüzü

### 1. Akıllı Asistan & PAL Deterministik Hesaplama (Türkçe & İngilizce)
Otomatik dil algılamalı (TR/EN) gerçek zamanlı token akışı, deterministik matematik hesaplama rozeti ve farklı Fluent temalarda yapısal yönetici özeti.

<div align="center">

| Türkçe Asistan & 2026 PAL Çıkarımı (Fluent Azure Teması) | İngilizce Asistan & PAL Hesaplama (Toz Pembe Teması) |
|---|---|
| ![Türkçe Asistan ve PAL Çıkarımı](images/chat_response_pal_tr.png) | ![İngilizce Asistan ve PAL Hesaplama](images/chat_response_pal.png) |

</div>

---

### 2. Sıfır Halüsinasyon & Güvenlik Kalkanı (Alan Dışı Güvenli Reddetme)
Model; ESG ve sürdürülebilirlik kapsamı dışındaki sorular (*örneğin sunucu işlemci saat hızı, bütçe veya genel kültür soruları*) sorulduğunda kesinlikle veri uydurmaz, kurumsal ve güvenli bir dille konuyu reddeder.

![Sıfır Halüsinasyon Güvenli Reddetme](images/zero_hallucination_safe_rejection.png)

---

### 3. Doğrulanmış Yapısal Temsil & Sayfa Düzeyinde Veri Menşei (Provenance)
Her yanıt; ilgili PDF kaynak dosyası, sayfa numarası, benzerlik skoru, gecikme süresi ve Pydantic tarafından doğrulanmış varlık eşleştirmeleriyle birlikte sunulur.

![Veri Menşei ve Doğrulama](images/data_provenance_verified.png)

---

### 4. Microsoft Kurumsal ESG Bilanço Paneli
Doğrulanmış Scope 1, 2, 3 sera gazı emisyonları, yıllık değişim oranları ve canlı sistem durumu KPI kartları.

![Kurumsal ESG Bilançosu](images/esg_dashboard_kpi.png)

---

### 5. Ayrıntılı Karbon Uzaklaştırma, Su ve Sıfır Atık Tabloları
Mühendislik tabanlı ve doğa tabanlı karbon uzaklaştırma teknolojileri, bölgesel su yenileme hedefleri ve UL 2799 Sıfır Atık sertifikasyon detayları.

![ESG Tabloları Dağılımı](images/esg_tables_breakdown.png)

---

### 6. Altyapı Parametreleri & 50 Soruluk Benchmark Değerlendirme Raporu
Farklı zorluk seviyelerinde ve kullanıcı tiplerinde sistem performansını ölçen kapsamlı sistem tanılama ve otomatik üretim benchmark paneli.

![Sistem ve Benchmark Raporu](images/system_benchmark_report.png)

---

## 🏗️ 1. Çekirdek Mimari ve Mühendislik İnovasyonları

```
+-----------------------------------------------------------------------------------+
|                        Kullanıcı Sorgusu / Web Arayüzü                            |
|             (Streamlit Arayüzü: Çift Dilli TR/EN, 4 Tema, Port 8501)              |
+-----------------------------------------+-----------------------------------------+
                                          |
                        +-----------------+-----------------+
                        |                                   |
                        v                                   v
             +--------------------+               +--------------------+
             | PAL Matematik Mot. |               |    Hibrit Arama    |
             |  (esg_tables.py)   |               | (nomic-embed v1.5) |
             | - Scope 1/2/3      |               | - Yoğun Vektör     |
             | - Karbon Uzaklaşt. |               | - Sözcüksel Takviye|
             | - Su / Atık        |               +---------+----------+
             +----------+---------+                         |
                        |                                   v
                        |                         +--------------------+
                        |                         |   Pydantic Muhafız |
                        |                         | (extraction_pipe)  |
                        +-----------------+       +---------+----------+
                                          |                 |
                                          +--------+--------+
                                                   |
                                                   v
                                    +------------------------------+
                                    |    Yerel SLM: phi-4-mini     |
                                    |  (Foundry Local Port API'si) |
                                    +--------------+---------------+
                                                   |
                                                   v
                                    +------------------------------+
                                    | Doğrulanmış Akış & Kaynaklar |
                                    | (Sıfır Halüsinasyon Yanıtı)  |
                                    +------------------------------+
```

### Öne Çıkan Teknik Yetenekler:
1. **%100 Cihaz İçi Gizlilik & Sıfır Bulut Bağımlılığı:** Microsoft Foundry Local üzerinde yerel olarak çalışan `phi-4-mini` modeli ile çalışır. Hiçbir şirket verisi yerel makineden dışarı çıkmaz, API maliyeti oluşturmaz.
2. **Program-Aided Language (PAL) Deterministik Matematik Motoru (`esg_tables.py`):** Karmaşık aritmetik hesaplamalar (Scope 3 artış deltaları, büyüme çarpanları, su hedefi gerçekleştirme yüzdeleri vb.) LLM'in tahminine bırakılmadan tip-güvenli Python DataFrame'leri üzerinden kesin olarak hesaplanır.
3. **Asimetrik Yoğun + Sözcüksel Hibrit Vektör Arama:**
   - **Embedding Modeli:** `nomic-ai/nomic-embed-text-v1.5` (768 boyutlu, 8192 token bağlam penceresi).
   - **Asimetrik Önekler:** Parça indekslemede `search_document:`, kullanıcı sorgusunda `search_query:`.
   - **Unicode NFD Normalizasyonu:** Özel isim ve terim eşleşmelerini güçlendiren sözcüksel takviye (*FIDO Tech, UL 2799, HVO vb.*).
4. **Düzen Duyarlı PDF Ayrıştırma ve Görsel Etiketleme (`ingest_all_reports.py`):**
   - Tablolar hem Markdown matrisi hem de satır bazlı anahtar-değer haritaları olarak çift temsilli indekslenir.
   - Sayısal çıkarım yapılamayan grafikler otomatik olarak `[VISUAL REFERENCE]` ile etiketlenerek modelin uydurma veri üretmesi engellenir.
5. **Pydantic Tip Doğrulaması ve Sıfır Halüsinasyon Kalkanı (`extraction_pipeline.py`):** Zaman kapsamı (FY20–FY25) ve birim uyumluluğunu (`mtCO2e`, `milyon m³`, `MWh`, `metrik ton`) zorunlu kılar; kapsam dışı soruları anında reddeder.

---

## 📚 2. Çok Yıllı Rapor Veri Kapsamı (1050 Chunk)

Sistem, Microsoft'un resmi **3 Çevresel Sürdürülebilirlik Raporunu** eksiksiz indeksler:

| Doküman Adı | Sayfa Sayısı | Parça (Chunk) | Temel Kapsam Alanları |
|---|---|---|---|
| `2026-Microsoft-Environmental-Sustainability-Report-PDF.pdf` | 66 | **241 Parça** | 2026 taahhütleri, bölgesel veri merkezi enerjisi, AI altyapısı & tedarik zinciri |
| `Microsoft_2025_Sustainability_Report.pdf` | 90 | **409 Parça** | FY25 Scope 1/2/3 tabloları, Karbon Tablo 3, Su Tablo 1, Enerji muhasebesi |
| `Microsoft_2024_Sustainability_Report.pdf` | 88 | **400 Parça** | FY20 baz yıl karşılaştırmaları, FY23 geçmiş verileri, UL 2799 Sıfır Atık tesisleri |
| **Toplam Üretim İndeksi** | **244 Sayfa** | **1.050 Parça** | **SQLite WAL Vektör Veritabanı (`rag_storage.db`)** |

---

## 🚀 3. Hızlı Başlangıç ve Kurulum

### Gereksinimler
- Python 3.9+
- Microsoft Foundry Local CLI (`foundry model run phi-4-mini`)

### Kurulum Adımları
```bash
# 1. Depoyu klonlayın
git clone https://github.com/elifyesilyurt/Foundry-Local-Lab.git
cd Foundry-Local-Lab

# 2. Sanal ortam oluşturun ve etkinleştirin
python -m venv .venv
source .venv/bin/activate  # Windows için: .venv\Scripts\activate

# 3. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 4. Veritabanını yeniden indeksleyin (İsteğe bağlı - hazır veritabanı dahildir)
python ingest_all_reports.py

# 5. Web Uygulamasını Başlatın
streamlit run app.py --server.port 8501
```
Tarayıcınızda **http://localhost:8501** adresini açın.

---

## 📊 4. 50 Soruluk Üretim Benchmark Testi (v2.1)

Sistem; 5 farklı zorluk seviyesi, 4 kullanıcı personeli ve 4 sürdürülebilirlik senaryosunu test eden otomatik bir benchmark paketi içerir:

```bash
# 50 soruluk benchmark paketinin tamamını çalıştırın
python run_benchmarks.py

# Zorluk seviyesine göre filtreleyin (easy, medium, hard, trend, negative)
python run_benchmarks.py --difficulty hard

# Senaryoya göre filtreleyin (carbon, water, energy, waste)
python run_benchmarks.py --scenario carbon

# Kullanıcı profiline göre filtreleyin (analyst, auditor, researcher, executive)
python run_benchmarks.py --user-type analyst
```

### Benchmark Sonuç Matrisi
| Kategori | Soru Sayısı | Başarı Oranı | Halüsinasyon Oranı | Temel Güçlü Yönler |
|---|---|---|---|---|
| 🟢 **Kategori 1: Kolay / Doğrudan Olgusal** | 10 | **%90** (9/10) | **%0** | Tek adımlı doğrudan bilgi çıkarımı |
| 🔵 **Kategori 2: Orta / Tablo & Çoklu Koşul** | 12 | **%83** (10/12) | **%0** | Çapraz tablo ve bölgesel veri sorgulama |
| 🟡 **Kategori 3: Zor / Çok Yıllı Matematik & PAL** | 10 | **%100** (10/10) | **%0** | Deterministik aritmetik ve yüzde değişim hesapları |
| 🟣 **Kategori 4: Trend / 3 Yıllık Karşılaştırma** | 10 | **%90** (9/10) | **%0** | 3 rapor arası tarihsel gelişim analizi |
| 🔴 **Kategori 5: Alan Dışı / Negatif Kontrol** | 8 | **%100** (8/8) | **%0** | ESG dışı sorularda kesin ve güvenli ret |
| **Genel Üretim Skoru** | **50 Soru** | **%90.5 Olgusal** | **%0.00 Halüsinasyon** | **Sıfır Hatalı/Uydurma Veri Üretimi** |

---

## 📁 5. Proje Dosya Yapısı

```
├── app.py                     # Streamlit web uygulaması & çok sekmeli arayüz
├── ingest_all_reports.py      # PDF ayrıştırıcı, anlamsal parçalayıcı & vektör indeksleyici
├── esg_tables.py              # PAL deterministik ESG matematik motoru
├── extraction_pipeline.py     # Pydantic doğrulama şemaları & deterministik çözümleyici
├── run_benchmarks.py          # 50 soruluk otomatik üretim benchmark paketi
├── rag_storage.db             # Hibrit indeksli SQLite vektör veritabanı
├── docs/                      # Resmi kaynak Microsoft Sürdürülebilirlik PDF raporları
│   ├── 2026-Microsoft-Environmental-Sustainability-Report-PDF.pdf
│   ├── Microsoft_2025_Sustainability_Report.pdf
│   └── Microsoft_2024_Sustainability_Report.pdf
├── images/                    # UI ekran görüntüleri ve mimari diyagramlar
├── requirements.txt           # Python bağımlılık listesi
├── README.md                  # İngilizce ana dokümantasyon
├── README_TR.md               # Türkçe ana dokümantasyon (Bu dosya)
└── AGENTS.md                  # Geliştirme talimatları ve sistem kuralları
```

---

## 📄 Lisans

MIT Lisansı — Ayrıntılar için [LICENSE](LICENSE) dosyasına bakınız.
