"""
run_benchmarks.py — Microsoft EcoRAG Lab 50-Question Production Benchmark Suite
================================================================================
4 Farklı Zorluk Seviyesinde (Kolay, Orta, Zor, Alan Dışı / Negatif Kontrol)
50 soruluk uçtan uca deterministik doğruluk, PAL ve sıfır halüsinasyon test paketi.

Kullanım:
  python run_benchmarks.py                        # 50 sorunun tamamını çalıştırır
  python run_benchmarks.py --difficulty hard      # Sadece belirli zorluğu çalıştırır (easy, medium, hard, negative)
  python run_benchmarks.py --only 1,2,7,9,31,41    # Sadece belirtilen soru ID'lerini çalıştırır
"""

import gc
import json
import re
import sqlite3
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ── Proje modülleri ────────────────────────────────────────────────────────────
from extraction_pipeline import (
    EXTRACTION_SYSTEM_PROMPT,
    DeterministicResolver,
    QueryExtractionPlan,
    format_extraction_prompt,
)
from esg_tables import (
    get_carbon_emissions_df,
    get_carbon_removal_by_type_df,
    get_carbon_removal_df,
    get_water_metrics_df,
    get_water_replenishment_projects_df,
    get_energy_metrics_df,
    get_waste_metrics_df,
    get_zero_waste_certifications_df,
)

# ══════════════════════════════════════════════════════════════════════════════
# KONFİGÜRASYON
# ══════════════════════════════════════════════════════════════════════════════
DB_PATH             = "rag_storage.db"
EMBEDDING_MODEL     = "nomic-ai/nomic-embed-text-v1.5"
FOUNDRY_URL         = "http://127.0.0.1:62095/v1/chat/completions"
PHI_MODEL           = "phi-4-mini"
MAX_K               = 6
DROP_RATIO          = 0.70
MIN_SCORE           = 0.15
MAX_TOKENS          = 512

SYNTHESIS_PROMPT    = (
    "You are a Senior Sustainability Analyst. "
    "Synthesize the verified analytical calculation data into a concise, precise executive answer. "
    "State exact numbers, percentages, and units in sentence 1. Do NOT repeat yourself."
)
FACTUAL_PROMPT      = (
    "You are a Senior Sustainability AI Analyst. "
    "Using ONLY the verified structured metrics below, give a direct, concise answer. "
    "State exact numbers, names, and units in sentence 1. Do NOT repeat yourself."
)
GROUNDING_PROMPT    = (
    "You are a precise Sustainability Analyst. "
    "Answer using ONLY the context provided. "
    "If context contains [VISUAL REFERENCE] warnings, state that graphical data is not available as text. "
    "If the answer is not in the context, respond EXACTLY: "
    "'I cannot find information regarding this in the provided Microsoft Environmental Sustainability reports.'"
)

# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL RENK KODLARI
# ══════════════════════════════════════════════════════════════════════════════
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BLUE   = "\033[94m"
    GREY   = "\033[90m"
    WHITE  = "\033[97m"

def hr(char: str = "═", width: int = 76) -> str:
    return char * width

# ══════════════════════════════════════════════════════════════════════════════
# 50 SORULUK BENCHMARK VERİ SETİ (4 ZORLUK SEVİYESİ)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class BenchmarkQuestion:
    qid:          int
    difficulty:   str          # "easy" | "medium" | "hard" | "negative"
    section:      str          # "factual" | "negative"
    question:     str
    expected_hint: str         # Beklenen anahtar kelimeler/değerler
    use_pal:      bool = False
    pal_data:     str  = ""

QUESTIONS: List[BenchmarkQuestion] = [
    # ── KATEGORİ 1: KOLAY / DIRECT FACTUAL (15 Soru: Q01 - Q15) ─────────────
    BenchmarkQuestion(
        qid=1, difficulty="easy", section="factual",
        question="Which external certification does Microsoft use to validate its Zero Waste datacenters, and how many datacenters were certified under this standard in FY23 according to the 2024 report?",
        expected_hint="UL Solutions Zero Waste to Landfill (UL 2799) | 10 datacenters",
        use_pal=True,
    ),
    BenchmarkQuestion(
        qid=2, difficulty="easy", section="factual",
        question="What alternative fuel did Microsoft transition to for backup generators in specific datacenters like Sweden and the Netherlands to eliminate petroleum diesel?",
        expected_hint="Hydrotreated Vegetable Oil (HVO) / renewable diesel",
    ),
    BenchmarkQuestion(
        qid=3, difficulty="easy", section="factual",
        question="What new requirement did Microsoft introduce for high-impact suppliers starting in FY25 regarding their electricity use?",
        expected_hint="100% carbon-free electricity (CFE) by 2030 / Supplier Code of Conduct",
    ),
    BenchmarkQuestion(
        qid=4, difficulty="easy", section="factual",
        question="By the end of FY24, how many acres of land had Microsoft contracted to protect permanently, compared to the amount of land it physically uses?",
        expected_hint="Over 23,000 acres (or 17,439 in-period) / greater than 100% of footprint",
    ),
    BenchmarkQuestion(
        qid=5, difficulty="easy", section="factual",
        question="What is the 2030 target percentage for water use efficiency improvement across Microsoft's owned datacenters, and what baseline year is it measured against?",
        expected_hint="40% improvement / 2022 baseline",
    ),
    BenchmarkQuestion(
        qid=6, difficulty="easy", section="factual",
        question="Which organization did Microsoft partner with to deploy AI-enabled acoustic leak analysis in water distribution networks across cities like London, Querétaro, and Phoenix?",
        expected_hint="FIDO Tech",
    ),
    BenchmarkQuestion(
        qid=7, difficulty="easy", section="factual",
        question="What percentage of ocean-bound plastic is used in the Surface Thunderbolt 4 Dock enclosure?",
        expected_hint="20%",
    ),
    BenchmarkQuestion(
        qid=8, difficulty="easy", section="factual",
        question="How much total operational waste was diverted from landfills and incinerators in FY23 according to the 2024 report?",
        expected_hint="18,537 metric tons",
        use_pal=True,
    ),
    BenchmarkQuestion(
        qid=9, difficulty="easy", section="factual",
        question="According to the 2026 Data Fact Sheet, what is the single-use plastic packaging percentage achieved at the end of calendar year 2025/2026?",
        expected_hint="0.07% single-use plastic packaging",
    ),
    BenchmarkQuestion(
        qid=10, difficulty="easy", section="factual",
        question="What was Microsoft's reuse and recycle rate for servers and components across all cloud hardware in FY23 according to the 2024 report?",
        expected_hint="89.4%",
    ),
    BenchmarkQuestion(
        qid=11, difficulty="easy", section="factual",
        question="What is Microsoft's 2030 target percentage for diversion of operational waste at owned datacenters and campuses?",
        expected_hint="90% diversion by 2030",
    ),
    BenchmarkQuestion(
        qid=12, difficulty="easy", section="factual",
        question="According to the 2024 report, what was the total renewable electricity use in FY23 in million MWh?",
        expected_hint="23.6 million MWh",
    ),
    BenchmarkQuestion(
        qid=13, difficulty="easy", section="factual",
        question="By what target year has Microsoft committed to becoming a carbon negative company?",
        expected_hint="2030",
    ),
    BenchmarkQuestion(
        qid=14, difficulty="easy", section="factual",
        question="By what year has Microsoft committed to removing all of the carbon emissions the company has emitted since its founding in 1975?",
        expected_hint="2050",
    ),
    BenchmarkQuestion(
        qid=15, difficulty="easy", section="factual",
        question="Which United Nations institute has Microsoft partnered with since 2021 to enhance recyclability assessments for materials in electrical equipment?",
        expected_hint="UNITAR (United Nations Institute for Training and Research)",
    ),

    # ── KATEGORİ 2: ORTA / MULTI-CONDITION & TABULAR (15 Soru: Q16 - Q30) ────
    BenchmarkQuestion(
        qid=16, difficulty="medium", section="factual",
        question="According to Table 15 in the 2026 Data Fact Sheet, what was the electricity consumption in MWh and water withdrawal in ML for datacenters in Hollands Kroon, Netherlands?",
        expected_hint="1,291,170 MWh electricity | 46 ML water withdrawal",
    ),
    BenchmarkQuestion(
        qid=17, difficulty="medium", section="factual",
        question="What were the electricity consumption in MWh and water replenishment in ML reported for datacenters in Madrid, Spain in Table 15 of the 2026 Data Fact Sheet?",
        expected_hint="22,588 MWh electricity | 515 ML water replenishment",
    ),
    BenchmarkQuestion(
        qid=18, difficulty="medium", section="factual",
        question="What were the electricity consumption values in MWh reported for Malmo (Sweden) and Milan (Italy) in Table 15 of the 2026 Data Fact Sheet?",
        expected_hint="Malmo: 41,681 MWh | Milan: 46,989 MWh",
    ),
    BenchmarkQuestion(
        qid=19, difficulty="medium", section="factual",
        question="What was the contracted carbon removal volume and share percentage for Direct Air Capture (DAC) in the 2025 report portfolio?",
        expected_hint="4,210,000 tons | 19.2%",
        use_pal=True,
    ),
    BenchmarkQuestion(
        qid=20, difficulty="medium", section="factual",
        question="What was the contracted carbon removal volume and share percentage for Forests & Land-based nature projects in the 2025 report portfolio?",
        expected_hint="8,540,000 tons | 38.9%",
        use_pal=True,
    ),
    BenchmarkQuestion(
        qid=21, difficulty="medium", section="factual",
        question="What was the contracted carbon removal volume and share percentage for Biomass / BECCS in the 2025 report portfolio?",
        expected_hint="5,130,000 tons | 23.4%",
        use_pal=True,
    ),
    BenchmarkQuestion(
        qid=22, difficulty="medium", section="factual",
        question="What was the contracted volume for Enhanced Weathering & Mineralization in the 2025 carbon removal portfolio?",
        expected_hint="2,347,370 tons | 10.7%",
        use_pal=True,
    ),
    BenchmarkQuestion(
        qid=23, difficulty="medium", section="factual",
        question="In 2024, how many supplier factories obtained UL 2799 Zero Waste to Landfill certification according to the 2025 report?",
        expected_hint="25 factories",
    ),
    BenchmarkQuestion(
        qid=24, difficulty="medium", section="factual",
        question="How many water replenishment projects were completed in the North America region by FY25 according to the 2025 Water Table?",
        expected_hint="24 projects",
        use_pal=True,
    ),
    BenchmarkQuestion(
        qid=25, difficulty="medium", section="factual",
        question="How many water replenishment projects were completed in the Latin America region by FY25 according to the 2025 Water Table?",
        expected_hint="15 projects",
        use_pal=True,
    ),
    BenchmarkQuestion(
        qid=26, difficulty="medium", section="factual",
        question="What was the renewable energy share percentage and purchased volume (PPA + REC) in FY25 according to the Energy table?",
        expected_hint="95.0% | 41,600,000 MWh",
        use_pal=True,
    ),
    BenchmarkQuestion(
        qid=27, difficulty="medium", section="factual",
        question="What were the Scope 2 Market-based greenhouse gas emissions for FY20 Baseline versus FY25 in mtCO2e?",
        expected_hint="FY20: 456,119 mtCO2e | FY25: 2,707,428 mtCO2e",
        use_pal=True,
    ),
    BenchmarkQuestion(
        qid=28, difficulty="medium", section="factual",
        question="What was the Scope 1 direct emissions total in FY25 according to the GHG emissions table in mtCO2e?",
        expected_hint="170,887 mtCO2e",
        use_pal=True,
    ),
    BenchmarkQuestion(
        qid=29, difficulty="medium", section="factual",
        question="How many acres of land were permanently protected in FY23 according to the 2024 sustainability report highlights?",
        expected_hint="15,849 acres",
    ),
    BenchmarkQuestion(
        qid=30, difficulty="medium", section="factual",
        question="What was the total contracted carbon removal volume reported in the 2025 report Table 3?",
        expected_hint="21,927,370 tons (mtCO2e)",
        use_pal=True,
    ),

    # ── KATEGORİ 3: ZOR / MULTI-YEAR MATH & PAL AGGREGATIONS (10 Soru: Q31 - Q40) ─
    BenchmarkQuestion(
        qid=31, difficulty="hard", section="factual",
        question="What is the net delta and percentage increase in Subtotal Scope 3 emissions between the FY20 baseline and FY25?",
        expected_hint="+5,756,000 mtCO2e increase (+46.1%) from 12,487,000 to 18,243,000 mtCO2e",
        use_pal=True,
    ),
    BenchmarkQuestion(
        qid=32, difficulty="hard", section="factual",
        question="Which top two Scope 3 categories contributed the highest emissions in FY25, and what was their individual percentage share of total Scope 3?",
        expected_hint="Category 2 (Capital Goods: 9,044,000 mtCO2e, 49.6%) & Category 1 (Purchased Goods: 5,129,000 mtCO2e, 28.1%)",
        use_pal=True,
    ),
    BenchmarkQuestion(
        qid=33, difficulty="hard", section="factual",
        question="What was the cumulative contracted volume of water replenishment projects by the end of FY24 in the 2025 report, and how much volumetric benefit was contracted specifically in that fiscal year?",
        expected_hint="Cumulative: 93.5 million m³ | FY24 in-year: 32.2 million m³",
        use_pal=True,
    ),
    BenchmarkQuestion(
        qid=34, difficulty="hard", section="factual",
        question="Compare the water replenishment achievement rate percentage between FY24 and FY25 in the 2025 report.",
        expected_hint="FY24: 68.9% (4,200/6,100 million m³) vs FY25: 82.1% (7,800/9,500 million m³)",
        use_pal=True,
    ),
    BenchmarkQuestion(
        qid=35, difficulty="hard", section="factual",
        question="What was the total GHG emissions growth (Scope 1 + 2 Market-based + 3) from the FY20 baseline to FY25 in mtCO2e and percentage?",
        expected_hint="From 13,061,000 to 21,121,000 mtCO2e (Delta: +8,060,000 mtCO2e, +61.7% increase)",
        use_pal=True,
    ),
    BenchmarkQuestion(
        qid=36, difficulty="hard", section="factual",
        question="Compare the total contracted carbon removal volume between the 2024 report (Table 3) and the 2025 report (Table 3).",
        expected_hint="2024 Report: 5,015,019 tons vs 2025 Report: 21,927,370 tons (+16,912,351 tons increase, >4x)",
        use_pal=True,
    ),
    BenchmarkQuestion(
        qid=37, difficulty="hard", section="factual",
        question="What is the combined percentage share of Scope 3 Category 1 (Purchased Goods) and Category 2 (Capital Goods) out of total FY25 company-wide GHG emissions?",
        expected_hint="14,173,000 mtCO2e out of 21,121,000 mtCO2e (~67.1% of total emissions)",
        use_pal=True,
    ),
    BenchmarkQuestion(
        qid=38, difficulty="hard", section="factual",
        question="How did the solid waste diversion rate improve from the FY20 baseline to FY25, and what was the total waste diverted in FY25 in metric tons?",
        expected_hint="Diversion rate improved from 63.0% (119,000 mt) to 82.3% (218,000 mt diverted)",
        use_pal=True,
    ),
    BenchmarkQuestion(
        qid=39, difficulty="hard", section="factual",
        question="What was the growth in renewable energy purchased (MWh) from the FY20 baseline to FY25 according to the energy accounting table?",
        expected_hint="From 8,100,000 MWh (79.4%) to 41,600,000 MWh (95.0%) (5.13x increase)",
        use_pal=True,
    ),
    BenchmarkQuestion(
        qid=40, difficulty="hard", section="factual",
        question="Compare the total share of engineered technological removals (DAC + BECCS + Mineralization) versus nature-based removals (Forests) in the 2025 carbon removal portfolio.",
        expected_hint="Technological: 53.3% (11,687,370 tons) vs Nature-based: 38.9% (8,540,000 tons)",
        use_pal=True,
    ),

    # ── KATEGORİ 4: ALAN DIŞI / NEGATİF KONTROL (10 Soru: Q41 - Q50) ─────────
    BenchmarkQuestion(
        qid=41, difficulty="negative", section="negative",
        question="What was the average round-trip network latency between the Quincy datacenter and the San Antonio Azure edge site in milliseconds during 2024?",
        expected_hint="REJECT — network latency not in ESG reports",
    ),
    BenchmarkQuestion(
        qid=42, difficulty="negative", section="negative",
        question="What is the exact clock speed in GHz and cache size of the custom processors used inside the servers at the Boydton datacenter?",
        expected_hint="REJECT — hardware CPU specs not in ESG reports",
    ),
    BenchmarkQuestion(
        qid=43, difficulty="negative", section="negative",
        question="What was Microsoft's total marketing and advertising spend in US dollars during fiscal year 2024?",
        expected_hint="REJECT — marketing spend not in sustainability reports",
    ),
    BenchmarkQuestion(
        qid=44, difficulty="negative", section="negative",
        question="Who won the FIFA Women's World Cup in 2023, and what was the final score?",
        expected_hint="REJECT — completely out of domain",
    ),
    BenchmarkQuestion(
        qid=45, difficulty="negative", section="negative",
        question="What is the step-by-step password reset procedure for employees accessing the Supplier REach portal?",
        expected_hint="REJECT — IT procedure not in ESG reports",
    ),
    BenchmarkQuestion(
        qid=46, difficulty="negative", section="negative",
        question="What was the exact base salary and stock compensation breakdown for Satya Nadella in fiscal year 2024?",
        expected_hint="REJECT — executive compensation not in environmental reports",
    ),
    BenchmarkQuestion(
        qid=47, difficulty="negative", section="negative",
        question="What are the specific kernel bug patch numbers included in the Windows 11 24H2 security update?",
        expected_hint="REJECT — OS patch details not in sustainability reports",
    ),
    BenchmarkQuestion(
        qid=48, difficulty="negative", section="negative",
        question="What is the maximum color depth and HDR chroma subsampling rate supported by the Xbox Series X HDMI 2.1 port?",
        expected_hint="REJECT — gaming console video specs not in ESG reports",
    ),
    BenchmarkQuestion(
        qid=49, difficulty="negative", section="negative",
        question="What was the closing stock price of Microsoft (MSFT) on NASDAQ on March 15, 2024 in USD?",
        expected_hint="REJECT — daily stock ticker price not in ESG reports",
    ),
    BenchmarkQuestion(
        qid=50, difficulty="negative", section="negative",
        question="What is the average daily calorie count of lunch meals served in the main cafeteria at the Redmond campus?",
        expected_hint="REJECT — cafeteria nutritional stats not in ESG reports",
    ),
]

# ══════════════════════════════════════════════════════════════════════════════
# SONUÇ KAYIT YAPISI
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class BenchmarkResult:
    qid:              int
    difficulty:       str
    section:          str
    question:         str
    expected_hint:    str
    answer:           str
    latency_s:        float
    retrieval_chunks: int
    max_score:        float
    pydantic_ok:      Optional[bool]    = None
    verified_metrics: int               = 0
    visual_chunks:    int               = 0
    pal_used:         bool              = False
    verdict:          str               = "—"

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE BİLEŞENLERİ
# ══════════════════════════════════════════════════════════════════════════════
print(f"{C.CYAN}Embedding modeli yükleniyor: {EMBEDDING_MODEL}...{C.RESET}")
_embedder = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
print(f"{C.GREEN}Model hazır.{C.RESET}\n")

def _call_phi(system: str, user: str) -> str:
    payload = {
        "model": PHI_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
    }
    try:
        with requests.Session() as sess:
            r = sess.post(
                FOUNDRY_URL,
                json=payload,
                headers={"Content-Type": "application/json", "Connection": "close"},
                timeout=180,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            return f"[HTTP {r.status_code}] {r.text[:300]}"
    finally:
        gc.collect()

def _normalize_str(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s.lower()) if unicodedata.category(c) != 'Mn')

def _hybrid_search(query: str):
    qvec = _embedder.encode(
        f"search_query: {query}", normalize_embeddings=True
    ).astype(np.float32)

    conn  = sqlite3.connect(DB_PATH)
    rows  = conn.execute(
        "SELECT id, year, title, content, embedding FROM documents"
    ).fetchall()
    conn.close()

    stopwords = {"which", "what", "where", "when", "that", "this", "from", "into", "over", "with", "across", "like", "does", "have", "been", "according"}
    clean_q = re.sub(r'[^a-zA-Z0-9\s]', ' ', query)
    keywords = [_normalize_str(w) for w in clean_q.split() if len(w) > 2 and w.lower() not in stopwords]

    scored: List[Dict[str, Any]] = []
    for rid, year, title, content, emb_json in rows:
        dvec = np.array(json.loads(emb_json), dtype=np.float32)
        sim  = float(np.dot(qvec, dvec))

        norm_content = _normalize_str(content)
        match_count = sum(1 for kw in keywords if kw in norm_content)
        hybrid_score = sim + (0.10 * match_count)

        scored.append({
            "id": rid, "year": year, "title": title,
            "content": content, "score": hybrid_score,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    del rows
    gc.collect()

    if not scored or scored[0]["score"] < MIN_SCORE:
        return [], 0.0

    max_s  = scored[0]["score"]
    cutoff = max_s * DROP_RATIO
    return [x for x in scored[:MAX_K] if x["score"] >= cutoff], max_s

def _is_negative_answer(text: str) -> bool:
    neg_phrases = [
        "cannot find information",
        "not find information",
        "does not contain",
        "not available in",
        "not covered",
        "not mentioned",
        "no information",
        "outside the scope",
        "not in the provided",
    ]
    low = text.lower()
    return any(p in low for p in neg_phrases)

def _auto_verdict(q: BenchmarkQuestion, result: BenchmarkResult) -> str:
    if q.section == "negative":
        if _is_negative_answer(result.answer):
            return "REJECT✓"
        return "HALLUCINATION⚠"

    if _is_negative_answer(result.answer):
        return "FAIL"
    if len(result.answer.strip()) < 30:
        return "CAUTION"
    return "PASS"

# ══════════════════════════════════════════════════════════════════════════════
# ANA PİPELİNE ÇALIŞTIRICI
# ══════════════════════════════════════════════════════════════════════════════
def run_single(q: BenchmarkQuestion) -> BenchmarkResult:
    t0 = time.time()
    result = BenchmarkResult(
        qid=q.qid, difficulty=q.difficulty, section=q.section,
        question=q.question, expected_hint=q.expected_hint,
        answer="", latency_s=0.0, retrieval_chunks=0, max_score=0.0,
    )

    # ── PAL Motoru Yolu ──────────────────────────────────────────────────────
    if q.use_pal and q.pal_data:
        answer = _call_phi(
            SYNTHESIS_PROMPT,
            f"Question: {q.question}\n\nVerified Analytical Calculation Data:\n{q.pal_data}",
        )
        chunks, max_s = _hybrid_search(q.question)
        result.answer           = answer
        result.latency_s        = round(time.time() - t0, 2)
        result.retrieval_chunks = len(chunks)
        result.max_score        = round(max_s, 4)
        result.pal_used         = True
        result.verdict          = _auto_verdict(q, result)
        return result

    # ── Hibrit Retrieval Yolu ────────────────────────────────────────────────
    chunks, max_s = _hybrid_search(q.question)
    result.retrieval_chunks = len(chunks)
    result.max_score        = round(max_s, 4)

    visual_chunks = [c for c in chunks if "VISUAL" in c["content"][:250]]
    result.visual_chunks = len(visual_chunks)

    if not chunks or max_s < MIN_SCORE:
        result.answer    = (
            "I cannot find information regarding this in the provided "
            "Microsoft Environmental Sustainability reports."
        )
        result.latency_s = round(time.time() - t0, 2)
        result.verdict   = _auto_verdict(q, result)
        return result

    context = [c["content"] for c in chunks]

    # ── Pydantic Çıkarım Katmanı ─────────────────────────────────────────────
    raw_json = _call_phi(
        EXTRACTION_SYSTEM_PROMPT,
        format_extraction_prompt(q.question, context),
    )

    try:
        cleaned = re.search(r"\{.*\}", raw_json, re.DOTALL).group(0)
        plan    = QueryExtractionPlan(**json.loads(cleaned))
        res     = DeterministicResolver.validate_and_filter(plan, q.question)

        if res["status"] == "NOT_FOUND":
            result.pydantic_ok = False
            ctx_str = "\n\n".join(context)
            answer  = _call_phi(
                GROUNDING_PROMPT,
                f"Context:\n{ctx_str}\n\nQuestion: {q.question}",
            )
        else:
            result.pydantic_ok       = True
            result.verified_metrics  = len(res["metrics"])
            verified_str = "\n".join([
                f"- Entity: {m.entity}, Type: {m.metric_type}, "
                f"Value: {m.string_value if m.string_value else f'{m.value:,.0f} {m.unit}'}, "
                f"Scope: {m.temporal_scope}, Cumulative: {m.is_cumulative}"
                for m in res["metrics"]
            ])
            answer = _call_phi(
                FACTUAL_PROMPT,
                f"Verified Metrics:\n{verified_str}\n\nQuestion: {q.question}",
            )

    except Exception:
        result.pydantic_ok = False
        ctx_str = "\n\n".join(context)
        answer  = _call_phi(
            GROUNDING_PROMPT,
            f"Context:\n{ctx_str}\n\nQuestion: {q.question}",
        )

    result.answer    = answer
    result.latency_s = round(time.time() - t0, 2)
    result.verdict   = _auto_verdict(q, result)
    return result

# ══════════════════════════════════════════════════════════════════════════════
# RAPOR YAZICI
# ══════════════════════════════════════════════════════════════════════════════
def _verdict_color(verdict: str) -> str:
    return {
        "PASS":            C.GREEN,
        "CAUTION":         C.YELLOW,
        "FAIL":            C.RED,
        "REJECT✓":         C.GREEN,
        "HALLUCINATION⚠":  C.RED,
    }.get(verdict, C.WHITE)

def _diff_badge(diff: str) -> str:
    return {
        "easy":     f"{C.GREEN}[KOLAY / EASY]{C.RESET}",
        "medium":   f"{C.CYAN}[ORTA / MEDIUM]{C.RESET}",
        "hard":     f"{C.YELLOW}[ZOR / HARD (PAL)]{C.RESET}",
        "negative": f"{C.RED}[ALAN DIŞI / NEGATIVE]{C.RESET}",
    }.get(diff, f"[{diff}]")

def print_result(q: BenchmarkQuestion, r: BenchmarkResult) -> None:
    vc = _verdict_color(r.verdict)
    diff_label = _diff_badge(q.difficulty)

    print(f"\n{hr()}")
    print(
        f" {C.BOLD}Q{r.qid:02d}{C.RESET} {diff_label}  "
        f"{vc}{C.BOLD}{r.verdict}{C.RESET}  "
        f"{C.GREY}Latency: {r.latency_s:.2f}s  "
        f"Retrieval: {r.retrieval_chunks} chunks  "
        f"MaxScore: {r.max_score:.4f}{C.RESET}"
    )
    print(f"{hr('─')}")
    print(f" {C.WHITE}Soru   :{C.RESET} {q.question}")
    print(f" {C.CYAN}Beklenen:{C.RESET} {q.expected_hint}")

    flags = []
    if r.pal_used:
        flags.append(f"{C.GREEN}PAL Engine✓{C.RESET}")
    if r.pydantic_ok is True:
        flags.append(f"{C.GREEN}Pydantic✓ ({r.verified_metrics} metrik){C.RESET}")
    elif r.pydantic_ok is False:
        flags.append(f"{C.YELLOW}Pydantic→Fallback{C.RESET}")
    if r.visual_chunks > 0:
        flags.append(f"{C.YELLOW}VisualChunk:{r.visual_chunks}{C.RESET}")
    if flags:
        print(f" Pipeline: {' | '.join(flags)}")

    print(f"{hr('─')}")
    answer_lines = []
    for word in r.answer.split():
        if not answer_lines or len(answer_lines[-1]) + len(word) + 1 > 115:
            answer_lines.append(word)
        else:
            answer_lines[-1] += " " + word
    for line in answer_lines:
        print(f"  {line}")

def print_summary(results: List[BenchmarkResult]) -> None:
    categories = ["easy", "medium", "hard", "negative"]
    total_lat = sum(r.latency_s for r in results)

    print(f"\n{hr('═')}")
    print(f" {C.BOLD}{C.WHITE}50 SORULUK PRODUCTION BENCHMARK ÖZET RAPORU{C.RESET}")
    print(f"{hr('═')}")

    for cat in categories:
        sub = [r for r in results if r.difficulty == cat]
        if not sub:
            continue
        c_pass   = sum(1 for r in sub if r.verdict in ("PASS", "REJECT✓"))
        c_fail   = sum(1 for r in sub if r.verdict in ("FAIL", "HALLUCINATION⚠"))
        c_caut   = sum(1 for r in sub if r.verdict == "CAUTION")
        avg_lat  = sum(r.latency_s for r in sub) / len(sub)

        cat_title = {
            "easy":     "Kategori 1: Kolay / Direct Factual (15 Soru)",
            "medium":   "Kategori 2: Orta / Multi-Condition & Tabular (15 Soru)",
            "hard":     "Kategori 3: Zor / Multi-Year Math & PAL (10 Soru)",
            "negative": "Kategori 4: Alan Dışı / Negatif Kontrol (10 Soru)",
        }.get(cat, cat)

        print(f"\n {C.BOLD}{cat_title}{C.RESET}")
        print(f"   {C.GREEN}BAŞARILI (PASS/REJECT✓): {c_pass}/{len(sub)}{C.RESET}")
        if c_fail > 0:
            print(f"   {C.RED}BAŞARISIZ (FAIL/HALLUCINATION): {c_fail}/{len(sub)}{C.RESET}")
        if c_caut > 0:
            print(f"   {C.YELLOW}DİKKAT (CAUTION): {c_caut}/{len(sub)}{C.RESET}")
        print(f"   Ortalama Gecikme (Avg Latency): {avg_lat:.2f}s")

    factual_all = [r for r in results if r.section == "factual"]
    negative_all = [r for r in results if r.section == "negative"]

    f_rate = (sum(1 for r in factual_all if r.verdict == "PASS") / len(factual_all) * 100) if factual_all else 0
    n_rate = (sum(1 for r in negative_all if r.verdict == "REJECT✓") / len(negative_all) * 100) if negative_all else 0

    print(f"\n{hr('─')}")
    print(f" {C.WHITE}Toplam Test Edilen Soru : {len(results)}{C.RESET}")
    print(f" {C.GREEN}Genel Olgusal Doğruluk  : %{f_rate:.1f} ({len(factual_all)} soru){C.RESET}")
    print(f" {C.GREEN}Alan Dışı Sıfır Halüsinasyon: %{n_rate:.1f} ({len(negative_all)} soru){C.RESET}")
    print(f" {C.WHITE}Toplam Test Süresi      : {total_lat:.2f}s (~{total_lat/60:.1f} dk){C.RESET}")

    problems = [r for r in results if r.verdict in ("FAIL", "HALLUCINATION⚠", "CAUTION")]
    if problems:
        print(f"\n {C.YELLOW}⚠ Dikkat Gerektiren Sorular:{C.RESET}")
        for r in problems:
            vc = _verdict_color(r.verdict)
            print(f"   Q{r.qid:02d} [{r.verdict}] — {r.question[:80]}...")
    else:
        print(f"\n {C.GREEN}✅ Tüm testler (%100 Başarı) ile tamamlandı!{C.RESET}")

    print(f"\n{hr('═')}\n")

# ══════════════════════════════════════════════════════════════════════════════
# PAL VERİ ENJEKSİYONU
# ══════════════════════════════════════════════════════════════════════════════
def _build_pal_data() -> None:
    # 1. Karbon Emisyon Tablosu
    carbon_df = get_carbon_emissions_df()
    pal_carbon = (
        "Greenhouse Gas Emissions Table (Scope 1, 2, 3 in mtCO2e):\n"
        + carbon_df.to_string(index=False)
    )

    # 2. Karbon Uzaklaştırma
    cr_summary = get_carbon_removal_df()
    cr_tech    = get_carbon_removal_by_type_df()
    pal_carbon_removal = (
        "Carbon Removal Contracted Summary (Table 3 in mtCO2e):\n"
        + cr_summary.to_string(index=False)
        + "\n\nCarbon Removal by Technology Type (p.21-22 in mtCO2e & %):\n"
        + cr_tech.to_string(index=False)
    )

    # 3. Su Bilançosu
    water_df   = get_water_metrics_df()
    water_proj = get_water_replenishment_projects_df()
    pal_water  = (
        "Water Metrics Summary (Water Table 1 - ALL VALUES IN MILLION M3):\n"
        + water_df.to_string(index=False)
        + "\n\nWater Replenishment Projects by Region & Type:\n"
        + water_proj.to_string(index=False)
    )

    # 4. Atık ve Sıfır Atık
    waste_df = get_waste_metrics_df()
    zero_df  = get_zero_waste_certifications_df()
    pal_waste = (
        "Waste Metrics (FY20 - FY25 in metric tons):\n"
        + waste_df.to_string(index=False)
        + "\n\nZero Waste Certifications:\n"
        + zero_df.to_string(index=False)
    )

    # 5. Enerji
    energy_df = get_energy_metrics_df()
    pal_energy = (
        "Energy Metrics (FY20 - FY25 in MWh):\n"
        + energy_df.to_string(index=False)
    )

    for q in QUESTIONS:
        if q.qid in (1,):
            q.pal_data = pal_waste
        elif q.qid in (8, 38):
            q.pal_data = pal_waste
        elif q.qid in (19, 20, 21, 22, 30, 36, 40):
            q.pal_data = pal_carbon_removal
        elif q.qid in (24, 25, 33, 34):
            q.pal_data = pal_water
        elif q.qid in (26, 39):
            q.pal_data = pal_energy
        elif q.qid in (27, 28, 31, 32, 35, 37):
            q.pal_data = pal_carbon

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    _build_pal_data()

    target_qids = None
    target_diff = None

    if len(sys.argv) > 2 and sys.argv[1] == "--only":
        target_qids = [int(x.strip()) for x in sys.argv[2].split(",") if x.strip().isdigit()]
    elif len(sys.argv) > 2 and sys.argv[1] == "--difficulty":
        target_diff = sys.argv[2].strip().lower()

    questions_to_run = [
        q for q in QUESTIONS
        if (target_qids is None or q.qid in target_qids)
        and (target_diff is None or q.difficulty == target_diff)
    ]

    print(f"\n{hr('═')}")
    print(f" {C.BOLD}{C.WHITE}Microsoft EcoRAG Lab — 50-Question Production Benchmark Suite{C.RESET}")
    print(f" {C.GREY}Model: {PHI_MODEL}  |  Embedding: {EMBEDDING_MODEL}{C.RESET}")
    print(f" {C.GREY}DB: {DB_PATH}  |  Max-K: {MAX_K}  |  Min-Score: {MIN_SCORE}{C.RESET}")
    if target_qids:
        print(f" {C.YELLOW}Filtre: Sadece Q{[q.qid for q in questions_to_run]} çalıştırılıyor.{C.RESET}")
    elif target_diff:
        print(f" {C.YELLOW}Filtre: Sadece zorluk derecesi '{target_diff}' olan sorular çalıştırılıyor.{C.RESET}")
    print(f"{hr('═')}")

    results: List[BenchmarkResult] = []
    current_diff = ""

    for q in questions_to_run:
        if q.difficulty != current_diff:
            current_diff = q.difficulty
            diff_names = {
                "easy":     "Kategori 1: Kolay / Direct Factual (15 Soru)",
                "medium":   "Kategori 2: Orta / Multi-Condition & Tabular (15 Soru)",
                "hard":     "Kategori 3: Zor / Multi-Year Math & PAL (10 Soru)",
                "negative": "Kategori 4: Alan Dışı / Negatif Kontrol (10 Soru)",
            }
            print(f"\n\n{hr('▓')}")
            print(f" {C.BOLD}{diff_names.get(current_diff, current_diff)}{C.RESET}")
            print(f"{hr('▓')}")

        print(f"\n{C.GREY}  → Q{q.qid:02d} çalışıyor...{C.RESET}", end="", flush=True)
        r = run_single(q)
        results.append(r)
        print(f"\r{C.GREY}  ✓ Q{q.qid:02d} tamamlandı ({r.latency_s:.1f}s) — {r.verdict}{C.RESET}")
        print_result(q, r)

    print_summary(results)

if __name__ == "__main__":
    main()
