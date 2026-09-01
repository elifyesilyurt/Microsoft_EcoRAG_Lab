"""
run_benchmarks.py — Microsoft EcoRAG Lab Production Benchmark Suite
====================================================================
app.py pipeline'ını doğrudan import ederek soruları uçtan uca test eder.

  Bölüm 1 : Factual & Deterministik Çıkarım Testleri (Q1–Q9)
  Bölüm 2 : Negatif Kontrol / Alan Dışı (Out-of-Domain) Testleri (Q10–Q14)

Kullanım:
  python run_benchmarks.py               # Tüm 14 soruyu çalıştırır
  python run_benchmarks.py --only 1,2,7,9 # Sadece belirtilen soruları çalıştırır
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
    "Synthesize the verified data into a concise, precise executive answer. "
    "State exact numbers and units clearly in sentence 1. Do NOT repeat yourself."
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

def hr(char: str = "═", width: int = 72) -> str:
    return char * width

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK VERİ SETİ
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class BenchmarkQuestion:
    qid:          int
    section:      str          # "factual" | "negative"
    question:     str
    expected_hint: str         # Beklenen cevabın anahtar kelimeleri
    use_pal:      bool = False
    pal_data:     str  = ""

QUESTIONS: List[BenchmarkQuestion] = [
    # ── Bölüm 1: Factual & Deterministik ────────────────────────────────────
    BenchmarkQuestion(
        qid=1, section="factual",
        question=(
            "Which external certification does Microsoft use to validate its Zero Waste "
            "datacenters, and how many datacenters were certified under this standard "
            "in FY23 according to the 2024 report?"
        ),
        expected_hint="UL Solutions Zero Waste to Landfill (Platinum/UL 2799) | 10 datacenters",
        use_pal=True,
    ),
    BenchmarkQuestion(
        qid=2, section="factual",
        question=(
            "What was the cumulative contracted volume of water replenishment projects "
            "by the end of FY24 in the 2025 report, and how much volumetric benefit "
            "was contracted specifically in that fiscal year?"
        ),
        expected_hint="Cumulative 93.5 million m³ | FY24 in-year 32.2 million m³",
        use_pal=True,
    ),
    BenchmarkQuestion(
        qid=3, section="factual",
        question=(
            "What alternative fuel did Microsoft transition to for backup generators "
            "in specific datacenters like Sweden and the Netherlands to eliminate "
            "petroleum diesel?"
        ),
        expected_hint="Hydrotreated Vegetable Oil (HVO) / renewable diesel",
    ),
    BenchmarkQuestion(
        qid=4, section="factual",
        question=(
            "What new requirement did Microsoft introduce for high-impact suppliers "
            "starting in FY25 regarding their electricity use?"
        ),
        expected_hint="100% carbon-free electricity (CFE) by 2030 / Supplier REach",
    ),
    BenchmarkQuestion(
        qid=5, section="factual",
        question=(
            "By the end of FY24, how many acres of land had Microsoft contracted to "
            "protect permanently, compared to the amount of land it physically uses?"
        ),
        expected_hint="Over 23,000 acres / greater than 100% of land footprint",
    ),
    BenchmarkQuestion(
        qid=6, section="factual",
        question=(
            "What is the 2030 target percentage for water use efficiency improvement "
            "across Microsoft's owned datacenters, and what baseline year is it "
            "measured against?"
        ),
        expected_hint="40% improvement / 2022 baseline",
    ),
    BenchmarkQuestion(
        qid=7, section="factual",
        question=(
            "Which organization did Microsoft partner with to deploy AI-enabled "
            "acoustic leak analysis in water distribution networks across cities "
            "like London, Querétaro, and Phoenix?"
        ),
        expected_hint="FIDO Tech",
    ),
    BenchmarkQuestion(
        qid=8, section="factual",
        question=(
            "What percentage of ocean-bound plastic is used in the Surface "
            "Thunderbolt 4 Dock enclosure?"
        ),
        expected_hint="20%",
    ),
    BenchmarkQuestion(
        qid=9, section="factual",
        question=(
            "How much total operational waste was diverted from landfills and "
            "incinerators in FY23 according to the 2024 report?"
        ),
        expected_hint="18,537 metric tons",
        use_pal=True,
    ),
    # ── Bölüm 2: Negatif Kontrol / Alan Dışı ────────────────────────────────
    BenchmarkQuestion(
        qid=10, section="negative",
        question=(
            "What was the average round-trip network latency between the Quincy "
            "datacenter and the San Antonio Azure edge site in milliseconds during 2024?"
        ),
        expected_hint="REJECT — network latency not in ESG reports",
    ),
    BenchmarkQuestion(
        qid=11, section="negative",
        question=(
            "What is the exact clock speed in GHz and cache size of the custom "
            "processors used inside the servers at the Boydton datacenter?"
        ),
        expected_hint="REJECT — hardware specs not in ESG reports",
    ),
    BenchmarkQuestion(
        qid=12, section="negative",
        question=(
            "What was Microsoft's total marketing and advertising spend in US dollars "
            "during fiscal year 2024?"
        ),
        expected_hint="REJECT — marketing spend not in sustainability reports",
    ),
    BenchmarkQuestion(
        qid=13, section="negative",
        question=(
            "Who won the FIFA Women's World Cup in 2023, and what was the final score?"
        ),
        expected_hint="REJECT — completely out of domain",
    ),
    BenchmarkQuestion(
        qid=14, section="negative",
        question=(
            "What is the step-by-step password reset procedure for employees accessing "
            "the Supplier REach portal?"
        ),
        expected_hint="REJECT — IT procedure not in ESG reports",
    ),
]

# ══════════════════════════════════════════════════════════════════════════════
# SONUÇ KAYIT YAPISI
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class BenchmarkResult:
    qid:              int
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
# PIPELINE BILEŞENLERI
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
    if len(result.answer.strip()) < 40:
        return "CAUTION"
    return "PASS"


# ══════════════════════════════════════════════════════════════════════════════
# ANA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_single(q: BenchmarkQuestion) -> BenchmarkResult:
    t0 = time.time()
    result = BenchmarkResult(
        qid=q.qid, section=q.section,
        question=q.question, expected_hint=q.expected_hint,
        answer="", latency_s=0.0, retrieval_chunks=0, max_score=0.0,
    )

    # ── PAL Motoru yolu ──────────────────────────────────────────────────────
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

    # ── Hibrit Retrieval yolu ────────────────────────────────────────────────
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


def print_result(q: BenchmarkQuestion, r: BenchmarkResult) -> None:
    vc = _verdict_color(r.verdict)
    section_label = (
        f"{C.BLUE}[FACTUAL]{C.RESET}"
        if q.section == "factual"
        else f"{C.RED}[NEGATIVE]{C.RESET}"
    )

    print(f"\n{hr()}")
    print(
        f" {C.BOLD}Q{r.qid:02d}{C.RESET} {section_label}  "
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
        flags.append(f"{C.GREEN}PAL✓{C.RESET}")
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
    factual  = [r for r in results if r.section == "factual"]
    negative = [r for r in results if r.section == "negative"]

    f_pass    = sum(1 for r in factual  if r.verdict == "PASS")
    f_caution = sum(1 for r in factual  if r.verdict == "CAUTION")
    f_fail    = sum(1 for r in factual  if r.verdict == "FAIL")
    n_reject  = sum(1 for r in negative if r.verdict == "REJECT✓")
    n_halluc  = sum(1 for r in negative if r.verdict == "HALLUCINATION⚠")

    avg_lat_f = sum(r.latency_s for r in factual)  / len(factual)  if factual  else 0
    avg_lat_n = sum(r.latency_s for r in negative) / len(negative) if negative else 0
    total_lat = sum(r.latency_s for r in results)

    print(f"\n{hr('═')}")
    print(f" {C.BOLD}{C.WHITE}BENCHMARK ÖZET RAPORU{C.RESET}")
    print(f"{hr('═')}")
    if factual:
        print(f"\n {C.BLUE}Bölüm 1 — Factual Testler ({len(factual)} soru){C.RESET}")
        print(f"   {C.GREEN}PASS   : {f_pass}{C.RESET}")
        print(f"   {C.YELLOW}CAUTION: {f_caution}{C.RESET}")
        print(f"   {C.RED}FAIL   : {f_fail}{C.RESET}")
        print(f"   Ort. Latency : {avg_lat_f:.2f}s")

    if negative:
        print(f"\n {C.RED}Bölüm 2 — Negatif Kontrol Testleri ({len(negative)} soru){C.RESET}")
        print(f"   {C.GREEN}REJECT✓ (Doğru Reddetme) : {n_reject}/{len(negative)}{C.RESET}")
        print(f"   {C.RED}HALLUCINATION⚠           : {n_halluc}/{len(negative)}{C.RESET}")
        print(f"   Ort. Latency : {avg_lat_n:.2f}s")

    print(f"\n {C.WHITE}Toplam Test Süresi  : {total_lat:.2f}s{C.RESET}")
    print(f" {C.WHITE}Toplam Soru Sayısı  : {len(results)}{C.RESET}")

    problems = [r for r in results if r.verdict in ("FAIL", "HALLUCINATION⚠", "CAUTION")]
    if problems:
        print(f"\n {C.YELLOW}⚠ Dikkat Gerektiren Sorular:{C.RESET}")
        for r in problems:
            vc = _verdict_color(r.verdict)
            print(f"   Q{r.qid:02d} [{r.verdict}] — {r.question[:80]}...")
    else:
        print(f"\n {C.GREEN}✅ Tüm testler başarıyla geçildi!{C.RESET}")

    print(f"\n{hr('═')}\n")


# ══════════════════════════════════════════════════════════════════════════════
# PAL VERİSİNİ HAZIRLA
# ══════════════════════════════════════════════════════════════════════════════

def _build_pal_data() -> None:
    # Q1: Zero Waste Certifications
    pal_q1 = (
        "Zero Waste & Datacenter Certifications Summary (2024 Report, p.5, 7, 36 & 2025 Report, p.47):\n"
        + get_zero_waste_certifications_df().to_string(index=False)
    )
    
    # Q2: Water Replenishment
    water_df   = get_water_metrics_df()
    water_proj = get_water_replenishment_projects_df()
    pal_water  = (
        "Water Metrics Summary (2025 Report, Water Table 1 - ALL VALUES IN MILLION M3):\n"
        + water_df.to_string(index=False)
        + "\n\nWater Replenishment Projects by Region & Type:\n"
        + water_proj.to_string(index=False)
    )

    # Q9: Waste Metrics & Operational Diversion
    pal_waste = (
        "Waste Metrics & Operational Diversion (2024 Report, p.5, 7, 36 & 2025 Report, p.40):\n"
        + get_waste_metrics_df().to_string(index=False)
        + "\n\nZero Waste Details:\n"
        + get_zero_waste_certifications_df().to_string(index=False)
    )

    for q in QUESTIONS:
        if q.qid == 1:
            q.pal_data = pal_q1
        elif q.qid == 2:
            q.pal_data = pal_water
        elif q.qid == 9:
            q.pal_data = pal_waste


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _build_pal_data()

    # CLI argüman kontrolü: --only 1,2,7,9
    target_qids = None
    if len(sys.argv) > 2 and sys.argv[1] == "--only":
        target_qids = [int(x.strip()) for x in sys.argv[2].split(",") if x.strip().isdigit()]

    questions_to_run = [q for q in QUESTIONS if target_qids is None or q.qid in target_qids]

    print(f"\n{hr('═')}")
    print(f" {C.BOLD}{C.WHITE}Microsoft EcoRAG Lab — Production Benchmark Suite{C.RESET}")
    print(f" {C.GREY}Model: {PHI_MODEL}  |  Embedding: {EMBEDDING_MODEL}{C.RESET}")
    print(f" {C.GREY}DB: {DB_PATH}  |  Max-K: {MAX_K}  |  Min-Score: {MIN_SCORE}{C.RESET}")
    if target_qids:
        print(f" {C.YELLOW}Filtre: Sadece Q{[q.qid for q in questions_to_run]} çalıştırılıyor.{C.RESET}")
    print(f"{hr('═')}")

    results: List[BenchmarkResult] = []
    current_section = ""

    for q in questions_to_run:
        if q.section != current_section:
            current_section = q.section
            label = (
                "BÖLÜM 1: Factual & Deterministik Testler"
                if q.section == "factual"
                else "BÖLÜM 2: Negatif Kontrol / Alan Dışı Testler"
            )
            print(f"\n\n{hr('▓')}")
            print(f" {C.BOLD}{label}{C.RESET}")
            print(f"{hr('▓')}")

        print(f"\n{C.GREY}  → Q{q.qid:02d} çalışıyor...{C.RESET}", end="", flush=True)
        r = run_single(q)
        results.append(r)
        print(f"\r{C.GREY}  ✓ Q{q.qid:02d} tamamlandı ({r.latency_s:.1f}s) — {r.verdict}{C.RESET}")
        print_result(q, r)

    print_summary(results)


if __name__ == "__main__":
    main()
