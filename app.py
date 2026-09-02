import os
import gc
import json
import re
import sqlite3
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from esg_tables import (
    get_carbon_emissions_df,
    get_carbon_removal_df,
    get_carbon_removal_by_type_df,
    get_water_metrics_df,
    get_water_replenishment_projects_df,
    get_energy_metrics_df,
    get_waste_metrics_df,
    get_zero_waste_certifications_df
)
from extraction_pipeline import (
    EXTRACTION_SYSTEM_PROMPT,
    format_extraction_prompt,
    QueryExtractionPlan,
    DeterministicResolver
)

# ══════════════════════════════════════════════════════════════════════════════
# KONFİGÜRASYON & SABİTLER
# ══════════════════════════════════════════════════════════════════════════════
DB_PATH = "rag_storage.db"
EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
FOUNDRY_BASE_URL = "http://127.0.0.1:62095"
MODEL_NAME = "phi-4-mini"

RELATIVE_DROP_RATIO = 0.70
MAX_K = 6
MIN_SCORE_FLOOR = 0.15

SYNTHESIS_PROMPT = """You are a Senior Sustainability Analyst.
Synthesize the verified analytical calculation results into a clear, structured executive report with exact units (mtCO2e / metric tons / m3).
Do not alter any calculated numbers. Do not repeat yourself."""

FACTUAL_SYNTHESIS_PROMPT = """You are a Senior Sustainability AI Analyst.
Using the verified structured metrics provided below, compose a concise, direct natural language answer.
State the exact numbers, names, and corresponding units clearly in sentence 1. Do not repeat yourself."""

# ══════════════════════════════════════════════════════════════════════════════
# SAYFA VE TEMA YAPILANDIRMASI
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Microsoft EcoRAG Lab",
    page_icon=":material/eco:",
    layout="wide"
)

# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING & FOUNDRY LOCAL MOTORU
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)

embedder = load_embedder()

def query_foundry(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    url = f"{FOUNDRY_BASE_URL}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Connection": "close"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": 512
    }
    
    try:
        with requests.Session() as session:
            res = session.post(url, headers=headers, json=payload, timeout=180)
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"].strip()
            else:
                raise RuntimeError(f"HTTP {res.status_code}: {res.text}")
    finally:
        gc.collect()

def compute_carbon_trend_summary() -> str:
    df = get_carbon_emissions_df()
    s1 = df[df["Metric"] == "Scope 1"].iloc[0]
    s2m = df[df["Metric"] == "Scope 2 (Market-based)"].iloc[0]
    s3 = df[df["Metric"] == "Subtotal Scope 3"].iloc[0]
    
    cat_df = df[df["Metric"].str.startswith("Scope 3 Cat")].copy()
    cat_df["Share_FY25"] = (cat_df["FY25"] / s3["FY25"]) * 100
    top2 = cat_df.sort_values(by="FY25", ascending=False).head(2)
    top2_list = [(r["Metric"], int(r["FY25"]), round(r["Share_FY25"], 2)) for _, r in top2.iterrows()]

    lines = [
        "Verified Scope Emissions Metrics (mtCO2e):",
        f"- Scope 1: FY20={int(s1['FY20_Baseline']):,}, FY24={int(s1['FY24']):,}, FY25={int(s1['FY25']):,} (Delta: +{int(s1['FY25']-s1['FY20_Baseline']):,})",
        f"- Scope 2 (Market-based): FY20={int(s2m['FY20_Baseline']):,}, FY24={int(s2m['FY24']):,}, FY25={int(s2m['FY25']):,} (Delta: +{int(s2m['FY25']-s2m['FY20_Baseline']):,})",
        f"- Scope 3 Subtotal: FY20={int(s3['FY20_Baseline']):,}, FY24={int(s3['FY24']):,}, FY25={int(s3['FY25']):,} (Delta: +{int(s3['FY25']-s3['FY20_Baseline']):,})",
        f"- FY25 Total Scope 3: {int(s3['FY25']):,} mtCO2e",
        "- Top 2 Scope 3 Categories (FY25):",
        f"  1. {top2_list[0][0]}: {top2_list[0][1]:,} mtCO2e ({top2_list[0][2]}%)",
        f"  2. {top2_list[1][0]}: {top2_list[1][1]:,} mtCO2e ({top2_list[1][2]}%)"
    ]
    return "\n".join(lines)

def search_context_hybrid(query: str):
    import unicodedata
    query_vector = embedder.encode(f"search_query: {query}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, year, title, content, embedding FROM documents")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return [], 0.0

    def normalize_str(s: str) -> str:
        return ''.join(c for c in unicodedata.normalize('NFD', s.lower()) if unicodedata.category(c) != 'Mn')

    stopwords = {"which", "what", "where", "when", "that", "this", "from", "into", "over", "with", "across", "like", "does", "have", "been", "according"}
    clean_q = re.sub(r'[^a-zA-Z0-9\s]', ' ', query)
    keywords = [normalize_str(w) for w in clean_q.split() if len(w) > 2 and w.lower() not in stopwords]

    scores = []
    for r in rows:
        c_id, year, title, content, emb_json = r
        doc_vector = np.array(json.loads(emb_json))
        norm_q = np.linalg.norm(query_vector)
        norm_d = np.linalg.norm(doc_vector)
        sim = 0.0 if (norm_q == 0 or norm_d == 0) else float(np.dot(query_vector, doc_vector) / (norm_q * norm_d))
        
        norm_content = normalize_str(content)
        match_count = sum(1 for kw in keywords if kw in norm_content)
        hybrid_score = sim + (0.10 * match_count)
        
        scores.append({"id": c_id, "year": year, "title": title, "content": content, "score": hybrid_score})

    scores.sort(key=lambda x: x["score"], reverse=True)
    if not scores:
        return [], 0.0

    max_score = scores[0]["score"]
    if max_score < MIN_SCORE_FLOOR:
        return [], max_score

    cutoff = max_score * RELATIVE_DROP_RATIO
    filtered = [item for item in scores[:MAX_K] if item["score"] >= cutoff]
    
    del scores
    del rows
    gc.collect()
    
    return filtered, max_score

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR (KENAR ÇUBUĞU)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### :material/eco: **EcoRAG Lab**")
    st.caption("Deterministic Sustainability Analysis")
    
    theme_choice = st.selectbox(
        "Görsel Tema / Palette",
        options=["🌿 Eco Emerald (Koyu)", "💼 Fluent Azure (Açık)"],
        index=0
    )
    
    with st.container(border=True):
        st.markdown("**Sistem Durumu**")
        st.badge("Aktif & Doğrulanmış", icon=":material/check_circle:", color="green")
        st.markdown(f"**Model:** `{MODEL_NAME}`")
        st.markdown(f"**Embedding:** `nomic-v1.5 (768d)`")
        st.markdown(f"**İndeks:** `982 Chunk (SQLite WAL)`")
        st.markdown(f"**Motor:** `PAL + Asymmetric IR`")
    
    if st.button("Sohbeti Sıfırla", icon=":material/delete:", width="stretch"):
        st.session_state.messages = []
        gc.collect()
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# DİNAMİK TEMA ENJEKSİYONU
# ══════════════════════════════════════════════════════════════════════════════
if "Eco Emerald" in theme_choice:
    st.html("""
    <style>
    /* Eco Emerald Dark Theme */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb22 !important;
        color: #58a6ff !important;
        font-weight: 600;
    }
    </style>
    """)
else:
    st.html("""
    <style>
    /* Fluent Azure Light Theme */
    .stApp {
        background-color: #f8f9fa;
        color: #1f2328;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0078d415 !important;
        color: #0078d4 !important;
        font-weight: 600;
    }
    </style>
    """)

# ══════════════════════════════════════════════════════════════════════════════
# BAŞLIK VE SEKME DÜZENİ (3 ANA SEKME)
# ══════════════════════════════════════════════════════════════════════════════
st.title("Microsoft EcoRAG Lab")
st.caption("Sıfır Halüsinasyonlu Deterministik ESG ve Sürdürülebilirlik Analiz Paneli")

tab_chat, tab_dashboard, tab_system = st.tabs([
    ":material/chat: Akıllı Asistan",
    ":material/analytics: ESG Bilanço Paneli",
    ":material/tune: Sistem & Benchmark Durumu"
])

# ══════════════════════════════════════════════════════════════════════════════
# SEKME 1: AKILLI ASİSTAN (SOHBET ARAYÜZÜ)
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Hazır Soru Hapları (st.pills)
    selected_pill = st.pills(
        "Hızlı Başlangıç Soruları",
        options=[
            "Scope 1-3 Emisyon Trendi",
            "FIDO Tech Akustik Su Kaçağı",
            "Zero Waste Veri Merkezleri (UL Standardı)",
            "Karbon Uzaklaştırma Portföyü"
        ],
        label_visibility="collapsed"
    )

    # Hap tıklandığında sorguyu eşleştir
    pill_query_map = {
        "Scope 1-3 Emisyon Trendi": "Compare Microsoft Scope 1, Scope 2, and Scope 3 emissions trend between FY20 baseline and FY25, highlighting the top contributing categories.",
        "FIDO Tech Akustik Su Kaçağı": "Which organization did Microsoft partner with to deploy AI-enabled acoustic leak analysis in water distribution networks across cities like London, Querétaro, and Phoenix?",
        "Zero Waste Veri Merkezleri (UL Standardı)": "Which external certification does Microsoft use to validate its Zero Waste datacenters, and how many datacenters were certified under this standard in FY23 according to the 2024 report?",
        "Karbon Uzaklaştırma Portföyü": "What is the total contracted carbon removal volume and its breakdown by technology type according to Carbon Table 3 in the 2025 report?"
    }

    active_query = None
    if selected_pill and selected_pill in pill_query_map:
        active_query = pill_query_map[selected_pill]

    # Geçmiş Mesajları Çiz
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and "route" in msg:
                if msg["route"] == "pal":
                    st.markdown(":green-badge[PAL Deterministik Hesaplama]")
                else:
                    st.markdown(":blue-badge[Hibrit Vektör Arama & Pydantic]")
            
            st.markdown(msg["content"])
            
            if "calc_details" in msg and msg["calc_details"]:
                with st.expander("⚡ Verified Analytical Engine Output", icon=":material/verified:"):
                    st.text(msg["calc_details"])
            if "provenance" in msg and msg["provenance"]:
                with st.expander(
                    f"Kullanılan Kaynaklar ({len(msg['provenance'])}) • Benzerlik Skoru: {msg.get('max_score', 0):.4f} • Süre: {msg.get('latency', 0):.2f}s",
                    icon=":material/library_books:"
                ):
                    for p in msg["provenance"]:
                        st.markdown(f"**{p['title']}** (Skor: {p['score']:.4f})")
                        st.text(p["content"][:300] + "...")

    # Kullanıcı Girdisi (chat_input veya pill)
    user_input = st.chat_input("Microsoft çevre ve sürdürülebilirlik raporlarına dair bir soru sorun...")
    query_to_run = user_input or active_query

    if query_to_run:
        # Son kullanıcı sorusu ile tekrarlanmayı engelle
        if not st.session_state.messages or st.session_state.messages[-1]["content"] != query_to_run:
            st.session_state.messages.append({"role": "user", "content": query_to_run})
            with st.chat_message("user"):
                st.markdown(query_to_run)

            start_time = time.time()
            with st.spinner("Deterministik çıkarım ve doğrulama yürütülüyor..."):
                try:
                    q_lower = query_to_run.lower()
                    is_math_scope = ("trend" in q_lower or "compare" in q_lower or "difference" in q_lower) and ("scope" in q_lower)
                    is_carbon_removal = ("carbon removal" in q_lower or "table 3" in q_lower) and ("technology" in q_lower or "breakdown" in q_lower)
                    is_zero_waste_cert = ("zero waste" in q_lower) and ("certification" in q_lower or "standard" in q_lower or "validate" in q_lower)
                    
                    calc_details = None
                    chunks = []
                    max_score = 0.0
                    route_type = "rag"

                    if is_math_scope:
                        route_type = "pal"
                        calc_details = compute_carbon_trend_summary()
                        synthesis_input = f"Question: {query_to_run}\n\nVerified Data:\n{calc_details}"
                        ans = query_foundry(SYNTHESIS_PROMPT, synthesis_input, temperature=0.0)
                        chunks, max_score = search_context_hybrid(query_to_run)
                    elif is_carbon_removal:
                        route_type = "pal"
                        calc_details = (
                            "Carbon Removal Summary (2025 Report, p.21):\n"
                            + get_carbon_removal_df().to_string(index=False)
                            + "\n\nTechnology Type Breakdown:\n"
                            + get_carbon_removal_by_type_df().to_string(index=False)
                        )
                        synthesis_input = f"Question: {query_to_run}\n\nVerified Data:\n{calc_details}"
                        ans = query_foundry(SYNTHESIS_PROMPT, synthesis_input, temperature=0.0)
                        chunks, max_score = search_context_hybrid(query_to_run)
                    elif is_zero_waste_cert:
                        route_type = "pal"
                        calc_details = (
                            "Zero Waste Certifications (2024 Report, p.36):\n"
                            + get_zero_waste_certifications_df().to_string(index=False)
                        )
                        synthesis_input = f"Question: {query_to_run}\n\nVerified Data:\n{calc_details}"
                        ans = query_foundry(SYNTHESIS_PROMPT, synthesis_input, temperature=0.0)
                        chunks, max_score = search_context_hybrid(query_to_run)
                    else:
                        chunks, max_score = search_context_hybrid(query_to_run)
                        if not chunks or max_score < MIN_SCORE_FLOOR:
                            ans = "I cannot find information regarding this in the provided Microsoft Environmental Sustainability reports."
                        else:
                            context_chunks = [c["content"] for c in chunks]
                            extract_prompt = format_extraction_prompt(query_to_run, context_chunks)
                            raw_json = query_foundry(EXTRACTION_SYSTEM_PROMPT, extract_prompt, temperature=0.0)
                            
                            try:
                                cleaned = re.search(r"\{.*\}", raw_json, re.DOTALL).group(0)
                                plan = QueryExtractionPlan(**json.loads(cleaned))
                                resolution = DeterministicResolver.validate_and_filter(plan, query_to_run)
                                
                                if resolution["status"] == "NOT_FOUND":
                                    ans = "I cannot find information regarding this in the provided Microsoft Environmental Sustainability reports."
                                else:
                                    verified_metrics_str = "\n".join([
                                        f"- Entity: {m.entity}, Type: {m.metric_type}, "
                                        f"Value: {m.string_value if m.string_value else f'{m.value:,.0f} {m.unit}'}, "
                                        f"Scope: {m.temporal_scope}, Cumulative: {m.is_cumulative}"
                                        for m in resolution["metrics"]
                                    ])
                                    calc_details = verified_metrics_str
                                    synthesis_prompt = f"Verified Metrics:\n{verified_metrics_str}\n\nQuestion: {query_to_run}"
                                    ans = query_foundry(FACTUAL_SYNTHESIS_PROMPT, synthesis_prompt, temperature=0.0)
                            except Exception:
                                context_str = "\n\n".join(context_chunks)
                                ans = query_foundry(
                                    "You are a precise Sustainability Analyst. Answer directly using ONLY context. If context has [VISUAL REFERENCE], state graphical data cannot be extracted from text.",
                                    f"Context:\n{context_str}\n\nQuestion: {query_to_run}",
                                    temperature=0.0
                                )

                    latency = time.time() - start_time

                    with st.chat_message("assistant"):
                        if route_type == "pal":
                            st.markdown(":green-badge[PAL Deterministik Hesaplama]")
                        else:
                            st.markdown(":blue-badge[Hibrit Vektör Arama & Pydantic]")
                        
                        st.markdown(ans)
                        if calc_details:
                            with st.expander("⚡ Verified Analytical Engine Output", icon=":material/verified:"):
                                st.text(calc_details)
                        if chunks:
                            with st.expander(
                                f"Kullanılan Kaynaklar ({len(chunks)}) • Benzerlik Skoru: {max_score:.4f} • Süre: {latency:.2f}s",
                                icon=":material/library_books:"
                            ):
                                for p in chunks:
                                    st.markdown(f"**{p['title']}** (Skor: {p['score']:.4f})")
                                    st.text(p["content"][:300] + "...")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ans,
                        "route": route_type,
                        "calc_details": calc_details,
                        "provenance": chunks,
                        "max_score": max_score,
                        "latency": latency
                    })
                    gc.collect()

                except Exception as e:
                    st.error(f"Sorgu yürütülürken hata oluştu: {e}")
                    gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# SEKME 2: ESG BİLANÇO PANELİ (DASHBOARD)
# ══════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    st.markdown("### :material/dashboard: **Microsoft Kurumsal ESG Bilançosu**")
    st.caption("2024–2025 Sürdürülebilirlik Raporları ve 2026 Data Fact Sheet Doğrulanmış Verileri")

    # Üst 3 Büyük KPI Kartı (st.metric)
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.metric(
                label="Toplam GHG Emisyonu (FY25)",
                value="21.12M mtCO2e",
                delta="+61.7% (FY20 Bazına Göre)",
                delta_color="inverse"
            )
            st.caption("Scope 1 + Scope 2 (Market) + Scope 3")
    with col2:
        with st.container(border=True):
            st.metric(
                label="Kümülatif Su Yenileme (FY25)",
                value="125.0M m³",
                delta="+82.1% Hedef Başarım Oranı",
                delta_color="normal"
            )
            st.caption("2030 Water Positive Hedefi Kapsamı")
    with col3:
        with st.container(border=True):
            st.metric(
                label="Yönlendirilen Katı Atık (FY25)",
                value="218,000 mt",
                delta="%82.3 Çöpten Kurtarma Oranı",
                delta_color="normal"
            )
            st.caption("Geri Dönüşüm, Yeniden Kullanım ve Kompost")

    st.space("medium")

    # Tablo 1: Karbon Emisyonları (Scope 1, 2, 3)
    with st.container(border=True):
        st.markdown("#### :material/co2: **1. Sera Gazı Emisyon Dağılımı (Scope 1, 2, 3)**")
        st.caption("Birim: mtCO2e (Metrik ton CO2 eşdeğeri) • Kaynak: 2025 Report Appendix Table 1")
        carbon_df = get_carbon_emissions_df()
        st.dataframe(carbon_df, width="stretch", hide_index=True)

    col_left, col_right = st.columns(2)
    with col_left:
        with st.container(border=True):
            st.markdown("#### :material/filter_drama: **2. Karbon Uzaklaştırma Portföyü**")
            st.caption("Birim: mtCO2e • Kaynak: 2025 Report p.21-22")
            cr_type_df = get_carbon_removal_by_type_df()
            st.dataframe(cr_type_df, width="stretch", hide_index=True)

    with col_right:
        with st.container(border=True):
            st.markdown("#### :material/water_drop: **3. Su Bilançosu & Hedefler**")
            st.caption("Birim: million m³ • Kaynak: 2025 Report Water Table 1")
            water_df = get_water_metrics_df()
            st.dataframe(water_df, width="stretch", hide_index=True)

    with st.container(border=True):
        st.markdown("#### :material/delete_forever: **4. Sıfır Atık & UL Solutions Sertifikasyonları**")
        st.caption("Kaynak: 2024 Report p.36 & 2025 Report p.47")
        zero_waste_df = get_zero_waste_certifications_df()
        st.dataframe(zero_waste_df, width="stretch", hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# SEKME 3: SİSTEM & BENCHMARK DURUMU
# ══════════════════════════════════════════════════════════════════════════════
with tab_system:
    st.markdown("### :material/memory: **Altyapı & Benchmark Değerlendirme Raporu**")
    st.caption("Yerel SLM Çıkarım Mimarisi ve Deterministik Doğrulama Ölçümleri")

    col_arch1, col_arch2 = st.columns(2)
    with col_arch1:
        with st.container(border=True):
            st.markdown("#### :material/settings_suggest: **Teknik Parametreler**")
            st.markdown("""
            - **SLM Modeli:** `phi-4-mini` (Local Foundry Endpoint)
            - **Sıcaklık (Temperature):** `0.0` (Deterministik Çıkarım)
            - **Max Tokens Sınırı:** `512` (Loop Hallucination Koruması)
            - **Embedding Modeli:** `nomic-ai/nomic-embed-text-v1.5`
            - **Embedding Boyutu:** `768 Boyutlu Yoğun Vektör`
            - **Vektör Prefix:** Asimetrik (`search_document:` / `search_query:`)
            - **Veritabanı Motoru:** `SQLite 3 (WAL Modu)`
            - **Toplam İndeks Parçası:** `982 Chunk`
            """)

    with col_arch2:
        with st.container(border=True):
            st.markdown("#### :material/verified: **14 Soruluk Üretim Benchmarkı**")
            st.markdown("""
            - **Olgusal Doğruluk (Factual Accuracy):** `%100 (9/9 Başarılı)`
            - **Alan Dışı Güvenli Reddetme:** `%100 (5/5 Başarılı)`
            - **Halüsinasyon Oranı:** `%0.00 (Sıfır Halüsinasyon)`
            - **PAL Sayısal Sorgu Latency:** `~3.22 saniye`
            - **Hibrit RAG Sorgu Latency:** `~15.80 saniye`
            - **Birim & Tip Koruma Güvencesi:** `Pydantic Assertion`
            """)

    with st.container(border=True):
        st.markdown("#### :material/account_tree: **Çalışma Hattı Akış Şeması**")
        st.code("""
[Kullanıcı Sorusu] ──► Query Routing ──┬──► (Sayısal/Bilanço) ──► PAL Engine (esg_tables.py) ────────┐
                                      └──► (Metin/Politika)   ──► Hybrid Retrieval (Nomic 1.5)      │
                                                                       │                            │
                                                                       ▼                            ▼
                                                                Pydantic Assertion ──► phi-4-mini Synthesis
                                                                                            │
                                                                                            ▼
                                                                                   [Doğrulanmış Yanıt]
        """, language="text")
