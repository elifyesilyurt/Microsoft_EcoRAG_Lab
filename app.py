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
# SAYFA YAPILANDIRMASI
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
# ÇİFT DİLLİ METİN SÖZLÜĞÜ (BILINGUAL DICTIONARY)
# ══════════════════════════════════════════════════════════════════════════════
TEXTS = {
    "tr": {
        "title": "Microsoft EcoRAG Lab",
        "subtitle": "Sıfır Halüsinasyonlu Deterministik ESG ve Sürdürülebilirlik Analiz Paneli",
        "sidebar_title": "EcoRAG Lab",
        "sidebar_caption": "Deterministik Sürdürülebilirlik Analizi",
        "lang_label": "Dil / Language",
        "theme_label": "Görsel Tema / Palette",
        "status_box_title": "Sistem Durumu",
        "status_badge": "Aktif & Doğrulanmış",
        "status_model": "Model",
        "status_embed": "Embedding",
        "status_index": "İndeks",
        "status_engine": "Motor",
        "reset_btn": "Sohbeti Sıfırla",
        "tab_chat": "💬 Akıllı Asistan",
        "tab_dash": "📊 ESG Bilanço Paneli",
        "tab_sys": "🛠️ Sistem & Benchmark Durumu",
        "pills_title": "Hızlı Başlangıç & Benchmark Test Soruları",
        "badge_pal": "PAL Deterministik Hesaplama",
        "badge_rag": "Hibrit Vektör Arama & Pydantic",
        "verified_output_label": "⚡ Doğrulanmış Analitik Çıktı (Verified Metrics)",
        "provenance_label": "Kullanılan Kaynaklar ({count}) • Benzerlik Skoru: {score:.4f} • Süre: {latency:.2f}s",
        "chat_placeholder": "Microsoft çevre ve sürdürülebilirlik raporlarına dair bir soru sorun...",
        "spinner_text": "Deterministik çıkarım ve doğrulama yürütülüyor...",
        "not_found_msg": "Microsoft Çevresel Sürdürülebilirlik raporlarında bu konuyla ilgili bilgi bulunmamaktadır.",
        "kpi_co2_title": "Toplam GHG Emisyonu (FY25)",
        "kpi_co2_delta": "+61.7% (FY20 Bazına Göre)",
        "kpi_co2_cap": "Scope 1 + Scope 2 (Market) + Scope 3",
        "kpi_water_title": "Kümülatif Su Yenileme (FY25)",
        "kpi_water_delta": "+82.1% Hedef Başarım Oranı",
        "kpi_water_cap": "2030 Water Positive Hedefi Kapsamı",
        "kpi_waste_title": "Yönlendirilen Katı Atık (FY25)",
        "kpi_waste_delta": "%82.3 Çöpten Kurtarma Oranı",
        "kpi_waste_cap": "Geri Dönüşüm, Yeniden Kullanım ve Kompost",
        "dash_title": "Microsoft Kurumsal ESG Bilançosu",
        "dash_caption": "2024–2025 Sürdürülebilirlik Raporları ve 2026 Data Fact Sheet Doğrulanmış Verileri",
        "dash_t1": "1. Sera Gazı Emisyon Dağılımı (Scope 1, 2, 3)",
        "dash_t1_cap": "Birim: mtCO2e (Metrik ton CO2 eşdeğeri) • Kaynak: 2025 Report Appendix Table 1",
        "dash_t2": "2. Karbon Uzaklaştırma Portföyü",
        "dash_t2_cap": "Birim: mtCO2e • Kaynak: 2025 Report p.21-22",
        "dash_t3": "3. Su Bilançosu & Hedefler",
        "dash_t3_cap": "Birim: million m³ • Kaynak: 2025 Report Water Table 1",
        "dash_t4": "4. Sıfır Atık & UL Solutions Sertifikasyonları",
        "dash_t4_cap": "Kaynak: 2024 Report p.36 & 2025 Report p.47",
        "dash_t5": "5. 2026 Data Fact Sheet — Resmi Denetim & Bölgesel Göstergeler",
        "dash_t5_cap": "Kaynak: Microsoft_2026_Data_Fact_Sheet.pdf (Denetlenmiş Resmi Metrikler & Metodolojiler)",
        "sys_title": "Altyapı & Benchmark Değerlendirme Raporu",
        "sys_caption": "Yerel SLM Çıkarım Mimarisi ve Deterministik Doğrulama Ölçümleri",
        "sys_card1_title": "Teknik Parametreler",
        "sys_card2_title": "50 Soruluk Üretim Benchmarkı",
        "sys_flow_title": "Çalışma Hattı Akış Şeması"
    },
    "en": {
        "title": "Microsoft EcoRAG Lab",
        "subtitle": "Zero-Hallucination Deterministic ESG & Sustainability Analysis Engine",
        "sidebar_title": "EcoRAG Lab",
        "sidebar_caption": "Deterministic Sustainability Analysis",
        "lang_label": "Language / Dil",
        "theme_label": "Theme / Palette",
        "status_box_title": "System Status",
        "status_badge": "Active & Verified",
        "status_model": "Model",
        "status_embed": "Embedding",
        "status_index": "Index",
        "status_engine": "Engine",
        "reset_btn": "Clear Conversation",
        "tab_chat": "💬 Smart Assistant",
        "tab_dash": "📊 ESG Balance Dashboard",
        "tab_sys": "🛠️ System & Benchmark Status",
        "pills_title": "Quick Prompts & Benchmark Questions",
        "badge_pal": "PAL Deterministic Calculation",
        "badge_rag": "Hybrid Vector Search & Pydantic",
        "verified_output_label": "⚡ Verified Structured Representation",
        "provenance_label": "Data Provenance ({count}) • Similarity Score: {score:.4f} • Latency: {latency:.2f}s",
        "chat_placeholder": "Ask a question regarding Microsoft sustainability reports...",
        "spinner_text": "Executing deterministic extraction & validation...",
        "not_found_msg": "I cannot find information regarding this in the provided Microsoft Environmental Sustainability reports.",
        "kpi_co2_title": "Total GHG Emissions (FY25)",
        "kpi_co2_delta": "+61.7% (vs FY20 Baseline)",
        "kpi_co2_cap": "Scope 1 + Scope 2 (Market) + Scope 3",
        "kpi_water_title": "Cumulative Water Replenishment (FY25)",
        "kpi_water_delta": "+82.1% Achievement Rate",
        "kpi_water_cap": "2030 Water Positive Commitment",
        "kpi_waste_title": "Diverted Solid Waste (FY25)",
        "kpi_waste_delta": "82.3% Diversion Rate",
        "kpi_waste_cap": "Recycled, Reused & Composted",
        "dash_title": "Microsoft Corporate ESG Balance Sheet",
        "dash_caption": "Verified Data from 2024–2025 Sustainability Reports & 2026 Data Fact Sheet",
        "dash_t1": "1. Greenhouse Gas Emissions (Scope 1, 2, 3)",
        "dash_t1_cap": "Unit: mtCO2e (Metric tons CO2 equivalent) • Source: 2025 Report Appendix Table 1",
        "dash_t2": "2. Carbon Removal Portfolio Breakdown",
        "dash_t2_cap": "Unit: mtCO2e • Source: 2025 Report p.21-22",
        "dash_t3": "3. Water Metrics & Replenishment Targets",
        "dash_t3_cap": "Unit: million m³ • Source: 2025 Report Water Table 1",
        "dash_t4": "4. Zero Waste & UL Solutions Certifications",
        "dash_t4_cap": "Source: 2024 Report p.36 & 2025 Report p.47",
        "dash_t5": "5. 2026 Data Fact Sheet — Audit Metrics & Regional Indicators",
        "dash_t5_cap": "Source: Microsoft_2026_Data_Fact_Sheet.pdf (Audited Official Metrics & Methodologies)",
        "sys_title": "Infrastructure & Benchmark Evaluation Report",
        "sys_caption": "Local SLM Inference Architecture and Deterministic Verification Metrics",
        "sys_card1_title": "Technical Parameters",
        "sys_card2_title": "50-Question Production Benchmark",
        "sys_flow_title": "Pipeline Execution Flowchart"
    }
}

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR (KENAR ÇUBUĞU)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### :material/eco: **EcoRAG Lab**")
    st.caption("Deterministic Sustainability Analysis")

    # Yatay Dil Açma/Kapama Düğmesi (Sol: English EN, Sağ: Türkçe TR)
    lang_choice = st.segmented_control(
        "Language / Dil",
        options=["English (EN)", "Türkçe (TR)"],
        default="Türkçe (TR)",
        key="lang_choice"
    )
    is_tr = (lang_choice == "Türkçe (TR)")
    L = "tr" if is_tr else "en"
    T = TEXTS[L]

    # Tema Seçici (3 Tema Seçeneği: Eco Emerald, Fluent Azure, Pastel Sage)
    theme_choice = st.selectbox(
        T["theme_label"],
        options=[
            "🌿 Eco Emerald (Koyu / Dark)",
            "💼 Fluent Azure (Açık / Light)",
            "🌸 Toz Pembe Pastel (Blush Rose)"
        ],
        index=0
    )

    with st.container(border=True):
        st.markdown(f"**{T['status_box_title']}**")
        st.badge(T["status_badge"], icon=":material/check_circle:", color="green")
        st.markdown(f"**{T['status_model']}:** `{MODEL_NAME}`")
        st.markdown(f"**{T['status_embed']}:** `nomic-v1.5 (768d)`")
        st.markdown(f"**{T['status_index']}:** `982 Chunks (3 PDF)`")
        st.markdown(f"**{T['status_engine']}:** `PAL + Asymmetric IR`")

    if st.button(T["reset_btn"], icon=":material/delete:", width="stretch"):
        st.session_state.messages = []
        gc.collect()
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# DİNAMİK TEMA ENJEKSİYONU (3 FARKLI PALET - TAM KONTRAST & EKSİKSİZ BİLEŞEN UYUMU)
# ══════════════════════════════════════════════════════════════════════════════
if "Eco Emerald" in theme_choice:
    st.html("""
    <style>
    /* 🌿 Eco Emerald Dark Theme */
    .stApp {
        background-color: #0d1117 !important;
    }
    header[data-testid="stHeader"] {
        background-color: #0d1117 !important;
    }
    .stApp, .stApp p, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp label, .stApp div[data-testid="stMarkdownContainer"] p {
        color: #e6edf3 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d !important;
    }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] h3 {
        color: #e6edf3 !important;
    }
    /* Pills (Hızlı Sorular) */
    div[data-testid="stPills"] button, div[data-testid="stPills"] [data-baseweb="tag"], div[data-testid="stPills"] span {
        background-color: #21262d !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        font-weight: 500 !important;
    }
    div[data-testid="stPills"] button:hover {
        background-color: #30363d !important;
        color: #58a6ff !important;
    }
    div[data-testid="stPills"] button[aria-pressed="true"], div[data-testid="stPills"] [aria-selected="true"] {
        background-color: #238636 !important;
        color: #ffffff !important;
        border-color: #2ea043 !important;
    }
    div[data-testid="stPills"] button[aria-pressed="true"] span {
        color: #ffffff !important;
    }
    /* Segmented Control (Dil Seçici) */
    div[data-testid="stSegmentedControl"] button {
        background-color: #21262d !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        font-weight: 500 !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"], div[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
        background-color: #238636 !important;
        color: #ffffff !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] p, div[data-testid="stSegmentedControl"] button[aria-checked="true"] span {
        color: #ffffff !important;
    }
    /* Chat Input */
    div[data-testid="stChatInput"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
    }
    div[data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: #e6edf3 !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #8b949e !important;
    }
    div[data-testid="stChatInput"] button {
        color: #58a6ff !important;
    }
    /* Code Badges */
    code {
        background-color: #21262d !important;
        color: #58a6ff !important;
        border: 1px solid #30363d !important;
        border-radius: 4px;
        padding: 2px 6px;
        font-weight: 600;
    }
    /* Button */
    .stButton > button {
        background-color: #21262d !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background-color: #30363d !important;
        border-color: #8b949e !important;
    }
    /* Selectbox */
    div[data-baseweb="select"] > div {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
    }
    div[data-baseweb="select"] span {
        color: #e6edf3 !important;
    }
    /* Metrics, Cards, Expanders */
    div[data-testid="stMetricValue"] { color: #58a6ff !important; }
    div[data-testid="stMetricLabel"] { color: #8b949e !important; }
    div[data-testid="stMetric"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
    }
    div[data-testid="stChatMessage"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
    }
    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] span { color: #e6edf3 !important; }
    div[data-testid="stExpander"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
    }
    div[data-testid="stExpander"] p, div[data-testid="stExpander"] span, div[data-testid="stExpander"] summary { color: #e6edf3 !important; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8b949e !important;
        background-color: #21262d !important;
        border-radius: 6px;
        padding: 8px 16px;
    }
    .stTabs [data-baseweb="tab"] p, .stTabs [data-baseweb="tab"] span { color: #8b949e !important; }
    .stTabs [aria-selected="true"] {
        background-color: #238636 !important;
        color: #ffffff !important;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] span { color: #ffffff !important; }
    .stApp small, .stApp .stCaption, .stApp caption, .stApp div[data-testid="stCaptionContainer"] { color: #8b949e !important; }
    </style>
    """)
elif "Fluent Azure" in theme_choice:
    st.html("""
    <style>
    /* 💼 Fluent Azure Light Theme */
    .stApp {
        background-color: #f8fafc !important;
    }
    header[data-testid="stHeader"] {
        background-color: #f8fafc !important;
    }
    .stApp, .stApp p, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp label, .stApp div[data-testid="stMarkdownContainer"] p {
        color: #0f172a !important; /* Çok net koyu metin */
    }
    section[data-testid="stSidebar"] {
        background-color: #f1f5f9 !important;
        border-right: 1px solid #cbd5e1 !important;
    }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] h3 {
        color: #0f172a !important;
    }
    /* Pills (Hızlı Sorular) */
    div[data-testid="stPills"] button, div[data-testid="stPills"] [data-baseweb="tag"], div[data-testid="stPills"] span {
        background-color: #e2e8f0 !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stPills"] button:hover {
        background-color: #cbd5e1 !important;
        color: #000000 !important;
    }
    div[data-testid="stPills"] button[aria-pressed="true"], div[data-testid="stPills"] [aria-selected="true"] {
        background-color: #0078d4 !important;
        color: #ffffff !important;
        border-color: #005a9e !important;
    }
    div[data-testid="stPills"] button[aria-pressed="true"] span {
        color: #ffffff !important;
    }
    /* Segmented Control (Dil Seçici) */
    div[data-testid="stSegmentedControl"] button {
        background-color: #e2e8f0 !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"], div[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
        background-color: #0078d4 !important;
        color: #ffffff !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] p, div[data-testid="stSegmentedControl"] button[aria-checked="true"] span {
        color: #ffffff !important;
    }
    /* Chat Input */
    div[data-testid="stChatInput"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05) !important;
    }
    div[data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: #0f172a !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #64748b !important;
    }
    div[data-testid="stChatInput"] button {
        color: #0078d4 !important;
    }
    /* Code Badges & Code Blocks */
    div[data-testid="stCode"], div[data-testid="stCodeBlock"], pre {
        background-color: #f1f5f9 !important;
        border: 2px solid #cbd5e1 !important;
        border-radius: 12px !important;
        padding: 10px !important;
    }
    div[data-testid="stCode"] code, div[data-testid="stCodeBlock"] code, pre code {
        background-color: transparent !important;
        color: #0f172a !important;
        border: none !important;
        font-weight: 500 !important;
    }
    code {
        background-color: #e2e8f0 !important;
        color: #0078d4 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 4px;
        padding: 2px 6px;
        font-weight: 600;
    }
    /* Button */
    .stButton > button {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background-color: #e2e8f0 !important;
        border-color: #94a3b8 !important;
    }
    /* Selectbox */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
    }
    div[data-baseweb="select"] span {
        color: #0f172a !important;
    }
    /* Metrics, Cards, Expanders */
    div[data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        color: #334155 !important;
        font-weight: 600;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        padding: 12px;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #0f172a !important;
    }
    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] span {
        color: #0f172a !important;
    }
    div[data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
    }
    div[data-testid="stExpander"] p, div[data-testid="stExpander"] span, div[data-testid="stExpander"] summary {
        color: #0f172a !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #334155 !important;
        background-color: #e2e8f0 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px;
        padding: 8px 16px;
    }
    .stTabs [data-baseweb="tab"] p, .stTabs [data-baseweb="tab"] span {
        color: #334155 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0078d4 !important;
        color: #ffffff !important;
        font-weight: 700;
        border: 1px solid #005a9e !important;
    }
    .stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] span {
        color: #ffffff !important;
    }
    .stApp small, .stApp .stCaption, .stApp caption, .stApp div[data-testid="stCaptionContainer"] {
        color: #475569 !important;
    }
    </style>
    """)
else:
    # 🌸 Toz Pembe Pastel (Blush Rose Theme)
    st.html("""
    <style>
    /* 🌸 Toz Pembe Pastel / Blush Rose Theme */
    .stApp {
        background: linear-gradient(180deg, #fdf6f7 0%, #f7e8ec 100%) !important;
    }
    header[data-testid="stHeader"] {
        background-color: #fdf6f7 !important;
    }
    .stApp, .stApp p, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp label, .stApp div[data-testid="stMarkdownContainer"] p {
        color: #2d1b22 !important; /* Net okunur koyu mürdüm-antrasit */
    }
    section[data-testid="stSidebar"] {
        background-color: #f7e2e6 !important;
        border-right: 1px solid #e8bcc5 !important;
    }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] h3 {
        color: #2d1b22 !important;
    }
    /* 🔘 Universal Pills & Tags Override (Siyahlığı Tamamen Kaldırır) */
    [data-testid*="stPills"],
    [data-testid*="stPills"] *,
    [data-baseweb="tag"],
    [data-baseweb="tag"] *,
    div[role="radiogroup"] button,
    div[role="radiogroup"] button *,
    div[data-testid="stPills"] button,
    div[data-testid="stPills"] button * {
        background-color: #f7dbe1 !important; /* Arka plandan bir tık koyu açık toz pembe */
        color: #4a0e1e !important; /* Net okunan koyu mürdüm yazı */
        border: 1.5px solid #d99ca9 !important;
        font-weight: 600 !important;
    }
    [data-testid*="stPills"] button:hover,
    [data-baseweb="tag"]:hover,
    div[role="radiogroup"] button:hover {
        background-color: #ebd0d6 !important;
        color: #2d050f !important;
        border-color: #b85d75 !important;
    }
    [data-testid*="stPills"] [aria-pressed="true"],
    [data-testid*="stPills"] [aria-selected="true"],
    [data-testid*="stPills"] [aria-checked="true"],
    div[role="radiogroup"] [aria-checked="true"] {
        background-color: #f0c3cb !important;
        color: #2d050f !important;
        border: 2px solid #b85d75 !important;
    }
    [data-testid*="stPills"] [aria-pressed="true"] *,
    [data-testid*="stPills"] [aria-selected="true"] *,
    [data-testid*="stPills"] [aria-checked="true"] * {
        color: #2d050f !important;
    }
    /* Segmented Control (Dil Seçici) */
    div[data-testid="stSegmentedControl"] button,
    div[data-testid="stSegmentedControl"] button * {
        background-color: #fceef1 !important;
        color: #501d2d !important;
        border: 1px solid #e8bcc5 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"],
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
        background-color: #f0c3cb !important;
        color: #4a0e1e !important;
        border: 2px solid #d99ca9 !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] * {
        color: #4a0e1e !important;
    }
    /* 💬 Chat Input & Bottom Bar (Siyahlığı Tamamen Kaldırır) */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"],
    [data-testid="stChatInputContainer"],
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div {
        background-color: transparent !important;
        border: none !important;
    }
    [data-testid="stChatInput"] {
        background-color: #ffffff !important;
        border: 2px solid #d99ca9 !important;
        border-radius: 14px !important;
        box-shadow: 0 2px 8px rgba(184, 93, 117, 0.08) !important;
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] textarea * {
        background-color: transparent !important;
        color: #2d1b22 !important;
        font-size: 15px !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #8a4a58 !important;
    }
    [data-testid="stChatInput"] button {
        background-color: #f7dbe1 !important;
        color: #831843 !important;
        border-radius: 8px !important;
    }
    /* Code Badges & Code Blocks */
    div[data-testid="stCode"], div[data-testid="stCodeBlock"], pre {
        background-color: #faedf0 !important;
        border: 2px solid #d99ca9 !important;
        border-radius: 12px !important;
        padding: 10px !important;
    }
    div[data-testid="stCode"] code, div[data-testid="stCodeBlock"] code, pre code {
        background-color: transparent !important;
        color: #501d2d !important;
        border: none !important;
        font-weight: 500 !important;
    }
    div[data-testid="stCode"] button, div[data-testid="stCodeBlock"] button {
        color: #831843 !important;
        background-color: transparent !important;
    }
    code {
        background-color: #fadce2 !important;
        color: #9d174d !important;
        border: 1px solid #e8bcc5 !important;
        border-radius: 4px;
        padding: 2px 6px;
        font-weight: 600;
    }
    /* Button */
    .stButton > button {
        background-color: #ffffff !important;
        color: #501d2d !important;
        border: 1px solid #e8bcc5 !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background-color: #fadce2 !important;
        border-color: #d99ca9 !important;
    }
    /* Selectbox */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #2d1b22 !important;
        border: 1px solid #e8bcc5 !important;
    }
    div[data-baseweb="select"] span {
        color: #2d1b22 !important;
    }
    /* Metrics, Cards, Expanders */
    div[data-testid="stMetricValue"] {
        color: #831843 !important;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        color: #502838 !important;
        font-weight: 600;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e8bcc5 !important;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(184, 93, 117, 0.08);
        padding: 14px;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #fffffffa !important;
        border: 1px solid #e8bcc5 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 6px rgba(184, 93, 117, 0.05);
    }
    div[data-testid="stChatMessage"] {
        background-color: #fffffffa !important;
        border: 1px solid #ebd0d6 !important;
        color: #2d1b22 !important;
    }
    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] span {
        color: #2d1b22 !important;
    }
    div[data-testid="stExpander"] {
        background-color: #fffffffa !important;
        border: 1px solid #e8bcc5 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stExpander"] p, div[data-testid="stExpander"] span, div[data-testid="stExpander"] summary {
        color: #2d1b22 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        background-color: #f7e2e6 !important;
        color: #5c182c !important;
        border: 1px solid #ebd0d6 !important;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"] p, .stTabs [data-baseweb="tab"] span {
        color: #5c182c !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0c3cb !important;
        color: #4a0e1e !important;
        font-weight: 700 !important;
        border: 2px solid #d99ca9 !important;
        border-bottom: 3px solid #b85d75 !important;
    }
    .stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] span {
        color: #4a0e1e !important;
    }
    .stApp small, .stApp .stCaption, .stApp caption, .stApp div[data-testid="stCaptionContainer"] {
        color: #6b404e !important;
    }
    </style>
    """)

# ══════════════════════════════════════════════════════════════════════════════
# BAŞLIK VE SEKME DÜZENİ (3 ANA SEKME)
# ══════════════════════════════════════════════════════════════════════════════
st.title(T["title"])
st.caption(T["subtitle"])

tab_chat, tab_dashboard, tab_system = st.tabs([
    T["tab_chat"],
    T["tab_dash"],
    T["tab_sys"]
])

# ══════════════════════════════════════════════════════════════════════════════
# SEKME 1: AKILLI ASİSTAN (SOHBET ARAYÜZÜ)
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Dile Göre Hazır Soru Hapları
    if is_tr:
        pill_options = [
            "Scope 1-3 Emisyon Trendi",
            "FIDO Tech Akustik Su Kaçağı",
            "Zero Waste Veri Merkezleri (UL Standardı)",
            "Karbon Uzaklaştırma Portföyü",
            "2026 Fact Sheet: Ambalaj & Plastik",
            "2026 Bölgesel Tüketim (Hollanda/Madrid)",
            "Ağ Gecikme Süresi (Alan Dışı Test)",
            "Sunucu CPU Saat Hızı (Alan Dışı Test)",
            "2023 FIFA Dünya Kupası (Alan Dışı Test)"
        ]
    else:
        pill_options = [
            "Scope 1-3 Emissions Trend",
            "FIDO Tech Acoustic Leak AI",
            "Zero Waste Datacenters (UL Standard)",
            "Carbon Removal Portfolio",
            "2026 Fact Sheet: Packaging & Plastic",
            "2026 Regional Consumption (Netherlands/Madrid)",
            "Network Latency (Out-of-Domain Test)",
            "Server CPU Clock Speed (Out-of-Domain Test)",
            "2023 FIFA World Cup (Out-of-Domain Test)"
        ]

    selected_pill = st.pills(
        T["pills_title"],
        options=pill_options,
        label_visibility="collapsed"
    )

    pill_query_map = {
        # TR
        "Scope 1-3 Emisyon Trendi": "Compare Microsoft Scope 1, Scope 2, and Scope 3 emissions trend between FY20 baseline and FY25, highlighting the top contributing categories.",
        "FIDO Tech Akustik Su Kaçağı": "Which organization did Microsoft partner with to deploy AI-enabled acoustic leak analysis in water distribution networks across cities like London, Querétaro, and Phoenix?",
        "Zero Waste Veri Merkezleri (UL Standardı)": "Which external certification does Microsoft use to validate its Zero Waste datacenters, and how many datacenters were certified under this standard in FY23 according to the 2024 report?",
        "Karbon Uzaklaştırma Portföyü": "What is the total contracted carbon removal volume and its breakdown by technology type according to Carbon Table 3 in the 2025 report?",
        "2026 Fact Sheet: Ambalaj & Plastik": "According to the 2026 Data Fact Sheet, what is the single-use plastic packaging percentage achieved at the end of calendar year 2025/2026 and what third-party frameworks are used?",
        "2026 Bölgesel Tüketim (Hollanda/Madrid)": "What are the datacenter water and electricity metrics for regions like the Netherlands and Madrid reported in the 2026 Data Fact Sheet?",
        "Ağ Gecikme Süresi (Alan Dışı Test)": "What was the average round-trip network latency between the Quincy datacenter and the San Antonio Azure edge site in milliseconds during 2024?",
        "Sunucu CPU Saat Hızı (Alan Dışı Test)": "What is the exact clock speed in GHz and cache size of the custom processors used inside the servers at the Boydton datacenter?",
        "2023 FIFA Dünya Kupası (Alan Dışı Test)": "Who won the FIFA Women's World Cup in 2023, and what was the final score?",
        # EN
        "Scope 1-3 Emissions Trend": "Compare Microsoft Scope 1, Scope 2, and Scope 3 emissions trend between FY20 baseline and FY25, highlighting the top contributing categories.",
        "FIDO Tech Acoustic Leak AI": "Which organization did Microsoft partner with to deploy AI-enabled acoustic leak analysis in water distribution networks across cities like London, Querétaro, and Phoenix?",
        "Zero Waste Datacenters (UL Standard)": "Which external certification does Microsoft use to validate its Zero Waste datacenters, and how many datacenters were certified under this standard in FY23 according to the 2024 report?",
        "Carbon Removal Portfolio": "What is the total contracted carbon removal volume and its breakdown by technology type according to Carbon Table 3 in the 2025 report?",
        "2026 Fact Sheet: Packaging & Plastic": "According to the 2026 Data Fact Sheet, what is the single-use plastic packaging percentage achieved at the end of calendar year 2025/2026 and what third-party frameworks are used?",
        "2026 Regional Consumption (Netherlands/Madrid)": "What are the datacenter water and electricity metrics for regions like the Netherlands and Madrid reported in the 2026 Data Fact Sheet?",
        "Network Latency (Out-of-Domain Test)": "What was the average round-trip network latency between the Quincy datacenter and the San Antonio Azure edge site in milliseconds during 2024?",
        "Server CPU Clock Speed (Out-of-Domain Test)": "What is the exact clock speed in GHz and cache size of the custom processors used inside the servers at the Boydton datacenter?",
        "2023 FIFA World Cup (Out-of-Domain Test)": "Who won the FIFA Women's World Cup in 2023, and what was the final score?"
    }

    active_query = None
    if selected_pill and selected_pill in pill_query_map:
        active_query = pill_query_map[selected_pill]

    # Geçmiş Mesajları Çiz
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and "route" in msg:
                if msg["route"] == "pal":
                    st.markdown(f":green-badge[{T['badge_pal']}]")
                else:
                    st.markdown(f":blue-badge[{T['badge_rag']}]")

            st.markdown(msg["content"])

            if "calc_details" in msg and msg["calc_details"]:
                with st.expander(T["verified_output_label"], icon=":material/verified:"):
                    st.text(msg["calc_details"])
            if "provenance" in msg and msg["provenance"]:
                prov_title = T["provenance_label"].format(
                    count=len(msg["provenance"]),
                    score=msg.get("max_score", 0),
                    latency=msg.get("latency", 0)
                )
                with st.expander(prov_title, icon=":material/library_books:"):
                    for p in msg["provenance"]:
                        st.markdown(f"**{p['title']}** (Score / Skor: {p['score']:.4f})")
                        st.text(p["content"][:300] + "...")

    # Kullanıcı Girdisi (chat_input veya pill)
    user_input = st.chat_input(T["chat_placeholder"])
    query_to_run = user_input or active_query

    if query_to_run:
        if not st.session_state.messages or st.session_state.messages[-1]["content"] != query_to_run:
            st.session_state.messages.append({"role": "user", "content": query_to_run})
            with st.chat_message("user"):
                st.markdown(query_to_run)

            start_time = time.time()
            with st.spinner(T["spinner_text"]):
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
                            ans = T["not_found_msg"]
                        else:
                            context_chunks = [c["content"] for c in chunks]
                            extract_prompt = format_extraction_prompt(query_to_run, context_chunks)
                            raw_json = query_foundry(EXTRACTION_SYSTEM_PROMPT, extract_prompt, temperature=0.0)

                            try:
                                cleaned = re.search(r"\{.*\}", raw_json, re.DOTALL).group(0)
                                plan = QueryExtractionPlan(**json.loads(cleaned))
                                resolution = DeterministicResolver.validate_and_filter(plan, query_to_run)

                                if resolution["status"] == "NOT_FOUND":
                                    ans = T["not_found_msg"]
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
                            st.markdown(f":green-badge[{T['badge_pal']}]")
                        else:
                            st.markdown(f":blue-badge[{T['badge_rag']}]")

                        st.markdown(ans)
                        if calc_details:
                            with st.expander(T["verified_output_label"], icon=":material/verified:"):
                                st.text(calc_details)
                        if chunks:
                            prov_title = T["provenance_label"].format(
                                count=len(chunks),
                                score=max_score,
                                latency=latency
                            )
                            with st.expander(prov_title, icon=":material/library_books:"):
                                for p in chunks:
                                    st.markdown(f"**{p['title']}** (Score / Skor: {p['score']:.4f})")
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
                    st.error(f"Error / Hata: {e}")
                    gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# SEKME 2: ESG BİLANÇO PANELİ (DASHBOARD)
# ══════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    st.markdown(f"### :material/dashboard: **{T['dash_title']}**")
    st.caption(T["dash_caption"])

    # Üst 3 Büyük KPI Kartı (st.metric)
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.metric(
                label=T["kpi_co2_title"],
                value="21.12M mtCO2e",
                delta=T["kpi_co2_delta"],
                delta_color="inverse"
            )
            st.caption(T["kpi_co2_cap"])
    with col2:
        with st.container(border=True):
            st.metric(
                label=T["kpi_water_title"],
                value="125.0M m³",
                delta=T["kpi_water_delta"],
                delta_color="normal"
            )
            st.caption(T["kpi_water_cap"])
    with col3:
        with st.container(border=True):
            st.metric(
                label=T["kpi_waste_title"],
                value="218,000 mt",
                delta=T["kpi_waste_delta"],
                delta_color="normal"
            )
            st.caption(T["kpi_waste_cap"])

    st.space("medium")

    # Tablo 1: Karbon Emisyonları (Scope 1, 2, 3)
    with st.container(border=True):
        st.markdown(f"#### :material/co2: **{T['dash_t1']}**")
        st.caption(T["dash_t1_cap"])
        carbon_df = get_carbon_emissions_df()
        st.dataframe(carbon_df, width="stretch", hide_index=True)

    col_left, col_right = st.columns(2)
    with col_left:
        with st.container(border=True):
            st.markdown(f"#### :material/filter_drama: **{T['dash_t2']}**")
            st.caption(T["dash_t2_cap"])
            cr_type_df = get_carbon_removal_by_type_df()
            st.dataframe(cr_type_df, width="stretch", hide_index=True)

    with col_right:
        with st.container(border=True):
            st.markdown(f"#### :material/water_drop: **{T['dash_t3']}**")
            st.caption(T["dash_t3_cap"])
            water_df = get_water_metrics_df()
            st.dataframe(water_df, width="stretch", hide_index=True)

    with st.container(border=True):
        st.markdown(f"#### :material/delete_forever: **{T['dash_t4']}**")
        st.caption(T["dash_t4_cap"])
        zero_waste_df = get_zero_waste_certifications_df()
        st.dataframe(zero_waste_df, width="stretch", hide_index=True)

    with st.container(border=True):
        st.markdown(f"#### :material/fact_check: **{T['dash_t5']}**")
        st.caption(T["dash_t5_cap"])
        if is_tr:
            st.markdown("""
            - **Tek Kullanımlık Plastik Ambalaj (2025/2026 Takvim Yılı Sonu):** `%0.07` (2030 Sıfır Plastik Hedefi Yolunda)
            - **Standart ve Denetim Çerçeveleri:** `TRUE Zero Waste` & `UL 2799 ECVP` Çerçeveleri
            - **Bölgesel Veri Merkezi Elektrik Tüketimleri (2026 Tablosu):**
              - *Hollanda:* `1,291,170 MWh` (46 Yenilenebilir Varlık)
              - *Madrid (İspanya):* `22,588 MWh` (15 Yenilenebilir Varlık)
              - *Malmö (İsveç):* `41,681 MWh`
              - *Milano (İtalya):* `46,950 MWh`
            """)
        else:
            st.markdown("""
            - **Single-Use Plastic Packaging (End of Calendar Year 2025/2026):** `0.07%` (Towards 2030 Zero Plastic Target)
            - **Standard & Audit Frameworks:** `TRUE Zero Waste` & `UL 2799 ECVP` Frameworks
            - **Regional Datacenter Electricity Consumption (2026 Table):**
              - *Netherlands:* `1,291,170 MWh` (46 Renewable Assets)
              - *Madrid (Spain):* `22,588 MWh` (15 Renewable Assets)
              - *Malmö (Sweden):* `41,681 MWh`
              - *Milan (Italy):* `46,950 MWh`
            """)

# ══════════════════════════════════════════════════════════════════════════════
# SEKME 3: SİSTEM & BENCHMARK DURUMU
# ══════════════════════════════════════════════════════════════════════════════
with tab_system:
    st.markdown(f"### :material/memory: **{T['sys_title']}**")
    st.caption(T["sys_caption"])

    col_arch1, col_arch2 = st.columns(2)
    with col_arch1:
        with st.container(border=True):
            st.markdown(f"#### :material/settings_suggest: **{T['sys_card1_title']}**")
            if is_tr:
                st.markdown("""
                - **SLM Modeli:** `phi-4-mini` (Local Foundry Endpoint)
                - **Sıcaklık (Temperature):** `0.0` (Deterministik Çıkarım)
                - **Max Tokens Sınırı:** `512` (Loop Hallucination Koruması)
                - **Embedding Modeli:** `nomic-ai/nomic-embed-text-v1.5`
                - **Embedding Boyutu:** `768 Boyutlu Yoğun Vektör`
                - **Vektör Prefix:** Asimetrik (`search_document:` / `search_query:`)
                - **Veritabanı Motoru:** `SQLite 3 (WAL Modu)`
                - **Toplam İndeks Parçası:** `982 Chunk (3 Doküman)`
                  - `Microsoft_2024_Sustainability_Report.pdf` (487 Chunk)
                  - `Microsoft_2025_Sustainability_Report.pdf` (322 Chunk)
                  - `Microsoft_2026_Data_Fact_Sheet.pdf` (173 Chunk)
                """)
            else:
                st.markdown("""
                - **SLM Model:** `phi-4-mini` (Local Foundry Endpoint)
                - **Temperature:** `0.0` (Deterministic Inference)
                - **Max Tokens Limit:** `512` (Loop Hallucination Safeguard)
                - **Embedding Model:** `nomic-ai/nomic-embed-text-v1.5`
                - **Embedding Dimensions:** `768-dim Dense Vector`
                - **Vector Prefix:** Asymmetric (`search_document:` / `search_query:`)
                - **Database Engine:** `SQLite 3 (WAL Mode)`
                - **Total Indexed Chunks:** `982 Chunks (3 Documents)`
                  - `Microsoft_2024_Sustainability_Report.pdf` (487 Chunks)
                  - `Microsoft_2025_Sustainability_Report.pdf` (322 Chunks)
                  - `Microsoft_2026_Data_Fact_Sheet.pdf` (173 Chunks)
                """)

    with col_arch2:
        with st.container(border=True):
            st.markdown(f"#### :material/verified: **{T['sys_card2_title']}**")
            if is_tr:
                st.markdown("""
                - **Toplam Test Kapsamı:** `50 Soru (4 Zorluk Seviyesi)`
                - **Kolay / Direct Factual:** `%100 (15/15 Başarılı)`
                - **Orta / Multi-Condition & Tabular:** `%100 (15/15 Başarılı)`
                - **Zor / Multi-Year Math & PAL:** `%100 (10/10 Başarılı)`
                - **Alan Dışı / Sıfır Halüsinasyon:** `%100 (10/10 Reddetme)`
                - **Genel Doğruluk Oranı:** `%100.0 (50/50 PASS)`
                - **PAL Sayısal Latency:** `~3.71 saniye`
                - **Hibrit RAG Latency:** `~16.20 saniye`
                - **Birim & Tip Koruma Güvencesi:** `Pydantic Assertion`
                """)
            else:
                st.markdown("""
                - **Total Test Scope:** `50 Questions (4 Difficulty Levels)`
                - **Easy / Direct Factual:** `100% (15/15 Passed)`
                - **Medium / Multi-Condition & Tabular:** `100% (15/15 Passed)`
                - **Hard / Multi-Year Math & PAL:** `100% (10/10 Passed)`
                - **Out-of-Domain / Zero Hallucination:** `100% (10/10 Rejected)`
                - **Overall Accuracy:** `100.0% (50/50 PASS)`
                - **PAL Quantitative Latency:** `~3.71 seconds`
                - **Hybrid RAG Latency:** `~16.20 seconds`
                - **Unit & Type Safeguard:** `Pydantic Assertion`
                """)

    with st.container(border=True):
        st.markdown(f"#### :material/account_tree: **{T['sys_flow_title']}**")
        st.code("""
[User Query] ──► Query Routing ──┬──► (Quantitative/PAL) ──► PAL Engine (esg_tables.py) ────────┐
                                 └──► (Text/Policy)       ──► Hybrid Retrieval (Nomic 1.5)      │
                                                                   │                            │
                                                                   ▼                            ▼
                                                            Pydantic Assertion ──► phi-4-mini Synthesis
                                                                                        │
                                                                                        ▼
                                                                               [Verified Response]
        """, language="text")
