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
from esg_tables import get_carbon_emissions_df, get_carbon_removal_df, get_water_metrics_df
from extraction_pipeline import (
    EXTRACTION_SYSTEM_PROMPT,
    format_extraction_prompt,
    QueryExtractionPlan,
    DeterministicResolver
)

DB_PATH = "rag_storage.db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FOUNDRY_BASE_URL = "http://127.0.0.1:62095"
MODEL_NAME = "phi-4-mini"

RELATIVE_DROP_RATIO = 0.70
MAX_K = 6
MIN_SCORE_FLOOR = 0.15

SYNTHESIS_PROMPT = """You are a Senior Sustainability Analyst.
Synthesize the verified analytical calculation results into a clear, structured executive report with exact units (mtCO2e / metric tons / m3).
Do not alter any calculated numbers."""

FACTUAL_SYNTHESIS_PROMPT = """You are a Senior Sustainability AI Analyst.
Using the verified structured metrics provided below, compose a concise, direct natural language answer.
State the exact numbers and their corresponding units clearly in sentence 1."""

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

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
        "max_tokens": 1024
    }
    
    try:
        with requests.Session() as session:
            res = session.post(url, headers=headers, json=payload, timeout=180)
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"].strip()
                return result
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
    query_vector = embedder.encode(query)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, year, title, content, embedding FROM documents")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return [], 0.0

    keywords = [w.lower() for w in query.replace("?", "").replace(",", "").split() if len(w) > 3]

    scores = []
    for r in rows:
        c_id, year, title, content, emb_json = r
        doc_vector = np.array(json.loads(emb_json))
        norm_q = np.linalg.norm(query_vector)
        norm_d = np.linalg.norm(doc_vector)
        sim = 0.0 if (norm_q == 0 or norm_d == 0) else float(np.dot(query_vector, doc_vector) / (norm_q * norm_d))
        
        content_lower = content.lower()
        match_count = sum(1 for kw in keywords if kw in content_lower)
        hybrid_score = sim + (0.05 * match_count)
        
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

st.set_page_config(page_title="Microsoft EcoRAG Lab", layout="wide")
st.title("🌱 Microsoft EcoRAG Lab")
st.caption("Structured Intermediate Representation • Deterministic Verification Engine")

with st.sidebar:
    st.header("Pipeline Mode")
    st.markdown(f"**Model:** `{MODEL_NAME}`")
    st.markdown(f"**Architecture:** `Structured IR + Assertion Engine`")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        gc.collect()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "calc_details" in msg and msg["calc_details"]:
            with st.expander("⚡ Verified Analytical Engine Output"):
                st.text(msg["calc_details"])
        if "provenance" in msg and msg["provenance"]:
            with st.expander(f"Data Provenance ({len(msg['provenance'])}) - Score: {msg.get('max_score', 0):.4f} - Latency: {msg.get('latency', 0):.2f}s"):
                for p in msg["provenance"]:
                    st.markdown(f"**{p['title']}** (Score: {p['score']:.4f})")
                    st.text(p["content"][:300] + "...")

if user_query := st.chat_input("Ask a question regarding Microsoft sustainability reports..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    start_time = time.time()
    
    with st.spinner("Executing structured extraction & validation..."):
        try:
            q_lower = user_query.lower()
            is_math_compare = ("trend" in q_lower or "compare" in q_lower or "difference" in q_lower) and ("scope" in q_lower)
            
            calc_details = None
            chunks = []
            max_score = 0.0
            
            if is_math_compare:
                calc_details = compute_carbon_trend_summary()
                synthesis_input = f"Question: {user_query}\n\nData:\n{calc_details}"
                ans = query_foundry(SYNTHESIS_PROMPT, synthesis_input, temperature=0.0)
                chunks, max_score = search_context_hybrid(user_query)
            else:
                chunks, max_score = search_context_hybrid(user_query)
                if not chunks or max_score < MIN_SCORE_FLOOR:
                    ans = "I cannot find information regarding this in the provided Microsoft Environmental Sustainability reports."
                else:
                    context_chunks = [c["content"] for c in chunks]
                    extract_prompt = format_extraction_prompt(user_query, context_chunks)
                    
                    raw_json = query_foundry(EXTRACTION_SYSTEM_PROMPT, extract_prompt, temperature=0.0)
                    
                    try:
                        cleaned = re.search(r"\{.*\}", raw_json, re.DOTALL).group(0)
                        plan_data = json.loads(cleaned)
                        plan = QueryExtractionPlan(**plan_data)
                        resolution = DeterministicResolver.validate_and_filter(plan, user_query)
                        
                        if resolution["status"] == "NOT_FOUND":
                            ans = "I cannot find information regarding this in the provided Microsoft Environmental Sustainability reports."
                        else:
                            verified_metrics_str = "\n".join([
                                f"- Entity: {m.entity}, Type: {m.metric_type}, Value: {m.value:,.0f} {m.unit}, Scope: {m.temporal_scope}, Cumulative: {m.is_cumulative}"
                                for m in resolution["metrics"]
                            ])
                            calc_details = verified_metrics_str
                            synthesis_prompt = f"Verified Metrics:\n{verified_metrics_str}\n\nOriginal User Question: {user_query}"
                            ans = query_foundry(FACTUAL_SYNTHESIS_PROMPT, synthesis_prompt, temperature=0.0)
                    except Exception as ex:
                        # Fallback to direct grounding
                        context_str = "\n\n".join(context_chunks)
                        ans = query_foundry(
                            "You are a precise Sustainability Analyst. Answer directly using ONLY context.",
                            f"Context:\n{context_str}\n\nQuestion: {user_query}",
                            temperature=0.0
                        )
            
            latency = time.time() - start_time

            with st.chat_message("assistant"):
                st.markdown(ans)
                if calc_details:
                    with st.expander("⚡ Verified Structured Representation"):
                        st.text(calc_details)
                if chunks:
                    with st.expander(f"Data Provenance ({len(chunks)}) - Score: {max_score:.4f} - Latency: {latency:.2f}s"):
                        for p in chunks:
                            st.markdown(f"**{p['title']}** (Score: {p['score']:.4f})")
                            st.text(p["content"][:300] + "...")

            st.session_state.messages.append({
                "role": "assistant",
                "content": ans,
                "calc_details": calc_details,
                "provenance": chunks,
                "max_score": max_score,
                "latency": latency
            })
            gc.collect()

        except Exception as e:
            st.error(f"Error: {e}")
            gc.collect()
