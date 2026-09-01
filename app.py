import os
import json
import sqlite3
import time
import requests
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

# Application Configuration
DB_PATH = "rag_storage.db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FOUNDRY_BASE_URL = "http://127.0.0.1:62095"
MODEL_NAME = "phi-4-mini"

# Retrieval Hyperparameters
RELATIVE_DROP_RATIO = 0.80
MAX_K = 12
MIN_SCORE_FLOOR = 0.15

SYSTEM_PROMPT = """You are a rigorous, production-grade Senior Sustainability Data Analyst.
Analyze the user query based ONLY on the provided context.

Strict Matrix and Tabular Verification Rules:
1. Column-to-Year Alignment: Ensure each metric is mapped to the exact target year column. Never substitute FY24 numbers for FY25, or vice-versa.
2. Row-to-Metric Alignment: Match exact row names. Distinguish 'Scope 2 Market-based' from 'Scope 2 Location-based', and 'Scope 3 Subtotal' from 'Total GHG Emissions'.
3. Arithmetic Integrity: When computing differences, verify both operands explicitly from the text/table (e.g., Target Year Value - Baseline Year Value).
4. Structured Presentation: Present multi-year trend queries using explicit year breakdowns (FY20 Baseline, FY24, FY25) followed by the delta.
5. Strict Faithfulness: Extract exact numbers without guessing or substituting adjacent cells."""

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

embedder = load_embedder()

def search_context(query: str):
    """Dynamic adaptive retrieval with domain keyword boosting."""
    query_vector = embedder.encode(query)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, year, title, content, embedding FROM documents")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return [], 0.0

    query_lower = query.lower()
    keywords = ["scope 1", "scope 2", "scope 3", "baseline", "emissions", "water", "waste", "table", "market-based", "location-based"]
    matched_keywords = [k for k in keywords if k in query_lower]

    scores = []
    for r in rows:
        c_id, year, title, content, emb_json = r
        doc_vector = np.array(json.loads(emb_json))
        norm_q = np.linalg.norm(query_vector)
        norm_d = np.linalg.norm(doc_vector)
        sim = 0.0 if (norm_q == 0 or norm_d == 0) else float(np.dot(query_vector, doc_vector) / (norm_q * norm_d))
        
        content_lower = content.lower()
        boost = sum(0.04 for kw in matched_keywords if kw in content_lower)
        adjusted_score = sim + boost

        scores.append({
            "id": c_id,
            "year": year,
            "title": title,
            "content": content,
            "score": adjusted_score,
            "base_sim": sim
        })

    scores.sort(key=lambda x: x["score"], reverse=True)
    if not scores:
        return [], 0.0

    max_score = scores[0]["base_sim"]
    if max_score < MIN_SCORE_FLOOR:
        return [], max_score

    cutoff = scores[0]["score"] * RELATIVE_DROP_RATIO
    dynamic_chunks = [item for item in scores[:MAX_K] if item["score"] >= cutoff]
    return dynamic_chunks, max_score

def query_foundry_model(prompt: str, context: str):
    """Sends payload to Foundry Local endpoint."""
    url = f"{FOUNDRY_BASE_URL}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
        ],
        "temperature": 0.1,
        "max_tokens": 1024
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=90)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

# Streamlit Interface
st.set_page_config(page_title="Microsoft EcoRAG Lab", layout="wide")
st.title("🌱 Microsoft EcoRAG Lab")
st.caption("Local SLM & Multi-Year Sustainability Analyst")

with st.sidebar:
    st.header("Settings")
    st.markdown(f"**Model:** `{MODEL_NAME}`")
    st.markdown(f"**Endpoint:** `{FOUNDRY_BASE_URL}`")
    st.markdown(f"**Retrieval:** `Dynamic Adaptive Threshold (80%)`")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "provenance" in msg and msg["provenance"]:
            with st.expander(f"Data Provenance ({len(msg['provenance'])}) - Max Score: {msg.get('max_score', 0):.4f} - Latency: {msg.get('latency', 0):.2f}s"):
                for p in msg["provenance"]:
                    st.markdown(f"**{p['title']}** (Score: {p['score']:.4f})")
                    st.text(p["content"][:300] + "...")

if user_query := st.chat_input("Ask a question regarding Microsoft sustainability reports..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    start_time = time.time()
    with st.spinner("Searching and generating response..."):
        try:
            chunks, max_score = search_context(user_query)
            if not chunks:
                ans = "I could not find sufficient matching records in the local ESG database."
                prov = []
            else:
                context_str = "\n\n".join([c["content"] for c in chunks])
                ans = query_foundry_model(user_query, context_str)
                prov = chunks

            latency = time.time() - start_time

            with st.chat_message("assistant"):
                st.markdown(ans)
                if prov:
                    with st.expander(f"Data Provenance ({len(prov)}) - Max Score: {max_score:.4f} - Latency: {latency:.2f}s"):
                        for p in prov:
                            st.markdown(f"**{p['title']}** (Score: {p['score']:.4f})")
                            st.text(p["content"][:300] + "...")

            st.session_state.messages.append({
                "role": "assistant",
                "content": ans,
                "provenance": prov,
                "max_score": max_score,
                "latency": latency
            })
        except Exception as e:
            st.error(f"Error during execution: {e}")
