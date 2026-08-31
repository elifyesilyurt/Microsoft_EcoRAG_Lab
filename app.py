import streamlit as st
import sqlite3
import json
import numpy as np
import requests
import subprocess
import re
import time
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# 1. Configuration and Metadata
# ---------------------------------------------------------
APP_VERSION = "v0.3.0-PoC"
DB_PATH = "rag_storage.db"
MODEL_NAME = "phi-3.5-mini-instruct"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OOD_THRESHOLD = 0.15

st.set_page_config(
    page_title="Microsoft EcoRAG Lab",
    layout="wide"
)

# ---------------------------------------------------------
# 2. Dynamic Foundry Local Port Resolution
# ---------------------------------------------------------
@st.cache_resource
def get_foundry_endpoint():
    """Dynamically resolves the active port from Foundry Local CLI status."""
    try:
        result = subprocess.run(["foundry", "status"], capture_output=True, text=True, check=False)
        match = re.search(r"http://127\.0\.0\.1:(\d+)", result.stdout + result.stderr)
        if match:
            return f"http://127.0.0.1:{match.group(1)}/v1/chat/completions"
    except Exception:
        pass
    return "http://127.0.0.1:49826/v1/chat/completions"

FOUNDRY_URL = get_foundry_endpoint()

# ---------------------------------------------------------
# 3. Model and Database Loaders
# ---------------------------------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

embedder = load_embedder()

def search_context(query, top_k=4):
    """Performs cosine similarity search over stored SQLite embeddings."""
    query_vector = embedder.encode(query)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, year, title, page, content, embedding FROM chunks")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return [], 0.0

    scores = []
    for r in rows:
        c_id, year, title, page, content, emb_json = r
        doc_vector = np.array(json.loads(emb_json))
        
        norm_q = np.linalg.norm(query_vector)
        norm_d = np.linalg.norm(doc_vector)
        sim = 0.0 if (norm_q == 0 or norm_d == 0) else np.dot(query_vector, doc_vector) / (norm_q * norm_d)
        
        scores.append({
            "id": c_id,
            "year": year,
            "title": title,
            "page": page,
            "content": content,
            "score": float(sim)
        })

    scores.sort(key=lambda x: x["score"], reverse=True)
    top_chunks = scores[:top_k]
    max_score = top_chunks[0]["score"] if top_chunks else 0.0
    return top_chunks, max_score

# ---------------------------------------------------------
# 4. Streamlit User Interface
# ---------------------------------------------------------
st.title("Microsoft EcoRAG Lab")
st.caption(f"Local SLM & Multi-Year Sustainability Analyst - Version: {APP_VERSION}")

with st.sidebar:
    st.header("System Status")
    st.info(f"**Architecture:** Local Edge (On-Device)\n\n"
            f"**LLM:** `{MODEL_NAME}` (Apple Metal)\n\n"
            f"**Embedding:** `{EMBEDDING_MODEL_NAME}` (384-d)\n\n"
            f"**Vector Store:** SQLite3 + NumPy Matrix\n\n"
            f"**OOD Threshold:** `{OOD_THRESHOLD}`\n\n"
            f"**Foundry Endpoint:**\n`{FOUNDRY_URL}`")
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "provenance" in msg and msg["provenance"]:
            with st.expander(f"Sources ({len(msg['provenance'])}) - {msg.get('latency', '')}"):
                for idx, src in enumerate(msg["provenance"], 1):
                    st.markdown(f"**Source {idx}** | Year: `{src['year']}` | Page: `{src['page']}` | Score: `{src['score']:.4f}`")
                    st.text(src["content"][:300] + "...")

# ---------------------------------------------------------
# 5. User Interaction and RAG Pipeline
# ---------------------------------------------------------
user_query = st.chat_input("Ask a question regarding Microsoft sustainability reports...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    start_time = time.time()

    with st.spinner("Retrieving semantic context..."):
        retrieved_chunks, max_similarity = search_context(user_query, top_k=4)

    with st.chat_message("assistant"):
        if max_similarity < OOD_THRESHOLD:
            answer = (
                f"This specific information is not available in the Microsoft Environmental Sustainability reports. "
                f"(Similarity Score: {max_similarity:.4f} < Threshold: {OOD_THRESHOLD})"
            )
            elapsed_time = f"{time.time() - start_time:.2f}s"
            st.markdown(answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "provenance": [],
                "latency": elapsed_time
            })
        else:
            context_text = "\n\n---\n\n".join([
                f"[Report: {c['year']}, Page: {c['page']}]\n{c['content']}" 
                for c in retrieved_chunks
            ])

            system_prompt = (
                "You are an expert enterprise sustainability analyst for Microsoft. "
                "Answer the question STRICTLY using only the provided context below. "
                "Do not hallucinate, speculate, or use outside knowledge. "
                "If the specific figures or answers are not in the context, state clearly: "
                "'This specific information is not available in the Microsoft Environmental Sustainability reports.'\n\n"
                f"CONTEXT:\n{context_text}"
            )

            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                "temperature": 0.0,
                "stream": False
            }

            try:
                response = requests.post(FOUNDRY_URL, json=payload, timeout=90)
                response.raise_for_status()
                res_data = response.json()
                answer = res_data["choices"][0]["message"]["content"]
            except Exception as e:
                answer = f"Model inference error: {str(e)}"

            elapsed_time = f"{time.time() - start_time:.2f}s"
            st.markdown(answer)

            with st.expander(f"Sources ({len(retrieved_chunks)}) - {elapsed_time}"):
                for idx, src in enumerate(retrieved_chunks, 1):
                    st.markdown(f"**Source {idx}** | Report: `{src['year']}` | Page: `{src['page']}` | Similarity Score: `{src['score']:.4f}`")
                    st.text(src["content"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "provenance": retrieved_chunks,
                "latency": elapsed_time
            })
