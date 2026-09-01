import streamlit as st
import sqlite3
import json
import numpy as np
import requests
import subprocess
import re
import time
from sentence_transformers import SentenceTransformer

APP_VERSION = "v0.3.0-PoC"
DB_PATH = "rag_storage.db"
MODEL_NAME = "phi-4-mini"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OOD_THRESHOLD = 0.20
TOP_K = 6

st.set_page_config(page_title="Microsoft EcoRAG Lab", layout="wide")

@st.cache_resource
def get_foundry_endpoint() -> str:
    try:
        result = subprocess.run(["foundry", "status"], capture_output=True, text=True, check=False)
        match = re.search(r"http://127\.0\.0\.1:(\d+)", result.stdout + result.stderr)
        if match:
            return f"http://127.0.0.1:{match.group(1)}/v1/chat/completions"
    except Exception:
        pass
    return "http://127.0.0.1:49812/v1/chat/completions"

FOUNDRY_URL = get_foundry_endpoint()

@st.cache_resource
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

embedder = load_embedder()

def search_context(query: str, top_k: int = TOP_K):
    query_vector = embedder.encode(query)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, year, title, content, embedding FROM documents")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return [], 0.0

    scores = []
    for r in rows:
        c_id, year, title, content, emb_json = r
        doc_vector = np.array(json.loads(emb_json))
        norm_q = np.linalg.norm(query_vector)
        norm_d = np.linalg.norm(doc_vector)
        sim = 0.0 if (norm_q == 0 or norm_d == 0) else np.dot(query_vector, doc_vector) / (norm_q * norm_d)
        scores.append({"id": c_id, "year": year, "title": title, "content": content, "score": float(sim)})

    scores.sort(key=lambda x: x["score"], reverse=True)
    top_chunks = scores[:top_k]
    max_score = top_chunks[0]["score"] if top_chunks else 0.0
    return top_chunks, max_score

st.title("Microsoft EcoRAG Lab")
st.caption(f"Local SLM & Multi-Year Sustainability Analyst - Version: {APP_VERSION}")

with st.sidebar:
    st.header("System Telemetry")
    st.info(
        f"**Architecture:** Local Edge (On-Device)\n\n"
        f"**LLM Backend:** `{MODEL_NAME}` (Apple Metal)\n\n"
        f"**Embeddings:** `{EMBEDDING_MODEL_NAME}` (384-d)\n\n"
        f"**Vector Store:** SQLite3 (`documents` schema)\n\n"
        f"**Retrieved Passages:** Top `{TOP_K}` Chunks\n\n"
        f"**OOD Filter Threshold:** `{OOD_THRESHOLD}`\n\n"
        f"**Inference Endpoint:**\n`{FOUNDRY_URL}`"
    )
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "provenance" in msg and msg["provenance"]:
            with st.expander(f"Data Provenance ({len(msg['provenance'])}) - Latency: {msg.get('latency', '')}"):
                for idx, src in enumerate(msg["provenance"], 1):
                    st.markdown(f"**Source {idx}** | Year: `{src['year']}` | Section: `{src['title']}` | Cosine Similarity: `{src['score']:.4f}`")
                    st.text(src["content"][:300] + "...")

user_query = st.chat_input("Ask a question regarding Microsoft sustainability reports...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    start_time = time.time()
    with st.spinner(f"Retrieving top {TOP_K} semantic contexts..."):
        retrieved_chunks, max_similarity = search_context(user_query, top_k=TOP_K)

    with st.chat_message("assistant"):
        if max_similarity < OOD_THRESHOLD:
            answer = (
                f"This specific information is not available in the Microsoft Environmental Sustainability reports. "
                f"(Max Similarity Score: {max_similarity:.4f} < Threshold: {OOD_THRESHOLD})"
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
                f"[Report Year: {c['year']}, Section: {c['title']}]\n{c['content']}" 
                for c in retrieved_chunks
            ])

            system_prompt = (
                "You are an enterprise sustainability analyst for Microsoft. "
                "Answer the user's question clearly, concisely, and analytically based STRICTLY on the provided context. "
                "Extract exact numerical figures, baseline comparisons, and step-by-step arithmetic without external speculation. "
                "If certain data points are missing from the context, explicitly state what is available and what is missing."
            )

            user_prompt = f"CONTEXT:\n{context_text}\n\nQUESTION:\n{user_query}\n\nANSWER:"

            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.0,
                "stream": True
            }

            placeholder = st.empty()
            accumulated_text = ""

            try:
                with requests.post(FOUNDRY_URL, json=payload, stream=True, timeout=90) as response:
                    if not response.ok:
                        accumulated_text = f"Foundry Local Error ({response.status_code}): {response.text}"
                        placeholder.markdown(accumulated_text)
                    else:
                        for line in response.iter_lines():
                            if line:
                                line_str = line.decode('utf-8')
                                if line_str.startswith("data: "):
                                    json_str = line_str[6:].strip()
                                    if json_str == "[DONE]":
                                        break
                                    try:
                                        chunk_json = json.loads(json_str)
                                        delta = chunk_json["choices"][0]["delta"]
                                        token = delta.get("content", "")
                                        accumulated_text += token
                                        placeholder.markdown(accumulated_text + " ▌")
                                    except Exception:
                                        pass
            except Exception as e:
                accumulated_text = f"Model inference error: {str(e)}"

            placeholder.markdown(accumulated_text)
            elapsed_time = f"{time.time() - start_time:.2f}s"

            with st.expander(f"Data Provenance ({len(retrieved_chunks)}) - Max Score: {max_similarity:.4f} - Latency: {elapsed_time}"):
                for idx, src in enumerate(retrieved_chunks, 1):
                    st.markdown(f"**Source {idx}** | Year: `{src['year']}` | Section: `{src['title']}` | Cosine Similarity: `{src['score']:.4f}`")
                    st.text(src["content"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": accumulated_text,
                "provenance": retrieved_chunks,
                "latency": elapsed_time
            })
