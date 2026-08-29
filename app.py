import streamlit as st
import sqlite3
import json
import time
import subprocess
import re
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Union

# -----------------------------------------------------------------------------
# 1. Page Configuration & Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Microsoft EcoRAG - Sustainability AI Studio",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    
    .hero-container {
        background: linear-gradient(135deg, #0d324d 0%, #0c7b93 50%, #00a896 100%);
        border-radius: 12px;
        padding: 24px 28px;
        color: white;
        margin-bottom: 20px;
    }
    .hero-title { font-size: 1.8rem; font-weight: 700; margin: 0; }
    .hero-subtitle { font-size: 0.95rem; opacity: 0.9; margin-top: 4px; }
    
    .metric-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.25);
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 8px;
    }

    .source-card {
        background: rgba(255, 255, 255, 0.04);
        border-left: 4px solid #00a896;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .year-tag {
        background: #00a896;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Configurations & Model Client
# -----------------------------------------------------------------------------
DB_PATH = "rag_storage.db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OUT_OF_DOMAIN_THRESHOLD = 0.15

def detect_foundry_url() -> str:
    try:
        out = subprocess.check_output(["foundry", "status"], text=True)
        match = re.search(r"http://127\.0\.0\.1:\d+", out)
        if match:
            return match.group(0)
    except Exception:
        pass
    return "http://127.0.0.1:49826"

@st.cache_resource
def load_embed_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def get_local_llm_client():
    base = detect_foundry_url()
    return openai.OpenAI(base_url=f"{base}/v1", api_key="foundry-local", timeout=60.0)

embed_model = load_embed_model()
client = get_local_llm_client()

# -----------------------------------------------------------------------------
# 3. Retrieval Engine
# -----------------------------------------------------------------------------
def get_db_stats() -> Tuple[int, List[int]]:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
        cursor.execute("SELECT DISTINCT year FROM documents ORDER BY year")
        years = [r[0] for r in cursor.fetchall() if r[0] is not None]
        conn.close()
        return count, years
    except Exception:
        return 0, []

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    dot = np.dot(vec_a, vec_b)
    norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return float(dot / norm) if norm > 0 else 0.0

def retrieve_strands(query: str, top_k: int = 3, filter_year: Union[str, int] = "All") -> List[Dict]:
    query_vector = embed_model.encode(query)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if filter_year != "All":
        cursor.execute("SELECT id, year, title, content, embedding FROM documents WHERE year = ?", (filter_year,))
    else:
        cursor.execute("SELECT id, year, title, content, embedding FROM documents")
    rows = cursor.fetchall()
    conn.close()
    
    scored = []
    for doc_id, year, title, content, emb_str in rows:
        score = cosine_similarity(query_vector, np.array(json.loads(emb_str)))
        scored.append({
            "id": doc_id,
            "year": year,
            "title": title,
            "content": content,
            "score": score
        })
        
    scored.sort(key=lambda x: x["score"], reverse=True)
    top_results = scored[:top_k]
    top_results.sort(key=lambda x: x["year"] if x["year"] else 0)
    return top_results

# -----------------------------------------------------------------------------
# 4. RAG Execution (Robust Non-Streaming with Simulated Typing)
# -----------------------------------------------------------------------------
def execute_rag(query: str, top_k: int = 3, filter_year: Union[str, int] = "All"):
    start_time = time.time()
    chunks = retrieve_strands(query, top_k=top_k, filter_year=filter_year)
    
    best_score = max([c["score"] for c in chunks]) if chunks else 0.0
    if best_score < OUT_OF_DOMAIN_THRESHOLD:
        def fallback():
            yield "This specific information is not available in the Microsoft Environmental Sustainability reports."
        return fallback(), [], start_time

    context_str = "\n\n".join(
        f"[{c.get('year', 'N/A')} Report | {c['title']}]\n{c['content']}" for c in chunks
    )
    
    system_prompt = (
        "You are an expert AI Sustainability Analyst specialized in Microsoft Environmental Reports. "
        "Strictly adhere to the following rules:\n"
        "1. Base your answer EXCLUSIVELY on the provided CONTEXT.\n"
        "2. Highlight chronological changes, metrics (Scope 1/2/3, water replenishment, clean energy contracts), and report years.\n"
        "3. Never hallucinate facts not present in the text.\n"
        "4. If context does not contain the answer, state: 'This specific information is not available in the Microsoft Environmental Sustainability reports.'"
    )
    
    user_payload = f"CONTEXT:\n{context_str}\n\nQUESTION: {query}\nDETAILED ANALYSIS:"
    
    try:
        models = client.models.list()
        model_name = models.data[0].id if models.data else "qwen2.5-0.5b"
    except Exception:
        model_name = "qwen2.5-0.5b"

    try:
        # stream=False kullanarak soket kapanma hatasını sıfıra indiriyoruz
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload}
            ],
            temperature=0.0,
            stream=False
        )
        full_text = response.choices[0].message.content or ""
        
        def word_stream():
            for word in full_text.split(" "):
                yield word + " "
                time.sleep(0.015)

        return word_stream(), chunks, start_time
    except Exception as err:
        err_msg = str(err)
        def err_stream():
            yield f"Context retrieved ({len(chunks)} chunks).\n\n"
            yield f"Yerel LLM servisine baglanirken durum olustu: {err_msg}"
        return err_stream(), chunks, start_time

# -----------------------------------------------------------------------------
# 5. UI Layout
# -----------------------------------------------------------------------------
total_chunks, available_years = get_db_stats()
endpoint_url = detect_foundry_url()

st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">Microsoft EcoRAG Studio</div>
    <div class="hero-subtitle">Cok Yillik Surdurulebilirlik Raporlari Yerel Analiz Motoru</div>
    <div>
        <span class="metric-badge">Endpoint: {endpoint_url}</span>
        <span class="metric-badge">Embedding: {EMBEDDING_MODEL_NAME}</span>
        <span class="metric-badge">Veritabani: {total_chunks} Parca</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Parametreler")
    year_options = ["All"] + [str(y) for y in available_years] if available_years else ["All"]
    selected_year = st.selectbox("Rapor Yili:", year_options, index=0)
    top_k = st.slider("Top-K Parca Sayisi:", min_value=1, max_value=5, value=3)
    
    st.markdown("---")
    st.info(f"Toplam Parca: {total_chunks}\nYillar: {', '.join(map(str, available_years))}")
    
    if st.button("Sohbeti Sifirla", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "chunks" in message and message["chunks"]:
            with st.expander(f"Kaynaklar ({len(message['chunks'])})"):
                for chunk in message["chunks"]:
                    st.markdown(f"""
                    <div class="source-card">
                        <span class="year-tag">{chunk.get('year', 'Rapor')}</span> <b>{chunk['title']}</b> 
                        <span style="float: right; opacity: 0.7;">Skor: {chunk['score']:.4f}</span>
                        <div style="font-size: 0.88rem; margin-top: 6px;">{chunk['content']}</div>
                    </div>
                    """, unsafe_allow_html=True)

if prompt := st.chat_input("Microsoft surdurulebilirlik raporlari hakkinda bir soru sorun..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream_or_gen, chunks, start_time = execute_rag(prompt, top_k=top_k, filter_year=selected_year)
        response_text = st.write_stream(stream_or_gen)
        latency = round(time.time() - start_time, 2)
        
        if chunks:
            with st.expander(f"Kaynaklar ({len(chunks)}) - {latency}s"):
                for chunk in chunks:
                    st.markdown(f"""
                    <div class="source-card">
                        <span class="year-tag">{chunk.get('year', 'Rapor')}</span> <b>{chunk['title']}</b> 
                        <span style="float: right; opacity: 0.7;">Skor: {chunk['score']:.4f}</span>
                        <div style="font-size: 0.88rem; margin-top: 6px;">{chunk['content']}</div>
                    </div>
                    """, unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "chunks": chunks,
        "latency": latency
    })
