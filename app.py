import streamlit as st
import sqlite3
import json
import time
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from foundry_local_sdk import Configuration, FoundryLocalManager

# Page Configuration
st.set_page_config(
    page_title="Foundry Local • AI Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Animated & Glassmorphic CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Animated Gradient Background for Header */
    .hero-container {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        border-radius: 20px;
        padding: 30px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 5px;
        font-weight: 300;
    }

    /* Metric Badges with Hover Glow */
    .metric-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 8px;
        backdrop-filter: blur(10px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .metric-badge:hover {
        transform: translateY(-2px);
        border-color: #23d5ab;
        box-shadow: 0 4px 15px rgba(35, 213, 171, 0.3);
    }

    /* Source Chunk Card Animation */
    .source-card {
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid #23a6d5;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .source-card:hover {
        transform: scale(1.01);
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        border-left-color: #23d5ab;
    }
</style>
""", unsafe_allow_html=True)

DB_PATH = "rag_storage.db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_ALIAS = "qwen2.5-0.5b"


@st.cache_resource
def load_rag_components():
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    config = Configuration(app_name="FoundryLocalWorkshop")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    manager.start_web_service()
    
    catalog = manager.get_catalog() if hasattr(manager, "get_catalog") else manager.catalog
    llm = catalog.get_model(LLM_ALIAS)
    llm.load()
    
    endpoint = manager.endpoint if hasattr(manager, "endpoint") else f"{manager.urls[0]}/v1"
    client = openai.OpenAI(base_url=endpoint, api_key="foundry-local")
    return embed_model, client


embed_model, client = load_rag_components()


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


def get_top_chunks(query: str, top_k: int = 2) -> list:
    query_vector = embed_model.encode(query)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, content, embedding FROM documents")
    rows = cursor.fetchall()
    conn.close()
    
    scored_chunks = []
    for doc_id, title, content, emb_str in rows:
        doc_vector = np.array(json.loads(emb_str))
        score = cosine_similarity(query_vector, doc_vector)
        scored_chunks.append({
            "id": doc_id,
            "title": title,
            "content": content,
            "score": score
        })
    
    scored_chunks.sort(key=lambda item: item["score"], reverse=True)
    return scored_chunks[:top_k]


def stream_rag_response(user_question: str, top_k: int = 2):
    start_time = time.time()
    retrieved_chunks = get_top_chunks(user_question, top_k=top_k)
    context_text = "\n\n".join(
        f"### {chunk['title']}\n{chunk['content']}" for chunk in retrieved_chunks
    )
    
    system_prompt = (
        "You are a helpful and accurate technical assistant. "
        "Answer the user's question using ONLY the provided context below. "
        "If the information is not present in the context, explicitly state: "
        "'This information is not available in the documents.'\n\n"
        f"Context:\n{context_text}"
    )
    
    stream = client.chat.completions.create(
        model=LLM_ALIAS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ],
        temperature=0.0,
        stream=True
    )
    
    return stream, retrieved_chunks, start_time


# Top Animated Hero Section
st.markdown("""
<div class="hero-container">
    <div class="hero-title">⚡ Foundry Local AI Studio</div>
    <div class="hero-subtitle">100% Yerel Donanım Hızlandırmalı RAG & Vektör Çıkarım Motoru</div>
    <div style="margin-top: 15px;">
        <span class="metric-badge">🟢 LLM: qwen2.5-0.5b</span>
        <span class="metric-badge">🧠 Embeddings: all-MiniLM-L6-v2</span>
        <span class="metric-badge">⚡ Engine: Foundry Local</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.markdown("### 🎛️ Motor Yapılandırması")
    top_k = st.slider("Alınacak Doküman Parçası (Top-K):", min_value=1, max_value=4, value=2)
    st.markdown("---")
    st.markdown("### 📊 Sistem Durumu")
    st.success("Yerel GPU Hızlandırıcı: **Aktif**")
    st.info("Vektör Veritabanı: **SQLite (Yerel)**")
    
    if st.button("🗑️ Sohbeti Sıfırla", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Initialize Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Historical Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "chunks" in message and message["chunks"]:
            with st.expander(f"🔍 Doğrulanan Kaynaklar ({len(message['chunks'])}) • Gecikme: {message.get('latency', 0)}s"):
                for chunk in message["chunks"]:
                    st.markdown(f"""
                    <div class="source-card">
                        <b>📄 {chunk['title']}</b> <span style="float: right; opacity: 0.7;">Benzerlik: {chunk['score']:.4f}</span>
                        <div style="font-size: 0.88rem; margin-top: 5px; opacity: 0.85;">{chunk['content']}</div>
                    </div>
                    """, unsafe_allow_html=True)

# User Chat Input
if prompt := st.chat_input("Teknik veya kavramsal bir soru sorun..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream, chunks, start_time = stream_rag_response(prompt, top_k=top_k)
        
        # Real-time Typewriter Streaming Effect
        def generate_stream():
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        response_text = st.write_stream(generate_stream())
        latency = round(time.time() - start_time, 2)
        
        with st.expander(f"🔍 Doğrulanan Kaynaklar ({len(chunks)}) • Gecikme: {latency}s"):
            for chunk in chunks:
                st.markdown(f"""
                <div class="source-card">
                    <b>📄 {chunk['title']}</b> <span style="float: right; opacity: 0.7;">Benzerlik: {chunk['score']:.4f}</span>
                    <div style="font-size: 0.88rem; margin-top: 5px; opacity: 0.85;">{chunk['content']}</div>
                </div>
                """, unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "chunks": chunks,
        "latency": latency
    })
