import streamlit as st
import sqlite3
import json
import time
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from foundry_local_sdk import Configuration, FoundryLocalManager
from typing import List, Tuple, Dict, Generator, Union

# -----------------------------------------------------------------------------
# 1. Page Configuration & UI Styles
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Foundry Local • AI Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    * { font-family: 'Outfit', sans-serif; }
    
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
    
    .hero-title { font-size: 2.2rem; font-weight: 700; margin: 0; }
    .hero-subtitle { font-size: 1rem; opacity: 0.9; margin-top: 5px; font-weight: 300; }

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
    }

    .source-card {
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid #23a6d5;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Global Configurations
# -----------------------------------------------------------------------------
DB_PATH = "rag_storage.db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_ALIAS = "qwen2.5-0.5b"
OUT_OF_DOMAIN_THRESHOLD = 0.12


# -----------------------------------------------------------------------------
# 3. Core Engine Initialization
# -----------------------------------------------------------------------------
@st.cache_resource
def load_rag_components() -> Tuple[SentenceTransformer, openai.OpenAI]:
    """
    Initializes the local embedding model and the Microsoft Foundry Local web service.
    Utilizes caching to prevent re-initialization on Streamlit reruns.
    
    Returns:
        Tuple containing the embedding model instance and the OpenAI client connected to Foundry Local.
    """
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    try:
        config = Configuration(app_name="FoundryLocalWorkshop")
        FoundryLocalManager.initialize(config)
    except Exception:
        pass  # Ignore Singleton re-initialization errors
        
    manager = FoundryLocalManager.instance
    try:
        manager.start_web_service()
    except Exception:
        pass
    
    catalog = manager.get_catalog() if hasattr(manager, "get_catalog") else manager.catalog
    llm = catalog.get_model(LLM_ALIAS)
    try:
        llm.load()
    except Exception:
        pass
    
    endpoint = manager.endpoint if hasattr(manager, "endpoint") else f"{manager.urls[0]}/v1"
    client = openai.OpenAI(base_url=endpoint, api_key="foundry-local")
    return embed_model, client


embed_model, client = load_rag_components()


# -----------------------------------------------------------------------------
# 4. Vector Operations
# -----------------------------------------------------------------------------
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two numeric vectors.
    
    Args:
        vec_a (np.ndarray): The first vector.
        vec_b (np.ndarray): The second vector.
        
    Returns:
        float: A similarity score between -1.0 and 1.0.
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


def get_top_chunks(query: str, top_k: int = 2) -> List[Dict]:
    """
    Encodes the user query and searches the SQLite database for the most semantically similar chunks.
    
    Args:
        query (str): The user's input question.
        top_k (int): The maximum number of documents to retrieve.
        
    Returns:
        List[Dict]: A sorted list of the top retrieved documents containing their id, title, content, and similarity score.
    """
    query_vector = embed_model.encode(query)
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, content, embedding FROM documents")
        rows = cursor.fetchall()
        conn.close()
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return []
    
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


# -----------------------------------------------------------------------------
# 5. RAG Pipeline & Inference
# -----------------------------------------------------------------------------
def stream_rag_response(user_question: str, top_k: int = 2) -> Tuple[Union[Generator, openai.Stream], List[Dict], float]:
    """
    Executes the Retrieval-Augmented Generation pipeline. Applies dynamic thresholding to mitigate hallucinations.
    
    Args:
        user_question (str): The user's input question.
        top_k (int): Number of chunks to retrieve.
        
    Returns:
        Tuple: A text generator or OpenAI stream, the retrieved source chunks, and the start timestamp.
    """
    start_time = time.time()
    retrieved_chunks = get_top_chunks(user_question, top_k=top_k)

    # Base Hallucination Guard (Catches out-of-domain questions instantly)
    best_score = retrieved_chunks[0]["score"] if retrieved_chunks else 0.0
    if best_score < OUT_OF_DOMAIN_THRESHOLD:
        def fallback_generator():
            yield "This information is not available in the documents."
        return fallback_generator(), [], start_time

    context_text = "\n\n".join(
        f"Document: {chunk['title']}\nContent: {chunk['content']}" for chunk in retrieved_chunks
    )
    
    # Strict Constraints Prompt
    system_prompt = (
        "You are an AI assistant designed exclusively to answer questions about technical documents. "
        "Strictly adhere to these rules:\n"
        "1. Base your answer EXCLUSIVELY on the facts provided in the CONTEXT.\n"
        "2. Never use pre-trained world knowledge, facts, or assumptions.\n"
        "3. If the context does not explicitly answer the question, output ONLY: "
        "'This information is not available in the documents.'"
    )
    
    user_payload = f"CONTEXT:\n{context_text}\n\nQUESTION: {user_question}\nANSWER:"
    
    stream = client.chat.completions.create(
        model=LLM_ALIAS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload}
        ],
        temperature=0.0,
        stream=True
    )
    
    return stream, retrieved_chunks, start_time


# -----------------------------------------------------------------------------
# 6. Streamlit User Interface
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">⚡ Foundry Local AI Studio</div>
    <div class="hero-subtitle">100% Yerel Donanım Hızlandırmalı & Halüsinasyon Korumalı RAG Motoru</div>
    <div style="margin-top: 15px;">
        <span class="metric-badge">🟢 LLM: qwen2.5-0.5b</span>
        <span class="metric-badge">🧠 Embeddings: all-MiniLM-L6-v2</span>
        <span class="metric-badge">🛡️ Dinamik Halüsinasyon Koruması: Aktif</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🎛️ Motor Yapılandırması")
    top_k = st.slider("Alınacak Doküman Parçası (Top-K):", min_value=1, max_value=4, value=2)
    st.markdown("---")
    st.markdown("### 📊 Güvenlik & Durum")
    st.success("Güvenlik Filtresi: **Otomatik & Dinamik (0 Ayar)**")
    
    if st.button("🗑️ Sohbeti Sıfırla", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
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

# Chat Input & Stream Processing
if prompt := st.chat_input("Teknik bir soru sorun..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream_or_gen, chunks, start_time = stream_rag_response(prompt, top_k=top_k)
        
        def generate_stream():
            for item in stream_or_gen:
                if isinstance(item, str):
                    yield item
                elif hasattr(item, "choices") and item.choices and item.choices[0].delta.content:
                    yield item.choices[0].delta.content

        response_text = st.write_stream(generate_stream())
        latency = round(time.time() - start_time, 2)
        
        if chunks:
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
