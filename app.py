import streamlit as st
import sqlite3
import json
import time
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from foundry_local_sdk import Configuration, FoundryLocalManager

# Streamlit UI Configuration
st.set_page_config(
    page_title="Foundry Local RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

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


def answer_query(user_question: str, top_k: int = 2) -> tuple:
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
    
    response = client.chat.completions.create(
        model=LLM_ALIAS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ],
        temperature=0.0
    )
    
    elapsed_time = round(time.time() - start_time, 2)
    return response.choices[0].message.content, retrieved_chunks, elapsed_time


# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ RAG Ayarları")
    top_k = st.slider("Alınacak Doküman Sayısı (Top-K):", min_value=1, max_value=4, value=2)
    if st.button("Sohbeti Temizle", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("🤖 Foundry Local RAG Assistant")
st.caption("100% Yerel Çalışan, Kaynak Gösterimli Soru-Cevap Arayüzü")

# Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "chunks" in message and message["chunks"]:
            with st.expander(f"📚 Kullanılan Kaynaklar ({len(message['chunks'])} Parça) - Süre: {message.get('latency', 0)}s"):
                for chunk in message["chunks"]:
                    st.markdown(f"**{chunk['title']}** (Benzerlik Skoru: `{chunk['score']:.4f}`)")
                    st.caption(chunk["content"])

# User Input
if prompt := st.chat_input("Bir soru sorun (Örn: What is the benefit of Foundry Local?)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Vektörler taranıyor ve yerel yanıt oluşturuluyor..."):
            answer, chunks, latency = answer_query(prompt, top_k=top_k)
            st.markdown(answer)
            with st.expander(f"📚 Kullanılan Kaynaklar ({len(chunks)} Parça) - Süre: {latency}s"):
                for chunk in chunks:
                    st.markdown(f"**{chunk['title']}** (Benzerlik Skoru: `{chunk['score']:.4f}`)")
                    st.caption(chunk["content"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chunks": chunks,
        "latency": latency
    })
