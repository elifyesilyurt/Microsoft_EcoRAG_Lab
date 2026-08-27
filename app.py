import streamlit as st
import sqlite3
import json
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from foundry_local_sdk import Configuration, FoundryLocalManager

# Streamlit Sayfa Yapılandırması
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
    
    return response.choices[0].message.content, retrieved_chunks


# UI Düzeni
st.title("🤖 Foundry Local RAG Assistant")
st.caption("100% Yerel Çalışan Vektör Tabanlı Soru-Cevap Motoru")

col_main, col_context = st.columns([3, 2])

with col_main:
    user_query = st.text_input("Sorunuzu girin:", placeholder="Örn: How does Microsoft Foundry Local ensure privacy?")
    top_k = st.slider("Alınacak en iyi belge parça sayısı (Top-K):", min_value=1, max_value=4, value=2)
    ask_button = st.button("Yanıt Üret", type="primary")

    if ask_button and user_query:
        with st.spinner("Vektör araması yapılıyor ve yerel model yanıtı üretiyor..."):
            answer, chunks = answer_query(user_query, top_k=top_k)
            st.session_state["last_answer"] = answer
            st.session_state["last_chunks"] = chunks

    if "last_answer" in st.session_state:
        st.subheader("Model Yanıtı")
        st.write(st.session_state["last_answer"])

with col_context:
    st.subheader("Kullanılan Kaynak Bağlamlar (Chunks)")
    if "last_chunks" in st.session_state:
        for chunk in st.session_state["last_chunks"]:
            with st.expander(f"📄 {chunk['title']} (Benzerlik: {chunk['score']:.4f})"):
                st.write(chunk["content"])
    else:
        st.info("Henüz bir soru sorulmadı. Sorgu yapıldığında getirilen doküman parçaları burada listelenecek.")
