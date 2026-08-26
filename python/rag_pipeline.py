import sqlite3
import json
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from foundry_local_sdk import Configuration, FoundryLocalManager

# 1. Configuration and Global Initialization
DB_PATH = "rag_storage.db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_ALIAS = "qwen2.5-0.5b"

print("1. Loading embedding model...")
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("2. Initializing Foundry Local LLM service...")
config = Configuration(app_name="FoundryLocalWorkshop")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance
manager.start_web_service()

catalog = manager.get_catalog() if hasattr(manager, "get_catalog") else manager.catalog
llm = catalog.get_model(LLM_ALIAS)
if not llm.is_cached:
    llm.download()
llm.load()

endpoint = manager.endpoint if hasattr(manager, "endpoint") else f"{manager.urls[0]}/v1"
client = openai.OpenAI(base_url=endpoint, api_key="foundry-local")


# 2. Vector Retrieval Functions
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


# 3. End-to-End RAG Query Engine
def answer_query(user_question: str, top_k: int = 2) -> str:
    # Step A: Retrieve relevant context
    retrieved_chunks = get_top_chunks(user_question, top_k=top_k)
    context_text = "\n\n".join(
        f"### {chunk['title']}\n{chunk['content']}" for chunk in retrieved_chunks
    )
    
    # Step B: Construct strictly grounded system prompt
    system_prompt = (
        "You are a helpful and accurate technical assistant. "
        "Answer the user's question using ONLY the provided context below. "
        "If the information is not present in the context, explicitly state: "
        "'This information is not available in the documents.'\n\n"
        f"Context:\n{context_text}"
    )
    
    # Step C: Generate grounded completion via Foundry Local
    response = client.chat.completions.create(
        model=LLM_ALIAS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ],
        temperature=0.0
    )
    
    return response.choices[0].message.content, retrieved_chunks


if __name__ == "__main__":
    test_questions = [
        "How does Microsoft Foundry Local ensure privacy and latency?",
        "What is the mathematical concept behind vector embeddings?",
        "What is the price of the enterprise tier of Foundry Local?"  # Should trigger 'not available'
    ]
    
    print("\n" + "=" * 60)
    print("STARTING END-TO-END RAG VALIDATION")
    print("=" * 60)
    
    for q in test_questions:
        print(f"\n[User Question]: {q}")
        answer, chunks = answer_query(q, top_k=2)
        print(f"[Used Context Sources]: {[c['title'] for c in chunks]}")
        print(f"[Assistant Answer]:\n{answer}")
        print("-" * 60)
