import sqlite3
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Configuration
DB_PATH = "rag_storage.db"
MODEL_NAME = "all-MiniLM-L6-v2"

# 2. Load the Embedding Model
model = SentenceTransformer(MODEL_NAME)

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Calculates cosine similarity between two 1D vectors."""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))

def get_top_chunks(query: str, top_k: int = 2) -> list:
    """
    Encodes the query and retrieves the top-k most relevant 
    chunks from SQLite storage based on semantic similarity.
    """
    # Step A: Generate embedding for the query (1 model execution)
    query_vector = model.encode(query)
    
    # Step B: Fetch stored document vectors from database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, content, embedding FROM documents")
    rows = cursor.fetchall()
    conn.close()
    
    # Step C: Compute similarity score for each document chunk
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
    
    # Step D: Sort by highest score and take top_k
    scored_chunks.sort(key=lambda item: item["score"], reverse=True)
    return scored_chunks[:top_k]

if __name__ == "__main__":
    # Test Queries
    test_queries = [
        "How do vector embeddings work for search?",
        "Can I run AI models on my local workstation without cloud?",
        "What is the role of context retrieval in generation?"
    ]
    
    for q in test_queries:
        print("=" * 60)
        print(f"Query: {q}")
        print("=" * 60)
        results = get_top_chunks(q, top_k=2)
        
        for rank, res in enumerate(results, start=1):
            print(f"Rank {rank} | Score: {res['score']:.4f} | Document: {res['title']}")
            print(f"Snippet: {res['content'][:120]}...\n")
