import sqlite3
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Yerel Veritabani Baglantisi ve Tablo Olusturma
DB_PATH = "rag_storage.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding TEXT NOT NULL
)
""")
conn.commit()

# 2. Embedding Modelini Yukle
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Ornek Metin Parcalari
sample_docs = [
    {
        "title": "Cat Habits",
        "content": "A cute cat is sleeping peacefully on the sofa and purring softly."
    },
    {
        "title": "Dog Training",
        "content": "The dog is running happily in the park and learning how to fetch the ball."
    },
    {
        "title": "AI & Embeddings",
        "content": "Natural language processing models convert raw text chunks into dense vector representations."
    }
]

# 4. Verileri ve Vektorleri Veritabanina Yaz
for doc in sample_docs:
    vector = model.encode(doc["content"]).tolist()
    vector_json = json.dumps(vector)
    cursor.execute(
        "INSERT INTO documents (title, content, embedding) VALUES (?, ?, ?)",
        (doc["title"], doc["content"], vector_json)
    )
conn.commit()

# 5. Veritabanindan Verileri Oku ve Dogrula
cursor.execute("SELECT id, title, content, embedding FROM documents")
rows = cursor.fetchall()

for row in rows:
    doc_id, title, content, raw_embedding_json = row
    loaded_vector = np.array(json.loads(raw_embedding_json))
    print(f"ID: {doc_id} | Baslik: {title}")
    print(f"Icerik: {content}")
    print(f"Vektor Boyutu: {loaded_vector.shape}\n")

conn.close()
