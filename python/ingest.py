import os
import glob
import sqlite3
import json
from sentence_transformers import SentenceTransformer

DB_PATH = "rag_storage.db"
DATA_DIR = "data"
CHUNK_SIZE = 150

model = SentenceTransformer("all-MiniLM-L6-v2")

def split_text_into_chunks(text, max_words=CHUNK_SIZE):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS documents")
cursor.execute("""
CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding TEXT NOT NULL
)
""")
conn.commit()

txt_files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
total_chunks_indexed = 0

for file_path in txt_files:
    filename = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        file_content = f.read()

    chunks = split_text_into_chunks(file_content)
    
    for idx, chunk in enumerate(chunks):
        vector = model.encode(chunk).tolist()
        vector_json = json.dumps(vector)
        doc_title = f"{filename}#part_{idx+1}"
        
        cursor.execute(
            "INSERT INTO documents (title, content, embedding) VALUES (?, ?, ?)",
            (doc_title, chunk, vector_json)
        )
        total_chunks_indexed += 1
        print(f"Indekslendi: {doc_title} ({len(chunk.split())} kelime)")

conn.commit()

cursor.execute("SELECT COUNT(*) FROM documents")
count = cursor.fetchone()[0]
conn.close()

print(f"\nToplam Eklenen Parca Sayisi: {count}")
