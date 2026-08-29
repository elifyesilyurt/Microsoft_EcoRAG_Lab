import os
import glob
import sqlite3
import json
import re
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer

DB_PATH = "rag_storage.db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DOCS_DIR = "docs"

def split_into_sentences(text: str) -> list:
    clean_text = " ".join(text.split())
    sentences = re.split(r'(?<=[.!?])\s+', clean_text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]

def semantic_chunking(sentences: list, embed_model: SentenceTransformer, similarity_threshold: float = 0.50, max_chunk_len: int = 750) -> list:
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences

    embeddings = embed_model.encode(sentences, show_progress_bar=False)
    chunks = []
    current_chunk = [sentences[0]]
    current_len = len(sentences[0])
    
    for i in range(len(sentences) - 1):
        vec_a = embeddings[i]
        vec_b = embeddings[i + 1]
        sim = float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-9))
        
        next_sentence = sentences[i + 1]
        next_len = len(next_sentence)
        
        if sim >= similarity_threshold and (current_len + next_len) <= max_chunk_len:
            current_chunk.append(next_sentence)
            current_len += next_len
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [next_sentence]
            current_len = next_len
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def table_to_markdown(table: list) -> str:
    cleaned_rows = []
    for row in table:
        cleaned_row = [str(cell).strip().replace("\n", " ") if cell is not None else "" for cell in row]
        if any(cleaned_row):
            cleaned_rows.append(cleaned_row)
            
    if not cleaned_rows or len(cleaned_rows) < 2:
        return ""
        
    header = "| " + " | ".join(cleaned_rows[0]) + " |"
    divider = "| " + " | ".join(["---"] * len(cleaned_rows[0])) + " |"
    body = ["| " + " | ".join(r) + " |" for r in cleaned_rows[1:]]
    return "\n".join([header, divider] + body)

def extract_year_from_filename(filename: str) -> int:
    match = re.search(r"202[0-9]", filename)
    return int(match.group(0)) if match else 2024

def run_hybrid_ingest():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        
    print(f"[INFO] Sifirdan temiz veritabani olusturuluyor: {DB_PATH}")
    print(f"[INFO] Yerel Embedding modeli yukleniyor: {EMBEDDING_MODEL_NAME}...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER,
            title TEXT,
            content TEXT,
            embedding TEXT
        )
    """)
    
    pdf_files = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    if not pdf_files:
        print("[ERROR] docs/ klasorunde PDF bulunamadi!")
        return

    print(f"[INFO] Toplam {len(pdf_files)} adet rapor (Tablo + Semantik Metin) taraniyor...")
    total_chunks = 0

    for pdf_path in pdf_files:
        doc_name = os.path.basename(pdf_path)
        doc_year = extract_year_from_filename(doc_name)
        print(f"\n--> Isleniyor [{doc_year} Raporu]: {doc_name}")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                print(f"    Toplam Sayfa: {total_pages}")
                
                for page_num, page in enumerate(pdf.pages):
                    # 1. Tablolari cikar ve Markdown olarak ekle
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            md_table = table_to_markdown(table)
                            if len(md_table) > 40:
                                vector = embed_model.encode(md_table, show_progress_bar=False).tolist()
                                title = f"{doc_name} [TABLO] (Yil: {doc_year}, s.{page_num + 1})"
                                cursor.execute(
                                    "INSERT INTO documents (year, title, content, embedding) VALUES (?, ?, ?, ?)",
                                    (doc_year, title, md_table, json.dumps(vector))
                                )
                                total_chunks += 1
                                
                    # 2. Metin paragraflarini semantik olarak ekle
                    text = page.extract_text()
                    if text and len(text.strip()) > 30:
                        sentences = split_into_sentences(text)
                        chunks = semantic_chunking(sentences, embed_model)
                        for chunk in chunks:
                            vector = embed_model.encode(chunk, show_progress_bar=False).tolist()
                            title = f"{doc_name} (Yil: {doc_year}, s.{page_num + 1})"
                            cursor.execute(
                                "INSERT INTO documents (year, title, content, embedding) VALUES (?, ?, ?, ?)",
                                (doc_year, title, chunk, json.dumps(vector))
                            )
                            total_chunks += 1
                            
                    if (page_num + 1) % 25 == 0 or (page_num + 1) == total_pages:
                        print(f"    Islendi: {page_num + 1}/{total_pages} sayfa ({total_chunks} toplam chunk)...")
        except Exception as e:
            print(f"[ERROR] Hata ({doc_name}): {e}")

    conn.commit()
    conn.close()
    print(f"\n[SUCCESS] Toplam {total_chunks} adet (Tablo + Metin) parca SQLite veritabanina kaydedildi.")

if __name__ == "__main__":
    run_hybrid_ingest()
