import os
import sqlite3
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer

DB_FILE = "rag_storage.db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

FILES_TO_PROCESS = [
    {
        "path": "Microsoft_2024_Sustainability_Report.pdf",
        "year": "2024",
        "title": "2024 Sustainability Report",
    },
    {
        "path": "Microsoft_2025_Sustainability_Report.pdf",
        "year": "2025",
        "title": "2025 Sustainability Report",
    },
    {
        "path": "Microsoft_2026_Data_Fact_Sheet.pdf",
        "year": "2026",
        "title": "2026 Data Fact Sheet",
    },
]

def init_database(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS documents")
    cursor.execute("""
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            year TEXT NOT NULL,
            page INTEGER NOT NULL,
            type TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    """)
    conn.commit()
    return conn

def clean_cell_text(text: str | None) -> str:
    if not text:
        return ""
    return str(text).replace("\n", " ").replace("|", "/").strip()

def extract_chunks_from_pdf(file_meta: dict) -> list[tuple]:
    chunks = []
    file_path = file_meta["path"]
    title = file_meta["title"]
    year = file_meta["year"]

    if not os.path.exists(file_path):
        print(f"[WARNING] File not found: {file_path}. Skipping.")
        return chunks

    with pdfplumber.open(file_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_num = page_idx + 1

            # A. Extract Full Structured Tables
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue

                cleaned_table = [
                    [clean_cell_text(cell) for cell in row] for row in table
                ]
                headers = cleaned_table[0]
                if not any(headers):
                    headers = [f"Col_{i+1}" for i in range(len(headers))]

                md_table = (
                    "| " + " | ".join(headers) + " |\n| "
                    + " | ".join(["---"] * len(headers)) + " |\n"
                )
                for row in cleaned_table[1:]:
                    padded_row = (row + [""] * len(headers))[:len(headers)]
                    md_table += "| " + " | ".join(padded_row) + " |\n"

                header_prefix = (
                    f"--- Document: {title} (Report Year: {year}, Page: {page_num}) | "
                    f"Content Type: [STRUCTURED TABLE] ---\n"
                )
                chunks.append((title, year, page_num, "table", header_prefix + md_table))

            # B. Extract Narrative Text
            page_text = page.extract_text()
            if page_text:
                cleaned_text = " ".join(page_text.split())
                chunk_size = 600
                overlap = 150
                stride = chunk_size - overlap

                for i in range(0, len(cleaned_text), stride):
                    sub_text = cleaned_text[i : i + chunk_size].strip()
                    if len(sub_text) > 80:
                        header_prefix = (
                            f"--- Document: {title} (Report Year: {year}, Page: {page_num}) | "
                            f"Content Type: [TEXT NARRATIVE] ---\n"
                        )
                        chunks.append((title, year, page_num, "text", header_prefix + sub_text))

    return chunks

def main():
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    conn = init_database(DB_FILE)
    cursor = conn.cursor()
    total_chunks_processed = 0

    for file_meta in FILES_TO_PROCESS:
        print(f"\nProcessing document: {file_meta['title']}...")
        extracted_chunks = extract_chunks_from_pdf(file_meta)
        print(f"  -> Extracted {len(extracted_chunks)} valid chunks.")

        if not extracted_chunks:
            continue

        contents = [item[4] for item in extracted_chunks]
        print("  -> Generating vector embeddings in batch...")
        embeddings = model.encode(
            contents,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        insert_records = []
        for (title, year, page, c_type, content), emb in zip(extracted_chunks, embeddings, strict=True):
            emb_blob = emb.astype(np.float32).tobytes()
            insert_records.append((title, year, page, c_type, content, emb_blob))

        cursor.executemany("""
            INSERT INTO documents (title, year, page, type, content, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        """, insert_records)

        conn.commit()
        total_chunks_processed += len(extracted_chunks)

    conn.close()
    print(f"\n[SUCCESS] Ingestion completed. Total indexed chunks: {total_chunks_processed}")

if __name__ == "__main__":
    main()
