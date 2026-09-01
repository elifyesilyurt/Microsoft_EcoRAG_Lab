import os
import glob
import re
import json
import sqlite3
import pdfplumber
from sentence_transformers import SentenceTransformer

DB_PATH = "rag_storage.db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

print(f"Loading embedding model ({EMBEDDING_MODEL_NAME})...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

def format_chunk_with_metadata(doc_title: str, year: str, page_num: int, content: str, is_table: bool = False) -> str:
    tag = "[TABLE DATA]" if is_table else "[TEXT SECTION]"
    header = f"--- Document: {doc_title} (Year: {year}, Page: {page_num}) | Type: {tag} ---\n"
    return header + content.strip()

def table_to_structured_repr(table: list[list[str]]) -> str:
    """Converts 2D table into both Markdown format and explicit Key-Value row mappings."""
    cleaned = []
    for row in table:
        cleaned_row = [re.sub(r'\s+', ' ', str(cell or '')).strip() for cell in row]
        if any(cleaned_row):
            cleaned.append(cleaned_row)
    
    if len(cleaned) < 2:
        return ""
    
    headers = cleaned[0]
    md_header = "| " + " | ".join(headers) + " |"
    md_sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    md_body = ["| " + " | ".join(r) + " |" for r in cleaned[1:]]
    md_table = "\n".join([md_header, md_sep] + md_body)
    
    kv_lines = ["\n[Structured Row Mappings]:"]
    for row in cleaned[1:]:
        row_title = row[0] if len(row) > 0 else "Metric"
        if not row_title:
            continue
        row_details = []
        for h, val in zip(headers[1:], row[1:]):
            if val:
                row_details.append(f"{h}: {val}")
        if row_details:
            kv_lines.append(f"- **{row_title}** -> ({', '.join(row_details)})")
            
    return md_table + "\n" + "\n".join(kv_lines)

def extract_year_from_filename(filename: str) -> str:
    match = re.search(r'20\d{2}', filename)
    return match.group(0) if match else "Unknown"

def process_pdf(pdf_path: str):
    doc_title = os.path.basename(pdf_path)
    year = extract_year_from_filename(doc_title)
    chunks = []

    print(f"Processing: {doc_title} (Year: {year})...")
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            # 1. Extract tables with row-centric mappings
            tables = page.extract_tables()
            for table in tables:
                structured_table = table_to_structured_repr(table)
                if structured_table:
                    chunk = format_chunk_with_metadata(doc_title, year, page_idx, structured_table, is_table=True)
                    chunks.append({"year": year, "title": f"{doc_title} (p.{page_idx})", "content": chunk})

            # 2. Extract text sections
            text = page.extract_text()
            if text:
                paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
                for p in paragraphs:
                    chunk = format_chunk_with_metadata(doc_title, year, page_idx, p, is_table=False)
                    chunks.append({"year": year, "title": f"{doc_title} (p.{page_idx})", "content": chunk})

    return chunks

def run_ingestion():
    pdf_files = sorted(glob.glob("docs/*.pdf"))
    if not pdf_files:
        print("No PDF files found in docs/ directory!")
        return

    print(f"Found {len(pdf_files)} PDF file(s): {[os.path.basename(p) for p in pdf_files]}")

    all_chunks = []
    for pdf_path in pdf_files:
        all_chunks.extend(process_pdf(pdf_path))

    print(f"\nTotal extracted chunks (Tables + Text): {len(all_chunks)}")
    print("Writing to database and generating embeddings...")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS documents")
    cursor.execute("""
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year TEXT,
            title TEXT,
            content TEXT,
            embedding TEXT
        )
    """)
    conn.commit()

    for idx, c in enumerate(all_chunks, start=1):
        vec = embedder.encode(c["content"]).tolist()
        cursor.execute(
            "INSERT INTO documents (year, title, content, embedding) VALUES (?, ?, ?, ?)",
            (c["year"], c["title"], c["content"], json.dumps(vec))
        )
        if idx % 100 == 0 or idx == len(all_chunks):
            print(f"  Indexed {idx}/{len(all_chunks)} chunks...")

    conn.commit()
    conn.close()
    print("\nIngestion completed successfully! Database rag_storage.db is ready.")

if __name__ == "__main__":
    run_ingestion()
