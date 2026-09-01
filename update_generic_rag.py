import re

# 1. Update ingest_all_reports.py with table_to_structured_repr
with open("ingest_all_reports.py", "r") as f:
    ingest_code = f.read()

new_table_func = '''def table_to_markdown(table: list[list[str]]) -> str:
    cleaned = []
    for row in table:
        cleaned_row = [re.sub(r'\\s+', ' ', str(cell or '')).strip() for cell in row]
        if any(cleaned_row):
            cleaned.append(cleaned_row)
    
    if len(cleaned) < 2:
        return ""
    
    headers = cleaned[0]
    md_header = "| " + " | ".join(headers) + " |"
    md_sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    md_body = ["| " + " | ".join(r) + " |" for r in cleaned[1:]]
    md_table = "\\n".join([md_header, md_sep] + md_body)
    
    kv_lines = ["\\n[Structured Row Mappings]:"]
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
            
    return md_table + "\\n" + "\\n".join(kv_lines)'''

ingest_code = re.sub(r'def table_to_markdown\(.*?\n(?=def extract_year)', new_table_func + '\n\n', ingest_code, flags=re.DOTALL)

with open("ingest_all_reports.py", "w") as f:
    f.write(ingest_code)

# 2. Update app.py with Universal System Prompt
with open("app.py", "r") as f:
    app_code = f.read()

universal_prompt = '''SYSTEM_PROMPT = \"\"\"You are a rigorous, production-grade Data Analyst AI.
Analyze the user query using ONLY the provided context.

Universal Tabular and Numerical Rules:
1. Matrix Accuracy: Match the exact metric row name with the exact year/category column. Never merge parent totals with subtotal categories.
2. Explicit Mapping: Use the provided [Structured Row Mappings] to verify row-to-column numerical pairs.
3. No Guesswork: State only the numbers explicitly presented in the tables or text.
4. Structured Output: When answering trend, multi-year, or cross-metric questions, summarize the data in a clear Markdown table or structured bullet points with exact units (e.g., mtCO2e, ML, %, MWh).
5. Provenance: Always reference the source document, year, and page header for each data point.\"\"\"'''

app_code = re.sub(r'SYSTEM_PROMPT = \"\"\".*?\"\"\"', universal_prompt, app_code, flags=re.DOTALL)

with open("app.py", "w") as f:
    f.write(app_code)

print("Ingestion and App scripts successfully updated with Universal Tabular Serializer!")
