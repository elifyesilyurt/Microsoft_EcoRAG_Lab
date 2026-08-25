import openai
from foundry_local_sdk import Configuration, FoundryLocalManager

alias = "qwen2.5-0.5b"

# 1. Servisi Başlat ve Modeli Yükle
print("1. Foundry Local servisi başlatılıyor...")
config = Configuration(app_name="FoundryLocalWorkshop")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance
manager.start_web_service()

catalog = manager.get_catalog() if hasattr(manager, "get_catalog") else manager.catalog
model = catalog.get_model(alias)
if not model.is_cached:
    model.download()
model.load()

endpoint = manager.endpoint if hasattr(manager, "endpoint") else f"{manager.urls[0]}/v1"
client = openai.OpenAI(base_url=endpoint, api_key="foundry-local")

# 2. Egzersiz 1: Bellekteki Bilgi Tabanı
KNOWLEDGE_BASE = [
    {
        "title": "Foundry Local Overview",
        "content": (
            "Foundry Local brings the power of Azure AI Foundry to your local "
            "device without requiring an Azure subscription. It provides a local "
            "OpenAI-compatible HTTP endpoint."
        ),
    },
    {
        "title": "Supported Hardware",
        "content": (
            "Foundry Local automatically selects the best model variant for "
            "your hardware. If you have an Nvidia CUDA GPU it downloads the "
            "CUDA-optimized model. On Apple Silicon, it utilizes Metal acceleration."
        ),
    },
    {
        "title": "Installation",
        "content": (
            "On Windows install Foundry Local with winget install Microsoft.FoundryLocal. "
            "On macOS use brew install microsoft/foundrylocal/foundrylocal."
        ),
    }
]

# 3. Egzersiz 2: Basit Anahtar Kelime Örtüşmesi (Keyword Overlap Retrieval)
def retrieve(query: str, top_k: int = 2) -> list:
    query_words = set(query.lower().split())
    scored = []
    for chunk in KNOWLEDGE_BASE:
        chunk_words = set(chunk["content"].lower().split())
        overlap = len(query_words & chunk_words)
        scored.append((overlap, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]

# 4. Soru ve Getirme
question = "How do I install Foundry Local and what hardware does it support?"
print(f"\nSoru: {question}\n")

context_chunks = retrieve(question, top_k=2)
context_text = "\n\n".join(
    f"### {c['title']}\n{c['content']}" for c in context_chunks
)

print("--- Getirilen Baglam (Retrieved Context) ---")
print(context_text)
print("-------------------------------------------\n")

# 5. Egzersiz 3: Dayanaklı Sistem Promptu ile Yanıt Üretme
system_prompt = (
    "You are a helpful assistant. Answer the user's question using ONLY "
    "the information provided in the context below. If the context does "
    "not contain enough information, say so.\n\n"
    f"Context:\n{context_text}"
)

response = client.chat.completions.create(
    model=alias,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ],
    temperature=0.0
)

print("Modelin Cevabi:")
print(response.choices[0].message.content)
