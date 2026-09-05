import openai
from foundry_local_sdk import Configuration, FoundryLocalManager

alias = "qwen2.5-0.5b"

# 1. Initialize Foundry Local Service
print("Connecting to Foundry Local service...")
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

# 2. Knowledge Base with Pricing Added
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
    },
    {
        "title": "Pricing",
        "content": "Foundry Local is completely free and open source under the MIT license.",
    }
]

def retrieve(query: str, top_k: int = 2) -> list:
    query_words = set(query.lower().split())
    scored = []
    for chunk in KNOWLEDGE_BASE:
        chunk_words = set(chunk["content"].lower().split())
        overlap = len(query_words & chunk_words)
        scored.append((overlap, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]

def ask_llm(system_prompt: str, user_question: str) -> str:
    response = client.chat.completions.create(
        model=alias,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content

print("\n=======================================================")
print("EXPERIMENT 1 & 2: Pricing Knowledge & Retrieval")
print("=======================================================")
q_pricing = "How much does Foundry Local cost?"
chunks_pricing = retrieve(q_pricing, top_k=1)
ctx_pricing = "\n\n".join(f"### {c['title']}\n{c['content']}" for c in chunks_pricing)

prompt_pricing = (
    "You are a helpful assistant. Answer the user's question using ONLY "
    "the information provided in the context below. If the context does "
    "not contain enough information, say so.\n\n"
    f"Context:\n{ctx_pricing}"
)
print(f"Question: {q_pricing}")
print(f"Retrieved Context:\n{ctx_pricing}")
print(f"Model Response:\n{ask_llm(prompt_pricing, q_pricing)}")

print("\n=======================================================")
print("EXPERIMENT 3: top_k=1 vs top_k=3 Comparison")
print("=======================================================")
q_complex = "How do I install Foundry Local and what hardware does it support?"
print(f"Question: {q_complex}\n")

# top_k = 1
chunks_k1 = retrieve(q_complex, top_k=1)
ctx_k1 = "\n\n".join(f"### {c['title']}\n{c['content']}" for c in chunks_k1)
print(f"--- [top_k = 1] Retrieved Titles: {[c['title'] for c in chunks_k1]} ---")

# top_k = 3
chunks_k3 = retrieve(q_complex, top_k=3)
ctx_k3 = "\n\n".join(f"### {c['title']}\n{c['content']}" for c in chunks_k3)
print(f"--- [top_k = 3] Retrieved Titles: {[c['title'] for c in chunks_k3]} ---")

prompt_k3 = (
    "You are a helpful assistant. Answer the user's question using ONLY "
    "the information provided in the context below. If the context does "
    "not contain enough information, say so.\n\n"
    f"Context:\n{ctx_k3}"
)
print(f"\n[top_k = 3] Model Response:\n{ask_llm(prompt_k3, q_complex)}")

print("\n=======================================================")
print("EXPERIMENT 4: Without Grounding Constraints (No Context)")
print("=======================================================")
q_unknown = "What is the warranty policy of Foundry Local?"
prompt_no_grounding = "You are a helpful assistant."

print(f"Question: {q_unknown}")
print("System Prompt: 'You are a helpful assistant.' (No Context)")
print(f"Model Response:\n{ask_llm(prompt_no_grounding, q_unknown)}")
