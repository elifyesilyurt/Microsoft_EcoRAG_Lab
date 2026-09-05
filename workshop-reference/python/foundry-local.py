import openai
from foundry_local_sdk import Configuration, FoundryLocalManager

alias = "qwen2.5-0.5b"

# Adım 1: Foundry Local servisini başlat
print("Starting Foundry Local service...")
config = Configuration(app_name="FoundryLocalWorkshop")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance
manager.start_web_service()

# Adım 2: Modeli katalogdan al ve önbelleği kontrol et
catalog = manager.get_catalog() if hasattr(manager, "get_catalog") else manager.catalog
model = catalog.get_model(alias)

if model.is_cached:
    print(f"Model already downloaded: {alias}")
else:
    print(f"Downloading model: {alias}...")
    model.download()
    print(f"Download complete: {alias}")

# Adım 3: Modeli belleğe yükle
print(f"Loading model: {alias}...")
model.load()

# Dinamik uç noktaya bağlanan OpenAI istemcisi oluştur
endpoint = manager.endpoint if hasattr(manager, "endpoint") else f"{manager.urls[0]}/v1"
client = openai.OpenAI(
    base_url=endpoint,
    api_key="foundry-local"
)

# Egzersiz 2 & 3: Sistem istemi ile Akışlı (Streaming) Sohbet Tamamlama
print("\n--- Akışlı Sohbet Yanıtı (Pirate Persona) ---\n")
stream = client.chat.completions.create(
    model=model.id,
    messages=[
        {"role": "system", "content": "You are a pirate. Answer everything in pirate speak."},
        {"role": "user", "content": "What is the golden ratio?"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)

print("\n\n--- Yanıt Tamamlandı ---")
