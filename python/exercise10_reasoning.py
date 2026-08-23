import openai
from foundry_local_sdk import Configuration, FoundryLocalManager

# 1. SDK ve Servisi Başlat
config = Configuration(app_name="FoundryLabDemo")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance
manager.start_web_service()

alias = "qwen2.5-0.5b"
catalog = manager.get_catalog() if hasattr(manager, "get_catalog") else manager.catalog
model = catalog.get_model(alias)

if not model.is_cached:
    model.download()
model.load()

# 2. OpenAI İstemcisi
base_url = manager.endpoint if hasattr(manager, "endpoint") else f"{manager.urls[0]}/v1"
client = openai.OpenAI(base_url=base_url, api_key="foundry-local")

messages = [
    {
        "role": "user",
        "content": "What is 17 * 23? Show your step-by-step thought process first, and conclude with 'Therefore, 17 times 23 equals ...'."
    }
]

print("--- İstek Gönderiliyor ---\n")
response = client.chat.completions.create(
    model=model.id,
    messages=messages,
    temperature=0.1
)

content = response.choices[0].message.content

# 3. Akıl Yürütme ve Sonucu Ayrıştırma (Parser Mantığı)
if "<think>" in content and "</think>" in content:
    think_start = content.index("<think>") + len("<think>")
    think_end = content.index("</think>")
    thinking = content[think_start:think_end].strip()
    answer = content[think_end + len("</think>"):].strip()
elif "Therefore," in content:
    parts = content.split("Therefore,")
    thinking = parts[0].strip()
    answer = "Therefore, " + parts[1].strip()
else:
    thinking = "Model adımları düz metin olarak üretti."
    answer = content

print("🧠 [Düşünce Süreci / Reasoning Process]:")
print(thinking)
print("\n" + "="*45 + "\n")
print("🎯 [Nihai Sonuç / Final Answer]:")
print(answer)
