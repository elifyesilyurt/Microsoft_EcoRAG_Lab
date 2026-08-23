from foundry_local_sdk import Configuration, FoundryLocalManager

alias = "qwen2.5-0.5b"

print("--- Hızlı Başlangıç (Bootstrap) Başlıyor ---")

# 1. Konfigürasyonu tanımla ve SDK'yı başlat
config = Configuration(app_name="FoundryLabDemo")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

# 2. Servisi çalıştır
manager.start_web_service()

# 3. Modeli bul, yoksa indir ve belleğe al (Tek akış)
catalog = manager.get_catalog() if hasattr(manager, "get_catalog") else manager.catalog
model = catalog.get_model(alias)

if not model.is_cached:
    print(f"-> Model indiriliyor: {alias}")
    model.download()

print("-> Model belleğe alınıyor...")
model.load()

# 4. Servis ve model bilgilerini yazdır
endpoint = manager.endpoint if hasattr(manager, "endpoint") else manager.urls[0]
print(f"\nEndpoint : {endpoint}")
print(f"Model ID : {model.id}")
print("\n-> Model başarıyla hazırlandı ve kullanıma açıldı!")
