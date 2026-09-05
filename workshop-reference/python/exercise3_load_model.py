from foundry_local_sdk import Configuration, FoundryLocalManager

# 1. SDK'yı başlat
config = Configuration(app_name="FoundryLabDemo")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

# 2. Servisi aç ve modeli katalogdan al
manager.start_web_service()
catalog = manager.get_catalog() if hasattr(manager, "get_catalog") else manager.catalog

alias = "qwen2.5-0.5b"
model = catalog.get_model(alias)

print(f"\nModel: {model.alias}")
print(f"Önbellekte (Diskte) var mı?: {model.is_cached}")

# 3. Diskte yoksa önce indir
if not model.is_cached:
    print(f"\n-> Model diskte yok, indiriliyor (download)...")
    model.download()
    print("-> İndirme tamamlandı!")

# 4. Şimdi belleğe yükle
print(f"\n-> Model belleğe yükleniyor (load)...")
model.load()
print(f"-> Model başarıyla belleğe yüklendi ve hazır!")
