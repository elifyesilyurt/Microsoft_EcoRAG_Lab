from foundry_local_sdk import Configuration, FoundryLocalManager

# 1. Konfigürasyonu tanımla ve SDK tekil örneğini başlat
config = Configuration(app_name="FoundryLabDemo")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

# 2. Web servisini başlat
manager.start_web_service()

# 3. Kataloğu al ve modelleri listele
catalog = manager.get_catalog() if hasattr(manager, "get_catalog") else manager.catalog
models = catalog.list_models() if hasattr(catalog, "list_models") else catalog.get_models()

print(f"\nKatalogda toplam {len(models)} model bulundu:\n")

for model in models:
    alias = getattr(model, "alias", "Bilinmiyor")
    model_id = getattr(model, "id", getattr(model, "model_id", "Bilinmiyor"))
    print(f"  - Model: {alias} ({model_id})")

print("-" * 50)
