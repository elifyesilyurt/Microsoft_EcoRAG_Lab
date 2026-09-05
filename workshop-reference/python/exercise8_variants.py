from foundry_local_sdk import Configuration, FoundryLocalManager

# 1. SDK'yı başlat
config = Configuration(app_name="FoundryLabDemo")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

# 2. Servisi başlat
manager.start_web_service()

alias = "qwen2.5-0.5b"
catalog = manager.get_catalog() if hasattr(manager, "get_catalog") else manager.catalog
model = catalog.get_model(alias)

# 3. Model ve Donanım Varyant Bilgilerini İncele
print("\n--- Donanım ve Model Varyant Bilgileri ---")
print(f"Model ID       : {getattr(model, 'id', getattr(model, 'model_id', 'Bilinmiyor'))}")
print(f"Takma Ad       : {model.alias}")
print(f"İndirildi mi   : {model.is_cached}")

# Seçilen varyant detayları
if hasattr(model, "_selected_variant") and model._selected_variant:
    variant = model._selected_variant
    print(f"Seçilen Varyant: {getattr(variant, 'id', 'Bilinmiyor')}")
    print(f"Cihaz Türü     : {getattr(variant, 'device_type', 'Otomatik/GPU')}")
    print(f"Sağlayıcı      : {getattr(variant, 'execution_provider', 'Varsayılan')}")
