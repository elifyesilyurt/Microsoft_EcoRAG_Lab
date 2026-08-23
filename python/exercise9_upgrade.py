from foundry_local_sdk import Configuration, FoundryLocalManager

# 1. SDK'yı başlat
config = Configuration(app_name="FoundryLabDemo")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

# 2. Servisi çalıştır
manager.start_web_service()

alias = "qwen2.5-0.5b"

# 3. Kataloğu sunucudan yenile
print("1. Model kataloğu güncelleniyor (refresh)...")
if hasattr(manager, "refresh_catalog"):
    manager.refresh_catalog()
else:
    catalog = manager.get_catalog() if hasattr(manager, "get_catalog") else manager.catalog
    if hasattr(catalog, "refresh"):
        catalog.refresh()
print("-> Katalog başarıyla yenilendi.")

# 4. Model güncelleme kontrolü
print(f"\n2. '{alias}' modeli için sürüm kontrolü yapılıyor...")
is_upgradeable = False
if hasattr(manager, "is_model_upgradeable"):
    is_upgradeable = manager.is_model_upgradeable(alias)
elif hasattr(manager, "catalog"):
    model = manager.catalog.get_model(alias)
    is_upgradeable = getattr(model, "is_upgradeable", False)

if is_upgradeable:
    print(f"-> {alias} için yeni bir sürüm mevcut! Güncelleniyor...")
    if hasattr(manager, "upgrade_model"):
        manager.upgrade_model(alias)
    print("-> Güncelleme tamamlandı.")
else:
    print(f"-> {alias} modeli zaten en güncel sürümde (up to date).")

