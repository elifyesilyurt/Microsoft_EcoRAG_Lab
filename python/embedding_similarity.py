import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Yerel Embedding Modelini Yükle
print("Embedding modeli yükleniyor...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Kosinüs Benzerliği Fonksiyonu
# Formül: (A . B) / (||A|| * ||B||)
def cosine_similarity(vec_a, vec_b):
    dot = np.dot(vec_a, vec_b)
    norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return float(dot / norm)

# 3. Belge Havuzumuz (Veritabanındaki metinler)
sentences = [
    "A cute cat is sleeping peacefully on the sofa.",
    "The dog is running happily in the park and playing fetch.",
    "Artificial intelligence and deep learning models process large datasets.",
    "Walking along the beach on a sunny summer day is relaxing.",
    "Natural language processing systems analyze text and generate embeddings."
]

# 4. Kullanıcı Sorgusu (Arama kutusuna yazılan)
query = "Pets having fun and resting at home"

print(f"\n--- Kullanıcı Sorgusu: '{query}' ---\n")

# 5. Vektörleri Üret (Embedding Çıkarımı)
query_vec = model.encode(query)
sentence_vecs = model.encode(sentences)

# 6. Benzerlik Skorlarını Hesapla
results = []
for sent, s_vec in zip(sentences, sentence_vecs):
    sim = cosine_similarity(query_vec, s_vec)
    results.append((sent, sim))

# En yüksek benzerlikten en düşüğe sırala
results.sort(key=lambda x: x[1], reverse=True)

print("Kosinüs Benzerliği Sıralaması:")
for rank, (sent, score) in enumerate(results, 1):
    print(f"{rank}. [Skor: {score:.4f}] -> {sent}")

print(f"\n🎯 En İlgili Eşleşme: '{results[0][0]}' (Benzerlik Skoru: {results[0][1]:.4f})")
