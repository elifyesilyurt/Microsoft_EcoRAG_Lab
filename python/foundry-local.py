import openai

client = openai.OpenAI(
    base_url="http://127.0.0.1:54137/v1",
    api_key="foundry"
)

print("\n--- Yerel Qwen2.5-0.5B Yanıtı Başlıyor ---\n")

try:
    stream = client.chat.completions.create(
        model="qwen2.5-0.5b",
        messages=[
            {"role": "user", "content": "Give me 3 concrete tips to stay disciplined and focused."}
        ],
        temperature=0.7,
        max_tokens=300,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n\n--- Yanıt Tamamlandı ---\n")

except Exception as e:
    print(f"Hata: {e}")
