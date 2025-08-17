# Kombin_Asistani

Kişisel kombin asistanı, gardırobunuzdaki öğeleri etiketleyerek saklar ve
ChatGPT tabanlı bir model yardımıyla belirttiğiniz etkinlik, zaman ve hava
durumuna göre kombin önerileri üretir.

## Kurulum

1. Gerekli paketleri yükleyin:

```bash
pip install -r requirements.txt
```

2. OpenAI API anahtarınızı ortam değişkeni olarak ayarlayın:

```bash
export OPENAI_API_KEY="sk-..."
```

3. **(İsteğe bağlı)** `kombin_assistant/trends.py` dosyasındaki URL'yi gerçek
moda trendleri sunan bir API ile değiştirin.

## Örnek Kullanım

```bash
python main.py
```

Bu komut, örnek bir gardırop oluşturur ve verilen açıklamaya göre kombin
önerisi yazdırır.
