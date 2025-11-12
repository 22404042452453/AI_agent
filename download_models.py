from langchain_huggingface import HuggingFaceEmbeddings
import os

def download_embeddings():
    """Предварительное скачивание модели эмбеддингов"""
    print("🔄 Скачивание модели эмбеддингов sentence-transformers/paraphrase-multilingual-mpnet-base-v2...")
    print("Это может занять несколько минут при первом запуске...")

    try:
        # Инициализация модели эмбеддингов
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )

        # Принудительная загрузка модели через тестовое эмбеддинг
        test_text = ["тестовый текст для загрузки модели"]
        embeddings.embed_documents(test_text)

        print("✅ Модель эмбеддингов успешно скачана и готова к использованию!")
        print(f"📁 Модель сохранена в кэше: {os.path.expanduser('~/.cache/huggingface/hub')}")

    except Exception as e:
        print(f"❌ Ошибка при скачивании модели: {e}")
        print("Проверьте подключение к интернету и повторите попытку.")

if __name__ == "__main__":
    download_embeddings()
