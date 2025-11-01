import os
from main import load_documents_from_directory

docs = load_documents_from_directory('files')
print(f'Найдено документов: {len(docs)}')
for doc in docs:
    print(f'Файл: {doc["metadata"]["filename"]}, Формат: {doc["metadata"]["format"]}, Длина: {len(doc["content"])} символов')
