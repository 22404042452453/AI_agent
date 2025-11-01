import os
import logging
import fitz
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from search_handler import setup_search_chain, handle_search_mode
from tt_handler import setup_tt_chain, handle_tt_mode

logging.basicConfig(filename='activity.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_section_references(text):
    """Extract GOST section references from text chunk"""
    # More specific patterns for GOST sections
    section_patterns = [
        r'\b(\d+\.\d+\.\d+)\b',  # subsections like 4.2.3 (word boundaries)
        r'\b(\d+\.\d+)\b',       # sections like 4.2 (word boundaries)
        r'\b(\d+)\.\s',          # main sections like "4. " (followed by space)
    ]

    sections = []
    for pattern in section_patterns:
        matches = re.findall(pattern, text)
        sections.extend(matches)

    # Filter out obvious non-section numbers (like years, page numbers, etc.)
    filtered_sections = []
    for section in sections:
        # Skip if it looks like a year (19xx, 20xx)
        if re.match(r'^(19|20)\d{2}$', section):
            continue
        # Skip single digits that are likely not sections (too generic)
        if re.match(r'^\d$', section) and not re.search(r'\b' + re.escape(section) + r'\.\s', text):
            continue
        filtered_sections.append(section)

    # Remove duplicates and sort by hierarchy
    sections = list(set(filtered_sections))
    sections.sort(key=lambda x: [int(i) for i in x.split('.')])

    return sections


def load_documents_from_directory(directory_path):
    documents = []

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                try:
                    doc = fitz.open(pdf_path)
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    documents.append({
                        'content': text,
                        'metadata': {'source': pdf_path, 'filename': file, 'format': 'pdf'}
                    })
                    doc.close()
                    print(f"Загружен PDF файл: {file}")
                except Exception as e:
                    print(f"Ошибка при загрузке {pdf_path}: {e}")
            elif file.endswith('.txt'):
                txt_path = os.path.join(root, file)
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    documents.append({
                        'content': text,
                        'metadata': {'source': txt_path, 'filename': file, 'format': 'text'}
                    })
                    print(f"Загружен текстовый файл: {file}")
                except Exception as e:
                    print(f"Ошибка при загрузке {txt_path}: {e}")

    return documents


def create_vectorstore(documents):
    # Use separators optimized for GOST standards with section preservation
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=[
            r"\n\d+\.\d+\.\d+\.?\s",  # GOST subsections like 4.2.3.
            r"\n\d+\.\d+\.?\s",       # GOST sections like 4.2.
            r"\n\d+\.?\s",            # GOST main sections like 4.
            "\n\n",                  # Paragraphs
            "\n",                    # Lines
            ". ",                    # Sentences
            " ",                     # Words
            ""                       # Characters
        ]
    )
    texts = [doc['content'] for doc in documents]
    metadatas = [doc['metadata'] for doc in documents]
    # Try different embeddings model for Python 3.14 compatibility
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Warning: Could not load preferred embeddings model: {e}")
        print("Falling back to basic embeddings...")
        # Fallback to a more basic model
        embeddings = HuggingFaceEmbeddings(model_name="distilbert-base-uncased")
    all_chunks = []
    all_metadatas = []
    for i, text in enumerate(texts):
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            # Extract section references for this chunk
            sections = extract_section_references(chunk)
            chunk_metadata = metadatas[i].copy()
            chunk_metadata['sections'] = sections
            all_chunks.append(chunk)
            all_metadatas.append(chunk_metadata)
    vectorstore = FAISS.from_texts(texts=all_chunks, embedding=embeddings, metadatas=all_metadatas)
    return vectorstore


def create_indexes_only():
    """Create indexes without starting interactive mode"""
    # Нормативные документы
    docs_dir = "files"
    normative_vectorstore = None
    index_faiss = "./faiss_index/index.faiss"
    index_pkl = "./faiss_index/index.pkl"

    if not (os.path.exists(index_faiss) and os.path.exists(index_pkl)):
        print("Создание векторного хранилища нормативных документов...")
        documents = load_documents_from_directory(docs_dir)
        if not documents:
            print("Нормативные документы не найдены!")
            return False
        normative_vectorstore = create_vectorstore(documents)
        normative_vectorstore.save_local("./faiss_index")
        print(f"Обработано {len(documents)} нормативных документов")
    else:
        print("Векторное хранилище нормативных документов уже существует.")

    # TT документы
    tt_docs_dir = "files_TT"
    tt_vectorstore = None
    tt_index_faiss = "./faiss_index_tt/index.faiss"
    tt_index_pkl = "./faiss_index_tt/index.pkl"

    if os.path.exists(tt_docs_dir):
        if not (os.path.exists(tt_index_faiss) and os.path.exists(tt_index_pkl)):
            print("Создание векторного хранилища ТТ документов...")
            tt_documents = load_documents_from_directory(tt_docs_dir)
            if tt_documents:
                tt_vectorstore = create_vectorstore(tt_documents)
                tt_vectorstore.save_local("./faiss_index_tt")
                print(f"Обработано {len(tt_documents)} ТТ документов")
            else:
                print("ТТ документы не найдены в папке files_TT")
        else:
            print("Векторное хранилище ТТ документов уже существует.")
    else:
        print("Папка files_TT не найдена. ТТ индекс не создан.")

    return True


def main():
    import sys

    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--create-indexes":
        success = create_indexes_only()
        sys.exit(0 if success else 1)

    # Нормативные документы
    docs_dir = "files"
    normative_vectorstore = None
    if not os.path.exists("./faiss_index"):
        print("Создание векторного хранилища нормативных документов...")
        documents = load_documents_from_directory(docs_dir)
        if not documents:
            print("Нормативные документы не найдены!")
            return
        normative_vectorstore = create_vectorstore(documents)
        normative_vectorstore.save_local("./faiss_index")
        print(f"Обработано {len(documents)} нормативных документов")
    else:
        print("Загрузка существующего векторного хранилища нормативных документов...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        normative_vectorstore = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)

    # TT документы
    tt_docs_dir = "files_TT"
    tt_vectorstore = None
    if os.path.exists(tt_docs_dir):
        if not os.path.exists("./faiss_index_tt"):
            print("Создание векторного хранилища ТТ документов...")
            tt_documents = load_documents_from_directory(tt_docs_dir)
            if tt_documents:
                tt_vectorstore = create_vectorstore(tt_documents)
                tt_vectorstore.save_local("./faiss_index_tt")
                print(f"Обработано {len(tt_documents)} ТТ документов")
            else:
                print("ТТ документы не найдены в папке files_TT")
                tt_vectorstore = normative_vectorstore  # fallback to normative
        else:
            print("Загрузка существующего векторного хранилища ТТ документов...")
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            tt_vectorstore = FAISS.load_local("./faiss_index_tt", embeddings, allow_dangerous_deserialization=True)
    else:
        print("Папка files_TT не найдена. Используем нормативные документы для ТТ.")
        tt_vectorstore = normative_vectorstore

    # Настройка цепочек через модули
    search_chain = setup_search_chain(normative_vectorstore)
    tt_chain = setup_tt_chain(tt_vectorstore)

    print("Система готова!")
    print("Выберите режим:")
    print("1. Поиск информации по нормативам")
    print("2. Генерация технических требований (ТТ)")
    print("Для выхода введите 'exit' в любое время")

    while True:
        choice = input("Ваш выбор (1 или 2): ").strip()
        if choice.lower() == 'exit':
            break
        elif choice == '1':
            should_continue = handle_search_mode(search_chain)
            if not should_continue:
                break
        elif choice == '2':
            should_continue = handle_tt_mode(tt_chain)
            if not should_continue:
                break
        else:
            print("Неверный выбор. Введите 1 или 2.\n")


if __name__ == "__main__":
    main()
