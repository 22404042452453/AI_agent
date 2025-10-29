import os
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def load_pdfs_from_directory(directory_path):
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
                        'metadata': {'source': pdf_path, 'filename': file}
                    })
                    doc.close()
                except Exception as e:
                    print(f"Ошибка при загрузке {pdf_path}: {e}")
    return documents


def create_vectorstore(documents):
    # Разделение текста на чанки (меньшие для точности)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Размер чанка
        chunk_overlap=100,  # Перекрытие
        separators=["\n\n", "\n", ".", " ", ""]
    )

    texts = [doc['content'] for doc in documents]
    metadatas = [doc['metadata'] for doc in documents]

    # Создание эмбеддингов
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Разбиение документов
    all_chunks = []
    all_metadatas = []
    for i, text in enumerate(texts):
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append(metadatas[i])

    # Создание векторного хранилища
    vectorstore = FAISS.from_texts(
        texts=all_chunks,
        embedding=embeddings,
        metadatas=all_metadatas
    )

    return vectorstore


def setup_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "lambda_mult": 0.7})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Настройка Ollama модели
    llm = OllamaLLM(model="qwen3:8b", temperature=0.0)  # Температура 0 для точных ответов

    # Шаблон промпта
    template = """Ты эксперт по ГОСТ Р 58669-2019 "Единая энергетическая система и изолированно работающие энергосистемы. Релейная защита и автоматика. Трансформаторы тока измерительные. Требования к характеристикам намагничивания и методам их определения".

Используй следующий контекст из документа для ответа на вопрос.

ВАЖНЫЕ ТРЕБОВАНИЯ:
1. Отвечай строго на основе предоставленного контекста из ГОСТ
2. Если информации нет в контексте, честно скажи об этом
3. Указывай номера разделов/пунктов ГОСТа, если они есть в контексте
4. Используй технические термины точно так, как они определены в ГОСТе
5. Для расчетов и формул приводи обозначения согласно стандарту
Для формул и расчетов используй ТОЛЬКО обычный текстовый формат:
   - НЕ используй LaTeX, MathML или другую математическую разметку
   - НЕ используй символы вида $, \\text, \\, и т.д.
   - Пиши формулы простым текстом: "t = 9.7 мс", "I = U/R", "K = 1.5"
   - Используй обычные символы: *, /, +, -, =, >=, <=
   - Для степеней используй ^: "x^2" вместо x²
   - Для индексов используй нижнее подчеркивание: "I_ном" вместо Iном
6. Ограничение 200 токенов

ТЕМАТИКА ОТВЕТОВ:
- Характеристики намагничивания ТТ
- Методы определения точки насыщения
- Расчет коэффициента насыщения
- Требования к вторичной нагрузке
- Методики испытаний и измерений
- Погрешности и допуски
- Классы точности для релейной защиты

Контекст из ГОСТ Р 58669-2019:
{context}

Вопрос: {question}

Ответ на русском языке (со ссылками на пункты ГОСТа, если возможно):"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # Создание RAG цепочки с LCEL
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain


def main():
    # Путь к папке с документами
    docs_dir = "files"

    if not os.path.exists("./faiss_index"):
        print("Создание векторного хранилища...")
        documents = load_pdfs_from_directory(docs_dir)

        if not documents:
            print("Документы не найдены!")
            return

        vectorstore = create_vectorstore(documents)
        vectorstore.save_local("./faiss_index")
        print(f"Обработано {len(documents)} документов")
    else:
        print("Загрузка существующего векторного хранилища...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Настройка RAG
    qa_chain = setup_rag_chain(vectorstore)

    print("RAG-система готова! Введите вопрос или 'exit' для выхода:")

    while True:
        question = input("Вопрос: ")
        if question.lower().strip() == 'exit':
            break

        try:
            answer = qa_chain.invoke(question)
            print(f"Ответ: {answer}\n")
        except Exception as e:
            print(f"Ошибка: {e}\n")


if __name__ == "__main__":
    main()
