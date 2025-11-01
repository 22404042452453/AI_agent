import streamlit as st
import os
import fitz
import time
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# Функции как в main.py
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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  chunk_overlap=100, separators=["\n\n", "\n", ".", " ", ""]
    )
    texts = [doc['content'] for doc in documents]
    metadatas = [doc['metadata'] for doc in documents]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    all_chunks = []
    all_metadatas = []
    for i, text in enumerate(texts):
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append(metadatas[i])
    vectorstore = FAISS.from_texts(texts=all_chunks, embedding=embeddings, metadatas=all_metadatas)
    return vectorstore


def setup_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 7, "lambda_mult": 0.7})
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    llm = OllamaLLM(model="qwen3:8b", temperature=0.0)
    template = """Ты эксперт по корпоративным нормативным документам и стандартам.

Используй следующий контекст из документов для ответа на вопрос.

ВАЖНЫЕ ТРЕБОВАНИЯ:
1. Отвечай строго на основе предоставленного контекста из нормативных документов
2. Если информации нет в контексте, честно скажи об этом
3. Указывай номера разделов/пунктов документов, если они есть в контексте
4. Используй технические термины точно так, как они определены в нормативных документах
5. Для расчетов и формул приводи обозначения согласно стандарту
Для формул и расчетов используй ТОЛЬКО обычный текстовый формат:
   - НЕ используй LaTeX, MathML или другую математическую разметку
   - НЕ используй символы вида $, \\text, \\, и т.д.
   - Пиши формулы простым текстом
   - Используй обычные символы: +, -, *, /, =, >=, <=
6. Цитируй источники с указанием названия документа и страницы, если возможно
7. Отвечай полно и точно, максимум 300 слов.

Контекст из нормативных документов:
{context}

Вопрос: {question}

Ответ на русском языке (со ссылками на пункты документов, если возможно):"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return qa_chain


def setup_tt_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "lambda_mult": 0.5})
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    llm = OllamaLLM(model="qwen3:8b", temperature=0.2)
    template = """Ты инженер-технолог, специализирующийся на создании технических требований (ТТ) на основе нормативных документов.

На основе следующего контекста из нормативных документов создай технические требования для запроса инженера.

СТРУКТУРА ТЕХНИЧЕСКИХ ТРЕБОВАНИЙ:
1. Общие положения
2. Технические характеристики
3. Требования к материалам
4. Процесс производства/испытаний
5. Нормы контроля качества
6. Упаковка и маркировка
7. Ссылки на нормы

ВАЖНЫЕ ПРАВИЛА:
- Используй точные термины из контекста нормативных документов
- Ссылайся на конкретные пункты и разделы документов
- Если информации недостаточно, укажи это и используй общепринятые стандарты
- Будь конкретен и измеряем
- Формат: разделенный абзацами, с заголовками пунктов

Контекст из нормативных документов:
{context}

Запрос: {question}

Сгенерируй технические требования на русском языке:"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    tt_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return tt_chain


# Streamlit app
def clear_chat():
    st.session_state.messages = []
    if os.path.exists("chat_history.json"):
        os.remove("chat_history.json")

def main():
    st.title("Чат-система поиска и анализа нормативов")

    # Инициализация чата
    if "messages" not in st.session_state:
        if os.path.exists("chat_history.json"):
            with open("chat_history.json", "r", encoding="utf-8") as f:
                st.session_state.messages = json.load(f)
        else:
            st.session_state.messages = []

    # Sidebar for new chat and mode selection
    with st.sidebar:
        if st.button("Новый чат"):
            clear_chat()
            st.rerun()

        st.header("Режим")
        mode = st.radio(
            "Выберите режим:",
            ["Автоматично", "Поиск информации", "Генерация ТТ"],
            index=0
        )
        st.session_state.mode = mode

        st.header("История чатов")
        if st.session_state.messages:
            first_msg = st.session_state.messages[0]['content'][:50] + "..." if len(st.session_state.messages[0]['content']) > 50 else st.session_state.messages[0]['content']
            st.write(f"Текущий чат: {first_msg}")
        else:
            st.write("История пуста")

    # Загрузка векторного хранилища
    if "vectorstore" not in st.session_state or "tt_vectorstore" not in st.session_state:
        with st.spinner("Загрузка индексов документов..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            # Загрузка нормативного индекса
            if not os.path.exists("./faiss_index"):
                st.error("Индекс нормативных документов не найден. Сначала запустите main.py для создания индексов.")
                return
            normative_vectorstore = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)
            st.session_state.vectorstore = normative_vectorstore

            # Загрузка TT индекса
            if os.path.exists("./faiss_index_tt"):
                tt_vectorstore = FAISS.load_local("./faiss_index_tt", embeddings, allow_dangerous_deserialization=True)
                st.session_state.tt_vectorstore = tt_vectorstore
            else:
                st.session_state.tt_vectorstore = normative_vectorstore  # fallback
                st.warning("Индекс ТТ документов не найден. Используется нормативный индекс для ТТ.")

            st.session_state.qa_chain = setup_rag_chain(st.session_state.vectorstore)
            st.session_state.tt_chain = setup_tt_chain(st.session_state.tt_vectorstore)

    # Отображение сообщений чата
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def handle_submit():
        prompt = st.session_state.user_input
        if prompt:
            # Добавление сообщения пользователя
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Определение режима
            if st.session_state.mode == "Автоматично":
                is_tt_mode = (
                    prompt.strip().startswith('/tt') or
                    any(word in prompt.lower() for word in ['требования', 'ТТ', 'технические требования', 'генерировать тт'])
                )
            elif st.session_state.mode == "Генерация ТТ":
                is_tt_mode = True
            else:
                is_tt_mode = False

            chain = st.session_state.tt_chain if is_tt_mode else st.session_state.qa_chain
            mode_name = "Генерация ТТ" if is_tt_mode else "Поиск информации"

            # Генерация ответа
            with st.spinner(f"{mode_name}..."):
                try:
                    response = chain.invoke(prompt)
                except Exception as e:
                    response = f"Ошибка: {e}"

            # Typewriter effect for displaying response
            with st.chat_message("assistant"):
                placeholder = st.empty()
                for i in range(1, len(response) + 1):
                    placeholder.write(response[:i])
                    time.sleep(0.02)  # Speed of text appearance

            # Add message to session state after displaying
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Save chat history
            with open("chat_history.json", "w", encoding="utf-8") as f:
                json.dump(st.session_state.messages, f, ensure_ascii=False)

    # Интерфейс ввода чата
    prompt = st.chat_input("Введите ваш запрос... (для ТТ укажите 'ТТ' или начните с '/tt')", key="user_input", on_submit=handle_submit)

if __name__ == "__main__":
    main()
