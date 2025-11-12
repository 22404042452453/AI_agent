from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM


def create_search_chain(vectorstore, **kwargs):
    """Создает цепочку для строгого поиска по нормативным документам"""
    # Параметры по умолчанию
    defaults = {
        "search_type": "mmr",
        "k": 6,
        "fetch_k": 60,
        "lambda_mult": 0.8,
        "model": "qwen3:8b",
        "temperature": 0.0
    }
    defaults.update(kwargs)

    retriever = vectorstore.as_retriever(
        search_type=defaults["search_type"],
        search_kwargs={"k": defaults["k"], "fetch_k": defaults["fetch_k"], "lambda_mult": defaults.get("lambda_mult", 0.5)}
    )

    def format_docs(docs):
        formatted_docs = []
        for doc in docs:
            content = doc.page_content
            metadata = doc.metadata
            sections = metadata.get('sections', [])
            filename = metadata.get('filename', 'Unknown')

            # Clean up filename to get document designation
            doc_name = filename.replace('.pdf', '').replace('_', ' ').strip()

            # Add section references to the content
            if sections:
                section_refs = ", ".join(sections)
                content = f"[Разделы: {section_refs}] {content}"

            # Add document reference
            content = f"[Документ: {doc_name}]\n{content}"

            formatted_docs.append(content)

        return "\n\n".join(formatted_docs)

    llm = OllamaLLM(model=defaults["model"], temperature=defaults["temperature"])

    template = """Ты - ПРЕЦИЗИОННЫЙ АНАЛИЗАТОР нормативных документов с максимальной точностью и релевантностью. Твоя задача - предоставлять ТОЛЬКО релевантную информацию из контекста, строго отвечая на вопрос.

КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА (НАРУШЕНИЕ НЕДОПУСТИМО):
1. ИСПОЛЬЗУЙ ТОЛЬКО информацию из предоставленного контекста, которая НАПРЯМУЮ относится к вопросу
2. ИСКЛЮЧИ любую информацию из контекста, которая не отвечает на поставленный вопрос
3. ЕСЛИ в контексте НЕТ информации, релевантной вопросу - ОБЯЗАТЕЛЬНО ответь: "Информация отсутствует в предоставленных нормативных документах"
4. ЕСЛИ релевантная информация ЕСТЬ - цитируй ТОЛЬКО её с ОБЯЗАТЕЛЬНЫМИ ссылками на источники
5. ВСЕГДА указывай: документ, раздел/пункт (например: СП 4.04.07-2025, п. 4.2.3)
6. ЗАПРЕЩЕНО добавлять постороннюю информацию из контекста, которая не относится к вопросу
7. ЗАПРЕЩЕНО добавлять собственные знания, интерпретации, выводы или внешнюю информацию
8. ЗАПРЕЩЕНО отвечать на вопросы вне темы нормативных документов
9. ЦИТИРУЙ текст документа СЛОВО В СЛОВО, без обобщений или сокращений
10. ПРОВЕРЯЙ каждый факт на соответствие вопросу и контексту перед ответом
11. ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ - ЗАПРЕЩЕНЫ ответы на английском или других языках
12. ЗАПРЕЩЕНО добавлять в ответ текст о скачивании, экспорте или форматировании в Word - это обрабатывается системой автоматически
13. Максимум 500 слов, но лучше меньше если точнее

КОНТЕКСТ ИЗ ДОКУМЕНТОВ:
{context}

ВОПРОС: {question}

ТОЧНЫЙ И РЕЛЕВАНТНЫЙ ОТВЕТ НА РУССКОМ ЯЗЫКЕ (только по контексту с обязательными ссылками):"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    search_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return search_chain


def create_rag_chain(vectorstore, **kwargs):
    """Создает цепочку RAG для экспертного анализа нормативных документов"""
    # Параметры по умолчанию
    defaults = {
        "search_type": "mmr",
        "k": 8,
        "lambda_mult": 0.8,
        "model": "qwen3:8b",
        "temperature": 0.0
    }
    defaults.update(kwargs)

    retriever = vectorstore.as_retriever(
        search_type=defaults["search_type"],
        search_kwargs={"k": defaults["k"], "lambda_mult": defaults["lambda_mult"]}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    llm = OllamaLLM(model=defaults["model"], temperature=defaults["temperature"])

    template = """Ты ПРЕЦИЗИОННЫЙ ЭКСПЕРТ по корпоративным нормативным документам с максимальной точностью и релевантностью ответов.

КРИТИЧЕСКИ ВАЖНЫЕ ТРЕБОВАНИЯ К ТОЧНОСТИ И РЕЛЕВАНТНОСТИ:
1. Отвечай ТОЛЬКО на основе информации из контекста, которая НАПРЯМУЮ относится к вопросу
2. ИСКЛЮЧИ всю информацию из контекста, которая не отвечает на поставленный вопрос
3. Если релевантной информации НЕТ в контексте - ОБЯЗАТЕЛЬНО скажи: "Информация отсутствует в предоставленных документах"
4. Если релевантная информация ЕСТЬ - цитируй ТОЛЬКО её с ОБЯЗАТЕЛЬНЫМИ ссылками
5. ОБЯЗАТЕЛЬНО указывай номера разделов/пунктов документов при каждой ссылке
6. Используй технические термины ТОЧНО так, как они определены в документах
7. Для расчетов и формул - копируй обозначения дословно из стандарта
Формулы ТОЛЬКО в текстовом формате:
   - НЕ используй LaTeX, MathML или символы $, \\text, \\ и т.д.
   - Пиши формулы простым текстом: +, -, *, /, =, >=, <=
8. ЗАПРЕЩЕНО добавлять постороннюю информацию из контекста, которая не относится к вопросу
9. Цитируй источники с точными ссылками на документы и разделы
10. ПРОВЕРЯЙ соответствие каждого утверждения вопросу и контексту перед ответом
11. ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ - ЗАПРЕЩЕНЫ ответы на английском или других языках
12. ЗАПРЕЩЕНО добавлять в ответ текст о скачивании, экспорте или форматировании в Word - это обрабатывается системой автоматически
13. Отвечай максимально точно, минимум 200 слов, максимум 400 слов.

Контекст из нормативных документов:
{context}

Вопрос: {question}

ТОЧНЫЙ И РЕЛЕВАНТНЫЙ ОТВЕТ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ (с обязательными ссылками на пункты документов):"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return rag_chain


def create_tt_chain(vectorstore, **kwargs):
    """Создает цепочку для генерации технических требований (ТТ)"""
    # Параметры по умолчанию
    defaults = {
        "search_type": "mmr",
        "k": 10,
        "lambda_mult": 0.5,
        "model": "qwen3:8b",
        "temperature": 0.2
    }
    defaults.update(kwargs)

    retriever = vectorstore.as_retriever(
        search_type=defaults["search_type"],
        search_kwargs={"k": defaults["k"], "lambda_mult": defaults["lambda_mult"]}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    llm = OllamaLLM(model=defaults["model"], temperature=defaults["temperature"])

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

КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:
- ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ - ЗАПРЕЩЕНЫ ответы на английском или других языках
- ЗАПРЕЩЕНО добавлять в ответ текст о скачивании, экспорте или форматировании в Word - это обрабатывается системой автоматически
- Используй точные термины из контекста нормативных документов
- Ссылайся на конкретные пункты и разделы документов
- Если информации недостаточно, укажи это и используй общепринятые стандарты
- Будь конкретен и измеряем
- Формат: разделенный абзацами, с заголовками пунктов

Контекст из нормативных документов:
{context}

Запрос: {question}

Сгенерируй технические требования ТОЛЬКО НА РУССКОМ ЯЗЫКЕ:"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    tt_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return tt_chain
