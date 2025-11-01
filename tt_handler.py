import os
import logging
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM


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


def handle_tt_mode(tt_chain):
    print("Режим генерации технических требований (ТТ)")
    request = input("Введите описание объекта или требования для генерации ТТ: ")
    if request.lower().strip() == 'exit':
        return False
    try:
        tt = tt_chain.invoke(request)
        print("Сгенерированные ТТ:")
        with open("generated_tt.txt", "w", encoding="utf-8") as f:
            f.write(tt)
        print("ТТ также сохранены в 'generated_tt.txt'\n")
        logging.info(f"TT Generation - Request: {request} - TT length: {len(tt)}")
    except Exception as e:
        print(f"Ошибка: {e}\n")
        logging.error(f"TT Generation error for request: {request} - {e}")
    return True
