import os
import logging
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM


def setup_search_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    def format_docs(docs):
        formatted_docs = []
        for doc in docs:
            content = doc.page_content
            metadata = doc.metadata
            sections = metadata.get('sections', [])
            filename = metadata.get('filename', 'Unknown')

            # Clean up filename to get document designation
            # Remove .pdf extension and clean up
            doc_name = filename.replace('.pdf', '').replace('_', ' ').strip()

            # Add section references to the content
            if sections:
                section_refs = ", ".join(sections)
                content = f"[Разделы: {section_refs}] {content}"

            # Add document reference
            content = f"[Документ: {doc_name}]\n{content}"

            formatted_docs.append(content)

        return "\n\n".join(formatted_docs)

    llm = OllamaLLM(model="qwen3:8b", temperature=0.0)

    template = """Ты - СТРОГИЙ АНАЛИЗАТОР нормативных документов. Твоя задача - отвечать ТОЛЬКО на основе предоставленного контекста.

ПРАВИЛА РАБОТЫ (ОБЯЗАТЕЛЬНЫ К ВЫПОЛНЕНИЮ):
1. ИСПОЛЬЗУЙ ТОЛЬКО информацию из предоставленного контекста
2. ЕСЛИ в контексте НЕТ НИКАКОЙ релевантной информации - ответь: "Информация отсутствует в предоставленных нормативных документах"
3. ЕСЛИ в контексте ЕСТЬ релевантная информация - предоставь её с обязательными ссылками на источники
4. ВСЕГДА указывай источник: документ и раздел (например: СП 4.04.07-2025, п. 4.2.3)
5. НЕ добавляй НИКАКИХ собственных знаний, интерпретаций или внешней информации
6. НЕ отвечай на вопросы вне темы нормативных документов
7. ЦИТИРУЙ текст документа дословно, не обобщай
8. Максимум 500 слов

КОНТЕКСТ ИЗ ДОКУМЕНТОВ:
{context}

ВОПРОС: {question}

ОТВЕТ (только по контексту с обязательными ссылками):"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    search_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return search_chain


def handle_search_mode(search_chain):
    print("Режим поиска информации по нормативам")
    question = input("Введите вопрос по нормативным документам: ")
    if question.lower().strip() == 'exit':
        return False
    try:
        answer = search_chain.invoke(question)
        print(f"Ответ: {answer}\n")
        logging.info(f"Search - Query: {question} - Answer length: {len(answer)}")
    except Exception as e:
        print(f"Ошибка: {e}\n")
        logging.error(f"Search error for query: {question} - {e}")
    return True
