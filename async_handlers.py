import os
import logging
import concurrent.futures
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

# Глобальный пул потоков для обработки запросов
REQUEST_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="llm-request")

def shutdown_request_executor():
    """Корректное завершение работы пула потоков"""
    REQUEST_EXECUTOR.shutdown(wait=True)

import atexit
atexit.register(shutdown_request_executor)


def process_search_request_async(search_chain, question):
    """Асинхронная обработка поискового запроса с использованием глобального ThreadPoolExecutor"""
    def _sync_invoke():
        try:
            return search_chain.invoke(question)
        except Exception as e:
            logging.error(f"Error in search request processing: {e}")
            return f"Ошибка при обработке поискового запроса: {str(e)}"

    # Используем глобальный пул потоков для эффективной обработки нескольких пользователей
    # Таймаут 240 секунд для предотвращения зависания
    future = REQUEST_EXECUTOR.submit(_sync_invoke)
    try:
        return future.result(timeout=240)
    except concurrent.futures.TimeoutError:
        return "Превышено время ожидания ответа. Попробуйте позже."
    except Exception as e:
        return f"Ошибка выполнения запроса: {str(e)}"


def process_tt_request_async(tt_chain, question):
    """Асинхронная обработка запроса на генерацию ТТ с использованием глобального ThreadPoolExecutor"""
    def _sync_invoke():
        try:
            return tt_chain.invoke(question)
        except Exception as e:
            logging.error(f"Error in TT request processing: {e}")
            return f"Ошибка при генерации технических требований: {str(e)}"

    # Используем глобальный пул потоков для эффективной обработки нескольких пользователей
    # Таймаут 240 секунд для ТТ (генерация более сложная)
    future = REQUEST_EXECUTOR.submit(_sync_invoke)
    try:
        return future.result(timeout=240)
    except concurrent.futures.TimeoutError:
        return "Превышено время ожидания генерации ТТ. Попробуйте сформулировать запрос проще."
    except Exception as e:
        return f"Ошибка генерации ТТ: {str(e)}"
