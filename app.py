import streamlit as st
import os
import fitz
import time
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from chain_factory import create_rag_chain, create_tt_chain
from async_handlers import process_search_request_async, process_tt_request_async
from web_interface import (
    load_css, init_theme, toggle_theme, apply_theme,
    load_chat_history, save_chat_history, create_new_chat, update_chat_title,
    check_word_export_request, generate_word_document
)

# Streamlit app
def clear_chat():
    st.session_state.messages = []
    if os.path.exists("chat_history.json"):
        os.remove("chat_history.json")

def main():
    st.set_page_config(layout="wide")

    # Initialize placeholders
    if 'status_placeholder' not in st.session_state:
        st.session_state.status_placeholder = st.empty()
    if 'progress_placeholder' not in st.session_state:
        st.session_state.progress_placeholder = st.empty()

    # Инициализация CSS и темы
    load_css()
    init_theme()
    apply_theme()

    # Переключатель темы убран по просьбе пользователя

    # Основной заголовок
    st.markdown("""
    <div class="main-header">
        <div class="main-title">Чат-система поиска и анализа нормативов</div>
        <div class="subtitle">ИИ-помощник для работы с нормативными документами</div>
    </div>
    """, unsafe_allow_html=True)

    # Focus on input field
    st.markdown('<script>document.querySelector(".custom-input-form-wrapper textarea")?.focus();</script>', unsafe_allow_html=True)

    # Auto-scroll to bottom of chat
    st.markdown('<script>document.querySelector(".chat-container")?.scrollTop = document.querySelector(".chat-container")?.scrollHeight;</script>', unsafe_allow_html=True)

    # Применяем кастомные стили к форме
    st.markdown("""
    <style>
        /* Стили для wrapper нашей формы */
        .custom-input-form-wrapper {
            width: calc(100% - 40px) !important;
            max-width: 800px !important;
            margin: 0 auto !important;
            background: transparent !important;
            border: none !important;
            border-radius: var(--border-radius) !important;
            padding: var(--spacing-md) !important;
            box-shadow: none !important;
            transition: var(--transition) !important;
        }

        .custom-input-form-wrapper:hover {
            transform: translateY(-2px) !important;
            box-shadow: var(--shadow-medium) !important;
        }

        /* Стили для text_area в нашей форме */
        .custom-input-form-wrapper .stTextArea textarea {
            background: transparent !important;
            border: none !important;
            outline: none !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            font-size: 15px !important;
            color: var(--text) !important;
            line-height: 1.4 !important;
            resize: vertical !important;
            min-height: 24px !important;
            max-height: 120px !important;
            overflow-y: auto !important;
        }

        .custom-input-form-wrapper .stTextArea textarea::placeholder {
            color: var(--text-secondary) !important;
            opacity: 0.7 !important;
        }

        .custom-input-form-wrapper .stTextArea textarea:focus {
            outline: none !important;
            box-shadow: none !important;
        }

        /* Скрываем label text_area в нашей форме */
        .custom-input-form-wrapper .stTextArea label {
            display: none !important;
        }

        /* Стили для кнопки отправки в нашей форме */
        .custom-input-form-wrapper .stButton button {
            background: var(--primary-gradient) !important;
            border: none !important;
            border-radius: 12px !important;
            width: 48px !important;
            height: 48px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            cursor: pointer !important;
            transition: var(--transition) !important;
            box-shadow: var(--shadow-light) !important;
            font-size: 18px !important;
            margin-top: 0 !important;
        }

        .custom-input-form-wrapper .stButton button:hover {
            transform: translateY(-2px) scale(1.05) !important;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15) !important;
        }

        .custom-input-form-wrapper .stButton button:active {
            transform: translateY(0) scale(0.95) !important;
        }

        /* Адаптивность для нашей формы */
        @media (max-width: 768px) {
            .custom-input-form-wrapper {
                margin: 0 var(--spacing-sm) var(--spacing-lg) var(--spacing-sm) !important;
                padding: var(--spacing-sm) !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)

    # Инициализация структуры чатов
    if "chat_data" not in st.session_state:
        st.session_state.chat_data = load_chat_history()

    # Если нет текущего чата, создаем новый
    if not st.session_state.chat_data.get("current_chat_id"):
        chat_id, chat = create_new_chat()
        st.session_state.chat_data["chats"][chat_id] = chat
        st.session_state.chat_data["current_chat_id"] = chat_id
        save_chat_history(st.session_state.chat_data)

    # Получаем текущий чат
    current_chat_id = st.session_state.chat_data["current_chat_id"]
    current_chat = st.session_state.chat_data["chats"][current_chat_id]
    st.session_state.messages = current_chat["messages"]

    # Set default mode if not set
    if "mode" not in st.session_state:
        st.session_state.mode = "Автоматично"

    # Sidebar for new chat and mode selection
    with st.sidebar:
        # Заголовок боковой панели
        st.markdown("""
        <div class="sidebar-header">
            <h3>📋 Управление чатами</h3>
        </div>
        """, unsafe_allow_html=True)

        # Новый чат
        if st.button("➕ Новый чат", use_container_width=True):
            chat_id, chat = create_new_chat()
            st.session_state.chat_data["chats"][chat_id] = chat
            st.session_state.chat_data["current_chat_id"] = chat_id
            save_chat_history(st.session_state.chat_data)
            st.rerun()

        # История чатов
        st.markdown("### 📂 История чатов")

        # Поиск по чатам
        search_query = st.text_input(
            "Поиск чатов",
            placeholder="🔍 Поиск чатов...",
            key="chat_search",
            label_visibility="collapsed"
        )

        # Список чатов с прокруткой
        chats = st.session_state.chat_data["chats"]
        if chats:
            # Сортируем чаты по времени обновления (новые сверху)
            sorted_chats = sorted(chats.items(), key=lambda x: x[1]["updated_at"], reverse=True)

            # Фильтруем чаты по поисковому запросу
            if search_query:
                filtered_chats = [
                    (chat_id, chat) for chat_id, chat in sorted_chats
                    if search_query.lower() in chat["title"].lower()
                ]
            else:
                filtered_chats = sorted_chats

            # Группируем чаты по датам
            from datetime import datetime, date, timedelta

            today = date.today()
            yesterday = today - timedelta(days=1)

            grouped_chats = {
                "Сегодня": [],
                "Вчера": [],
                "Ранее": []
            }

            for chat_id, chat in filtered_chats:
                try:
                    updated_date = datetime.fromisoformat(chat["updated_at"]).date()
                    if updated_date == today:
                        grouped_chats["Сегодня"].append((chat_id, chat))
                    elif updated_date == yesterday:
                        grouped_chats["Вчера"].append((chat_id, chat))
                    else:
                        grouped_chats["Ранее"].append((chat_id, chat))
                except:
                    grouped_chats["Ранее"].append((chat_id, chat))

            # Контейнер с прокруткой для списка чатов
            with st.container(height=500):
                for group_name, group_chats in grouped_chats.items():
                    if group_chats:  # Показываем группу только если есть чаты
                        st.markdown(f"**{group_name}**")

                        for chat_id, chat in group_chats:
                            # Определяем, является ли этот чат текущим
                            is_current = chat_id == st.session_state.chat_data["current_chat_id"]

                            # Компактная карточка чата
                            col1, col2 = st.columns([1, 0.2])
                            with col1:
                                if st.button(
                                    chat['title'][:30] + "..." if len(chat['title']) > 30 else chat['title'],
                                    key=f"chat_{chat_id}",
                                    use_container_width=True,
                                    type="primary" if is_current else "secondary"
                                ):
                                    # Переключаемся на выбранный чат
                                    st.session_state.chat_data["current_chat_id"] = chat_id
                                    save_chat_history(st.session_state.chat_data)
                                    st.rerun()

                            with col2:
                                # Кнопка удаления чата
                                if st.button("🗑️", key=f"delete_{chat_id}", help="Удалить чат"):
                                    if chat_id in st.session_state.chat_data["chats"]:
                                        del st.session_state.chat_data["chats"][chat_id]
                                        # Если удаляем текущий чат, переключаемся на другой
                                        if chat_id == st.session_state.chat_data["current_chat_id"]:
                                            remaining_chats = list(st.session_state.chat_data["chats"].keys())
                                            if remaining_chats:
                                                st.session_state.chat_data["current_chat_id"] = remaining_chats[0]
                                            else:
                                                # Создаем новый чат если не осталось
                                                new_chat_id, new_chat = create_new_chat()
                                                st.session_state.chat_data["chats"][new_chat_id] = new_chat
                                                st.session_state.chat_data["current_chat_id"] = new_chat_id
                                        save_chat_history(st.session_state.chat_data)
                                        st.rerun()

                        st.markdown("---")  # Разделитель между группами
        else:
            st.write("История пуста")

        # Переключатель темы
        st.markdown("---")
        st.markdown("**Тема**")
        theme_icon = "☀️" if st.session_state.get("theme", "light") == "light" else "🌙"
        if st.button(theme_icon, key="theme_toggle", help="Переключить тему"):
            toggle_theme()
            st.rerun()

    # Загрузка векторного хранилища
    if "vectorstore" not in st.session_state or "tt_vectorstore" not in st.session_state:
        with st.spinner(""):
            # Показываем прогресс бар
            loading_progress_placeholder = st.empty()
            loading_status_placeholder = st.empty()
            loading_progress_placeholder.progress(0)
            loading_status_placeholder.markdown("""
            <div class="status-indicator">
                <div class="status-dot"></div>
                <span>Загрузка индексов документов...</span>
            </div>
            """, unsafe_allow_html=True)

            try:
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
            except Exception as e:
                st.error(f"Ошибка загрузки модели эмбеддингов: {e}")
                st.error("Попробуйте перезапустить приложение или проверить подключение к интернету.")
                return

            # Загрузка нормативного индекса
            loading_progress_placeholder.progress(25)
            if not os.path.exists("./faiss_index"):
                st.error("Индекс нормативных документов не найден. Сначала запустите main.py для создания индексов.")
                return
            normative_vectorstore = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)
            st.session_state.vectorstore = normative_vectorstore

            # Загрузка TT индекса
            loading_progress_placeholder.progress(75)
            if os.path.exists("./faiss_index_tt"):
                tt_vectorstore = FAISS.load_local("./faiss_index_tt", embeddings, allow_dangerous_deserialization=True)
                st.session_state.tt_vectorstore = tt_vectorstore
            else:
                st.session_state.tt_vectorstore = normative_vectorstore  # fallback
                st.warning("Индекс ТТ документов не найден. Используется нормативный индекс для ТТ.")

            loading_progress_placeholder.progress(100)
            st.session_state.qa_chain = create_rag_chain(st.session_state.vectorstore)
            st.session_state.tt_chain = create_tt_chain(st.session_state.tt_vectorstore)

            loading_progress_placeholder.empty()
            loading_status_placeholder.empty()

            # Устанавливаем флаг, что загрузка завершена
            st.session_state.indexes_loaded = True
            st.rerun()  # Перезагружаем страницу, чтобы скрыть сообщение о загрузке

    # Отображение сообщений чата
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for message in st.session_state.messages:
        message_class = "user" if message["role"] == "user" else "assistant"
        avatar_class = "user-avatar" if message["role"] == "user" else "assistant-avatar"
        avatar_text = "U" if message["role"] == "user" else "AI"

        st.markdown(f"""
        <div class="chat-message {message_class}">
            <div class="message-avatar {avatar_class}">{avatar_text}</div>
            <div class="message-content">
                {message["content"].replace(chr(10), '<br>')}
                <div class="message-time">{datetime.now().strftime("%H:%M")}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Placeholder for status during search
    st.session_state.status_placeholder = st.empty()
    st.session_state.progress_placeholder = st.empty()

    st.markdown('</div>', unsafe_allow_html=True)

    # Кастомное поле ввода внизу
    st.markdown('<div class="custom-input-form-wrapper">', unsafe_allow_html=True)
    with st.form(key="message_form", clear_on_submit=True):
        col1, col2 = st.columns([1, 0.1])
        with col1:
            prompt = st.text_area(
                "Сообщение",
                placeholder="Например: 'Что такое трансформатор?' или 'Экспортируй в Word'",
                key="user_input",
                label_visibility="collapsed",
                height=56
            )
        with col2:
            submit_clicked = st.form_submit_button("📤", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Auto-scroll to bottom of chat and focus on input
    st.markdown('''
    <script>
    setTimeout(() => {
        const chatContainer = document.querySelector(".chat-container");
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        const input = document.querySelector(".custom-input-form-wrapper textarea");
        if (input) {
            input.focus();
        }
    }, 100);
    </script>
    ''', unsafe_allow_html=True)

    if submit_clicked and prompt and st.session_state.get("indexes_loaded", False):
        # Проверка на запрос экспорта в Word
        word_export_requested = check_word_export_request(prompt)

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
        with st.spinner(""):
            # Статус индикатор
            st.session_state.status_placeholder.markdown(f"""
            <div class="status-indicator">
                <div class="status-dot"></div>
                <span>{mode_name}...</span>
            </div>
            """, unsafe_allow_html=True)

            # Прогресс бар
            st.session_state.progress_placeholder.markdown("""
            <div class="progress-container">
                <div class="progress-bar" style="width: 30%;"></div>
                </div>
            """, unsafe_allow_html=True)

            try:
                if is_tt_mode:
                    response = process_tt_request_async(chain, prompt)
                else:
                    response = process_search_request_async(chain, prompt)

                # Обновляем прогресс
                st.session_state.progress_placeholder.markdown("""
                <div class="progress-container">
                    <div class="progress-bar" style="width: 100%;"></div>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(0.5)

            except Exception as e:
                response = f"Ошибка: {e}"

            # Очищаем статус и прогресс
            st.session_state.status_placeholder.empty()
            st.session_state.progress_placeholder.empty()

        # Add message to session state
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Если запрошен экспорт в Word, показываем кнопку скачивания
        if word_export_requested and not response.startswith("Ошибка:"):
            # Генерируем Word документ
            word_buffer = generate_word_document(response, prompt, mode_name)

            # Кнопка скачивания
            st.download_button(
                label="📝 Скачать в Word",
                data=word_buffer,
                file_name=f"ответ_системы_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

        # Обновляем текущий чат в структуре данных
        current_chat_id = st.session_state.chat_data["current_chat_id"]
        st.session_state.chat_data["chats"][current_chat_id]["messages"] = st.session_state.messages
        st.session_state.chat_data["chats"][current_chat_id]["updated_at"] = datetime.now().isoformat()

        # Обновляем заголовок чата, если это первое сообщение
        if len(st.session_state.messages) == 2:  # пользователь + ответ
            new_title = update_chat_title(current_chat_id, st.session_state.messages)
            st.session_state.chat_data["chats"][current_chat_id]["title"] = new_title

        # Сохраняем всю структуру чатов
        save_chat_history(st.session_state.chat_data)

        # Перезагружаем страницу, чтобы отобразить новое сообщение в чате
        st.rerun()

if __name__ == "__main__":
    main()
