//! Автоматическое название чата: отдельный дешёвый вызов модели после
//! первого обмена, тем же приёмом, что `facts::update_after_exchange`
//! (design.md, решение 5).

use crate::state::AppState;
use agentcore::agent::{Agent, Message};
use agentcore::config::{ChatSettings, ReasoningMode, ThinkingMode};

/// Отличает вызов генерации названия от обычного диалогового вызова в
/// тестах, мокающих провайдера по содержимому тела запроса (по образцу
/// `facts::FACTS_UPDATE_MARKER`).
#[cfg_attr(not(test), allow(dead_code))]
pub const TITLE_UPDATE_MARKER: &str = "Придумай короткое название чата";

/// Запускать ли генерацию названия для этого обмена (specs/chat-title,
/// «Название генерируется один раз после первого обмена», «Название,
/// заданное клиентом, не перезаписывается»): включено оператором, чат ещё
/// на умолчании сервиса, и записанное сообщение пользователя — первое.
pub fn should_generate(state: &AppState, chat_title: &str, user_seq: i64) -> bool {
    state.config.auto_title && chat_title == crate::app::DEFAULT_CHAT_TITLE && user_seq == 1
}

fn build_prompt(system_text: &str, user_message: &str) -> String {
    format!(
        "{TITLE_UPDATE_MARKER} (не длиннее {} символов) по системному описанию чата и первому \
сообщению пользователя. Ответь только названием, без кавычек, пояснений и точки в конце.\n\n\
Системное описание:\n{system_text}\n\n\
Первое сообщение пользователя:\n{user_message}",
        crate::app::MAX_CHAT_TITLE_LEN,
    )
}

/// Обрамляющая пара кавычек, которую нормализация снимает с ответа модели
/// (specs/chat-title, «Обрамляющие кавычки... SHALL удаляться»).
const QUOTE_PAIRS: [(char, char); 3] = [('"', '"'), ('\'', '\''), ('«', '»')];

fn strip_quotes(text: &str) -> &str {
    for (open, close) in QUOTE_PAIRS {
        let mut chars = text.chars();
        if chars.next() == Some(open) && chars.next_back() == Some(close) {
            return chars.as_str();
        }
    }
    text
}

fn truncate_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    text.chars().take(max).collect()
}

/// Нормализация ответа модели: первая строка, без обрамляющих кавычек и
/// завершающей точки, усечённая до `MAX_CHAT_TITLE_LEN` (specs/chat-title,
/// «Форма и границы сгенерированного названия»; design.md, решение 7).
/// Пустой после нормализации ответ — `None`.
fn normalize_title(raw: &str) -> Option<String> {
    let first_line = raw.lines().next().unwrap_or("").trim();
    let without_dot = first_line.trim_end_matches('.').trim();
    let unquoted = strip_quotes(without_dot).trim();
    if unquoted.is_empty() {
        return None;
    }
    Some(truncate_chars(unquoted, crate::app::MAX_CHAT_TITLE_LEN))
}

/// Запрашивает у провайдера название и сохраняет его, если чат к этому
/// моменту всё ещё на умолчании сервиса (specs/chat-title, «Ручное
/// переименование во время генерации»; design.md, решение 6). Неудача
/// вызова, разбора или записи только логируется — вызывающий код (фоновый
/// `tokio::spawn`) не ждёт результата (specs/chat-title, «Неудача генерации
/// не влияет на диалог»).
pub async fn generate_and_save(
    state: AppState,
    chat_id: String,
    owner: String,
    settings: ChatSettings,
    system_text: String,
    user_message: String,
) {
    let prompt = build_prompt(&system_text, &user_message);
    let mut title_settings = settings;
    if let Some(model) = &state.config.title_model {
        title_settings.model = Some(model.clone());
    }
    title_settings.reasoning = ReasoningMode::Default;
    title_settings.thinking = ThinkingMode::Disabled;
    title_settings.custom_response_mode = false;

    let reply = match state.agent.ask(&[Message::user(prompt)], &title_settings).await {
        Ok(reply) => reply,
        Err(err) => {
            tracing::warn!(chat_id = %chat_id, error = %err, "не удалось сгенерировать название чата: вызов модели не удался");
            return;
        }
    };

    let Some(title) = normalize_title(&reply.content) else {
        tracing::warn!(chat_id = %chat_id, "не удалось сгенерировать название чата: пустой ответ после нормализации");
        return;
    };

    match crate::store::set_title_if_default(&state.db, &owner, &chat_id, &title, crate::app::DEFAULT_CHAT_TITLE)
        .await
    {
        Ok(true) => {}
        Ok(false) => {
            tracing::info!(chat_id = %chat_id, "название чата уже изменено клиентом — сгенерированное отброшено");
        }
        Err(err) => {
            tracing::warn!(chat_id = %chat_id, error = %err, "не удалось сохранить сгенерированное название чата");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Нормализация ответа модели (specs/chat-title, «Форма и границы») ---

    #[test]
    fn strips_wrapping_quotes_and_trailing_dot() {
        assert_eq!(normalize_title("\"Название чата\".").as_deref(), Some("Название чата"));
        assert_eq!(normalize_title("«Название чата».").as_deref(), Some("Название чата"));
        assert_eq!(normalize_title("'Название чата'").as_deref(), Some("Название чата"));
    }

    #[test]
    fn keeps_only_first_line() {
        assert_eq!(
            normalize_title("Название чата\nвторая строка, которую не берём").as_deref(),
            Some("Название чата")
        );
    }

    #[test]
    fn truncates_to_max_title_length() {
        let long = "a".repeat(crate::app::MAX_CHAT_TITLE_LEN + 50);
        let title = normalize_title(&long).expect("название");
        assert_eq!(title.chars().count(), crate::app::MAX_CHAT_TITLE_LEN);
    }

    #[test]
    fn empty_after_normalization_gives_none() {
        assert_eq!(normalize_title(""), None);
        assert_eq!(normalize_title("   \n"), None);
        assert_eq!(normalize_title("\"\""), None);
    }

    // --- Условие запуска генерации (specs/chat-title) ---

    #[tokio::test]
    async fn should_generate_only_for_first_message_on_default_title() {
        let mut state = AppState::for_tests().await;
        std::sync::Arc::make_mut(&mut state.config).auto_title = true;
        assert!(should_generate(&state, crate::app::DEFAULT_CHAT_TITLE, 1));
        assert!(!should_generate(&state, crate::app::DEFAULT_CHAT_TITLE, 2));
        assert!(!should_generate(&state, "Название клиента", 1));
    }

    #[tokio::test]
    async fn should_generate_is_false_when_auto_title_disabled() {
        let mut state = AppState::for_tests().await;
        std::sync::Arc::make_mut(&mut state.config).auto_title = false;
        assert!(!should_generate(&state, crate::app::DEFAULT_CHAT_TITLE, 1));
    }
}
