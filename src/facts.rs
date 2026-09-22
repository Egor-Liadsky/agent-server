//! Стратегия контекста `facts`: устойчивые факты «ключ-значение» хранятся
//! отдельно от сообщений, обновляются после каждого сообщения пользователя
//! отдельным дешёвым вызовом модели и уходят провайдеру вместе с хвостом
//! истории (specs/context-facts).

use crate::dto::ContextDto;
use crate::state::AppState;
use crate::store::{self, ChatMessage};
use agentcore::agent::{Agent, Message};
use agentcore::config::{ChatSettings, ReasoningMode, ThinkingMode};
use serde::Deserialize;

/// Заголовок раздела фактов в системном сообщении (specs/context-facts,
/// «Раздел SHALL быть явно озаглавлен как факты чата»).
const FACTS_SECTION_HEADING: &str = "Факты чата:";
/// Отличает вызов обновления фактов от обычного диалогового вызова в
/// тестах, мокающих провайдера по содержимому тела запроса.
#[cfg_attr(not(test), allow(dead_code))]
pub const FACTS_UPDATE_MARKER: &str = "Верни JSON-массив операций";

pub struct FactsOutcome {
    pub history: Vec<Message>,
    /// Раздел фактов для системного сообщения запроса. `None` — фактов нет
    /// (specs/context-facts, «Пустой набор фактов не добавляет раздела»).
    pub facts_section: Option<String>,
    pub sent_messages: u32,
    pub dropped_messages: u32,
    pub facts_applied: u32,
}

fn message_from_stored(stored: ChatMessage) -> Message {
    stored.into_message()
}

/// Текст раздела фактов для системного сообщения — пустой набор не создаёт
/// раздела совсем (specs/context-facts, «Пустой набор фактов не добавляет
/// раздела»).
fn facts_section(facts: &[store::Fact]) -> Option<String> {
    if facts.is_empty() {
        return None;
    }
    let lines: Vec<String> = facts.iter().map(|f| format!("- {}: {}", f.key, f.value)).collect();
    Some(format!("{FACTS_SECTION_HEADING}\n{}", lines.join("\n")))
}

/// История для провайдера: последние `window_size` сообщений, затем новое
/// сообщение — блок фактов уходит отдельно, разделом системного сообщения
/// (specs/context-facts, «Блок фактов передаётся системным сообщением»).
pub async fn assemble(
    state: &AppState,
    chat_id: &str,
    window_size: u32,
    stored: Vec<ChatMessage>,
    new_messages: Vec<Message>,
) -> FactsOutcome {
    let facts = match store::load_facts(&state.db, chat_id).await {
        Ok(facts) => facts,
        Err(err) => {
            tracing::warn!(chat_id, error = %err, "не удалось прочитать факты чата");
            Vec::new()
        }
    };
    let facts_applied = facts.len() as u32;
    let facts_section = facts_section(&facts);

    let boundary = crate::summary::tail_boundary(&stored, window_size);
    let dropped_messages = boundary as u32;
    let tail = &stored[boundary..];
    let sent_messages = tail.len() as u32;

    let mut history = Vec::with_capacity(tail.len() + 1);
    history.extend(tail.iter().cloned().map(message_from_stored));
    history.extend(new_messages);

    FactsOutcome {
        history,
        facts_section,
        sent_messages,
        dropped_messages,
        facts_applied,
    }
}

/// `facts_updated` не заполняется здесь: он относится к обновлению фактов
/// ПОСЛЕ ответа (`update_after_exchange`, вызывается после записи обмена),
/// а сборка истории происходит раньше. Вызывающий код (`app.rs`)
/// подставляет фактический результат в `ContextDto.facts_updated` отдельно.
pub fn context_dto(outcome: &FactsOutcome) -> ContextDto {
    ContextDto::for_facts(outcome.sent_messages, outcome.dropped_messages, outcome.facts_applied, false)
}

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum FactOp {
    Set { key: String, value: String },
    Delete { key: String },
}

fn update_prompt(existing: &[store::Fact], user_message: &str) -> String {
    let facts_text = if existing.is_empty() {
        "(факты пока не заданы)".to_string()
    } else {
        existing.iter().map(|f| format!("- {}: {}", f.key, f.value)).collect::<Vec<_>>().join("\n")
    };
    format!(
        "Текущие факты диалога:\n{facts_text}\n\n\
         Новое сообщение пользователя:\n{user_message}\n\n\
         Верни JSON-массив операций над фактами — только то, что нужно изменить. \
         Каждый элемент — {{\"op\":\"set\",\"key\":\"...\",\"value\":\"...\"}} для новой или изменённой \
         записи, либо {{\"op\":\"delete\",\"key\":\"...\"}} для удаления ключа, которого сообщение больше \
         не подтверждает. Если менять ничего не нужно, верни пустой массив []. Ответь только JSON-массивом, \
         без пояснений."
    )
}

/// Разбирает ответ модели как список операций. Незаданный или некорректный
/// JSON — `None`: неудача разбора не должна ронять основной ответ
/// пользователю (specs/context-facts, «Неудача обновления фактов не ломает
/// ответ»).
fn parse_operations(content: &str) -> Option<Vec<FactOp>> {
    let start = content.find('[')?;
    let end = content.rfind(']')?;
    if end < start {
        return None;
    }
    serde_json::from_str(&content[start..=end]).ok()
}

/// Обновляет факты чата после ответа пользователю: отдельный вызов модели,
/// операции применяются к существующему набору, усечённому по операторским
/// потолкам (specs/context-facts, «Факты обновляются после каждого
/// сообщения пользователя»; design.md, решение 5). Возвращает `true`, если
/// набор фактов действительно обновился.
pub async fn update_after_exchange(
    state: &AppState,
    chat_id: &str,
    settings: &ChatSettings,
    user_message: &str,
    through_seq: i64,
) -> bool {
    let existing = match store::load_facts(&state.db, chat_id).await {
        Ok(facts) => facts,
        Err(err) => {
            tracing::warn!(chat_id, error = %err, "не удалось прочитать факты перед обновлением");
            return false;
        }
    };

    let prompt = update_prompt(&existing, user_message);
    let mut facts_settings = settings.clone();
    if let Some(model) = &state.config.facts_model {
        facts_settings.model = Some(model.clone());
    }
    facts_settings.reasoning = ReasoningMode::Default;
    facts_settings.thinking = ThinkingMode::Disabled;
    facts_settings.custom_response_mode = false;

    let reply = match state.agent.ask(&[Message::user(prompt)], &facts_settings).await {
        Ok(reply) => reply,
        Err(err) => {
            tracing::warn!(chat_id, error = %err, "не удалось обновить факты чата: вызов модели не удался");
            return false;
        }
    };

    let Some(operations) = parse_operations(&reply.content) else {
        tracing::warn!(chat_id, "не удалось обновить факты чата: ответ модели не разобран как список операций");
        return false;
    };
    if operations.is_empty() {
        return false;
    }

    let mut by_key: std::collections::BTreeMap<String, String> =
        existing.into_iter().map(|f| (f.key, f.value)).collect();
    for op in operations {
        match op {
            FactOp::Set { key, value } => {
                let truncated = truncate_chars(&value, state.config.fact_value_max_chars as usize);
                by_key.insert(key, truncated);
            }
            FactOp::Delete { key } => {
                by_key.remove(&key);
            }
        }
    }

    // Потолок числа фактов: усекается сервисом, а не отклоняет запрос
    // (specs/context-facts, «Автоматическое обновление усекается»).
    let max_facts = state.config.max_facts as usize;
    let truncated_keys: Vec<String> = by_key.keys().take(max_facts).cloned().collect();

    let mut updated = false;
    for key in &truncated_keys {
        let value = &by_key[key];
        if let Err(err) = store::set_fact(&state.db, chat_id, key, value, through_seq).await {
            tracing::warn!(chat_id, key, error = %err, "не удалось сохранить факт");
            continue;
        }
        updated = true;
    }
    updated
}

fn truncate_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    text.chars().take(max).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use agentcore::agent::Role;

    fn fact(key: &str, value: &str) -> store::Fact {
        store::Fact {
            key: key.to_string(),
            value: value.to_string(),
            through_seq: 1,
            updated_at: 0,
        }
    }

    fn message(role: Role, seq: i64, content: &str) -> ChatMessage {
        ChatMessage {
            seq,
            role,
            content: content.to_string(),
            reasoning: None,
            meta: None,
            created_at: 0,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_name: None,
        }
    }

    // --- 6.2 Разбор операций ---

    #[test]
    fn parses_set_and_delete_operations() {
        let content = r#"вот ответ: [{"op":"set","key":"budget","value":"200000"},{"op":"delete","key":"old"}]"#;
        let ops = parse_operations(content).expect("операции разобраны");
        assert_eq!(ops.len(), 2);
        assert!(matches!(&ops[0], FactOp::Set { key, value } if key == "budget" && value == "200000"));
        assert!(matches!(&ops[1], FactOp::Delete { key } if key == "old"));
    }

    #[test]
    fn unparseable_response_is_none() {
        assert!(parse_operations("не JSON вовсе").is_none());
    }

    #[test]
    fn empty_operations_array_parses_as_empty() {
        let ops = parse_operations("[]").expect("пустой массив — валидный ответ");
        assert!(ops.is_empty());
    }

    // --- 6.4 Сборка истории ---

    #[test]
    fn facts_section_is_absent_when_empty() {
        assert!(facts_section(&[]).is_none());
    }

    #[test]
    fn facts_section_lists_each_fact() {
        let section = facts_section(&[fact("budget", "200000"), fact("deadline", "март")]).expect("раздел фактов");
        assert!(section.contains(FACTS_SECTION_HEADING));
        assert!(section.contains("budget: 200000"));
        assert!(section.contains("deadline: март"));
    }

    #[tokio::test]
    async fn assemble_orders_facts_section_then_tail_then_new_message() {
        let state = crate::state::AppState::for_tests().await;
        let chat = store::create_chat(&state.db, "owner-1", "Чат", &ChatSettings::default())
            .await
            .expect("чат");
        store::set_fact(&state.db, &chat.id, "budget", "200000", 1)
            .await
            .expect("факт сохранён");

        let stored = vec![message(Role::User, 1, "первое сообщение")];
        let outcome = assemble(&state, &chat.id, 6, stored, vec![Message::user("новое")]).await;

        assert_eq!(outcome.facts_applied, 1);
        assert_eq!(outcome.history.len(), 2);
        let section = outcome.facts_section.expect("раздел фактов собран");
        assert!(section.contains(FACTS_SECTION_HEADING));
        assert!(section.contains("budget: 200000"));
        assert_eq!(outcome.history[0].content, "первое сообщение");
        assert_eq!(outcome.history[1].content, "новое");
    }
}
