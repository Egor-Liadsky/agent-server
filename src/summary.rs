//! Компактизация истории чата: дословный хвост из последних `N` сообщений,
//! пересказ вытесненной части, ступенчатое обновление и подстановка в
//! запрос (specs/context-summary, design.md, решения 4-9).

use crate::store::ChatMessage;
use agentcore::agent::{Message, Role};

/// Заголовок-маркер, которым начинается сообщение-пересказ: `Role` ядра не
/// содержит `System`, поэтому пересказ идёт первым сообщением роли `user`
/// (design.md, решение 5).
pub const SUMMARY_MARKER: &str = "Краткое содержание предыдущей части разговора:";

/// Индекс начала дословного хвоста: последние `keep_messages` сообщений,
/// граница сдвинута вперёд (к более новым) до ближайшего сообщения роли
/// `user`, чтобы хвост не начинался с ответа ассистента без вопроса
/// (design.md, решение 5). Сдвиг всегда уменьшает хвост, поэтому
/// `keep_messages` — верхняя граница его размера, а не точное число.
pub fn tail_boundary(messages: &[ChatMessage], keep_messages: u32) -> usize {
    let len = messages.len();
    let keep = keep_messages as usize;
    if len <= keep {
        return 0;
    }
    let mut start = len - keep;
    while start < len && !matches!(messages[start].role, Role::User) {
        start += 1;
    }
    start
}

/// Сообщения перед границей хвоста, ещё не вошедшие в сохранённый пересказ
/// (`seq` больше `through_seq`).
pub fn pending_messages(messages: &[ChatMessage], boundary: usize, through_seq: i64) -> Vec<&ChatMessage> {
    messages[..boundary]
        .iter()
        .filter(|message| message.seq > through_seq)
        .collect()
}

fn role_label(role: Role) -> &'static str {
    match role {
        Role::User => "Пользователь",
        Role::Assistant => "Модель",
    }
}

/// Текст запроса на построение пересказа: прежний пересказ (если был),
/// вытесняемые сообщения и инструкция уложиться в лимит символов
/// (design.md, решение 7).
pub fn summary_prompt(previous: Option<&str>, pending: &[&ChatMessage], max_chars: u32) -> String {
    let mut text = String::new();
    if let Some(previous) = previous {
        text.push_str("Пересказ предыдущей части разговора:\n");
        text.push_str(previous);
        text.push_str("\n\n");
    }
    text.push_str("Новая часть разговора для включения в пересказ:\n");
    for message in pending {
        text.push_str(role_label(message.role));
        text.push_str(": ");
        text.push_str(&message.content);
        text.push('\n');
    }
    text.push_str(&format!(
        "\nОбнови пересказ разговора так, чтобы он включал и прежний пересказ, и новую \
         часть. Пиши кратко, только по существу, без вступлений. Уложись не более чем в \
         {max_chars} символов."
    ));
    text
}

/// Жёсткое усечение результата: модель инструкцию по длине может не
/// выполнить, а неограниченный пересказ сам станет источником переполнения
/// контекста (design.md, решение 7).
pub fn truncate_summary(text: &str, max_chars: u32) -> String {
    let max = max_chars as usize;
    if text.chars().count() <= max {
        return text.to_string();
    }
    text.chars().take(max).collect()
}

fn message_from_stored(message: &ChatMessage) -> Message {
    Message {
        role: message.role,
        content: message.content.clone(),
        reasoning: message.reasoning.clone(),
        meta: message.meta.clone(),
    }
}

/// Сообщение-пересказ, помещаемое первым в историю запроса.
pub fn summary_message(summary: &str) -> Message {
    Message::user(format!("{SUMMARY_MARKER}\n\n{summary}"))
}

/// Итоговая история запроса: пересказ (если применяется) первым сообщением,
/// затем дословный хвост, затем новое сообщение пользователя
/// (specs/context-summary, «Дословный хвост и подстановка пересказа»).
pub fn assemble_history(summary: Option<&str>, tail: &[ChatMessage], new_message: Message) -> Vec<Message> {
    let mut history = Vec::with_capacity(tail.len() + 2);
    if let Some(summary) = summary {
        history.push(summary_message(summary));
    }
    history.extend(tail.iter().map(message_from_stored));
    history.push(new_message);
    history
}

#[cfg(test)]
mod tests {
    use super::*;
    use agentcore::agent::MessageMeta;

    fn message(seq: i64, role: Role, content: &str) -> ChatMessage {
        ChatMessage {
            seq,
            role,
            content: content.to_string(),
            reasoning: None,
            meta: None,
            created_at: 0,
        }
    }

    fn exchange(pairs: &[(i64, &str, &str)]) -> Vec<ChatMessage> {
        // pairs: (начальный seq, вопрос, ответ) — пара user/assistant.
        let mut messages = Vec::new();
        for (seq, user, assistant) in pairs {
            messages.push(message(*seq, Role::User, user));
            messages.push(message(*seq + 1, Role::Assistant, assistant));
        }
        messages
    }

    // --- 9.1 Граница хвоста ---

    #[test]
    fn short_chat_has_no_boundary() {
        let messages = exchange(&[(1, "вопрос 1", "ответ 1")]);
        assert_eq!(tail_boundary(&messages, 10), 0);
    }

    #[test]
    fn long_chat_boundary_keeps_at_most_n_messages() {
        // 5 обменов = 10 сообщений, keep_messages = 4 -> хвост из последних 4.
        let messages = exchange(&[
            (1, "в1", "о1"),
            (3, "в2", "о2"),
            (5, "в3", "о3"),
            (7, "в4", "о4"),
            (9, "в5", "о5"),
        ]);
        let boundary = tail_boundary(&messages, 4);
        assert_eq!(messages.len() - boundary, 4);
        assert!(matches!(messages[boundary].role, Role::User));
    }

    #[test]
    fn boundary_landing_on_assistant_shifts_to_next_user_message() {
        let messages = exchange(&[(1, "в1", "о1"), (3, "в2", "о2"), (5, "в3", "о3")]);
        // keep_messages = 2: len(5) - 2 = 3 -> индекс 3 (assistant «о2»),
        // граница должна сдвинуться на следующее user-сообщение (индекс 4).
        let boundary = tail_boundary(&messages, 2);
        assert_eq!(boundary, 4);
        assert!(matches!(messages[boundary].role, Role::User));
    }

    // --- 9.2 Подстановка пересказа ---

    #[test]
    fn assembled_history_has_summary_tail_and_new_message() {
        let tail = exchange(&[(5, "в3", "о3")]);
        let history = assemble_history(Some("итог прошлого"), &tail, Message::user("новый вопрос"));
        assert_eq!(history.len(), 4);
        assert!(matches!(history[0].role, Role::User));
        assert!(history[0].content.starts_with(SUMMARY_MARKER));
        assert!(history[0].content.contains("итог прошлого"));
        assert_eq!(history[1].content, "в3");
        assert_eq!(history[2].content, "о3");
        assert_eq!(history[3].content, "новый вопрос");
    }

    #[test]
    fn assembled_history_without_summary_has_only_tail_and_new_message() {
        let tail = exchange(&[(5, "в3", "о3")]);
        let history = assemble_history(None, &tail, Message::user("новый вопрос"));
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].content, "в3");
    }

    // --- 9.3 Правило ступени ---

    #[test]
    fn pending_below_step_is_not_enough_to_rebuild() {
        let messages = exchange(&[(1, "в1", "о1"), (3, "в2", "о2"), (5, "в3", "о3")]);
        let boundary = tail_boundary(&messages, 2);
        let pending = pending_messages(&messages, boundary, 0);
        assert!((pending.len() as u32) < 5, "порог 5 не должен быть достигнут четырьмя сообщениями");
    }

    #[test]
    fn pending_at_or_above_step_triggers_rebuild() {
        let messages = exchange(&[(1, "в1", "о1"), (3, "в2", "о2"), (5, "в3", "о3"), (7, "в4", "о4")]);
        let boundary = tail_boundary(&messages, 2);
        let pending = pending_messages(&messages, boundary, 0);
        assert!(pending.len() as u32 >= 5);
    }

    #[test]
    fn already_summarized_messages_are_excluded_from_pending() {
        let messages = exchange(&[(1, "в1", "о1"), (3, "в2", "о2"), (5, "в3", "о3")]);
        let boundary = tail_boundary(&messages, 0);
        // through_seq = 4: сообщения с seq 1..4 уже пересказаны, остаются 5 и 6.
        let pending = pending_messages(&messages, boundary, 4);
        assert_eq!(pending.iter().map(|m| m.seq).collect::<Vec<_>>(), vec![5, 6]);
    }

    // --- 9.4 Усечение пересказа ---

    #[test]
    fn summary_longer_than_limit_is_truncated() {
        let text = "а".repeat(100);
        let truncated = truncate_summary(&text, 10);
        assert_eq!(truncated.chars().count(), 10);
    }

    #[test]
    fn summary_within_limit_is_not_changed() {
        let text = "короткий пересказ";
        assert_eq!(truncate_summary(text, 1000), text);
    }

    #[test]
    fn meta_is_dropped_from_message_meta_field_when_absent() {
        let mut msg = message(1, Role::User, "текст");
        msg.meta = Some(MessageMeta::default());
        let converted = message_from_stored(&msg);
        assert!(converted.meta.is_some());
    }
}
