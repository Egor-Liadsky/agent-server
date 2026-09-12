//! Стратегия контекста `sliding_window`: провайдеру уходят только последние
//! N сохранённых сообщений чата, старше — отбрасываются без замены
//! пересказом (specs/context-sliding-window).

use crate::dto::ContextDto;
use crate::store::ChatMessage;
use agentcore::agent::Message;

fn message_from_stored(stored: ChatMessage) -> Message {
    Message {
        role: stored.role,
        content: stored.content,
        reasoning: stored.reasoning,
        meta: stored.meta,
    }
}

/// Итог сборки: история для провайдера и число отброшенных сообщений.
pub struct WindowOutcome {
    pub history: Vec<Message>,
    pub sent_messages: u32,
    pub dropped_messages: u32,
}

/// Последние `window_size` сообщений чата плюс новое сообщение
/// пользователя, с границей, сдвинутой к сообщению пользователя
/// (specs/context-sliding-window, «Окно начинается с сообщения
/// пользователя»).
pub fn assemble(stored: Vec<ChatMessage>, window_size: u32, new_message: Message) -> WindowOutcome {
    let boundary = crate::summary::tail_boundary(&stored, window_size);
    let dropped_messages = boundary as u32;
    let tail = &stored[boundary..];
    let sent_messages = tail.len() as u32;

    let mut history: Vec<Message> = tail.iter().cloned().map(message_from_stored).collect();
    history.push(new_message);

    WindowOutcome {
        history,
        sent_messages,
        dropped_messages,
    }
}

pub fn context_dto(outcome: &WindowOutcome) -> ContextDto {
    ContextDto::for_window(outcome.sent_messages, outcome.dropped_messages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agentcore::agent::Role;

    fn message(role: Role, seq: i64, content: &str) -> ChatMessage {
        ChatMessage {
            seq,
            role,
            content: content.to_string(),
            reasoning: None,
            meta: None,
            created_at: 0,
        }
    }

    // --- 5.1 Границы окна ---

    #[test]
    fn history_shorter_than_window_is_sent_whole() {
        let stored = vec![message(Role::User, 1, "привет"), message(Role::Assistant, 2, "привет!")];
        let outcome = assemble(stored, 6, Message::user("как дела?"));
        assert_eq!(outcome.sent_messages, 2);
        assert_eq!(outcome.dropped_messages, 0);
        assert_eq!(outcome.history.len(), 3);
    }

    #[test]
    fn long_history_is_truncated_to_window() {
        let stored: Vec<ChatMessage> = (1..=20)
            .map(|seq| {
                if seq % 2 == 1 {
                    message(Role::User, seq, &format!("вопрос {seq}"))
                } else {
                    message(Role::Assistant, seq, &format!("ответ {seq}"))
                }
            })
            .collect();
        let outcome = assemble(stored, 6, Message::user("новое сообщение"));
        assert_eq!(outcome.sent_messages, 6);
        assert_eq!(outcome.dropped_messages, 14);
        assert_eq!(outcome.history.len(), 7);
        assert!(matches!(outcome.history[0].role, Role::User));
    }

    #[test]
    fn boundary_landing_on_assistant_message_shifts_to_next_user_message() {
        // seq 15 (граница по счёту) — ответ модели: должен сдвинуться на 16.
        let stored: Vec<ChatMessage> = (1..=20)
            .map(|seq| {
                if seq % 2 == 1 {
                    message(Role::Assistant, seq, &format!("ответ {seq}"))
                } else {
                    message(Role::User, seq, &format!("вопрос {seq}"))
                }
            })
            .collect();
        let outcome = assemble(stored, 6, Message::user("новое сообщение"));
        // Отброшено на одно больше расчётного окна — граница сдвинута.
        assert_eq!(outcome.dropped_messages, 15);
        assert_eq!(outcome.sent_messages, 5);
        assert!(matches!(outcome.history[0].role, Role::User));
    }

    #[test]
    fn window_exactly_matches_history_length() {
        let stored = vec![message(Role::User, 1, "привет"), message(Role::Assistant, 2, "привет!")];
        let outcome = assemble(stored, 2, Message::user("ещё"));
        assert_eq!(outcome.dropped_messages, 0);
        assert_eq!(outcome.sent_messages, 2);
    }
}
