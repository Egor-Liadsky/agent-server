//! Сборка системного сообщения запроса: базовый текст оператора плюс
//! разделы, которые формируют действующие стратегии контекста
//! (specs/context-strategies, design.md, решение 3).

use agentcore::agent::Message;

/// Системное сообщение — всегда одно и всегда первое в истории запроса,
/// даже если разделов нет вовсе (specs/context-strategies, «Первым
/// сообщением истории идёт системное сообщение»).
pub fn build(base: &str, sections: &[String]) -> Message {
    let mut text = base.to_string();
    for section in sections {
        if !text.is_empty() {
            text.push_str("\n\n");
        }
        text.push_str(section);
    }
    Message::system(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_only_without_sections() {
        let message = build("базовый текст", &[]);
        assert_eq!(message.content, "базовый текст");
    }

    #[test]
    fn sections_join_base_with_blank_line() {
        let message = build(
            "базовый текст",
            &["раздел один".to_string(), "раздел два".to_string()],
        );
        assert_eq!(message.content, "базовый текст\n\nраздел один\n\nраздел два");
    }

    #[test]
    fn empty_base_and_sections_gives_empty_system_message() {
        let message = build("", &[]);
        assert_eq!(message.content, "");
    }
}
