//! Стратегия контекста `branching`: провайдеру уходит история активной
//! ветки — сообщения родительских веток до их точек ветвления, затем
//! собственные сообщения ветки, затем новое сообщение
//! (specs/chat-branching, «История собирается по цепочке ветки»).

use crate::state::AppState;
use crate::store::{self, ChatMessage};
use agentcore::agent::Message;

pub struct BranchOutcome {
    pub history: Vec<Message>,
    pub sent_messages: u32,
    pub branch_id: String,
}

fn message_from_stored(stored: ChatMessage) -> Message {
    stored.into_message()
}

/// `stored` здесь не используется: история ветки собирается заново по её
/// цепочке родителей, а не из полной выгрузки чата, которую даёт общий путь
/// `load_messages` для остальных стратегий (design.md, решение 4 — ветки
/// влияют на сборку истории только при стратегии `branching`).
pub async fn assemble(state: &AppState, chat: &store::Chat, _stored: Vec<ChatMessage>, new_messages: Vec<Message>) -> BranchOutcome {
    let branch_id = chat.active_branch.clone();
    let history = match store::load_branch_history(&state.db, &chat.id, &branch_id, state.config.max_branch_depth).await
    {
        Ok(messages) => messages,
        Err(err) => {
            tracing::warn!(chat_id = %chat.id, branch_id, error = %err, "не удалось собрать историю ветки");
            Vec::new()
        }
    };
    let sent_messages = history.len() as u32;
    let mut history: Vec<Message> = history.into_iter().map(message_from_stored).collect();
    history.extend(new_messages);

    BranchOutcome {
        history,
        sent_messages,
        branch_id,
    }
}
