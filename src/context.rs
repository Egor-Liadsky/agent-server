//! Единая точка выбора стратегии управления контекстом. Стратегия — трейт
//! со сборкой истории, а не условия в обработчике: новая стратегия
//! добавляется реализацией трейта, а не правкой `handle_chat_in_existing`
//! (design.md, решение 2 — тот же приём, что и трейт `Agent` в `agentcore`
//! и стадии `pipeline.rs`).

use crate::app::compact_history;
use crate::dto::ContextDto;
use crate::state::AppState;
use crate::store;
use agentcore::agent::Message;
use agentcore::config::{ChatSettings, ContextStrategy};
use async_trait::async_trait;

/// История для провайдера и блок наблюдаемости `context` этого запроса.
pub struct Assembled {
    pub history: Vec<Message>,
    pub context: ContextDto,
    /// Разделы служебных блоков стратегии для системного сообщения —
    /// стратегия отдаёт их отдельно, а не готовую строку, чтобы порядок и
    /// оформление были едины для всех стратегий (design.md, решение 3).
    sections: Vec<String>,
}

/// Всё, что реализации стратегии могут понадобиться для сборки истории.
/// `chat` — чат из хранилища целиком: стратегии `branching` нужна его
/// активная ветка, остальным — только `id`.
pub struct StrategyCtx<'a> {
    pub state: &'a AppState,
    pub chat: &'a store::Chat,
    pub settings: &'a ChatSettings,
}

#[async_trait]
trait ContextStrategyImpl {
    async fn assemble(&self, ctx: &StrategyCtx<'_>, stored: Vec<store::ChatMessage>, new_message: Message) -> Assembled;
}

struct SummaryImpl;

#[async_trait]
impl ContextStrategyImpl for SummaryImpl {
    async fn assemble(&self, ctx: &StrategyCtx<'_>, stored: Vec<store::ChatMessage>, new_message: Message) -> Assembled {
        let compaction = compact_history(ctx.state, &ctx.chat.id, ctx.settings, stored, new_message).await;
        Assembled {
            context: ContextDto::for_summary(compaction.replaced_messages, compaction.summary_built),
            sections: compaction.summary_section.into_iter().collect(),
            history: compaction.history,
        }
    }
}

struct WindowImpl;

#[async_trait]
impl ContextStrategyImpl for WindowImpl {
    async fn assemble(&self, ctx: &StrategyCtx<'_>, stored: Vec<store::ChatMessage>, new_message: Message) -> Assembled {
        let outcome = crate::window::assemble(stored, effective_window(ctx.state, ctx.settings), new_message);
        Assembled {
            context: crate::window::context_dto(&outcome),
            sections: Vec::new(),
            history: outcome.history,
        }
    }
}

struct FactsImpl;

#[async_trait]
impl ContextStrategyImpl for FactsImpl {
    async fn assemble(&self, ctx: &StrategyCtx<'_>, stored: Vec<store::ChatMessage>, new_message: Message) -> Assembled {
        let window_size = effective_window(ctx.state, ctx.settings);
        let outcome = crate::facts::assemble(ctx.state, &ctx.chat.id, window_size, stored, new_message).await;
        let context = crate::facts::context_dto(&outcome);
        Assembled {
            context,
            sections: outcome.facts_section.into_iter().collect(),
            history: outcome.history,
        }
    }
}

struct BranchingImpl;

#[async_trait]
impl ContextStrategyImpl for BranchingImpl {
    async fn assemble(&self, ctx: &StrategyCtx<'_>, stored: Vec<store::ChatMessage>, new_message: Message) -> Assembled {
        let outcome = crate::branch::assemble(ctx.state, ctx.chat, stored, new_message).await;
        Assembled {
            context: ContextDto::for_branching(outcome.sent_messages, outcome.branch_id),
            sections: Vec::new(),
            history: outcome.history,
        }
    }
}

/// Единственная точка ветвления по стратегии (specs/context-strategies,
/// «Стратегия контекста задаётся настройкой чата»).
fn impl_for(strategy: ContextStrategy) -> Box<dyn ContextStrategyImpl + Send + Sync> {
    match strategy {
        ContextStrategy::Summary => Box::new(SummaryImpl),
        ContextStrategy::SlidingWindow => Box::new(WindowImpl),
        ContextStrategy::Facts => Box::new(FactsImpl),
        ContextStrategy::Branching => Box::new(BranchingImpl),
        // Полноценная реализация — предмет отдельного изменения
        // add-memory-layers (src/memory.rs, MemoryLayersImpl); до его
        // применения стратегия ведёт себя как sliding_window, чтобы
        // ContextStrategy оставался исчерпывающим match уже сейчас (ядро
        // agent-cli, откуда пришло значение MemoryLayers, — общая
        // зависимость обоих изменений).
        ContextStrategy::MemoryLayers => Box::new(WindowImpl),
    }
}

/// Первым сообщением истории всегда идёт системное сообщение — при любой
/// стратегии, включая те, что не формируют служебных разделов
/// (specs/context-strategies, «Первым сообщением истории идёт системное
/// сообщение»; design.md, решение 3: `sliding_window` и `branching`
/// получают его без своего кода).
pub async fn assemble(
    state: &AppState,
    chat: &store::Chat,
    settings: &ChatSettings,
    strategy: ContextStrategy,
    stored: Vec<store::ChatMessage>,
    new_message: Message,
) -> Assembled {
    let ctx = StrategyCtx { state, chat, settings };
    let mut assembled = impl_for(strategy).assemble(&ctx, stored, new_message).await;
    let system_message = crate::system_message::build(&state.config.system_prompt, &assembled.sections);
    assembled.history.insert(0, system_message);
    assembled
}

/// Эффективный размер окна: клиентское значение чата поверх операторского
/// умолчания (specs/context-sliding-window, «Размер окна настраивается»).
/// Границы уже проверены на путях сохранения и вызова (`apply_context_strategy`).
pub fn effective_window(state: &AppState, settings: &ChatSettings) -> u32 {
    settings
        .context_window_messages
        .unwrap_or(state.config.context_window_messages)
}

/// Действующая стратегия запроса: настройка чата поверх операторского
/// умолчания (specs/context-strategies, «Операторские умолчание и список
/// разрешённых стратегий»).
pub fn effective_strategy(state: &AppState, settings: &ChatSettings) -> ContextStrategy {
    settings.context_strategy.unwrap_or(state.config.context_strategy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agentcore::agent::Role;

    async fn chat_with_strategy(state: &AppState, strategy: ContextStrategy) -> store::Chat {
        let settings = ChatSettings {
            context_strategy: Some(strategy),
            ..ChatSettings::default()
        };
        store::create_chat(&state.db, "owner-1", "Чат", &settings)
            .await
            .expect("чат")
    }

    // --- Системное сообщение есть при любой стратегии (specs/context-strategies) ---

    #[tokio::test]
    async fn sliding_window_gets_system_message_without_own_sections() {
        let state = AppState::for_tests().await;
        let chat = chat_with_strategy(&state, ContextStrategy::SlidingWindow).await;
        let assembled = assemble(
            &state,
            &chat,
            &chat.settings,
            ContextStrategy::SlidingWindow,
            Vec::new(),
            Message::user("новое"),
        )
        .await;
        assert!(matches!(assembled.history[0].role, Role::System));
        assert_eq!(assembled.history[0].content, state.config.system_prompt);
    }

    #[tokio::test]
    async fn branching_gets_system_message_without_own_sections() {
        let state = AppState::for_tests().await;
        let chat = chat_with_strategy(&state, ContextStrategy::Branching).await;
        let assembled = assemble(
            &state,
            &chat,
            &chat.settings,
            ContextStrategy::Branching,
            Vec::new(),
            Message::user("новое"),
        )
        .await;
        assert!(matches!(assembled.history[0].role, Role::System));
        assert_eq!(assembled.history[0].content, state.config.system_prompt);
    }

    #[tokio::test]
    async fn facts_section_lands_inside_system_message_not_as_user_message() {
        let state = AppState::for_tests().await;
        let chat = chat_with_strategy(&state, ContextStrategy::Facts).await;
        store::set_fact(&state.db, &chat.id, "budget", "200000", 1)
            .await
            .expect("факт сохранён");
        let assembled = assemble(
            &state,
            &chat,
            &chat.settings,
            ContextStrategy::Facts,
            Vec::new(),
            Message::user("новое"),
        )
        .await;
        assert!(matches!(assembled.history[0].role, Role::System));
        assert!(assembled.history[0].content.contains("Факты чата:"));
        assert!(assembled.history.iter().skip(1).all(|m| !m.content.contains("Факты чата:")));
    }
}
