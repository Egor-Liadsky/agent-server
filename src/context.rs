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
        Assembled {
            context: crate::facts::context_dto(&outcome),
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
    }
}

pub async fn assemble(
    state: &AppState,
    chat: &store::Chat,
    settings: &ChatSettings,
    strategy: ContextStrategy,
    stored: Vec<store::ChatMessage>,
    new_message: Message,
) -> Assembled {
    let ctx = StrategyCtx { state, chat, settings };
    impl_for(strategy).assemble(&ctx, stored, new_message).await
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
