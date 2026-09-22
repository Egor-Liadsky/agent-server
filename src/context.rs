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
    async fn assemble(&self, ctx: &StrategyCtx<'_>, stored: Vec<store::ChatMessage>, new_messages: Vec<Message>) -> Assembled;
}

struct SummaryImpl;

#[async_trait]
impl ContextStrategyImpl for SummaryImpl {
    async fn assemble(&self, ctx: &StrategyCtx<'_>, stored: Vec<store::ChatMessage>, new_messages: Vec<Message>) -> Assembled {
        let compaction = compact_history(ctx.state, &ctx.chat.id, ctx.settings, stored, new_messages).await;
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
    async fn assemble(&self, ctx: &StrategyCtx<'_>, stored: Vec<store::ChatMessage>, new_messages: Vec<Message>) -> Assembled {
        let outcome = crate::window::assemble(stored, effective_window(ctx.state, ctx.settings), new_messages);
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
    async fn assemble(&self, ctx: &StrategyCtx<'_>, stored: Vec<store::ChatMessage>, new_messages: Vec<Message>) -> Assembled {
        let window_size = effective_window(ctx.state, ctx.settings);
        let outcome = crate::facts::assemble(ctx.state, &ctx.chat.id, window_size, stored, new_messages).await;
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
    async fn assemble(&self, ctx: &StrategyCtx<'_>, stored: Vec<store::ChatMessage>, new_messages: Vec<Message>) -> Assembled {
        let outcome = crate::branch::assemble(ctx.state, ctx.chat, stored, new_messages).await;
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
    }
}

/// Первым сообщением истории всегда идёт системное сообщение — при любой
/// стратегии, включая те, что не формируют служебных разделов
/// (specs/context-strategies, «Первым сообщением истории идёт системное
/// сообщение»; design.md, решение 3: `sliding_window` и `branching`
/// получают его без своего кода). Слоистая память, если включена, оборачивает
/// результат стратегии: её разделы идут ПЕРЕД разделами стратегии, а поля
/// памяти дополняют `ContextDto` стратегии, не подменяя его
/// (design.md — decouple-memory-layers, решение 1).
pub async fn assemble(
    state: &AppState,
    chat: &store::Chat,
    settings: &ChatSettings,
    owner: &str,
    strategy: ContextStrategy,
    stored: Vec<store::ChatMessage>,
    new_messages: Vec<Message>,
) -> Assembled {
    let ctx = StrategyCtx { state, chat, settings };
    let mut assembled = impl_for(strategy).assemble(&ctx, stored, new_messages).await;
    if crate::memory::effective_layers_enabled(state, settings) {
        let memory = crate::memory::layers(state, chat, settings, owner).await;
        crate::memory::merge_into_context(&mut assembled.context, &memory, &assembled.history);
        let mut sections = memory.sections;
        sections.extend(std::mem::take(&mut assembled.sections));
        assembled.sections = sections;
        // Счётчики маршрутизатора относятся к маршрутизации ПРЕДЫДУЩЕГО
        // сообщения этого чата (design.md, решение 5): маршрутизатор ещё не
        // отработал на момент сборки текущего запроса, поэтому здесь читается
        // результат прошлого фонового прогона (`AppState.memory_route_outcomes`).
        let router = state
            .memory_route_outcomes
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .get(&chat.id)
            .copied()
            .unwrap_or_default();
        crate::memory::merge_router_counters(&mut assembled.context, router);
    }
    // Раздел состояния задачи соседствует с разделами памяти
    // (specs/memory-layers, «Модифицированная возможность memory-layers»):
    // читается заново на каждый запрос, поэтому поля `context` не отстают,
    // в отличие от счётчиков фонового трекера ниже (design.md, решение 5).
    if crate::task::effective_enabled(state, settings) {
        if let Ok(task) = store::load_task_state(&state.db, owner, &chat.id).await {
            crate::task::merge_into_context(&mut assembled.context, &task);
            assembled.sections.push(crate::task::system_message_section(&task));
            let tracker = state
                .task_track_outcomes
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .get(&chat.id)
                .copied()
                .unwrap_or_default();
            crate::task::merge_tracker_counters(&mut assembled.context, tracker);
        }
    }
    // Раздел профиля встаёт ПЕРЕД разделами памяти и стратегии — после того,
    // как они уже собраны, но до сборки системного сообщения (design.md,
    // решение 4): роль и ограничения должны задавать трактовку всего
    // остального.
    if let Some(profile_id) = crate::profile::effective_profile_id(state, settings) {
        if let Ok(Some(profile)) = crate::profile::find(state, owner, &profile_id).await {
            let built = crate::profile::build_section(&profile, state.config.profile_max_chars);
            if state.config.log_content {
                tracing::info!(
                    profile_id = %profile_id,
                    persona = %profile.persona,
                    style = %profile.style,
                    format = %profile.format,
                    constraints = ?profile.constraints,
                    chars = built.chars,
                    truncated = built.truncated,
                    "профиль применён к запросу"
                );
            } else {
                tracing::info!(
                    profile_id = %profile_id,
                    chars = built.chars,
                    truncated = built.truncated,
                    "профиль применён к запросу"
                );
            }
            assembled.context.profile_id = Some(profile_id);
            assembled.context.profile_chars = Some(built.chars);
            assembled.sections.insert(0, built.section);
        }
    }
    let system_message = crate::system_message::build(&state.config.system_prompt, &assembled.sections);
    assembled.history.insert(0, system_message);
    // Висячий вызов инструмента остаётся в истории, если клиент упал
    // посреди хода или ветка создана от середины хода; провайдер такую
    // историю отвергает, поэтому она выравнивается перед каждым вызовом.
    assembled.history = agentcore::agent::close_dangling_tool_calls(&assembled.history);
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
            "owner-1",
            ContextStrategy::SlidingWindow,
            Vec::new(),
            vec![Message::user("новое")],
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
            "owner-1",
            ContextStrategy::Branching,
            Vec::new(),
            vec![Message::user("новое")],
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
            "owner-1",
            ContextStrategy::Facts,
            Vec::new(),
            vec![Message::user("новое")],
        )
        .await;
        assert!(matches!(assembled.history[0].role, Role::System));
        assert!(assembled.history[0].content.contains("Факты чата:"));
        assert!(assembled.history.iter().skip(1).all(|m| !m.content.contains("Факты чата:")));
    }

    // --- Слоистая память как обёртка над стратегией (decouple-memory-layers) ---

    fn dummy_message(role: Role, seq: i64, content: &str) -> store::ChatMessage {
        store::ChatMessage {
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

    #[tokio::test]
    async fn memory_sections_come_before_summary_section_in_that_order() {
        let _guard = crate::state::test_lock();
        let state = AppState::for_tests().await;
        let settings = ChatSettings {
            context_strategy: Some(ContextStrategy::Summary),
            summary_enabled: Some(true),
            memory_layers_enabled: Some(true),
            ..ChatSettings::default()
        };
        let chat = store::create_chat(&state.db, "owner-1", "Чат", &settings).await.expect("чат");
        store::set_long_term_memory(&state.db, "owner-1", "decision", Some("auth"), "Clerk", "manual", None, 1)
            .await
            .expect("долговременная запись");
        store::set_working_memory(&state.db, &chat.id, &chat.active_task_id, "target", "iOS 17+", "manual", 1)
            .await
            .expect("рабочая запись");

        // Достаточно сообщений, чтобы хвост стратегии summary не покрыл всю
        // историю (умолчание `AGENTD_SUMMARY_KEEP_MESSAGES` — 20), и уже
        // сохранённый пересказ, который подставляется без вызова модели —
        // порог перестройки (`AGENTD_SUMMARY_STEP_MESSAGES` — 10) не достигнут.
        let stored: Vec<store::ChatMessage> = (1..=25)
            .map(|seq| {
                if seq % 2 == 1 {
                    dummy_message(Role::User, seq, &format!("вопрос {seq}"))
                } else {
                    dummy_message(Role::Assistant, seq, &format!("ответ {seq}"))
                }
            })
            .collect();
        store::save_summary(&state.db, &chat.id, "итог прошлого", stored[4].seq).await.expect("пересказ сохранён");

        let assembled = assemble(
            &state,
            &chat,
            &settings,
            "owner-1",
            ContextStrategy::Summary,
            stored,
            vec![Message::user("новое")],
        )
        .await;

        let system = &assembled.history[0].content;
        let long_term_at = system.find("Долговременная память:").expect("раздел долговременной памяти");
        let working_at = system.find("Рабочая память задачи:").expect("раздел рабочей памяти");
        let summary_at = system.find(crate::summary::SUMMARY_MARKER).expect("раздел пересказа");
        assert!(long_term_at < working_at, "долговременная память должна идти раньше рабочей");
        assert!(working_at < summary_at, "разделы памяти должны идти раньше раздела стратегии");
    }

    // --- Профиль как раздел, встающий первым (specs/user-profiles) ---

    #[tokio::test]
    async fn profile_section_precedes_memory_and_strategy_sections() {
        let _guard = crate::state::test_lock();
        let state = AppState::for_tests().await;
        let settings = ChatSettings {
            context_strategy: Some(ContextStrategy::Summary),
            summary_enabled: Some(true),
            memory_layers_enabled: Some(true),
            profile_id: Some("teacher".to_string()),
            ..ChatSettings::default()
        };
        let chat = store::create_chat(&state.db, "owner-1", "Чат", &settings).await.expect("чат");
        store::set_long_term_memory(&state.db, "owner-1", "decision", Some("auth"), "Clerk", "manual", None, 1)
            .await
            .expect("долговременная запись");
        store::set_working_memory(&state.db, &chat.id, &chat.active_task_id, "target", "iOS 17+", "manual", 1)
            .await
            .expect("рабочая запись");
        let stored: Vec<store::ChatMessage> = (1..=25)
            .map(|seq| {
                if seq % 2 == 1 {
                    dummy_message(Role::User, seq, &format!("вопрос {seq}"))
                } else {
                    dummy_message(Role::Assistant, seq, &format!("ответ {seq}"))
                }
            })
            .collect();
        store::save_summary(&state.db, &chat.id, "итог прошлого", stored[4].seq).await.expect("пересказ сохранён");

        let assembled = assemble(
            &state,
            &chat,
            &settings,
            "owner-1",
            ContextStrategy::Summary,
            stored,
            vec![Message::user("новое")],
        )
        .await;

        let system = &assembled.history[0].content;
        let profile_at = system.find("Профиль:").expect("раздел профиля");
        let long_term_at = system.find("Долговременная память:").expect("раздел долговременной памяти");
        let working_at = system.find("Рабочая память задачи:").expect("раздел рабочей памяти");
        let summary_at = system.find(crate::summary::SUMMARY_MARKER).expect("раздел пересказа");
        assert!(profile_at < long_term_at, "профиль должен идти раньше долговременной памяти");
        assert!(long_term_at < working_at);
        assert!(working_at < summary_at);
        assert_eq!(assembled.context.profile_id.as_deref(), Some("teacher"));
        assert!(assembled.context.profile_chars.unwrap_or(0) > 0);
    }

    #[tokio::test]
    async fn operator_default_profile_applies_when_chat_settings_leave_it_unset() {
        let _guard = crate::state::test_lock();
        let state = AppState::with_env(&[
            (crate::config::API_KEY_VAR, "test-key-value"),
            ("AGENTD_UPSTREAM_BASE_URL", "http://127.0.0.1:1"),
            ("AGENTD_MODEL", "test-model"),
            ("AGENTD_DEFAULT_PROFILE", "teacher"),
        ])
        .await;
        let settings = ChatSettings::default();
        let chat = store::create_chat(&state.db, "owner-1", "Чат", &settings).await.expect("чат");
        let assembled = assemble(
            &state,
            &chat,
            &settings,
            "owner-1",
            ContextStrategy::Summary,
            Vec::new(),
            vec![Message::user("новое")],
        )
        .await;
        assert_eq!(assembled.context.profile_id.as_deref(), Some("teacher"));
        assert!(assembled.history[0].content.contains("Профиль:"));
    }

    #[tokio::test]
    async fn chat_setting_overrides_operator_default_profile() {
        let _guard = crate::state::test_lock();
        let state = AppState::with_env(&[
            (crate::config::API_KEY_VAR, "test-key-value"),
            ("AGENTD_UPSTREAM_BASE_URL", "http://127.0.0.1:1"),
            ("AGENTD_MODEL", "test-model"),
            ("AGENTD_DEFAULT_PROFILE", "teacher"),
        ])
        .await;
        let settings = ChatSettings { profile_id: Some("reviewer".to_string()), ..ChatSettings::default() };
        let chat = store::create_chat(&state.db, "owner-1", "Чат", &settings).await.expect("чат");
        let assembled = assemble(
            &state,
            &chat,
            &settings,
            "owner-1",
            ContextStrategy::Summary,
            Vec::new(),
            vec![Message::user("новое")],
        )
        .await;
        assert_eq!(assembled.context.profile_id.as_deref(), Some("reviewer"));
    }

    #[tokio::test]
    async fn no_profile_leaves_section_order_unchanged() {
        let _guard = crate::state::test_lock();
        let state = AppState::for_tests().await;
        let settings = ChatSettings {
            context_strategy: Some(ContextStrategy::Summary),
            summary_enabled: Some(true),
            memory_layers_enabled: Some(true),
            ..ChatSettings::default()
        };
        let chat = store::create_chat(&state.db, "owner-1", "Чат", &settings).await.expect("чат");
        let assembled = assemble(
            &state,
            &chat,
            &settings,
            "owner-1",
            ContextStrategy::Summary,
            Vec::new(),
            vec![Message::user("новое")],
        )
        .await;
        let system = &assembled.history[0].content;
        assert!(!system.contains("Профиль:"));
        assert!(assembled.context.profile_id.is_none());
        assert!(assembled.context.profile_chars.is_none());
    }

    /// Профиль подставляется при каждой стратегии контекста и вместе со
    /// слоистой памятью, не меняя счётчиков стратегии (specs/user-profiles,
    /// «Профиль работает при каждой стратегии контекста», «Профиль и
    /// слоистая память применяются вместе»).
    #[tokio::test]
    async fn profile_applies_over_every_strategy_without_changing_strategy_counters() {
        for strategy in ContextStrategy::ALL {
            let _guard = crate::state::test_lock();
            let state = AppState::for_tests().await;
            let without_profile = ChatSettings { context_strategy: Some(strategy), ..ChatSettings::default() };
            let chat_a = store::create_chat(&state.db, "owner-1", "Чат", &without_profile).await.expect("чат");
            let baseline = assemble(
                &state,
                &chat_a,
                &without_profile,
                "owner-1",
                strategy,
                Vec::new(),
                vec![Message::user("новое")],
            )
            .await;

            let with_profile = ChatSettings {
                context_strategy: Some(strategy),
                profile_id: Some("teacher".to_string()),
                ..ChatSettings::default()
            };
            let chat_b = store::create_chat(&state.db, "owner-1", "Чат", &with_profile).await.expect("чат");
            let with_profile_assembled = assemble(
                &state,
                &chat_b,
                &with_profile,
                "owner-1",
                strategy,
                Vec::new(),
                vec![Message::user("новое")],
            )
            .await;

            assert!(
                with_profile_assembled.history[0].content.contains("Профиль:"),
                "стратегия {strategy:?}: раздел профиля должен присутствовать"
            );
            let mut baseline_context = baseline.context.clone();
            baseline_context.profile_id = with_profile_assembled.context.profile_id.clone();
            baseline_context.profile_chars = with_profile_assembled.context.profile_chars;
            // У ветвления `branch_id` — случайный идентификатор новой ветки
            // каждого чата, к профилю отношения не имеющий.
            baseline_context.branch_id = with_profile_assembled.context.branch_id.clone();
            assert_eq!(
                baseline_context, with_profile_assembled.context,
                "стратегия {strategy:?}: профиль не должен менять счётчики стратегии"
            );
        }
    }

    #[tokio::test]
    async fn profile_and_memory_layers_apply_together() {
        let _guard = crate::state::test_lock();
        let state = AppState::for_tests().await;
        let settings = ChatSettings {
            context_strategy: Some(ContextStrategy::Summary),
            memory_layers_enabled: Some(true),
            profile_id: Some("teacher".to_string()),
            ..ChatSettings::default()
        };
        let chat = store::create_chat(&state.db, "owner-1", "Чат", &settings).await.expect("чат");
        store::set_long_term_memory(&state.db, "owner-1", "decision", Some("auth"), "Clerk", "manual", None, 1)
            .await
            .expect("долговременная запись");
        store::set_working_memory(&state.db, &chat.id, &chat.active_task_id, "target", "iOS 17+", "manual", 1)
            .await
            .expect("рабочая запись");

        let assembled = assemble(
            &state,
            &chat,
            &settings,
            "owner-1",
            ContextStrategy::Summary,
            Vec::new(),
            vec![Message::user("новое")],
        )
        .await;
        let system = &assembled.history[0].content;
        assert!(system.contains("Профиль:"));
        assert!(system.contains("Долговременная память:"));
        assert!(system.contains("Рабочая память задачи:"));
    }

    #[tokio::test]
    async fn disabled_memory_layers_add_no_sections_for_any_strategy() {
        for strategy in ContextStrategy::ALL {
            let _guard = crate::state::test_lock();
            let state = AppState::for_tests().await;
            let settings = ChatSettings { context_strategy: Some(strategy), ..ChatSettings::default() };
            let chat = store::create_chat(&state.db, "owner-1", "Чат", &settings).await.expect("чат");
            store::set_long_term_memory(&state.db, "owner-1", "decision", Some("auth"), "Clerk", "manual", None, 1)
                .await
                .expect("долговременная запись");
            store::set_working_memory(&state.db, &chat.id, &chat.active_task_id, "target", "iOS 17+", "manual", 1)
                .await
                .expect("рабочая запись");

            let assembled =
                assemble(&state, &chat, &settings, "owner-1", strategy, Vec::new(), vec![Message::user("новое")]).await;

            let system = &assembled.history[0].content;
            assert!(
                !system.contains("Долговременная память:") && !system.contains("Рабочая память задачи:"),
                "стратегия {strategy:?}: разделы памяти не должны появляться при выключенной слоистой памяти"
            );
            assert!(assembled.context.memory_long_term_entries.is_none(), "стратегия {strategy:?}");
        }
    }
}
