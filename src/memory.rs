//! Стратегия контекста `memory_layers`: три слоя памяти с раздельным
//! хранением и разной областью жизни — краткосрочный (хвост сообщений),
//! рабочий (чат + задача) и долговременный (владелец) — и автоматический
//! маршрутизатор, распределяющий записи по слоям после каждого сообщения
//! пользователя (specs/memory-layers, design.md).

use crate::dto::ContextDto;
use crate::state::AppState;
use crate::store::{self, ChatMessage};
use agentcore::agent::{Agent, Message};
use agentcore::config::{ChatSettings, ReasoningMode, ThinkingMode};
use serde::Deserialize;
use std::collections::HashSet;

const LONG_TERM_SECTION_HEADING: &str = "Долговременная память:";
const WORKING_SECTION_HEADING: &str = "Рабочая память задачи:";

/// Отличает вызов маршрутизатора памяти от обычного диалогового вызова в
/// тестах, мокающих провайдера по содержимому тела запроса (по образцу
/// `facts::FACTS_UPDATE_MARKER`).
#[cfg_attr(not(test), allow(dead_code))]
pub const MEMORY_ROUTER_MARKER: &str = "Верни JSON-массив операций над памятью";

// --- Операции маршрутизатора (design.md, решение 3) ---

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum RawMemoryOp {
    Set {
        layer: String,
        #[serde(default)]
        key: Option<String>,
        #[serde(default)]
        value: Option<String>,
        #[serde(default)]
        entry_type: Option<String>,
        #[serde(default)]
        carry_forward: bool,
        #[serde(default)]
        reason: Option<String>,
    },
    Delete {
        layer: String,
        #[serde(default)]
        key: Option<String>,
        #[serde(default)]
        reason: Option<String>,
    },
    FinishTask {
        #[serde(default)]
        reason: Option<String>,
    },
}

/// Разбирает ответ модели как список операций памяти — тот же приём, что
/// `facts::parse_operations`: найти первую `[` и последнюю `]`, распарсить
/// внутри; неудача разбора — `None`, вызывающий код оставляет память без
/// изменений (specs/memory-layers, «Отказ маршрутизатора не блокирует ответ
/// пользователю»).
fn parse_operations(content: &str) -> Option<Vec<RawMemoryOp>> {
    let start = content.find('[')?;
    let end = content.rfind(']')?;
    if end < start {
        return None;
    }
    serde_json::from_str(&content[start..=end]).ok()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layer {
    Working,
    LongTerm,
}

impl Layer {
    fn parse(value: &str) -> Option<Layer> {
        match value {
            "working" => Some(Layer::Working),
            "long_term" => Some(Layer::LongTerm),
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq)]
enum ValidatedOp {
    Set { layer: Layer, key: String, value: String, entry_type: Option<String>, carry_forward: bool },
    Delete { layer: Layer, key: String },
    FinishTask,
}

/// Причина отказа валидации — записывается в журнал (specs/memory-layers,
/// «Автоматический маршрутизатор распределяет записи по слоям»).
#[derive(Debug, PartialEq)]
enum RejectReason {
    UnknownLayer,
    EmptyKey,
    KeyTooLong,
    MissingValue,
    ValueTooLong,
    UnknownEntryType,
}

impl RejectReason {
    fn as_str(&self) -> &'static str {
        match self {
            RejectReason::UnknownLayer => "unknown_layer",
            RejectReason::EmptyKey => "empty_key",
            RejectReason::KeyTooLong => "key_too_long",
            RejectReason::MissingValue => "missing_value",
            RejectReason::ValueTooLong => "value_too_long",
            RejectReason::UnknownEntryType => "unknown_entry_type",
        }
    }
}

struct Limits {
    working_key_max_chars: usize,
    working_value_max_chars: usize,
    long_term_key_max_chars: usize,
    long_term_value_max_chars: usize,
}

impl Limits {
    fn from_config(config: &crate::config::AgentdConfig) -> Self {
        Self {
            working_key_max_chars: config.memory_working_key_max_chars as usize,
            working_value_max_chars: config.memory_working_value_max_chars as usize,
            long_term_key_max_chars: config.memory_long_term_key_max_chars as usize,
            long_term_value_max_chars: config.memory_long_term_value_max_chars as usize,
        }
    }
}

/// Валидирует одну операцию перед применением (specs/memory-layers):
/// неизвестный слой, пустой ключ, превышение операторских лимитов длины —
/// отбрасывают операцию независимо от остальных в пакете.
fn validate(op: RawMemoryOp, limits: &Limits) -> Result<ValidatedOp, RejectReason> {
    match op {
        RawMemoryOp::FinishTask { .. } => Ok(ValidatedOp::FinishTask),
        RawMemoryOp::Delete { layer, key, .. } => {
            let layer = Layer::parse(&layer).ok_or(RejectReason::UnknownLayer)?;
            let key = key.unwrap_or_default();
            if key.trim().is_empty() {
                return Err(RejectReason::EmptyKey);
            }
            let key_max = match layer {
                Layer::Working => limits.working_key_max_chars,
                Layer::LongTerm => limits.long_term_key_max_chars,
            };
            if key.chars().count() > key_max {
                return Err(RejectReason::KeyTooLong);
            }
            Ok(ValidatedOp::Delete { layer, key })
        }
        RawMemoryOp::Set { layer, key, value, entry_type, carry_forward, .. } => {
            let layer = Layer::parse(&layer).ok_or(RejectReason::UnknownLayer)?;
            let key = key.unwrap_or_default();
            if key.trim().is_empty() {
                return Err(RejectReason::EmptyKey);
            }
            let (key_max, value_max) = match layer {
                Layer::Working => (limits.working_key_max_chars, limits.working_value_max_chars),
                Layer::LongTerm => (limits.long_term_key_max_chars, limits.long_term_value_max_chars),
            };
            if key.chars().count() > key_max {
                return Err(RejectReason::KeyTooLong);
            }
            let value = value.ok_or(RejectReason::MissingValue)?;
            if value.chars().count() > value_max {
                return Err(RejectReason::ValueTooLong);
            }
            if layer == Layer::LongTerm {
                let entry_type_str = entry_type.as_deref().unwrap_or("knowledge");
                if !matches!(entry_type_str, "profile" | "decision" | "knowledge") {
                    return Err(RejectReason::UnknownEntryType);
                }
            }
            Ok(ValidatedOp::Set { layer, key, value, entry_type, carry_forward })
        }
    }
}

fn router_prompt(long_term: &[store::LongTermMemoryEntry], working: &[store::WorkingMemoryEntry], user_message: &str) -> String {
    let long_term_text = if long_term.is_empty() {
        "(пока пусто)".to_string()
    } else {
        long_term
            .iter()
            .map(|e| format!("- [{}] {}: {}", e.entry_type, e.key.as_deref().unwrap_or("(без ключа)"), e.value))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let working_text = if working.is_empty() {
        "(пока пусто)".to_string()
    } else {
        working.iter().map(|e| format!("- {}: {}", e.key, e.value)).collect::<Vec<_>>().join("\n")
    };
    format!(
        "Текущая долговременная память (профиль, решения, знания):\n{long_term_text}\n\n\
         Текущая рабочая память активной задачи:\n{working_text}\n\n\
         Новое сообщение пользователя:\n{user_message}\n\n\
         {MEMORY_ROUTER_MARKER} — только то, что нужно изменить. Каждый элемент — \
         {{\"layer\":\"working\"|\"long_term\",\"op\":\"set\",\"key\":\"...\",\"value\":\"...\",\
         \"entry_type\":\"profile\"|\"decision\"|\"knowledge\" (только для long_term),\
         \"carry_forward\":true|false (только для working, по умолчанию false),\"reason\":\"...\"}} для новой \
         или изменённой записи; {{\"layer\":\"working\"|\"long_term\",\"op\":\"delete\",\"key\":\"...\"}} для \
         удаления ключа; {{\"op\":\"finish_task\"}} для явного завершения текущей задачи (рабочая память \
         очищается, записи с carry_forward из этого же пакета переносятся в долговременную память). Если \
         менять ничего не нужно, верни пустой массив []. Ответь только JSON-массивом, без пояснений."
    )
}

/// Что сделал маршрутизатор за один вызов — счётчики уходят в блок
/// `context` СЛЕДУЮЩЕГО ответа (design.md, решение 5).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RouteOutcome {
    pub applied_set: u32,
    pub applied_update: u32,
    pub applied_delete: u32,
    pub rejected: u32,
}

/// Запускает маршрутизатор после записи обмена: читает текущее состояние
/// памяти, вызывает модель, разбирает и применяет операции по одной с
/// валидацией. Отказ вызова или разбора логируется и не меняет память
/// (specs/memory-layers, «Отказ маршрутизатора не блокирует ответ
/// пользователю»). Вызывается после `append_exchange`, как
/// `facts::update_after_exchange`.
pub async fn route_after_exchange(
    state: &AppState,
    chat: &store::Chat,
    settings: &ChatSettings,
    owner: &str,
    user_message: &str,
) -> RouteOutcome {
    let limits = Limits::from_config(&state.config);
    let long_term = store::load_long_term_memory(&state.db, owner, effective_long_term_max_entries(state, settings))
        .await
        .unwrap_or_default();
    let working = store::load_working_memory(&state.db, &chat.id, &chat.active_task_id).await.unwrap_or_default();

    let prompt = router_prompt(&long_term, &working, user_message);
    let mut router_settings = settings.clone();
    if let Some(model) = &state.config.memory_router_model {
        router_settings.model = Some(model.clone());
    }
    router_settings.reasoning = ReasoningMode::Default;
    router_settings.thinking = ThinkingMode::Disabled;
    router_settings.custom_response_mode = false;

    let reply = match state.agent.ask(&[Message::user(prompt)], &router_settings).await {
        Ok(reply) => reply,
        Err(err) => {
            tracing::warn!(chat_id = %chat.id, error = %err, "не удалось запросить операции маршрутизатора памяти: вызов модели не удался");
            return RouteOutcome::default();
        }
    };
    let Some(operations) = parse_operations(&reply.content) else {
        tracing::warn!(chat_id = %chat.id, "не удалось разобрать ответ маршрутизатора памяти как список операций");
        return RouteOutcome::default();
    };

    let working_keys: HashSet<String> = working.iter().map(|e| e.key.clone()).collect();
    let long_term_keys: HashSet<String> = long_term.iter().filter_map(|e| e.key.clone()).collect();

    let mut outcome = RouteOutcome::default();
    let now = agentcore::agent::now_secs();
    let mut carry_forward_keys = Vec::new();

    for raw in operations {
        match validate(raw, &limits) {
            Err(reason) => {
                let value_logged = state.config.log_content;
                tracing::info!(
                    chat_id = %chat.id,
                    reject_reason = reason.as_str(),
                    accepted = false,
                    log_content = value_logged,
                    "операция маршрутизатора памяти отброшена"
                );
                outcome.rejected += 1;
            }
            Ok(ValidatedOp::FinishTask) => {
                if let Err(err) = store::finish_task(&state.db, owner, &chat.id, &carry_forward_keys).await {
                    tracing::warn!(chat_id = %chat.id, error = %err, "не удалось выполнить finish_task маршрутизатора памяти");
                }
            }
            Ok(ValidatedOp::Set { layer, key, value, entry_type, carry_forward }) => {
                log_applied(state, chat, layer, "set", &key, &value);
                if layer == Layer::Working && carry_forward {
                    carry_forward_keys.push(key.clone());
                }
                let existed = match layer {
                    Layer::Working => working_keys.contains(&key),
                    Layer::LongTerm => long_term_keys.contains(&key),
                };
                let applied = match layer {
                    Layer::Working => {
                        store::set_working_memory(&state.db, &chat.id, &chat.active_task_id, &key, &value, "router", now).await
                    }
                    Layer::LongTerm => {
                        let entry_type = entry_type.as_deref().unwrap_or("knowledge");
                        store::set_long_term_memory(&state.db, owner, entry_type, Some(&key), &value, "router", Some(&chat.id), now)
                            .await
                            .map(|(_, applied)| applied)
                    }
                };
                match applied {
                    Ok(true) if existed => outcome.applied_update += 1,
                    Ok(true) => outcome.applied_set += 1,
                    Ok(false) => {}
                    Err(err) => tracing::warn!(chat_id = %chat.id, error = %err, "не удалось применить операцию set маршрутизатора памяти"),
                }
            }
            Ok(ValidatedOp::Delete { layer, key }) => {
                log_applied(state, chat, layer, "delete", &key, "");
                let result = match layer {
                    Layer::Working => store::delete_working_memory(&state.db, &chat.id, &chat.active_task_id, &key).await,
                    Layer::LongTerm => store::delete_long_term_memory_by_key(&state.db, owner, &key).await,
                };
                match result {
                    Ok(()) => outcome.applied_delete += 1,
                    Err(store::StoreError::NotFound) => {}
                    Err(err) => tracing::warn!(chat_id = %chat.id, error = %err, "не удалось применить операцию delete маршрутизатора памяти"),
                }
            }
        }
    }
    outcome
}

/// Значение (`value`) в журнале — только при `AGENTD_LOG_CONTENT=true`, тем
/// же приёмом, что и остальное содержимое (specs/memory-layers, «Тексты
/// записей памяти маскируются в журнале»).
fn log_applied(state: &AppState, chat: &store::Chat, layer: Layer, op: &str, key: &str, value: &str) {
    let layer_str = match layer {
        Layer::Working => "working",
        Layer::LongTerm => "long_term",
    };
    if state.config.log_content {
        tracing::info!(chat_id = %chat.id, layer = layer_str, op, key, value, accepted = true, "операция маршрутизатора памяти применена");
    } else {
        tracing::info!(chat_id = %chat.id, layer = layer_str, op, key, accepted = true, "операция маршрутизатора памяти применена");
    }
}

// --- Стратегия контекста memory_layers (design.md, решение 5) ---

fn message_from_stored(stored: ChatMessage) -> Message {
    Message {
        role: stored.role,
        content: stored.content,
        reasoning: stored.reasoning,
        meta: stored.meta,
    }
}

fn long_term_section(entries: &[store::LongTermMemoryEntry]) -> Option<String> {
    if entries.is_empty() {
        return None;
    }
    let lines: Vec<String> = entries
        .iter()
        .map(|e| format!("- [{}] {}: {}", e.entry_type, e.key.as_deref().unwrap_or("(без ключа)"), e.value))
        .collect();
    Some(format!("{LONG_TERM_SECTION_HEADING}\n{}", lines.join("\n")))
}

fn working_section(entries: &[store::WorkingMemoryEntry]) -> Option<String> {
    if entries.is_empty() {
        return None;
    }
    let lines: Vec<String> = entries.iter().map(|e| format!("- {}: {}", e.key, e.value)).collect();
    Some(format!("{WORKING_SECTION_HEADING}\n{}", lines.join("\n")))
}

fn chars_of(text: &str) -> u32 {
    text.chars().count() as u32
}

/// Итог сборки истории и разделов стратегии `memory_layers`.
pub struct MemoryLayersOutcome {
    pub history: Vec<Message>,
    pub sections: Vec<String>,
    pub long_term_entries: u32,
    pub long_term_chars: u32,
    pub working_entries: u32,
    pub working_chars: u32,
    pub short_term_messages: u32,
    pub short_term_chars: u32,
}

/// Сборка стратегии `memory_layers`: долговременная и рабочая память —
/// разделами системного сообщения (`[long_term_section, working_section]`,
/// пропуская пустые), хвост краткосрочной памяти — прямой вызов
/// `window::assemble` (design.md, решение 5: переиспользование, а не
/// копирование логики усечения окна).
pub async fn assemble(
    state: &AppState,
    chat: &store::Chat,
    settings: &ChatSettings,
    owner: &str,
    stored: Vec<store::ChatMessage>,
    new_message: Message,
) -> MemoryLayersOutcome {
    let long_term = store::load_long_term_memory(&state.db, owner, effective_long_term_max_entries(state, settings))
        .await
        .unwrap_or_default();
    let working = store::load_working_memory(&state.db, &chat.id, &chat.active_task_id).await.unwrap_or_default();
    let working: Vec<_> = working
        .into_iter()
        .take(effective_working_max_entries(state, settings) as usize)
        .collect();

    let long_term_chars: u32 = long_term.iter().map(|e| chars_of(&e.value)).sum();
    let working_chars: u32 = working.iter().map(|e| chars_of(&e.value)).sum();
    let long_term_entries = long_term.len() as u32;
    let working_entries = working.len() as u32;

    let mut sections = Vec::new();
    if let Some(section) = long_term_section(&long_term) {
        sections.push(section);
    }
    if let Some(section) = working_section(&working) {
        sections.push(section);
    }

    let tail_size = effective_short_term_tail(state, settings);
    let boundary = crate::summary::tail_boundary(&stored, tail_size);
    let short_term_messages = (stored.len() - boundary) as u32;
    let short_term_chars: u32 = stored[boundary..].iter().map(|m| chars_of(&m.content)).sum();

    let mut history: Vec<Message> = stored.into_iter().skip(boundary).map(message_from_stored).collect();
    history.push(new_message);

    MemoryLayersOutcome {
        history,
        sections,
        long_term_entries,
        long_term_chars,
        working_entries,
        working_chars,
        short_term_messages,
        short_term_chars,
    }
}

/// `context: ContextDto` — счётчики применённых/отброшенных операций
/// относятся к маршрутизации ПРЕДЫДУЩЕГО сообщения (маршрутизатор ещё не
/// отработал на момент сборки истории, design.md, решение 5); вызывающий
/// код подставляет их отдельно, как `facts_updated` у стратегии `facts`.
pub fn context_dto(outcome: &MemoryLayersOutcome, router: RouteOutcome) -> ContextDto {
    ContextDto::for_memory_layers(
        outcome.long_term_entries,
        outcome.long_term_chars,
        outcome.working_entries,
        outcome.working_chars,
        outcome.short_term_messages,
        outcome.short_term_chars,
        router.applied_set,
        router.applied_update,
        router.applied_delete,
        router.rejected,
    )
}

// --- Операторские умолчания и переопределения на чат (design.md, решение 6) ---

pub fn effective_router_enabled(state: &AppState, settings: &ChatSettings) -> bool {
    settings.memory_router_enabled.unwrap_or(state.config.memory_router_enabled)
}

pub fn effective_short_term_tail(state: &AppState, settings: &ChatSettings) -> u32 {
    settings.memory_short_term_tail.unwrap_or(state.config.memory_short_term_tail_messages)
}

pub fn effective_working_max_entries(state: &AppState, settings: &ChatSettings) -> u32 {
    settings.memory_working_max_entries.unwrap_or(state.config.memory_working_max_entries)
}

pub fn effective_long_term_max_entries(state: &AppState, settings: &ChatSettings) -> u32 {
    settings.memory_long_term_max_entries.unwrap_or(state.config.memory_long_term_max_entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agentcore::agent::Role;

    fn limits() -> Limits {
        Limits {
            working_key_max_chars: 10,
            working_value_max_chars: 20,
            long_term_key_max_chars: 10,
            long_term_value_max_chars: 20,
        }
    }

    fn working_entry(key: &str, value: &str) -> store::WorkingMemoryEntry {
        store::WorkingMemoryEntry { key: key.to_string(), value: value.to_string(), source: "manual".to_string(), updated_at: 0 }
    }

    fn long_term_entry(entry_type: &str, key: &str, value: &str) -> store::LongTermMemoryEntry {
        store::LongTermMemoryEntry {
            id: "id-1".to_string(),
            entry_type: entry_type.to_string(),
            key: Some(key.to_string()),
            value: value.to_string(),
            source: "manual".to_string(),
            source_chat_id: None,
            created_at: 0,
            updated_at: 0,
        }
    }

    fn message(role: Role, seq: i64, content: &str) -> ChatMessage {
        ChatMessage { seq, role, content: content.to_string(), reasoning: None, meta: None, created_at: 0 }
    }

    // --- 3.1 Разбор операций ---

    #[test]
    fn parses_set_delete_and_finish_task_operations() {
        let content = r#"вот ответ: [
            {"layer":"working","op":"set","key":"target","value":"iOS 17+"},
            {"layer":"long_term","op":"set","entry_type":"decision","key":"auth","value":"Clerk"},
            {"layer":"working","op":"delete","key":"draft"},
            {"op":"finish_task"}
        ]"#;
        let ops = parse_operations(content).expect("операции разобраны");
        assert_eq!(ops.len(), 4);
        assert!(matches!(&ops[0], RawMemoryOp::Set { layer, key: Some(k), .. } if layer == "working" && k == "target"));
        assert!(matches!(&ops[3], RawMemoryOp::FinishTask { .. }));
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

    // --- 3.2 Валидация операций ---

    #[test]
    fn unknown_layer_is_rejected() {
        let op = RawMemoryOp::Set {
            layer: "mystery".to_string(),
            key: Some("k".to_string()),
            value: Some("v".to_string()),
            entry_type: None,
            carry_forward: false,
            reason: None,
        };
        assert_eq!(validate(op, &limits()), Err(RejectReason::UnknownLayer));
    }

    #[test]
    fn empty_key_is_rejected() {
        let op = RawMemoryOp::Set {
            layer: "working".to_string(),
            key: Some("".to_string()),
            value: Some("v".to_string()),
            entry_type: None,
            carry_forward: false,
            reason: None,
        };
        assert_eq!(validate(op, &limits()), Err(RejectReason::EmptyKey));
    }

    #[test]
    fn key_over_limit_is_rejected() {
        let op = RawMemoryOp::Set {
            layer: "working".to_string(),
            key: Some("k".repeat(11)),
            value: Some("v".to_string()),
            entry_type: None,
            carry_forward: false,
            reason: None,
        };
        assert_eq!(validate(op, &limits()), Err(RejectReason::KeyTooLong));
    }

    #[test]
    fn value_over_limit_is_rejected() {
        let op = RawMemoryOp::Set {
            layer: "working".to_string(),
            key: Some("k".to_string()),
            value: Some("v".repeat(21)),
            entry_type: None,
            carry_forward: false,
            reason: None,
        };
        assert_eq!(validate(op, &limits()), Err(RejectReason::ValueTooLong));
    }

    #[test]
    fn unknown_entry_type_for_long_term_is_rejected() {
        let op = RawMemoryOp::Set {
            layer: "long_term".to_string(),
            key: Some("k".to_string()),
            value: Some("v".to_string()),
            entry_type: Some("mystery".to_string()),
            carry_forward: false,
            reason: None,
        };
        assert_eq!(validate(op, &limits()), Err(RejectReason::UnknownEntryType));
    }

    #[test]
    fn valid_set_operation_is_accepted() {
        let op = RawMemoryOp::Set {
            layer: "working".to_string(),
            key: Some("k".to_string()),
            value: Some("v".to_string()),
            entry_type: None,
            carry_forward: true,
            reason: None,
        };
        let validated = validate(op, &limits()).expect("валидная операция");
        assert!(matches!(validated, ValidatedOp::Set { carry_forward: true, .. }));
    }

    #[test]
    fn finish_task_is_always_valid() {
        let op = RawMemoryOp::FinishTask { reason: None };
        assert!(matches!(validate(op, &limits()), Ok(ValidatedOp::FinishTask)));
    }

    // --- 4.1/4.3 Сборка разделов ---

    #[test]
    fn long_term_section_is_absent_when_empty() {
        assert!(long_term_section(&[]).is_none());
    }

    #[test]
    fn working_section_is_absent_when_empty() {
        assert!(working_section(&[]).is_none());
    }

    #[test]
    fn long_term_section_lists_entries_with_type() {
        let section = long_term_section(&[long_term_entry("decision", "auth", "Clerk")]).expect("раздел");
        assert!(section.contains(LONG_TERM_SECTION_HEADING));
        assert!(section.contains("decision"));
        assert!(section.contains("auth: Clerk"));
    }

    #[test]
    fn working_section_lists_entries() {
        let section = working_section(&[working_entry("target", "iOS 17+")]).expect("раздел");
        assert!(section.contains(WORKING_SECTION_HEADING));
        assert!(section.contains("target: iOS 17+"));
    }

    // --- 4.2 Порядок разделов ---

    #[tokio::test]
    async fn assemble_orders_long_term_section_before_working_section() {
        let state = AppState::for_tests().await;
        let chat = store::create_chat(&state.db, "owner-1", "Чат", &ChatSettings::default()).await.expect("чат");
        store::set_long_term_memory(&state.db, "owner-1", "decision", Some("auth"), "Clerk", "manual", None, 1)
            .await
            .expect("долговременная запись");
        store::set_working_memory(&state.db, &chat.id, &chat.active_task_id, "target", "iOS 17+", "manual", 1)
            .await
            .expect("рабочая запись");

        let outcome = assemble(&state, &chat, &chat.settings, "owner-1", Vec::new(), Message::user("новое")).await;
        assert_eq!(outcome.sections.len(), 2);
        assert!(outcome.sections[0].starts_with(LONG_TERM_SECTION_HEADING));
        assert!(outcome.sections[1].starts_with(WORKING_SECTION_HEADING));
    }

    #[tokio::test]
    async fn assemble_short_term_tail_is_limited_to_configured_size() {
        let state = AppState::for_tests().await;
        let chat = store::create_chat(&state.db, "owner-1", "Чат", &ChatSettings::default()).await.expect("чат");
        let stored: Vec<ChatMessage> = (1..=20)
            .map(|seq| {
                if seq % 2 == 1 {
                    message(Role::User, seq, &format!("вопрос {seq}"))
                } else {
                    message(Role::Assistant, seq, &format!("ответ {seq}"))
                }
            })
            .collect();
        let mut settings = chat.settings.clone();
        settings.memory_short_term_tail = Some(6);

        let outcome = assemble(&state, &chat, &settings, "owner-1", stored, Message::user("новое")).await;
        assert_eq!(outcome.short_term_messages, 6);
        assert_eq!(outcome.history.len(), 7, "6 сообщений хвоста плюс новое");
    }

    // --- 4.5 Счётчики блока context ---

    #[test]
    fn context_dto_counters_reflect_actual_assembly() {
        let outcome = MemoryLayersOutcome {
            history: Vec::new(),
            sections: Vec::new(),
            long_term_entries: 2,
            long_term_chars: 40,
            working_entries: 1,
            working_chars: 10,
            short_term_messages: 3,
            short_term_chars: 30,
        };
        let router = RouteOutcome { applied_set: 1, applied_update: 0, applied_delete: 0, rejected: 1 };
        let dto = context_dto(&outcome, router);
        assert_eq!(dto.memory_long_term_entries, Some(2));
        assert_eq!(dto.memory_working_entries, Some(1));
        assert_eq!(dto.memory_short_term_messages, Some(3));
        assert_eq!(dto.memory_router_applied_set, Some(1));
        assert_eq!(dto.memory_router_rejected, Some(1));
    }

    // --- 5.2 Настройка чата переопределяет операторское умолчание ---

    #[tokio::test]
    async fn chat_setting_overrides_operator_default_short_term_tail() {
        let state = AppState::for_tests().await;
        assert_eq!(effective_short_term_tail(&state, &ChatSettings::default()), state.config.memory_short_term_tail_messages);
        let settings = ChatSettings { memory_short_term_tail: Some(4), ..ChatSettings::default() };
        assert_eq!(effective_short_term_tail(&state, &settings), 4);
    }

    // --- 7.1 Значение записи маскируется в журнале по умолчанию ---

    fn captured(f: impl FnOnce()) -> String {
        let capture = crate::telemetry::capture::Capture::default();
        let subscriber = tracing_subscriber::fmt().json().with_writer(capture.clone()).finish();
        tracing::subscriber::with_default(subscriber, f);
        capture.text()
    }

    #[tokio::test]
    async fn applied_operation_value_is_absent_from_log_by_default() {
        let mut state = AppState::for_tests().await;
        std::sync::Arc::make_mut(&mut state.config).log_content = false;
        let chat = store::create_chat(&state.db, "owner-1", "Чат", &ChatSettings::default()).await.expect("чат");

        let output = captured(|| {
            log_applied(&state, &chat, Layer::Working, "set", "budget", "секретное значение 200000");
        });
        assert!(output.contains("\"key\":\"budget\""), "ключ должен остаться в журнале: {output}");
        assert!(!output.contains("секретное значение"), "значение не должно попасть в журнал: {output}");
    }

    #[tokio::test]
    async fn applied_operation_value_is_logged_when_enabled() {
        let mut state = AppState::for_tests().await;
        std::sync::Arc::make_mut(&mut state.config).log_content = true;
        let chat = store::create_chat(&state.db, "owner-1", "Чат", &ChatSettings::default()).await.expect("чат");

        let output = captured(|| {
            log_applied(&state, &chat, Layer::Working, "set", "budget", "видимое значение");
        });
        assert!(output.contains("видимое значение"), "значение должно попасть в журнал при AGENTD_LOG_CONTENT=true: {output}");
    }
}
