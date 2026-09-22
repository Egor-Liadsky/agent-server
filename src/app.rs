//! Сборка HTTP-приложения: маршруты, слои и обработчики.

use crate::dto::{
    AppendMessagesRequest, AppendMessagesResponse, BranchDto, BranchesResponse, ChatDto, ChatRequest,
    ChatResponse, ChatWithMessagesResponse, CreateBranchRequest, CreateChatRequest, CreateProfileRequest,
    DeleteLongTermMemoryQuery, DeleteWorkingMemoryQuery, FactDto, FactsResponse, FinishTaskRequest, GetChatQuery,
    ListChatsQuery, ListChatsResponse, LongTermMemoryEntryDto, LongTermMemoryResponse, MessageView, ModelsResponse,
    ProfileDto, ProfilesResponse, SetFactRequest, SetLongTermMemoryRequest, SetWorkingMemoryRequest,
    TaskStateDto, TaskTransitionDto, TaskTransitionRequest, TaskTransitionResponse, UpdateChatRequest,
    UpdateProfileRequest, WorkingMemoryEntryDto, WorkingMemoryResponse,
};
use crate::context;
use crate::error::ApiError;
use crate::middleware::{assign_request_id, authenticate, normalize_errors, ClientId, Owner, RequestId};
use crate::state::AppState;
use crate::store;
use crate::summary;
use crate::telemetry::{log_exchange, ExchangeRecord};
use agentcore::agent::{Agent, Message, Role, ToolCall, ToolSpec};
use agentcore::config::{ChatSettings, Provider, ReasoningMode, ThinkingMode};
use agentcore::pipeline::{Pipeline, PipelineOutcome, RequestContext};
use axum::extract::{Extension, Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use axum::error_handling::HandleErrorLayer;
use tower::limit::ConcurrencyLimitLayer;
use tower::load_shed::LoadShedLayer;
use tower::timeout::TimeoutLayer;
use tower::{BoxError, ServiceBuilder};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;

/// История для вызова модели читается целиком: обычная переписка укладывается
/// в этот предел с большим запасом, а бесконечный рост чата — вне области
/// изменения (design.md, non-goal «ретеншен, обрезка истории»).
const MAX_HISTORY_MESSAGES: u32 = 100_000;
const DEFAULT_CHATS_LIMIT: u32 = 50;
const MAX_CHATS_LIMIT: u32 = 200;
const DEFAULT_MESSAGES_LIMIT: u32 = 200;
const MAX_MESSAGES_LIMIT: u32 = 500;
pub(crate) const DEFAULT_CHAT_TITLE: &str = "Новый чат";
pub(crate) const MAX_CHAT_TITLE_LEN: usize = 200;
/// Предел одной дозаписи: обмен с локальной моделью — это две реплики, а
/// запас нужен только на повтор после неудачи, не на выгрузку истории.
const MAX_APPEND_MESSAGES: usize = 100;
/// Предел числа инструментов в одном запросе: описания уходят провайдеру в
/// каждом запросе хода, и длинный список лишь съедает контекст.
const MAX_TOOLS: usize = 128;
/// Предел длины имени инструмента — тот же, что у провайдеров
/// OpenAI-совместимого формата.
const MAX_TOOL_NAME_LEN: usize = 64;

/// Живость процесса. Никаких обращений к провайдеру: эндпоинт отвечает,
/// пока жив сам процесс.
async fn healthz() -> &'static str {
    "ok"
}

/// Готовность: конфигурация валидна и хранилище отвечает на проверочный
/// запрос. `healthz` выше сюда намеренно не заходит: живость процесса не
/// должна зависеть от базы.
async fn readyz(State(state): State<AppState>) -> Response {
    if !state.config.is_ready() {
        return ApiError::not_ready().into_response();
    }
    if crate::store::ping(&state.db).await.is_err() {
        return ApiError::not_ready().into_response();
    }
    (StatusCode::OK, "ready").into_response()
}

async fn models(State(state): State<AppState>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        models: state.config.allowed_models.clone(),
    })
}

/// Ключ провайдера принадлежит сервису: тело с `api_key` отклоняется явно,
/// а не обрабатывается с игнорированием поля.
fn contains_api_key(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Object(map) => {
            map.contains_key("api_key")
                || map.values().any(contains_api_key)
        }
        serde_json::Value::Array(items) => items.iter().any(contains_api_key),
        _ => false,
    }
}

/// Инварианты — конфигурация оператора: тело запроса с полем `invariants`
/// отклоняется явно, тем же приёмом, что и клиентский `api_key` (spec.md,
/// «Клиент не может подменить или дополнить инварианты»).
fn contains_invariants_field(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Object(map) => {
            map.contains_key("invariants")
                || map.values().any(contains_invariants_field)
        }
        serde_json::Value::Array(items) => items.iter().any(contains_invariants_field),
        _ => false,
    }
}

/// Приближённая оценка размера истории в токенах: точного токенизатора
/// провайдера (`deepseek-v4-flash`/`deepseek-v4-pro` через `agentupstream`, либо Ollama) у сервиса
/// нет, поэтому размер оценивается по длине текста, а не подсчитывается
/// точно (design.md, решение «Оценка размера — по длине текста»).
fn estimate_tokens(history: &[Message]) -> u32 {
    let chars: usize = history
        .iter()
        .map(|message| {
            let calls = if message.tool_calls.is_empty() {
                0
            } else {
                serde_json::to_string(&message.tool_calls)
                    .map(|json| json.chars().count())
                    .unwrap_or(0)
            };
            role_str(&message.role).len() + message.content.chars().count() + calls
        })
        .sum();
    (chars as u64).div_ceil(4) as u32
}

fn role_str(role: &agentcore::agent::Role) -> &'static str {
    match role {
        agentcore::agent::Role::User => "user",
        agentcore::agent::Role::Assistant => "assistant",
        agentcore::agent::Role::System => "system",
        agentcore::agent::Role::Tool => "tool",
    }
}

/// Проверка описаний инструментов из запроса. Пустой список равносилен
/// отсутствию поля.
fn validate_tools(tools: Option<Vec<ToolSpec>>) -> Result<Vec<ToolSpec>, ApiError> {
    let tools = tools.unwrap_or_default();
    if tools.len() > MAX_TOOLS {
        return Err(ApiError::tools_invalid(format!(
            "инструментов не может быть больше {MAX_TOOLS}"
        )));
    }
    let mut names = std::collections::HashSet::new();
    for tool in &tools {
        let valid_name = !tool.name.is_empty()
            && tool.name.len() <= MAX_TOOL_NAME_LEN
            && tool
                .name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
        if !valid_name {
            return Err(ApiError::tools_invalid(format!(
                "недопустимое имя инструмента «{}»: разрешены латинские буквы, цифры, _ и -, до {MAX_TOOL_NAME_LEN} символов",
                tool.name
            )));
        }
        if !names.insert(tool.name.as_str()) {
            return Err(ApiError::tools_invalid(format!(
                "инструмент {} описан дважды",
                tool.name
            )));
        }
        if !tool.parameters.is_object() {
            return Err(ApiError::tools_invalid(format!(
                "parameters инструмента {} должен быть JSON-объектом",
                tool.name
            )));
        }
    }
    Ok(tools)
}

/// Проверка связей «вызов → результат» в истории, присланной целиком: каждый
/// результат роли `tool` отвечает на вызов ответа модели, после которого
/// стоит (в сплошной группе результатов), по `tool_call_id`, а без него — по
/// имени инструмента.
fn validate_tool_messages(history: &[Message]) -> Result<(), ApiError> {
    let mut calls: Option<&[ToolCall]> = None;
    for message in history {
        match message.role {
            Role::Tool => {
                let matched = calls.is_some_and(|calls| {
                    calls.iter().any(|call| match &message.tool_call_id {
                        Some(id) => &call.id == id,
                        None => message.tool_name.as_deref() == Some(call.name.as_str()),
                    })
                });
                if !matched {
                    return Err(ApiError::tool_results_mismatch(
                        "результат инструмента не ссылается на вызов предыдущего ответа модели",
                    ));
                }
            }
            _ => {
                calls = (!message.tool_calls.is_empty()).then_some(message.tool_calls.as_slice());
            }
        }
    }
    Ok(())
}

/// Эффективный лимит контекстного окна на этот запрос: минимум из заданных
/// значений. Оператор задаёт только потолок — клиентское значение действует
/// самостоятельно и тогда, когда операторский лимит не настроен
/// (specs/chat-context-limit, «Эффективный лимит запроса»; design.md,
/// решение 1).
fn effective_context_limit(
    operator_default: Option<u32>,
    client_value: Option<u32>,
) -> Result<Option<u32>, ApiError> {
    match (operator_default, client_value) {
        (None, None) => Ok(None),
        (Some(default), None) => Ok(Some(default)),
        (None, Some(client)) => Ok(Some(client)),
        (Some(default), Some(client)) if client <= default => Ok(Some(client)),
        (Some(default), Some(_client)) => Err(ApiError::context_limit_invalid(format!(
            "max_context_tokens не может превышать операторский лимит {default}"
        ))),
    }
}

/// Проверка эффективного лимита перед вызовом конвейера: если оценка размера
/// истории превышает лимит, запрос отклоняется до обращения к провайдеру
/// (specs/context-limit, «Отказ при превышении эффективного лимита»).
fn check_context_limit(history: &[Message], limit: Option<u32>) -> Result<(), ApiError> {
    let Some(limit) = limit else {
        return Ok(());
    };
    let estimated = estimate_tokens(history);
    if estimated > limit {
        return Err(ApiError::context_limit_exceeded(format!(
            "оценка размера истории ({estimated} токенов, приближённо) превышает лимит контекстного окна ({limit})"
        )));
    }
    Ok(())
}

/// Эффективные настройки компактизации на этот запрос: клиентское значение
/// поверх операторского умолчания — то же правило присутствия, что и у
/// `max_context_tokens` (specs/context-summary, «Настройки компактизации на
/// уровне чата»).
struct EffectiveSummarySettings {
    enabled: bool,
    keep_messages: u32,
    step_messages: u32,
}

fn effective_summary_settings(state: &AppState, settings: &ChatSettings) -> EffectiveSummarySettings {
    EffectiveSummarySettings {
        enabled: settings.summary_enabled.unwrap_or(state.config.summary_enabled),
        keep_messages: settings
            .summary_keep_messages
            .unwrap_or(state.config.summary_keep_messages),
        step_messages: settings
            .summary_step_messages
            .unwrap_or(state.config.summary_step_messages),
    }
}

/// Итог компактизации: история, которую увидит провайдер, и сведения для
/// блока `context` в ответе (specs/context-summary, «Наблюдаемость
/// компактизации»).
pub(crate) struct CompactionOutcome {
    pub(crate) history: Vec<Message>,
    pub(crate) replaced_messages: u32,
    pub(crate) summary_built: bool,
    /// Раздел пересказа для системного сообщения запроса. `None` — чат
    /// целиком укладывается в дословный хвост (specs/context-summary,
    /// «Короткий чат не получает раздела пересказа»).
    pub(crate) summary_section: Option<String>,
}

fn history_without_compaction(stored: Vec<store::ChatMessage>, new_messages: Vec<Message>) -> CompactionOutcome {
    let mut history: Vec<Message> = stored.into_iter().map(message_from_stored).collect();
    history.extend(new_messages);
    CompactionOutcome {
        history,
        replaced_messages: 0,
        summary_built: false,
        summary_section: None,
    }
}

/// Компактизация истории чата перед вызовом модели: последние `N` сообщений
/// дословно, старше — пересказом, обновляемым ступенчато. Выполняется до
/// проверки лимита контекста, чтобы спасать чат от `context_limit_exceeded`
/// (specs/context-summary; design.md, решения 5-9).
pub(crate) async fn compact_history(
    state: &AppState,
    chat_id: &str,
    settings: &ChatSettings,
    stored: Vec<store::ChatMessage>,
    new_messages: Vec<Message>,
) -> CompactionOutcome {
    let effective = effective_summary_settings(state, settings);
    if !effective.enabled {
        return history_without_compaction(stored, new_messages);
    }
    let boundary = summary::tail_boundary(&stored, effective.keep_messages);
    if boundary == 0 {
        return history_without_compaction(stored, new_messages);
    }

    let existing = match store::load_summary(&state.db, chat_id).await {
        Ok(summary) => summary,
        Err(err) => {
            tracing::warn!(chat_id, error = %err, "не удалось прочитать пересказ чата");
            None
        }
    };
    let (previous_summary, through_seq) = match &existing {
        Some(s) => (Some(s.summary.clone()), s.through_seq),
        None => (None, 0),
    };

    let pending = summary::pending_messages(&stored, boundary, through_seq);
    let reached_step = pending.len() as u32 >= effective.step_messages;

    if !reached_step {
        return match previous_summary {
            // Прежний пересказ есть — подставляем его без обращения к модели.
            Some(text) => {
                let tail = &stored[boundary..];
                let history = summary::assemble_history(tail, new_messages);
                CompactionOutcome {
                    history,
                    replaced_messages: boundary as u32,
                    summary_built: false,
                    summary_section: Some(summary::summary_section(&text)),
                }
            }
            // Пересказа ещё нет, а порог не достигнут: компактизация ещё не
            // начала действовать — история уходит целиком.
            None => history_without_compaction(stored, new_messages),
        };
    }

    let prompt = summary::summary_prompt(previous_summary.as_deref(), &pending, state.config.summary_max_chars);
    let mut summary_settings = settings.clone();
    if let Some(model) = &state.config.summary_model {
        summary_settings.model = Some(model.clone());
    }
    summary_settings.reasoning = ReasoningMode::Default;
    summary_settings.thinking = ThinkingMode::Disabled;
    summary_settings.custom_response_mode = false;

    let built = state.agent.ask(&[Message::user(prompt)], &summary_settings).await;
    let new_through_seq = stored[boundary - 1].seq;
    let (final_summary, summary_built) = match built {
        Ok(reply) => {
            let truncated = summary::truncate_summary(&reply.content, state.config.summary_max_chars);
            match store::save_summary(&state.db, chat_id, &truncated, new_through_seq).await {
                Ok(()) => {
                    tracing::info!(
                        chat_id,
                        replaced_messages = boundary,
                        through_seq = new_through_seq,
                        "пересказ истории чата построен"
                    );
                    if state.config.log_content {
                        tracing::info!(chat_id, summary = %truncated, "текст построенного пересказа");
                    }
                    (Some(truncated), true)
                }
                Err(err) => {
                    tracing::warn!(chat_id, error = %err, "не удалось сохранить пересказ чата");
                    (previous_summary, false)
                }
            }
        }
        Err(err) => {
            tracing::warn!(chat_id, error = %err, "не удалось построить пересказ чата");
            (previous_summary, false)
        }
    };

    match final_summary {
        Some(text) => {
            let tail = &stored[boundary..];
            let history = summary::assemble_history(tail, new_messages);
            CompactionOutcome {
                history,
                replaced_messages: boundary as u32,
                summary_built,
                summary_section: Some(summary::summary_section(&text)),
            }
        }
        None => history_without_compaction(stored, new_messages),
    }
}

fn history_from(request: &ChatRequest) -> Result<Vec<Message>, ApiError> {
    let prompt = request
        .prompt
        .as_ref()
        .map(|prompt| prompt.trim())
        .filter(|prompt| !prompt.is_empty());
    let messages = request
        .messages
        .as_ref()
        .filter(|messages| !messages.is_empty());

    match (prompt, messages) {
        (Some(_), Some(_)) => Err(ApiError::invalid_request(
            "задайте ровно одно из полей prompt и messages",
        )),
        (None, None) => Err(ApiError::invalid_request(
            "задайте одно из полей prompt или messages",
        )),
        (Some(prompt), None) => Ok(vec![Message::user(prompt)]),
        (None, Some(messages)) => {
            let history: Vec<Message> = messages
                .iter()
                .cloned()
                .map(|message| message.into_message())
                .collect();
            validate_tool_messages(&history)?;
            Ok(history)
        }
    }
}

/// Настройки сервиса по умолчанию для нового чата или разового запроса без
/// чата: только заданная сервисом модель, всё остальное — умолчания ядра.
fn server_defaults(state: &AppState) -> ChatSettings {
    ChatSettings {
        model: Some(state.config.model.clone()),
        ..ChatSettings::default()
    }
}

/// Проверка настроек перед вызовом модели: белый список моделей и наличие
/// своего адреса Ollama — про то, что сервис собирается вызывать сам.
fn validate_settings_for_call(state: &AppState, settings: &ChatSettings) -> Result<(), ApiError> {
    if settings.provider == Provider::Ollama {
        // Локальную модель пользователя сервис вызвать не может: без своего
        // адреса Ollama диалог в таком чате идёт мимо сервиса.
        if state.config.ollama_url.is_none() {
            return Err(ApiError::invalid_request(
                "провайдер ollama не настроен на этом сервисе",
            ));
        }
        return Ok(());
    }
    let model = settings
        .model
        .clone()
        .unwrap_or_else(|| state.config.model.clone());
    if !state.config.is_model_allowed(&model) {
        return Err(ApiError::invalid_request(format!(
            "модель {model} не разрешена конфигурацией сервиса"
        )));
    }
    Ok(())
}

/// Проверка настроек перед сохранением в чате. Чат с локальным провайдером
/// обслуживает клиент, а сервис для него — только хранилище, поэтому ни
/// белый список моделей, ни свой адрес Ollama к нему не применяются
/// (specs/chat-message-api, «Настройки чата с локальным провайдером
/// принимаются на хранение»).
fn validate_settings_for_storage(state: &AppState, settings: &ChatSettings) -> Result<(), ApiError> {
    if settings.provider == Provider::Ollama {
        return Ok(());
    }
    validate_settings_for_call(state, settings)
}

/// Накладывает необязательный DTO настроек поверх базовых значений и
/// проверяет результат. `base` — умолчания сервиса при создании чата или
/// разовом запросе без чата, сохранённые настройки чата — при работе с
/// существующим чатом.
async fn merge_settings(
    state: &AppState,
    owner: &str,
    base: ChatSettings,
    dto: Option<crate::dto::ChatSettingsDto>,
    purpose: SettingsPurpose,
) -> Result<ChatSettings, ApiError> {
    let context_strategy_override = dto.as_ref().and_then(|d| d.context_strategy.clone());
    let context_window_override = dto.as_ref().and_then(|d| d.context_window_messages);
    let profile_override = dto.as_ref().and_then(|d| d.profile_id.clone());
    let mut settings = match dto {
        Some(dto) => dto.apply_to(base).map_err(ApiError::invalid_request)?,
        None => base,
    };
    // Лимит контекста относится к размеру истории, а не к способу вызова
    // модели: проверяется при обоих SettingsPurpose и для любого провайдера,
    // включая ollama (specs/chat-context-limit, «Проверка лимита при
    // сохранении настроек чата»; design.md, решение 2).
    effective_context_limit(state.config.max_context_tokens, settings.max_context_tokens)?;
    validate_summary_settings(state, &settings)?;
    apply_context_strategy(state, &mut settings, context_strategy_override, context_window_override)?;
    apply_profile(state, owner, &mut settings, profile_override).await?;
    match purpose {
        SettingsPurpose::Call => validate_settings_for_call(state, &settings)?,
        SettingsPurpose::Storage => validate_settings_for_storage(state, &settings)?,
    }
    Ok(settings)
}

/// Разбор и проверка `profile_id`, вне `ChatSettingsDto::apply_to`: тот же
/// приём, что и у `context_strategy` — собственный код ошибки
/// (`profile_invalid`), а не общий `invalid_request`
/// (specs/user-profiles, «Неизвестный профиль отклоняется явной ошибкой»;
/// design.md, решение 5). Проверяется на пути и сохранения, и вызова:
/// сохранить ссылку на несуществующий или чужой профиль, которая сорвёт
/// каждый следующий запрос, нельзя.
async fn apply_profile(
    state: &AppState,
    owner: &str,
    settings: &mut ChatSettings,
    profile_override: Option<Option<String>>,
) -> Result<(), ApiError> {
    if let Some(value) = profile_override {
        settings.profile_id = value;
    }
    if let Some(id) = settings.profile_id.clone() {
        let found = crate::profile::find(state, owner, &id)
            .await
            .map_err(ApiError::from_store_error)?;
        if found.is_none() {
            return Err(ApiError::profile_invalid(format!(
                "неизвестный профиль: {id}"
            )));
        }
    }
    Ok(())
}

/// Разбор и проверка `context_strategy`/`context_window_messages`, вне
/// `ChatSettingsDto::apply_to`: у этих полей собственные коды ошибок
/// (`context_strategy_invalid`, `context_strategy_not_allowed`,
/// `context_window_invalid`), а не общий `invalid_request`
/// (specs/context-strategies, specs/context-sliding-window). Проверяется на
/// пути и сохранения, и вызова: сохранить запрещённую стратегию, которая
/// сорвёт каждый следующий запрос, нельзя (design.md, решение 6).
fn apply_context_strategy(
    state: &AppState,
    settings: &mut ChatSettings,
    strategy_override: Option<Option<String>>,
    window_override: Option<Option<u32>>,
) -> Result<(), ApiError> {
    if let Some(value) = strategy_override {
        settings.context_strategy = match value {
            None => None,
            Some(name) => Some(
                agentcore::config::ContextStrategy::parse(&name).ok_or_else(|| {
                    ApiError::context_strategy_invalid(format!("неизвестная стратегия контекста: {name}"))
                })?,
            ),
        };
    }
    if let Some(value) = window_override {
        settings.context_window_messages = value;
    }

    let effective = context::effective_strategy(state, settings);
    if !state.config.is_context_strategy_allowed(effective) {
        return Err(ApiError::context_strategy_not_allowed(format!(
            "стратегия {} не входит в список стратегий, разрешённых оператором",
            effective.as_str()
        )));
    }

    if let Some(window) = settings.context_window_messages {
        if window == 0 || window > state.config.context_window_messages {
            return Err(ApiError::context_window_invalid(format!(
                "context_window_messages не может быть нулём или превышать операторский потолок {}",
                state.config.context_window_messages
            )));
        }
    }
    Ok(())
}

/// Границы клиентских настроек компактизации: клиент может только сузить
/// дословный хвост (не больше операторского потолка) и только разредить шаг
/// пересказа (не меньше операторской нижней границы); ноль недопустим ни для
/// одного значения (specs/context-summary, «Границы клиентских настроек
/// компактизации»).
fn validate_summary_settings(state: &AppState, settings: &ChatSettings) -> Result<(), ApiError> {
    if let Some(keep) = settings.summary_keep_messages
        && (keep == 0 || keep > state.config.summary_keep_messages)
    {
        return Err(ApiError::summary_settings_invalid(format!(
            "summary_keep_messages не может быть нулём или превышать операторский потолок {}",
            state.config.summary_keep_messages
        )));
    }
    if let Some(step) = settings.summary_step_messages
        && (step == 0 || step < state.config.summary_step_messages)
    {
        return Err(ApiError::summary_settings_invalid(format!(
            "summary_step_messages не может быть нулём или быть меньше операторской нижней границы {}",
            state.config.summary_step_messages
        )));
    }
    Ok(())
}

/// Зачем проверяются настройки: сервис собирается вызвать модель сам или
/// только сохранить настройки чата.
#[derive(Debug, Clone, Copy)]
enum SettingsPurpose {
    Call,
    Storage,
}

fn validate_title(title: Option<String>) -> Result<Option<String>, ApiError> {
    match title {
        None => Ok(None),
        Some(title) => {
            let trimmed = title.trim();
            if trimmed.is_empty() || trimmed.chars().count() > MAX_CHAT_TITLE_LEN {
                return Err(ApiError::invalid_request(format!(
                    "заголовок чата должен быть непустым и не длиннее {MAX_CHAT_TITLE_LEN} символов"
                )));
            }
            Ok(Some(trimmed.to_string()))
        }
    }
}

fn message_from_stored(message: store::ChatMessage) -> Message {
    message.into_message()
}

/// Текст запроса для журнала: он записывается только при включённом
/// признаке записи содержимого.
fn prompt_preview(body: &serde_json::Value) -> String {
    if let Some(prompt) = body.get("prompt").and_then(|value| value.as_str()) {
        return prompt.to_string();
    }
    // Результаты инструментов — содержимое запроса наравне с `prompt`:
    // попадают в журнал только при включённой записи содержимого.
    if let Some(results) = body.get("tool_results").and_then(|value| value.as_array()) {
        return results
            .iter()
            .filter_map(|result| result.get("content").and_then(|content| content.as_str()))
            .collect::<Vec<_>>()
            .join("\n");
    }
    body.get("messages")
        .and_then(|value| value.as_array())
        .and_then(|messages| messages.last())
        .and_then(|message| message.get("content"))
        .and_then(|content| content.as_str())
        .unwrap_or_default()
        .to_string()
}

async fn chat(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
    client: Option<Extension<ClientId>>,
    Json(body): Json<serde_json::Value>,
) -> Response {
    let started_at = std::time::Instant::now();
    let client = client
        .map(|Extension(ClientId(id))| id)
        .unwrap_or_else(|| crate::middleware::ANONYMOUS_CLIENT.to_string());
    let log_content = state.config.log_content;
    let prompt = prompt_preview(&body);

    let result = handle_chat(state, request_id.clone(), owner, body).await;
    let duration_ms = started_at.elapsed().as_millis() as u64;

    match result {
        Ok(response) => {
            log_exchange(ExchangeRecord {
                request_id: &request_id,
                client: &client,
                model: &response.model,
                duration_ms,
                status: StatusCode::OK.as_u16(),
                prompt_tokens: response.usage.prompt_tokens,
                completion_tokens: response.usage.completion_tokens,
                total_tokens: response.usage.total_tokens,
                prompt: log_content.then_some(prompt.as_str()),
                response: log_content.then_some(response.content.as_str()),
            });
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(err) => {
            let err = err.with_request_id(request_id.clone());
            log_exchange(ExchangeRecord {
                request_id: &request_id,
                client: &client,
                model: "",
                duration_ms,
                status: err.status.as_u16(),
                prompt: log_content.then_some(prompt.as_str()),
                ..ExchangeRecord::default()
            });
            err.into_response()
        }
    }
}

async fn handle_chat(
    state: AppState,
    request_id: String,
    owner: String,
    body: serde_json::Value,
) -> Result<ChatResponse, ApiError> {
    if contains_api_key(&body) {
        return Err(ApiError::invalid_request(
            "ключ провайдера задаёт сервис: поле api_key в запросе не принимается",
        ));
    }
    if contains_invariants_field(&body) {
        return Err(ApiError::invalid_request(
            "инварианты задаёт конфигурация сервиса: поле invariants в запросе не принимается",
        ));
    }
    let request: ChatRequest = serde_json::from_value(body)
        .map_err(|err| ApiError::invalid_request(format!("тело запроса не разобрано: {err}")))?;

    match request.chat_id.clone() {
        Some(chat_id) => handle_chat_in_existing(state, request_id, owner, chat_id, request).await,
        None => handle_chat_without_storage(state, request_id, owner, request).await,
    }
}

/// `POST /v1/chat` без `chat_id`: поведение не меняется — история приходит в
/// теле, ничего не пишется в хранилище (specs/chat-api, «Разовый вызов без
/// чата сохраняется»).
async fn handle_chat_without_storage(
    state: AppState,
    request_id: String,
    owner: String,
    request: ChatRequest,
) -> Result<ChatResponse, ApiError> {
    if request.tool_results.as_ref().is_some_and(|results| !results.is_empty()) {
        return Err(ApiError::invalid_request(
            "tool_results принимаются только вместе с chat_id: без чата результаты передаются сообщениями роли tool в messages",
        ));
    }
    let tools = validate_tools(request.tools.clone())?;
    let history = history_from(&request)?;
    let settings =
        merge_settings(&state, &owner, server_defaults(&state), request.settings, SettingsPurpose::Call).await?;
    let model = settings
        .model
        .clone()
        .unwrap_or_else(|| state.config.model.clone());

    let limit = effective_context_limit(state.config.max_context_tokens, settings.max_context_tokens)?;
    check_context_limit(&history, limit)?;

    let pipeline = Pipeline::new(state.agent.clone())
        .with_output_policies(agentcore::invariants::default_output_policies(state.agent.clone()));
    let context = RequestContext::new(request_id.clone(), history, settings)
        .with_invariants((*state.invariants).clone())
        .with_tools(tools);
    let (reply, policy) = run_pipeline(&pipeline, context, &request_id).await?;
    Ok(ChatResponse::new(request_id, model, &reply, policy))
}

/// Что клиент прислал в существующий чат: новую реплику или результаты
/// инструментов, продолжающие незавершённый ход.
enum TurnInput {
    Prompt(String),
    ToolResults(Vec<crate::dto::ToolResultDto>),
}

fn turn_input(request: &ChatRequest) -> Result<TurnInput, ApiError> {
    if request.messages.is_some() {
        return Err(ApiError::invalid_request(
            "поле messages вместе с chat_id не принимается: история берётся из чата",
        ));
    }
    let prompt = request
        .prompt
        .as_deref()
        .map(str::trim)
        .filter(|prompt| !prompt.is_empty());
    let results = request
        .tool_results
        .as_ref()
        .filter(|results| !results.is_empty());
    match (prompt, results) {
        (Some(_), Some(_)) => Err(ApiError::invalid_request(
            "задайте ровно одно из полей prompt и tool_results",
        )),
        (None, None) => Err(ApiError::invalid_request(
            "для chat_id обязательно поле prompt или tool_results",
        )),
        (Some(prompt), None) => Ok(TurnInput::Prompt(prompt.to_string())),
        (None, Some(results)) => Ok(TurnInput::ToolResults(results.clone())),
    }
}

/// Результаты инструментов как сообщения истории. Принимаются, только если
/// последнее сообщение активной ветки — ответ модели с вызовами, а
/// множество присланных `tool_call_id` совпадает с множеством `id` этих
/// вызовов: иначе модель получила бы ответ не на свой вопрос.
fn tool_results_for(
    stored: &[store::ChatMessage],
    results: Vec<crate::dto::ToolResultDto>,
) -> Result<Vec<Message>, ApiError> {
    let calls = match stored.last() {
        Some(last) if matches!(last.role, Role::Assistant) && !last.tool_calls.is_empty() => &last.tool_calls,
        _ => {
            return Err(ApiError::tool_results_mismatch(
                "последнее сообщение чата — не ответ модели с вызовами инструментов",
            ));
        }
    };
    let expected: std::collections::HashSet<&str> = calls.iter().map(|call| call.id.as_str()).collect();
    let received: std::collections::HashSet<&str> =
        results.iter().map(|result| result.tool_call_id.as_str()).collect();
    if received.len() != results.len() || expected != received {
        return Err(ApiError::tool_results_mismatch(
            "tool_call_id результатов должны совпадать с идентификаторами вызовов последнего ответа модели",
        ));
    }
    Ok(results
        .into_iter()
        .map(|result| {
            let name = result.name.unwrap_or_else(|| {
                calls
                    .iter()
                    .find(|call| call.id == result.tool_call_id)
                    .map(|call| call.name.clone())
                    .unwrap_or_default()
            });
            Message::tool_result(result.tool_call_id, name, result.content)
        })
        .collect())
}

/// `POST /v1/chat` с `chat_id`: история берётся из хранилища, клиент
/// присылает только новое сообщение (или результаты инструментов), обмен
/// пишется одной транзакцией после успешного ответа модели (specs/chat-api,
/// «Диалог в существующем чате»; design.md, решение 9).
async fn handle_chat_in_existing(
    state: AppState,
    request_id: String,
    owner: String,
    chat_id: String,
    request: ChatRequest,
) -> Result<ChatResponse, ApiError> {
    let input = turn_input(&request)?;
    let tools = validate_tools(request.tools.clone())?;

    let defaults = server_defaults(&state);
    let chat = store::load_chat(&state.db, &owner, &chat_id, &defaults)
        .await
        .map_err(ApiError::from_store_error)?;
    let settings =
        merge_settings(&state, &owner, chat.settings.clone(), request.settings, SettingsPurpose::Call).await?;
    let model = settings
        .model
        .clone()
        .unwrap_or_else(|| state.config.model.clone());

    let strategy = context::effective_strategy(&state, &settings);
    let stored = store::load_messages(
        &state.db,
        &chat_id,
        0,
        MAX_HISTORY_MESSAGES,
        None,
        state.config.max_branch_depth,
    )
    .await
    .map_err(ApiError::from_store_error)?;

    // Новые сообщения хода и реплика пользователя, с которой ход начался:
    // фоновым задачам нужна именно она, а не результаты инструментов.
    let (new_messages, prompt, turn_user_seq) = match input {
        TurnInput::Prompt(prompt) => (vec![Message::user(prompt.clone())], prompt, None),
        TurnInput::ToolResults(results) => {
            let messages = tool_results_for(&stored.messages, results)?;
            let turn_user = stored
                .messages
                .iter()
                .rev()
                .find(|message| matches!(message.role, Role::User));
            let prompt = turn_user.map(|m| m.content.clone()).unwrap_or_default();
            (messages, prompt, Some(turn_user.map(|m| m.seq).unwrap_or(0)))
        }
    };

    // Сборка истории — до проверки лимита контекста, чтобы стратегия успела
    // спасти чат от context_limit_exceeded, а не сработать после отказа
    // (design.md, решение 6, распространено на все стратегии решением 7).
    let mut assembled = context::assemble(
        &state,
        &chat,
        &settings,
        &owner,
        strategy,
        stored.messages,
        new_messages.clone(),
    )
    .await;
    let history = std::mem::take(&mut assembled.history);
    // Системное сообщение уходит первым при любой стратегии (design.md,
    // решение 3): его текст переиспользуется как контекст для генерации
    // названия чата, не задерживая ответ пользователю вторым обращением к
    // системному сообщению.
    let system_text = history.first().map(|m| m.content.clone()).unwrap_or_default();

    let limit = effective_context_limit(state.config.max_context_tokens, settings.max_context_tokens)?;
    check_context_limit(&history, limit)?;

    let pipeline = Pipeline::new(state.agent.clone())
        .with_output_policies(agentcore::invariants::default_output_policies(state.agent.clone()));
    let pipeline_context = RequestContext::new(request_id.clone(), history, settings.clone())
        .with_invariants((*state.invariants).clone())
        .with_tools(tools);
    let (reply, policy) = run_pipeline(&pipeline, pipeline_context, &request_id).await?;

    let mut assistant_meta = reply.meta.clone();
    if assistant_meta.model.is_none() {
        assistant_meta.model = reply.model.clone().or_else(|| Some(model.clone()));
    }
    // Ход с инструментами пишется частями, сразу после каждого ответа
    // модели, а не в конце хода: пишущий инструмент мог уже изменить
    // репозиторий, и история чата обязана это отражать, даже если клиент
    // потом упадёт.
    let mut rows: Vec<store::NewMessage> = new_messages.into_iter().map(store::NewMessage::from_message).collect();
    rows.push(store::NewMessage {
        role: Role::Assistant,
        content: reply.content.clone(),
        reasoning: reply.reasoning.clone(),
        meta: Some(assistant_meta),
        tool_calls: reply.tool_calls.clone(),
        tool_call_id: None,
        tool_name: None,
    });
    let mut written = store::append_messages(&state.db, &owner, &chat_id, rows)
        .await
        .map_err(ApiError::from_store_error)?;
    let assistant_row = written.pop().expect("ответ модели записан последним");
    let user_seq = turn_user_seq.unwrap_or_else(|| written.first().map(|row| row.seq).unwrap_or(0));

    let response = ChatResponse::new(request_id, model, &reply, policy).with_chat(chat_id.clone(), assistant_row.seq);
    // Пока модель собирает данные инструментами, ход не завершён: факты,
    // память, трекер задачи и название считаются по окончательному ответу,
    // иначе каждая итерация цикла стоила бы им лишнего вызова модели.
    if !reply.tool_calls.is_empty() {
        return Ok(response.with_context(assembled.context));
    }

    // Факты обновляются ПОСЛЕ записи обмена — задерживать ответ пользователю
    // вторым вызовом модели незачем, а сообщение уже целиком доступно в
    // хвосте истории (design.md, решение 5).
    if strategy == agentcore::config::ContextStrategy::Facts {
        let facts_updated =
            crate::facts::update_after_exchange(&state, &chat_id, &settings, &prompt, user_seq).await;
        assembled.context.facts_updated = Some(facts_updated);
    }

    // Маршрутизатор памяти — тоже фоновый вызов, после записи обмена, вне
    // ответа пользователю (design.md, решение 5, «Риски»): результат этого
    // прогона попадёт в блок `context` СЛЕДУЮЩЕГО ответа этого чата. Запуск
    // зависит от независимого переключателя слоистой памяти, а не от
    // действующей стратегии контекста (decouple-memory-layers, решение 3).
    if crate::memory::effective_layers_enabled(&state, &settings)
        && crate::memory::effective_router_enabled(&state, &settings)
    {
        let route_state = state.clone();
        let route_chat = chat.clone();
        let route_settings = settings.clone();
        let route_owner = owner.clone();
        let route_prompt = prompt.clone();
        tokio::spawn(async move {
            let outcome =
                crate::memory::route_after_exchange(&route_state, &route_chat, &route_settings, &route_owner, &route_prompt)
                    .await;
            route_state
                .memory_route_outcomes
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .insert(route_chat.id.clone(), outcome);
        });
    }

    // Трекер состояния задачи — тоже фоновый вызов после записи обмена, не
    // задерживающий ответ пользователю; результат прогона попадёт в блок
    // `context` СЛЕДУЮЩЕГО ответа этого чата (design.md, решение 5;
    // specs/task-state, «Автоматический трекер предлагает переход»).
    if crate::task::effective_enabled(&state, &settings) && crate::task::effective_auto_enabled(&state, &settings) {
        let track_state = state.clone();
        let track_chat = chat.clone();
        let track_settings = settings.clone();
        let track_owner = owner.clone();
        // Трекеру нужны обе реплики обмена: критерий перехода в `execution` —
        // подтверждение пользователя, и оно есть только в его сообщении
        // (specs/task-state, «Вход трекера содержит обе реплики обмена»).
        let track_prompt = prompt.clone();
        let assistant_reply = reply.content.clone();
        tokio::spawn(async move {
            let outcome = crate::task::track_after_exchange(
                &track_state,
                &track_chat,
                &track_settings,
                &track_owner,
                &track_prompt,
                &assistant_reply,
            )
            .await;
            track_state
                .task_track_outcomes
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .insert(track_chat.id.clone(), outcome);
        });
    }

    // Название генерируется фоном, вне ответа пользователю (design.md,
    // решение 5): ответ уже сформирован, ждать вызов модели незачем.
    if crate::title::should_generate(&state, &chat.title, user_seq) {
        let state = state.clone();
        let title_chat_id = chat_id.clone();
        let owner = owner.clone();
        let settings = settings.clone();
        tokio::spawn(crate::title::generate_and_save(
            state,
            title_chat_id,
            owner,
            settings,
            system_text,
            prompt.clone(),
        ));
    }

    Ok(response.with_context(assembled.context))
}

async fn run_pipeline(
    pipeline: &Pipeline,
    context: RequestContext,
    request_id: &str,
) -> Result<(agentcore::agent::AgentReply, agentcore::pipeline::PolicyLog), ApiError> {
    match pipeline.run(context).await {
        Ok(PipelineOutcome::Completed { reply, policy }) => Ok((*reply, policy)),
        Ok(PipelineOutcome::Rejected {
            stage, code, reason, ..
        }) => {
            tracing::info!(%request_id, %stage, %code, "запрос отклонён политикой");
            Err(ApiError::policy_rejected(code, reason))
        }
        Err(err) => Err(ApiError::from_agent_error(&err)),
    }
}

// --- Управление чатами ---

async fn create_chat(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
    Json(body): Json<CreateChatRequest>,
) -> Response {
    match handle_create_chat(state, owner, body).await {
        Ok(dto) => (StatusCode::CREATED, Json(dto)).into_response(),
        Err(err) => err.with_request_id(request_id).into_response(),
    }
}

async fn handle_create_chat(
    state: AppState,
    owner: String,
    body: CreateChatRequest,
) -> Result<ChatDto, ApiError> {
    let title = validate_title(body.title)?.unwrap_or_else(|| DEFAULT_CHAT_TITLE.to_string());
    let settings =
        merge_settings(&state, &owner, server_defaults(&state), body.settings, SettingsPurpose::Storage).await?;
    let chat = store::create_chat(&state.db, &owner, &title, &settings)
        .await
        .map_err(ApiError::from_store_error)?;
    Ok(ChatDto::from(chat))
}

fn clamp_query_error(limit: u32, max: u32) -> Result<(), ApiError> {
    if limit == 0 || limit > max {
        return Err(ApiError::invalid_request(format!(
            "limit должен быть в диапазоне от 1 до {max}"
        )));
    }
    Ok(())
}

async fn list_chats_handler(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
    Query(query): Query<ListChatsQuery>,
) -> Response {
    match handle_list_chats(state, owner, query).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(err) => err.with_request_id(request_id).into_response(),
    }
}

async fn handle_list_chats(
    state: AppState,
    owner: String,
    query: ListChatsQuery,
) -> Result<ListChatsResponse, ApiError> {
    let limit = query.limit.unwrap_or(DEFAULT_CHATS_LIMIT);
    clamp_query_error(limit, MAX_CHATS_LIMIT)?;
    let defaults = server_defaults(&state);
    let page = store::list_chats(&state.db, &owner, limit, query.cursor.as_deref(), &defaults)
        .await
        .map_err(ApiError::from_store_error)?;
    let mut chats = Vec::with_capacity(page.chats.len());
    for chat in page.chats {
        chats.push(chat_dto_with_summary(&state, chat).await?);
    }
    Ok(ListChatsResponse {
        chats,
        next_cursor: page.next_cursor,
    })
}

/// `ChatDto` вместе с пересказом чата, если он построен (specs/context-summary,
/// «Наблюдаемость компактизации»).
async fn chat_dto_with_summary(state: &AppState, chat: store::Chat) -> Result<ChatDto, ApiError> {
    let chat_id = chat.id.clone();
    let summary = store::load_summary(&state.db, &chat_id)
        .await
        .map_err(ApiError::from_store_error)?;
    Ok(ChatDto::from(chat).with_summary(summary))
}

async fn get_chat_handler(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
    Path(id): Path<String>,
    Query(query): Query<GetChatQuery>,
) -> Response {
    match handle_get_chat(state, owner, id, query).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(err) => err.with_request_id(request_id).into_response(),
    }
}

async fn handle_get_chat(
    state: AppState,
    owner: String,
    id: String,
    query: GetChatQuery,
) -> Result<ChatWithMessagesResponse, ApiError> {
    let limit = query.limit.unwrap_or(DEFAULT_MESSAGES_LIMIT);
    clamp_query_error(limit, MAX_MESSAGES_LIMIT)?;
    let after = query.after.unwrap_or(0);
    let defaults = server_defaults(&state);
    let chat = store::load_chat(&state.db, &owner, &id, &defaults)
        .await
        .map_err(ApiError::from_store_error)?;
    let page = store::load_messages(
        &state.db,
        &id,
        after,
        limit,
        query.branch.as_deref(),
        state.config.max_branch_depth,
    )
    .await
    .map_err(ApiError::from_store_error)?;
    Ok(ChatWithMessagesResponse {
        chat: chat_dto_with_summary(&state, chat).await?,
        messages: page.messages.into_iter().map(MessageView::from).collect(),
        next_after: page.next_after,
        branch_id: page.branch_id,
    })
}

async fn patch_chat_handler(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
    Path(id): Path<String>,
    Json(body): Json<UpdateChatRequest>,
) -> Response {
    match handle_patch_chat(state, owner, id, body).await {
        Ok(dto) => (StatusCode::OK, Json(dto)).into_response(),
        Err(err) => err.with_request_id(request_id).into_response(),
    }
}

async fn handle_patch_chat(
    state: AppState,
    owner: String,
    id: String,
    body: UpdateChatRequest,
) -> Result<ChatDto, ApiError> {
    if body.title.is_none() && body.settings.is_none() {
        return Err(ApiError::invalid_request(
            "тело изменения чата должно задавать title, settings или оба поля",
        ));
    }
    let title = validate_title(body.title)?;

    let defaults = server_defaults(&state);
    let settings = match body.settings {
        Some(dto) => {
            let chat = store::load_chat(&state.db, &owner, &id, &defaults)
                .await
                .map_err(ApiError::from_store_error)?;
            Some(merge_settings(&state, &owner, chat.settings, Some(dto), SettingsPurpose::Storage).await?)
        }
        None => None,
    };

    let updated = store::update_chat(&state.db, &owner, &id, title.as_deref(), settings.as_ref(), &defaults)
        .await
        .map_err(ApiError::from_store_error)?;
    chat_dto_with_summary(&state, updated).await
}

/// `POST /v1/chats/{id}/messages`: дозапись готовых реплик. Провайдер и
/// стадии конвейера здесь не участвуют — ответ модели клиент получил сам
/// (specs/chat-message-api).
async fn append_messages_handler(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
    Path(id): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> Response {
    match handle_append_messages(state, request_id.clone(), owner, id, body).await {
        Ok(response) => (StatusCode::CREATED, Json(response)).into_response(),
        Err(err) => err.with_request_id(request_id).into_response(),
    }
}

async fn handle_append_messages(
    state: AppState,
    request_id: String,
    owner: String,
    chat_id: String,
    body: serde_json::Value,
) -> Result<AppendMessagesResponse, ApiError> {
    if contains_api_key(&body) {
        return Err(ApiError::invalid_request(
            "ключ провайдера задаёт сервис: поле api_key в запросе не принимается",
        ));
    }
    let request: AppendMessagesRequest = serde_json::from_value(body)
        .map_err(|err| ApiError::invalid_request(format!("тело запроса не разобрано: {err}")))?;

    if request.messages.is_empty() {
        return Err(ApiError::invalid_request(
            "поле messages должно содержать хотя бы одно сообщение",
        ));
    }
    if request.messages.len() > MAX_APPEND_MESSAGES {
        return Err(ApiError::invalid_request(format!(
            "за одну дозапись принимается не больше {MAX_APPEND_MESSAGES} сообщений"
        )));
    }
    for message in &request.messages {
        // Ответ модели из одних вызовов и результат инструмента (у
        // `git_add` вывод пуст) законно приходят без текста.
        let may_be_empty = !message.tool_calls.is_empty() || matches!(message.role, crate::dto::RoleDto::Tool);
        if message.content.trim().is_empty() && !may_be_empty {
            return Err(ApiError::invalid_request(
                "текст сообщения не должен быть пустым",
            ));
        }
    }

    let count = request.messages.len();
    let messages: Vec<store::NewMessage> = request
        .messages
        .into_iter()
        .map(crate::dto::NewMessageDto::into_new_message)
        .collect();

    let rows = store::append_messages(&state.db, &owner, &chat_id, messages)
        .await
        .map_err(ApiError::from_store_error)?;

    // Тексты сообщений в журнал не попадают: их запись включает только
    // AGENTD_LOG_CONTENT, и у дозаписи нет обмена с провайдером, который
    // журнал обмена мог бы описать.
    tracing::info!(%request_id, chat_id = %chat_id, messages = count, "сообщения дозаписаны в чат");

    Ok(AppendMessagesResponse {
        request_id,
        chat_id,
        seqs: rows.into_iter().map(|row| row.seq).collect(),
    })
}

/// Отказ хранилища вне владения чатом: `NotFound` здесь значит «факта/ветки
/// нет», а не «чужой или несуществующий чат», поэтому код ответа —
/// вызывающий сам подставляет уместный (specs/context-facts,
/// specs/chat-branching).
fn map_store_error(err: store::StoreError, not_found: ApiError) -> ApiError {
    match err {
        store::StoreError::NotFound => not_found,
        other => ApiError::from_store_error(other),
    }
}

// --- Факты (specs/context-facts) ---

async fn get_facts_handler(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
    Path(id): Path<String>,
) -> Response {
    match handle_get_facts(state, owner, id).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(err) => err.with_request_id(request_id).into_response(),
    }
}

async fn handle_get_facts(state: AppState, owner: String, chat_id: String) -> Result<FactsResponse, ApiError> {
    let defaults = server_defaults(&state);
    store::load_chat(&state.db, &owner, &chat_id, &defaults)
        .await
        .map_err(ApiError::from_store_error)?;
    let facts = store::load_facts(&state.db, &chat_id)
        .await
        .map_err(ApiError::from_store_error)?;
    Ok(FactsResponse {
        facts: facts.into_iter().map(FactDto::from).collect(),
    })
}

async fn put_fact_handler(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
    Path((id, key)): Path<(String, String)>,
    Json(body): Json<SetFactRequest>,
) -> Response {
    match handle_set_fact(state, owner, id, key, body.value).await {
        Ok(dto) => (StatusCode::OK, Json(dto)).into_response(),
        Err(err) => err.with_request_id(request_id).into_response(),
    }
}

async fn handle_set_fact(
    state: AppState,
    owner: String,
    chat_id: String,
    key: String,
    value: String,
) -> Result<FactDto, ApiError> {
    let defaults = server_defaults(&state);
    store::load_chat(&state.db, &owner, &chat_id, &defaults)
        .await
        .map_err(ApiError::from_store_error)?;

    if value.chars().count() as u32 > state.config.fact_value_max_chars {
        return Err(ApiError::facts_limit_exceeded(format!(
            "значение факта не может превышать {} символов",
            state.config.fact_value_max_chars
        )));
    }
    let existing = store::load_facts(&state.db, &chat_id)
        .await
        .map_err(ApiError::from_store_error)?;
    let already_exists = existing.iter().any(|fact| fact.key == key);
    if !already_exists && existing.len() as u32 >= state.config.max_facts {
        return Err(ApiError::facts_limit_exceeded(format!(
            "число фактов чата не может превышать операторский потолок {}",
            state.config.max_facts
        )));
    }

    // Ручная правка не привязана к сообщению: through_seq = 0.
    let fact = store::set_fact(&state.db, &chat_id, &key, &value, 0)
        .await
        .map_err(ApiError::from_store_error)?;
    Ok(FactDto::from(fact))
}

async fn delete_fact_handler(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
    Path((id, key)): Path<(String, String)>,
) -> Response {
    match handle_delete_fact(state, owner, id, key).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(err) => err.with_request_id(request_id).into_response(),
    }
}

async fn handle_delete_fact(state: AppState, owner: String, chat_id: String, key: String) -> Result<(), ApiError> {
    let defaults = server_defaults(&state);
    store::load_chat(&state.db, &owner, &chat_id, &defaults)
        .await
        .map_err(ApiError::from_store_error)?;
    store::delete_fact(&state.db, &chat_id, &key)
        .await
        .map_err(|err| map_store_error(err, ApiError::fact_not_found()))
}

// --- Ветки (specs/chat-branching) ---

async fn get_branches_handler(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
    Path(id): Path<String>,
) -> Response {
    match handle_get_branches(state, owner, id).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(err) => err.with_request_id(request_id).into_response(),
    }
}

async fn handle_get_branches(state: AppState, owner: String, chat_id: String) -> Result<BranchesResponse, ApiError> {
    let branches = store::list_branches(&state.db, &owner, &chat_id)
        .await
        .map_err(ApiError::from_store_error)?;
    Ok(BranchesResponse {
        branches: branches.into_iter().map(BranchDto::from).collect(),
    })
}

async fn create_branch_handler(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
    Path(id): Path<String>,
    Json(body): Json<CreateBranchRequest>,
) -> Response {
    match handle_create_branch(state, owner, id, body).await {
        Ok(dto) => (StatusCode::CREATED, Json(dto)).into_response(),
        Err(err) => err.with_request_id(request_id).into_response(),
    }
}

async fn handle_create_branch(
    state: AppState,
    owner: String,
    chat_id: String,
    body: CreateBranchRequest,
) -> Result<BranchDto, ApiError> {
    let name = body
        .name
        .map(|n| n.trim().to_string())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| format!("ветка от {}", body.from_seq));
    let branch = store::create_branch(&state.db, &owner, &chat_id, body.from_seq, &name)
        .await
        .map_err(|err| map_store_error(err, ApiError::chat_not_found()))?;
    Ok(BranchDto::from(branch))
}

async fn activate_branch_handler(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
    Path((id, branch_id)): Path<(String, String)>,
) -> Response {
    match store::activate_branch(&state.db, &owner, &id, &branch_id).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(err) => map_store_error(err, ApiError::branch_not_found())
            .with_request_id(request_id)
            .into_response(),
    }
}

// --- Память (specs/memory-layers) ---

async fn get_working_memory_handler(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
    Path(id): Path<String>,
) -> Response {
    match handle_get_working_memory(state, owner, id).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(err) => err.with_request_id(request_id).into_response(),
    }
}

async fn handle_get_working_memory(
    state: AppState,
    owner: String,
    chat_id: String,
) -> Result<WorkingMemoryResponse, ApiError> {
    let defaults = server_defaults(&state);
    let chat = store::load_chat(&state.db, &owner, &chat_id, &defaults)
        .await
        .map_err(ApiError::from_store_error)?;
    let entries = store::load_working_memory(&state.db, &chat_id, &chat.active_task_id)
        .await
        .map_err(ApiError::from_store_error)?;
    Ok(WorkingMemoryResponse { entries: entries.into_iter().map(WorkingMemoryEntryDto::from).collect() })
}

async fn set_working_memory_handler(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
    Path(id): Path<String>,
    Json(body): Json<SetWorkingMemoryRequest>,
) -> Response {
    match handle_set_working_memory(state, owner, id, body).await {
        Ok(dto) => (StatusCode::OK, Json(dto)).into_response(),
        Err(err) => err.with_request_id(request_id).into_response(),
    }
}

async fn handle_set_working_memory(
    state: AppState,
    owner: String,
    chat_id: String,
    body: SetWorkingMemoryRequest,
) -> Result<WorkingMemoryEntryDto, ApiError> {
    let defaults = server_defaults(&state);
    let chat = store::load_chat(&state.db, &owner, &chat_id, &defaults)
        .await
        .map_err(ApiError::from_store_error)?;
    if body.key.chars().count() as u32 > state.config.memory_working_key_max_chars {
        return Err(ApiError::memory_limit_exceeded(format!(
            "ключ рабочей памяти не может превышать {} символов",
            state.config.memory_working_key_max_chars
        )));
    }
    if body.value.chars().count() as u32 > state.config.memory_working_value_max_chars {
        return Err(ApiError::memory_limit_exceeded(format!(
            "значение рабочей памяти не может превышать {} символов",
            state.config.memory_working_value_max_chars
        )));
    }
    let now = agentcore::agent::now_secs();
    store::set_working_memory(&state.db, &chat_id, &chat.active_task_id, &body.key, &body.value, "manual", now)
        .await
        .map_err(ApiError::from_store_error)?;
    Ok(WorkingMemoryEntryDto {
        key: body.key,
        value: body.value,
        source: "manual".to_string(),
        updated_at: now,
    })
}

async fn delete_working_memory_handler(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
    Path(id): Path<String>,
    Query(query): Query<DeleteWorkingMemoryQuery>,
) -> Response {
    match handle_delete_working_memory(state, owner, id, query.key).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(err) => err.with_request_id(request_id).into_response(),
    }
}

async fn handle_delete_working_memory(
    state: AppState,
    owner: String,
    chat_id: String,
    key: String,
) -> Result<(), ApiError> {
    let defaults = server_defaults(&state);
    let chat = store::load_chat(&state.db, &owner, &chat_id, &defaults)
        .await
        .map_err(ApiError::from_store_error)?;
    store::delete_working_memory(&state.db, &chat_id, &chat.active_task_id, &key)
        .await
        .map_err(|err| map_store_error(err, ApiError::memory_entry_not_found()))
}

async fn finish_task_handler(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
    Path(id): Path<String>,
    Json(body): Json<FinishTaskRequest>,
) -> Response {
    match handle_finish_task(state, owner, id, body.carry_forward_keys).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(err) => err.with_request_id(request_id).into_response(),
    }
}

async fn handle_finish_task(
    state: AppState,
    owner: String,
    chat_id: String,
    carry_forward_keys: Vec<String>,
) -> Result<LongTermMemoryResponse, ApiError> {
    let defaults = server_defaults(&state);
    store::load_chat(&state.db, &owner, &chat_id, &defaults)
        .await
        .map_err(ApiError::from_store_error)?;
    let result = store::finish_task(
        &state.db,
        &owner,
        &chat_id,
        &carry_forward_keys,
        "manual",
        "явное завершение задачи",
    )
    .await
    .map_err(ApiError::from_store_error)?;
    Ok(LongTermMemoryResponse { entries: result.transferred.into_iter().map(LongTermMemoryEntryDto::from).collect() })
}

// --- Состояние задачи (specs/task-state) ---

/// Отображает `TaskError` в `ApiError`: чужой/несуществующий чат неотличим
/// от отсутствующего (specs/task-state, «Чужой чат недоступен»), остальные
/// исходы — один код `task_transition_invalid` с уточняющим текстом.
fn map_task_error(err: crate::task::TaskError) -> ApiError {
    match err {
        crate::task::TaskError::NotFound => ApiError::chat_not_found(),
        crate::task::TaskError::Backend(err) => ApiError::from_store_error(err),
        crate::task::TaskError::UnknownStage => {
            ApiError::task_transition_invalid("неизвестный этап задачи")
        }
        crate::task::TaskError::InvalidEdge => {
            ApiError::task_transition_invalid("переход не входит в допустимые рёбра автомата")
        }
        crate::task::TaskError::Paused => {
            ApiError::task_transition_invalid("задача на паузе: переход отклонён до снятия с паузы")
        }
        crate::task::TaskError::StepTooLong => {
            ApiError::task_transition_invalid("текст шага превышает операторский лимит длины")
        }
        crate::task::TaskError::ExpectedActionTooLong => {
            ApiError::task_transition_invalid("текст ожидаемого действия превышает операторский лимит длины")
        }
    }
}

async fn task_state_dto(state: &AppState, owner: &str, chat_id: &str) -> Result<TaskStateDto, ApiError> {
    let task = store::load_task_state(&state.db, owner, chat_id)
        .await
        .map_err(ApiError::from_store_error)?;
    let transitions = store::load_task_transitions(&state.db, owner, chat_id)
        .await
        .map_err(ApiError::from_store_error)?;
    Ok(TaskStateDto {
        id: task.id,
        stage: task.stage,
        step: task.step,
        expected_action: task.expected_action,
        paused: task.paused,
        resume_brief: task.resume_brief,
        transitions: transitions.into_iter().map(TaskTransitionDto::from).collect(),
    })
}

async fn get_task_handler(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
    Path(id): Path<String>,
) -> Response {
    match task_state_dto(&state, &owner, &id).await {
        Ok(dto) => (StatusCode::OK, Json(dto)).into_response(),
        Err(err) => err.with_request_id(request_id).into_response(),
    }
}

async fn task_transition_handler(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
    Path(id): Path<String>,
    Json(body): Json<TaskTransitionRequest>,
) -> Response {
    match handle_task_transition(state, owner, id, body).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(err) => err.with_request_id(request_id).into_response(),
    }
}

async fn handle_task_transition(
    state: AppState,
    owner: String,
    chat_id: String,
    body: TaskTransitionRequest,
) -> Result<TaskTransitionResponse, ApiError> {
    match body.stage.as_deref() {
        Some("done") => {
            store::load_chat(&state.db, &owner, &chat_id, &server_defaults(&state))
                .await
                .map_err(ApiError::from_store_error)?;
            // Завершение идёт отдельным путём, но подчиняется тому же
            // автомату: ребро `validation → done` и отсутствие паузы
            // (specs/task-state, «Переход в done завершает задачу»).
            crate::task::check_can_finish(&state, &owner, &chat_id).await.map_err(map_task_error)?;
            let result = store::finish_task(
                &state.db,
                &owner,
                &chat_id,
                &body.carry_forward_keys,
                "manual",
                "переход состояния задачи в done",
            )
            .await
            .map_err(ApiError::from_store_error)?;
            let task = task_state_dto(&state, &owner, &chat_id).await?;
            Ok(TaskTransitionResponse { task, new_task_id: Some(result.new_task_id) })
        }
        Some(stage) => {
            crate::task::apply_manual_transition(
                &state,
                &owner,
                &chat_id,
                stage,
                body.step.as_deref(),
                body.expected_action.as_deref(),
            )
            .await
            .map_err(map_task_error)?;
            let task = task_state_dto(&state, &owner, &chat_id).await?;
            Ok(TaskTransitionResponse { task, new_task_id: None })
        }
        None => {
            crate::task::update_fields(&state, &owner, &chat_id, body.step.as_deref(), body.expected_action.as_deref())
                .await
                .map_err(map_task_error)?;
            let task = task_state_dto(&state, &owner, &chat_id).await?;
            Ok(TaskTransitionResponse { task, new_task_id: None })
        }
    }
}

async fn task_pause_handler(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
    Path(id): Path<String>,
) -> Response {
    match crate::task::pause(&state, &owner, &id).await {
        Ok(_) => match task_state_dto(&state, &owner, &id).await {
            Ok(dto) => (StatusCode::OK, Json(dto)).into_response(),
            Err(err) => err.with_request_id(request_id).into_response(),
        },
        Err(err) => map_task_error(err).with_request_id(request_id).into_response(),
    }
}

async fn task_resume_handler(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
    Path(id): Path<String>,
) -> Response {
    match crate::task::resume(&state, &owner, &id).await {
        Ok(_) => match task_state_dto(&state, &owner, &id).await {
            Ok(dto) => (StatusCode::OK, Json(dto)).into_response(),
            Err(err) => err.with_request_id(request_id).into_response(),
        },
        Err(err) => map_task_error(err).with_request_id(request_id).into_response(),
    }
}

async fn get_long_term_memory_handler(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
) -> Response {
    match handle_get_long_term_memory(state, owner).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(err) => err.with_request_id(request_id).into_response(),
    }
}

/// Потолок числа записей на ручное чтение — заметно шире, чем лимит
/// подстановки в контекст (`AGENTD_MEMORY_LONG_TERM_MAX_ENTRIES`): просмотр
/// не должен незаметно обрезать хранимые данные операторским лимитом,
/// рассчитанным на стоимость запроса к модели.
const MANUAL_LONG_TERM_LIST_LIMIT: u32 = 1000;

async fn handle_get_long_term_memory(state: AppState, owner: String) -> Result<LongTermMemoryResponse, ApiError> {
    let entries = store::load_long_term_memory(&state.db, &owner, MANUAL_LONG_TERM_LIST_LIMIT)
        .await
        .map_err(ApiError::from_store_error)?;
    Ok(LongTermMemoryResponse { entries: entries.into_iter().map(LongTermMemoryEntryDto::from).collect() })
}

async fn set_long_term_memory_handler(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
    Json(body): Json<SetLongTermMemoryRequest>,
) -> Response {
    match handle_set_long_term_memory(state, owner, body).await {
        Ok(dto) => (StatusCode::OK, Json(dto)).into_response(),
        Err(err) => err.with_request_id(request_id).into_response(),
    }
}

async fn handle_set_long_term_memory(
    state: AppState,
    owner: String,
    body: SetLongTermMemoryRequest,
) -> Result<LongTermMemoryEntryDto, ApiError> {
    if !matches!(body.entry_type.as_str(), "profile" | "decision" | "knowledge") {
        return Err(ApiError::invalid_request(format!(
            "неизвестный entry_type долговременной памяти: {}",
            body.entry_type
        )));
    }
    if let Some(key) = &body.key {
        if key.chars().count() as u32 > state.config.memory_long_term_key_max_chars {
            return Err(ApiError::memory_limit_exceeded(format!(
                "ключ долговременной памяти не может превышать {} символов",
                state.config.memory_long_term_key_max_chars
            )));
        }
    }
    if body.value.chars().count() as u32 > state.config.memory_long_term_value_max_chars {
        return Err(ApiError::memory_limit_exceeded(format!(
            "значение долговременной памяти не может превышать {} символов",
            state.config.memory_long_term_value_max_chars
        )));
    }
    let now = agentcore::agent::now_secs();
    let (id, _applied) = store::set_long_term_memory(
        &state.db,
        &owner,
        &body.entry_type,
        body.key.as_deref(),
        &body.value,
        "manual",
        None,
        now,
    )
    .await
    .map_err(ApiError::from_store_error)?;
    Ok(LongTermMemoryEntryDto {
        id,
        entry_type: body.entry_type,
        key: body.key,
        value: body.value,
        source: "manual".to_string(),
        source_chat_id: None,
        created_at: now,
        updated_at: now,
    })
}

async fn delete_long_term_memory_handler(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
    Query(query): Query<DeleteLongTermMemoryQuery>,
) -> Response {
    match store::delete_long_term_memory(&state.db, &owner, &query.id).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(err) => map_store_error(err, ApiError::memory_entry_not_found())
            .with_request_id(request_id)
            .into_response(),
    }
}

// --- Профили (specs/user-profiles) ---

async fn get_profiles_handler(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
) -> Response {
    match handle_get_profiles(state, owner).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(err) => err.with_request_id(request_id).into_response(),
    }
}

async fn handle_get_profiles(state: AppState, owner: String) -> Result<ProfilesResponse, ApiError> {
    let profiles = crate::profile::list(&state, &owner)
        .await
        .map_err(ApiError::from_store_error)?;
    Ok(ProfilesResponse { profiles: profiles.into_iter().map(ProfileDto::from).collect() })
}

async fn create_profile_handler(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
    Json(body): Json<CreateProfileRequest>,
) -> Response {
    match handle_create_profile(state, owner, body).await {
        Ok(dto) => (StatusCode::CREATED, Json(dto)).into_response(),
        Err(err) => err.with_request_id(request_id).into_response(),
    }
}

async fn handle_create_profile(
    state: AppState,
    owner: String,
    body: CreateProfileRequest,
) -> Result<ProfileDto, ApiError> {
    if body.name.trim().is_empty() {
        return Err(ApiError::invalid_request("название профиля не может быть пустым"));
    }
    if !crate::profile::has_any_preference(&body.persona, &body.style, &body.format, &body.constraints) {
        return Err(ApiError::profile_rejected(
            "профиль должен содержать хотя бы одно непустое поле предпочтений",
        ));
    }
    let existing = store::count_owner_profiles(&state.db, &owner)
        .await
        .map_err(ApiError::from_store_error)?;
    if existing >= state.config.max_profiles {
        return Err(ApiError::profile_rejected(format!(
            "превышен предел собственных профилей владельца ({})",
            state.config.max_profiles
        )));
    }
    let created = store::create_owner_profile(
        &state.db,
        &owner,
        &body.name,
        &body.persona,
        &body.style,
        &body.format,
        &body.constraints,
    )
    .await
    .map_err(ApiError::from_store_error)?;
    Ok(ProfileDto::from(crate::profile::Profile {
        id: created.id,
        name: created.name,
        persona: created.persona,
        style: created.style,
        format: created.format,
        constraints: created.constraints,
        built_in: false,
    }))
}

async fn get_profile_handler(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
    Path(id): Path<String>,
) -> Response {
    match handle_get_profile(state, owner, id).await {
        Ok(dto) => (StatusCode::OK, Json(dto)).into_response(),
        Err(err) => err.with_request_id(request_id).into_response(),
    }
}

async fn handle_get_profile(state: AppState, owner: String, id: String) -> Result<ProfileDto, ApiError> {
    let profile = crate::profile::find(&state, &owner, &id)
        .await
        .map_err(ApiError::from_store_error)?
        .ok_or_else(ApiError::profile_not_found)?;
    Ok(ProfileDto::from(profile))
}

async fn patch_profile_handler(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
    Path(id): Path<String>,
    Json(body): Json<UpdateProfileRequest>,
) -> Response {
    match handle_patch_profile(state, owner, id, body).await {
        Ok(dto) => (StatusCode::OK, Json(dto)).into_response(),
        Err(err) => err.with_request_id(request_id).into_response(),
    }
}

async fn handle_patch_profile(
    state: AppState,
    owner: String,
    id: String,
    body: UpdateProfileRequest,
) -> Result<ProfileDto, ApiError> {
    if crate::profile::find_built_in(&id).is_some() {
        return Err(ApiError::profile_rejected("встроенный профиль нельзя изменить"));
    }
    if let Some(name) = &body.name
        && name.trim().is_empty()
    {
        return Err(ApiError::invalid_request("название профиля не может быть пустым"));
    }
    let updated = store::update_owner_profile(
        &state.db,
        &owner,
        &id,
        body.name.as_deref(),
        body.persona.as_deref(),
        body.style.as_deref(),
        body.format.as_deref(),
        body.constraints.as_deref(),
    )
    .await
    .map_err(|err| map_store_error(err, ApiError::profile_not_found()))?;
    if !crate::profile::has_any_preference(&updated.persona, &updated.style, &updated.format, &updated.constraints)
    {
        return Err(ApiError::profile_rejected(
            "профиль должен содержать хотя бы одно непустое поле предпочтений",
        ));
    }
    Ok(ProfileDto::from(crate::profile::Profile {
        id: updated.id,
        name: updated.name,
        persona: updated.persona,
        style: updated.style,
        format: updated.format,
        constraints: updated.constraints,
        built_in: false,
    }))
}

async fn delete_profile_handler(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
    Path(id): Path<String>,
) -> Response {
    if crate::profile::find_built_in(&id).is_some() {
        return ApiError::profile_rejected("встроенный профиль нельзя удалить")
            .with_request_id(request_id)
            .into_response();
    }
    match store::delete_owner_profile(&state.db, &owner, &id).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(err) => map_store_error(err, ApiError::profile_not_found())
            .with_request_id(request_id)
            .into_response(),
    }
}

async fn delete_chat_handler(
    State(state): State<AppState>,
    Extension(RequestId(request_id)): Extension<RequestId>,
    Extension(Owner(owner)): Extension<Owner>,
    Path(id): Path<String>,
) -> Response {
    match store::delete_chat(&state.db, &owner, &id).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(err) => ApiError::from_store_error(err)
            .with_request_id(request_id)
            .into_response(),
    }
}

/// Исходы слоёв в коды состояния. Тело собирает `normalize_errors`, чтобы
/// в конверт попал идентификатор запроса.
async fn handle_layer_error(err: BoxError) -> StatusCode {
    if err.is::<tower::load_shed::error::Overloaded>() {
        StatusCode::TOO_MANY_REQUESTS
    } else if err.is::<tower::timeout::error::Elapsed>() {
        StatusCode::GATEWAY_TIMEOUT
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    }
}

pub fn router(state: AppState) -> Router {
    let max_body_bytes = state.config.max_body_bytes;
    let max_concurrency = state.config.max_concurrency;
    let request_timeout = state.config.request_timeout;

    let v1 = Router::new()
        .route("/chat", post(chat))
        .route("/models", get(models))
        .route("/chats", post(create_chat).get(list_chats_handler))
        .route(
            "/chats/{id}",
            get(get_chat_handler).patch(patch_chat_handler).delete(delete_chat_handler),
        )
        .route("/chats/{id}/messages", post(append_messages_handler))
        .route(
            "/chats/{id}/facts",
            get(get_facts_handler),
        )
        .route(
            "/chats/{id}/facts/{key}",
            axum::routing::put(put_fact_handler).delete(delete_fact_handler),
        )
        .route(
            "/chats/{id}/branches",
            get(get_branches_handler).post(create_branch_handler),
        )
        .route(
            "/chats/{id}/branches/{branch_id}/activate",
            post(activate_branch_handler),
        )
        .route(
            "/chats/{id}/memory/working",
            get(get_working_memory_handler)
                .post(set_working_memory_handler)
                .delete(delete_working_memory_handler),
        )
        .route(
            "/chats/{id}/memory/working/finish-task",
            post(finish_task_handler),
        )
        .route("/chats/{id}/task", get(get_task_handler))
        .route("/chats/{id}/task/transition", post(task_transition_handler))
        .route("/chats/{id}/task/pause", post(task_pause_handler))
        .route("/chats/{id}/task/resume", post(task_resume_handler))
        .route(
            "/memory/long-term",
            get(get_long_term_memory_handler)
                .post(set_long_term_memory_handler)
                .delete(delete_long_term_memory_handler),
        )
        .route(
            "/profiles",
            get(get_profiles_handler).post(create_profile_handler),
        )
        .route(
            "/profiles/{id}",
            get(get_profile_handler)
                .patch(patch_profile_handler)
                .delete(delete_profile_handler),
        )
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            authenticate,
        ));

    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .nest("/v1", v1)
        // Снаружи внутрь: журнал, идентификатор запроса, конверт ошибок,
        // лимит тела, лимит конкурентности с отказом вместо очереди
        // (`LoadShedLayer` снаружи `ConcurrencyLimitLayer`, иначе избыточные
        // запросы вставали бы в очередь вместо 429) и общий таймаут.
        .layer(
            ServiceBuilder::new()
                .layer(HandleErrorLayer::new(handle_layer_error))
                .layer(LoadShedLayer::new())
                .layer(ConcurrencyLimitLayer::new(max_concurrency))
                // Общий таймаут страхует зависание в стадиях конвейера и
                // намеренно длиннее провайдерского: иначе он срабатывал бы
                // раньше и скрывал причину 504.
                .layer(TimeoutLayer::new(
                    request_timeout + std::time::Duration::from_secs(5),
                )),
        );

    // Журнал тел ставится внутрь лимита размера и идентификатора запроса:
    // слишком большое тело отсекается до буферизации, а `request_id` к этому
    // моменту уже присвоен.
    let app = if state.config.debug {
        app.layer(axum::middleware::from_fn(crate::middleware::log_bodies))
    } else {
        app
    };

    app.layer(RequestBodyLimitLayer::new(max_body_bytes))
        .layer(axum::middleware::from_fn(normalize_errors))
        .layer(axum::middleware::from_fn(assign_request_id))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

#[cfg(test)]
mod context_limit_tests {
    use super::*;

    #[test]
    fn estimate_tokens_rounds_up_at_boundaries() {
        // "user" (4 символа) + пустое содержимое = 4 символа -> ceil(4/4) = 1.
        assert_eq!(estimate_tokens(&[Message::user("")]), 1);
        // "user" (4) + 1 символ содержимого = 5 символов -> ceil(5/4) = 2.
        assert_eq!(estimate_tokens(&[Message::user("a")]), 2);
        // "user" (4) + 4 символа содержимого = 8 символов -> ceil(8/4) = 2,
        // граница деления без остатка не должна округляться лишний раз.
        assert_eq!(estimate_tokens(&[Message::user("abcd")]), 2);
        assert_eq!(estimate_tokens(&[]), 0);
    }

    #[test]
    fn effective_context_limit_without_operator_default_or_client_value_is_none() {
        assert_eq!(effective_context_limit(None, None).unwrap(), None);
    }

    #[test]
    fn effective_context_limit_uses_operator_default_without_client_value() {
        assert_eq!(effective_context_limit(Some(100), None).unwrap(), Some(100));
    }

    #[test]
    fn effective_context_limit_accepts_narrower_client_value() {
        assert_eq!(effective_context_limit(Some(100), Some(50)).unwrap(), Some(50));
        assert_eq!(effective_context_limit(Some(100), Some(100)).unwrap(), Some(100));
    }

    #[test]
    fn effective_context_limit_rejects_wider_client_value() {
        assert!(effective_context_limit(Some(100), Some(101)).is_err());
    }

    #[test]
    fn effective_context_limit_accepts_client_value_without_operator_default() {
        // Клиентское значение действует самостоятельно: отсутствие
        // операторского лимита не повод для отказа (design.md, решение 1).
        assert_eq!(effective_context_limit(None, Some(4000)).unwrap(), Some(4000));
    }

    #[test]
    fn effective_context_limit_narrower_client_wins_over_wider_operator_default() {
        assert_eq!(
            effective_context_limit(Some(8000), Some(4000)).unwrap(),
            Some(4000)
        );
    }

    #[test]
    fn effective_context_limit_wider_client_than_operator_default_is_invalid() {
        let err = effective_context_limit(Some(4000), Some(8000)).unwrap_err();
        assert_eq!(err.code, "context_limit_invalid");
    }

    #[test]
    fn effective_context_limit_without_either_value_is_none() {
        assert_eq!(effective_context_limit(None, None).unwrap(), None);
    }
}
