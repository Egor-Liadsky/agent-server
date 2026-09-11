//! Сборка HTTP-приложения: маршруты, слои и обработчики.

use crate::dto::{
    AppendMessagesRequest, AppendMessagesResponse, ChatDto, ChatRequest, ChatResponse,
    ChatWithMessagesResponse, CreateChatRequest, GetChatQuery, ListChatsQuery, ListChatsResponse,
    MessageView, ModelsResponse, UpdateChatRequest,
};
use crate::error::ApiError;
use crate::middleware::{assign_request_id, authenticate, normalize_errors, ClientId, Owner, RequestId};
use crate::store;
use crate::telemetry::{log_exchange, ExchangeRecord};
use crate::state::AppState;
use agentcore::agent::Message;
use agentcore::config::{ChatSettings, Provider};
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
const DEFAULT_CHAT_TITLE: &str = "Новый чат";
const MAX_CHAT_TITLE_LEN: usize = 200;
/// Предел одной дозаписи: обмен с локальной моделью — это две реплики, а
/// запас нужен только на повтор после неудачи, не на выгрузку истории.
const MAX_APPEND_MESSAGES: usize = 100;

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

/// Приближённая оценка размера истории в токенах: точного токенизатора
/// провайдера (`deepseek-chat` через `agentupstream`, либо Ollama) у сервиса
/// нет, поэтому размер оценивается по длине текста, а не подсчитывается
/// точно (design.md, решение «Оценка размера — по длине текста»).
fn estimate_tokens(history: &[Message]) -> u32 {
    let chars: usize = history
        .iter()
        .map(|message| role_str(&message.role).len() + message.content.chars().count())
        .sum();
    (chars as u64).div_ceil(4) as u32
}

fn role_str(role: &agentcore::agent::Role) -> &'static str {
    match role {
        agentcore::agent::Role::User => "user",
        agentcore::agent::Role::Assistant => "assistant",
    }
}

/// Эффективный лимит контекстного окна на этот запрос: клиентское поле может
/// только сузить операторский лимит по умолчанию (specs/context-limit,
/// «Клиентское сужение лимита на один запрос»).
fn effective_context_limit(
    operator_default: Option<u32>,
    client_value: Option<u32>,
) -> Result<Option<u32>, ApiError> {
    match (operator_default, client_value) {
        (None, None) => Ok(None),
        (Some(default), None) => Ok(Some(default)),
        (Some(default), Some(client)) if client <= default => Ok(Some(client)),
        (Some(default), Some(_client)) => Err(ApiError::context_limit_invalid(format!(
            "max_context_tokens не может превышать операторский лимит {default}"
        ))),
        (None, Some(_client)) => Err(ApiError::context_limit_invalid(
            "max_context_tokens задан, но операторский лимит контекстного окна не настроен",
        )),
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
        (None, Some(messages)) => Ok(messages
            .iter()
            .cloned()
            .map(|message| message.into_message())
            .collect()),
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
fn merge_settings(
    state: &AppState,
    base: ChatSettings,
    dto: Option<crate::dto::ChatSettingsDto>,
    purpose: SettingsPurpose,
) -> Result<ChatSettings, ApiError> {
    let settings = match dto {
        Some(dto) => dto.apply_to(base).map_err(ApiError::invalid_request)?,
        None => base,
    };
    match purpose {
        SettingsPurpose::Call => validate_settings_for_call(state, &settings)?,
        SettingsPurpose::Storage => validate_settings_for_storage(state, &settings)?,
    }
    Ok(settings)
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
    Message {
        role: message.role,
        content: message.content,
        reasoning: message.reasoning,
        meta: message.meta,
    }
}

/// Текст запроса для журнала: он записывается только при включённом
/// признаке записи содержимого.
fn prompt_preview(body: &serde_json::Value) -> String {
    if let Some(prompt) = body.get("prompt").and_then(|value| value.as_str()) {
        return prompt.to_string();
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
    let request: ChatRequest = serde_json::from_value(body)
        .map_err(|err| ApiError::invalid_request(format!("тело запроса не разобрано: {err}")))?;

    match request.chat_id.clone() {
        Some(chat_id) => handle_chat_in_existing(state, request_id, owner, chat_id, request).await,
        None => handle_chat_without_storage(state, request_id, request).await,
    }
}

/// `POST /v1/chat` без `chat_id`: поведение не меняется — история приходит в
/// теле, ничего не пишется в хранилище (specs/chat-api, «Разовый вызов без
/// чата сохраняется»).
async fn handle_chat_without_storage(
    state: AppState,
    request_id: String,
    request: ChatRequest,
) -> Result<ChatResponse, ApiError> {
    let history = history_from(&request)?;
    let settings = merge_settings(&state, server_defaults(&state), request.settings, SettingsPurpose::Call)?;
    let model = settings
        .model
        .clone()
        .unwrap_or_else(|| state.config.model.clone());

    let limit = effective_context_limit(state.config.max_context_tokens, settings.max_context_tokens)?;
    check_context_limit(&history, limit)?;

    let pipeline = Pipeline::new(state.agent.clone());
    let context = RequestContext::new(request_id.clone(), history, settings);
    let (reply, policy) = run_pipeline(&pipeline, context, &request_id).await?;
    Ok(ChatResponse::new(request_id, model, &reply, policy))
}

/// `POST /v1/chat` с `chat_id`: история берётся из хранилища, клиент
/// присылает только новое сообщение, обмен пишется одной транзакцией после
/// успешного ответа модели (specs/chat-api, «Диалог в существующем чате»;
/// design.md, решение 9).
async fn handle_chat_in_existing(
    state: AppState,
    request_id: String,
    owner: String,
    chat_id: String,
    request: ChatRequest,
) -> Result<ChatResponse, ApiError> {
    if request.messages.is_some() {
        return Err(ApiError::invalid_request(
            "поле messages вместе с chat_id не принимается: история берётся из чата",
        ));
    }
    let prompt = request
        .prompt
        .as_deref()
        .map(str::trim)
        .filter(|prompt| !prompt.is_empty())
        .ok_or_else(|| ApiError::invalid_request("для chat_id обязательно поле prompt"))?
        .to_string();

    let defaults = server_defaults(&state);
    let chat = store::load_chat(&state.db, &owner, &chat_id, &defaults)
        .await
        .map_err(ApiError::from_store_error)?;
    let settings = merge_settings(&state, chat.settings.clone(), request.settings, SettingsPurpose::Call)?;
    let model = settings
        .model
        .clone()
        .unwrap_or_else(|| state.config.model.clone());

    let stored = store::load_messages(&state.db, &chat_id, 0, MAX_HISTORY_MESSAGES)
        .await
        .map_err(ApiError::from_store_error)?;
    let user_message = Message::user(prompt);
    let mut history: Vec<Message> = stored.messages.into_iter().map(message_from_stored).collect();
    history.push(user_message.clone());

    let limit = effective_context_limit(state.config.max_context_tokens, settings.max_context_tokens)?;
    check_context_limit(&history, limit)?;

    let pipeline = Pipeline::new(state.agent.clone());
    let context = RequestContext::new(request_id.clone(), history, settings);
    let (reply, policy) = run_pipeline(&pipeline, context, &request_id).await?;

    let mut assistant_meta = reply.meta.clone();
    if assistant_meta.model.is_none() {
        assistant_meta.model = reply.model.clone().or_else(|| Some(model.clone()));
    }
    let (_, assistant_row) = store::append_exchange(
        &state.db,
        &owner,
        &chat_id,
        store::NewMessage::from_message(user_message),
        store::NewMessage {
            role: agentcore::agent::Role::Assistant,
            content: reply.content.clone(),
            reasoning: reply.reasoning.clone(),
            meta: Some(assistant_meta),
        },
    )
    .await
    .map_err(ApiError::from_store_error)?;

    Ok(ChatResponse::new(request_id, model, &reply, policy).with_chat(chat_id, assistant_row.seq))
}

async fn run_pipeline(
    pipeline: &Pipeline,
    context: RequestContext,
    request_id: &str,
) -> Result<(agentcore::agent::AgentReply, agentcore::pipeline::PolicyLog), ApiError> {
    match pipeline.run(context).await {
        Ok(PipelineOutcome::Completed { reply, policy }) => Ok((reply, policy)),
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
    let settings = merge_settings(&state, server_defaults(&state), body.settings, SettingsPurpose::Storage)?;
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
    Ok(ListChatsResponse {
        chats: page.chats.into_iter().map(ChatDto::from).collect(),
        next_cursor: page.next_cursor,
    })
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
    let page = store::load_messages(&state.db, &id, after, limit)
        .await
        .map_err(ApiError::from_store_error)?;
    Ok(ChatWithMessagesResponse {
        chat: ChatDto::from(chat),
        messages: page.messages.into_iter().map(MessageView::from).collect(),
        next_after: page.next_after,
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
            Some(merge_settings(&state, chat.settings, Some(dto), SettingsPurpose::Storage)?)
        }
        None => None,
    };

    let updated = store::update_chat(&state.db, &owner, &id, title.as_deref(), settings.as_ref(), &defaults)
        .await
        .map_err(ApiError::from_store_error)?;
    Ok(ChatDto::from(updated))
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
        if message.content.trim().is_empty() {
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
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            authenticate,
        ));

    Router::new()
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
        )
        .layer(RequestBodyLimitLayer::new(max_body_bytes))
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
    fn effective_context_limit_rejects_client_value_without_operator_default() {
        assert!(effective_context_limit(None, Some(50)).is_err());
    }
}
