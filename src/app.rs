//! Сборка HTTP-приложения: маршруты, слои и обработчики.

use crate::dto::{ChatRequest, ChatResponse, ModelsResponse};
use crate::error::ApiError;
use crate::middleware::{assign_request_id, authenticate, normalize_errors, ClientId, RequestId};
use crate::telemetry::{log_exchange, ExchangeRecord};
use crate::state::AppState;
use agentcore::agent::Message;
use agentcore::config::{ChatSettings, Provider};
use agentcore::pipeline::{Pipeline, PipelineOutcome, RequestContext};
use axum::extract::{Extension, State};
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

/// Живость процесса. Никаких обращений к провайдеру: эндпоинт отвечает,
/// пока жив сам процесс.
async fn healthz() -> &'static str {
    "ok"
}

/// Готовность: ключ провайдера задан и базовый адрес разбирается.
async fn readyz(State(state): State<AppState>) -> Response {
    if state.config.is_ready() {
        (StatusCode::OK, "ready").into_response()
    } else {
        ApiError::not_ready().into_response()
    }
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

fn settings_from(state: &AppState, request: ChatRequest) -> Result<ChatSettings, ApiError> {
    let defaults = ChatSettings {
        model: Some(state.config.model.clone()),
        ..ChatSettings::default()
    };
    let settings = match request.settings {
        Some(dto) => dto.apply_to(defaults).map_err(ApiError::invalid_request)?,
        None => defaults,
    };

    let model = settings
        .model
        .clone()
        .unwrap_or_else(|| state.config.model.clone());
    if !state.config.is_model_allowed(&model) {
        return Err(ApiError::invalid_request(format!(
            "модель {model} не разрешена конфигурацией сервиса"
        )));
    }
    if settings.provider == Provider::Ollama && state.config.ollama_url.is_none() {
        return Err(ApiError::invalid_request(
            "провайдер ollama не настроен на этом сервисе",
        ));
    }
    Ok(settings)
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
    client: Option<Extension<ClientId>>,
    Json(body): Json<serde_json::Value>,
) -> Response {
    let started_at = std::time::Instant::now();
    let client = client
        .map(|Extension(ClientId(id))| id)
        .unwrap_or_else(|| crate::middleware::ANONYMOUS_CLIENT.to_string());
    let log_content = state.config.log_content;
    let prompt = prompt_preview(&body);

    let result = handle_chat(state, request_id.clone(), body).await;
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
    body: serde_json::Value,
) -> Result<ChatResponse, ApiError> {
    if contains_api_key(&body) {
        return Err(ApiError::invalid_request(
            "ключ провайдера задаёт сервис: поле api_key в запросе не принимается",
        ));
    }
    let request: ChatRequest = serde_json::from_value(body)
        .map_err(|err| ApiError::invalid_request(format!("тело запроса не разобрано: {err}")))?;

    let history = history_from(&request)?;
    let settings = settings_from(&state, request)?;
    let model = settings
        .model
        .clone()
        .unwrap_or_else(|| state.config.model.clone());

    // Вся логика стадий живёт в конвейере: обработчик только собирает
    // контекст и отображает исход в HTTP.
    let pipeline = Pipeline::new(state.agent.clone());
    let context = RequestContext::new(request_id.clone(), history, settings);

    match pipeline.run(context).await {
        Ok(PipelineOutcome::Completed { reply, policy }) => {
            Ok(ChatResponse::new(request_id, model, &reply, policy))
        }
        Ok(PipelineOutcome::Rejected {
            stage, code, reason, ..
        }) => {
            tracing::info!(%request_id, %stage, %code, "запрос отклонён политикой");
            Err(ApiError::policy_rejected(code, reason))
        }
        Err(err) => Err(ApiError::from_agent_error(&err)),
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
