//! Тесты контракта `/v1`: проверяются через `oneshot`, без реального сокета,
//! а провайдер поднимается на `wiremock`.

use crate::app::router;
use crate::config::API_KEY_VAR;
use crate::dto::{ChatRequest, ChatResponse};
use crate::error::ErrorEnvelope;
use crate::middleware::REQUEST_ID_HEADER;
use crate::state::{test_lock, AppState};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Запрос произвольного метода с необязательными телом и токеном — общий
/// строитель для тестов управления чатами.
fn request(method: &str, uri: &str, body: Option<serde_json::Value>, token: Option<&str>) -> Request<Body> {
    let mut builder = Request::builder().method(method).uri(uri);
    if let Some(token) = token {
        builder = builder.header("authorization", format!("Bearer {token}"));
    }
    match body {
        Some(body) => builder
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .expect("запрос"),
        None => builder.body(Body::empty()).expect("запрос"),
    }
}

pub struct Sent {
    pub status: StatusCode,
    pub body: serde_json::Value,
    pub request_id_header: String,
}

pub async fn send(state: AppState, request: Request<Body>) -> Sent {
    let response = router(state).oneshot(request).await.expect("ответ");
    let status = response.status();
    let request_id_header = response
        .headers()
        .get(REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_string();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("тело");
    let body = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    Sent {
        status,
        body,
        request_id_header,
    }
}

pub fn post_chat(body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("запрос")
}

pub fn get(uri: &str) -> Request<Body> {
    Request::builder()
        .uri(uri)
        .body(Body::empty())
        .expect("запрос")
}

/// Ответ провайдера в формате Chat Completions.
pub fn provider_reply(content: &str) -> serde_json::Value {
    serde_json::json!({
        "choices": [{ "message": { "content": content, "reasoning_content": null } }],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "completion_tokens_details": { "reasoning_tokens": 3 }
        }
    })
}

/// Различает вызов модели ради пересказа (тело содержит инструкцию
/// компактизации) и обычный диалоговый вызов: оба идут на один и тот же
/// `/chat/completions`, поэтому маршрутизация мока — по содержимому тела
/// запроса (specs/context-summary, интеграционные тесты компактизации).
struct BodyContains(&'static str);

impl wiremock::Match for BodyContains {
    fn matches(&self, request: &wiremock::Request) -> bool {
        String::from_utf8_lossy(&request.body).contains(self.0)
    }
}

struct BodyLacks(&'static str);

impl wiremock::Match for BodyLacks {
    fn matches(&self, request: &wiremock::Request) -> bool {
        !String::from_utf8_lossy(&request.body).contains(self.0)
    }
}

const SUMMARY_CALL_MARKER: &str = "Обнови пересказ разговора";

/// Дозаписывает `pairs` обменов (вопрос/ответ) в чат через
/// `POST /v1/chats/{id}/messages`, не обращаясь к провайдеру: удобно для
/// подготовки длинной истории перед проверкой компактизации.
async fn seed_exchanges(state: &AppState, chat_id: &str, pairs: &[(&str, &str)]) {
    let messages: Vec<serde_json::Value> = pairs
        .iter()
        .flat_map(|(question, answer)| {
            [
                serde_json::json!({ "role": "user", "content": question }),
                serde_json::json!({ "role": "assistant", "content": answer }),
            ]
        })
        .collect();
    let sent = send(
        state.clone(),
        append(chat_id, serde_json::json!({ "messages": messages }), None),
    )
    .await;
    assert_eq!(sent.status, StatusCode::CREATED, "подготовка истории: {}", sent.body);
}

/// Сервис, у которого апстрим — заданный `wiremock`.
pub async fn state_with_provider(server: &MockServer, extra: &[(&str, &str)]) -> AppState {
    let uri = server.uri();
    let mut pairs: Vec<(&str, &str)> = vec![
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_UPSTREAM_BASE_URL", uri.as_str()),
        ("AGENTD_MODEL", "model-a"),
    ];
    pairs.extend_from_slice(extra);
    AppState::with_env(&pairs).await
}

// --- 5.1 Сериализация контракта ---

#[test]
fn request_with_prompt_is_parsed() {
    let request: ChatRequest =
        serde_json::from_value(serde_json::json!({ "prompt": "привет" })).expect("запрос");
    assert_eq!(request.prompt.as_deref(), Some("привет"));
    assert!(request.messages.is_none());
}

#[test]
fn request_with_messages_is_parsed() {
    let request: ChatRequest = serde_json::from_value(serde_json::json!({
        "messages": [
            { "role": "user", "content": "вопрос" },
            { "role": "assistant", "content": "ответ" }
        ],
        "settings": { "model": "model-a", "temperature": 0.3 },
        "metadata": { "trace": "abc" }
    }))
    .expect("запрос");
    let messages = request.messages.expect("история");
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[1].content, "ответ");
    let settings = request.settings.expect("настройки");
    assert_eq!(settings.model.as_deref(), Some("model-a"));
    assert_eq!(settings.temperature, Some(Some(0.3)));
}

#[test]
fn successful_response_has_contract_shape() {
    let response = ChatResponse {
        request_id: "req-1".to_string(),
        content: "ответ".to_string(),
        reasoning: None,
        model: "model-a".to_string(),
        usage: Default::default(),
        timing: Default::default(),
        policy: Default::default(),
        chat_id: None,
        seq: None,
        context: None,
    };
    let value = serde_json::to_value(&response).expect("сериализация");
    assert_eq!(value["request_id"], "req-1");
    assert_eq!(value["content"], "ответ");
    assert!(value["reasoning"].is_null());
    assert!(value["chat_id"].is_null());
    assert_eq!(value["model"], "model-a");
    assert!(value["usage"].is_object());
    assert!(value["timing"].is_object());
    assert_eq!(value["policy"]["input"], serde_json::json!([]));
    assert_eq!(value["policy"]["output"], serde_json::json!([]));
    assert!(value["policy"]["judge"].is_null());
}

// --- 5.2 Валидация тела ---

async fn expect_invalid_request(body: serde_json::Value, state: AppState) -> ErrorEnvelope {
    let sent = send(state, post_chat(body)).await;
    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);
    let envelope: ErrorEnvelope = serde_json::from_value(sent.body).expect("конверт");
    assert_eq!(envelope.error.code, "invalid_request");
    envelope
}

#[tokio::test]
async fn both_prompt_and_messages_are_rejected() {
    let _guard = test_lock();
    expect_invalid_request(
        serde_json::json!({
            "prompt": "привет",
            "messages": [{ "role": "user", "content": "привет" }]
        }),
        AppState::for_tests().await,
    )
    .await;
}

#[tokio::test]
async fn neither_prompt_nor_messages_is_rejected() {
    let _guard = test_lock();
    expect_invalid_request(serde_json::json!({ "prompt": "  " }), AppState::for_tests().await).await;
}

#[tokio::test]
async fn api_key_in_body_is_rejected() {
    let _guard = test_lock();
    let envelope = expect_invalid_request(
        serde_json::json!({ "prompt": "привет", "settings": { "api_key": "sk-1" } }),
        AppState::for_tests().await,
    )
    .await;
    assert!(envelope.error.message.contains("api_key"));
}

#[tokio::test]
async fn unknown_model_is_rejected() {
    let _guard = test_lock();
    expect_invalid_request(
        serde_json::json!({ "prompt": "привет", "settings": { "model": "model-x" } }),
        AppState::with_env(&[
            (API_KEY_VAR, "secret-key-value"),
            ("AGENTD_MODEL", "model-a"),
        ])
        .await,
    )
    .await;
}

#[tokio::test]
async fn ollama_without_url_is_rejected() {
    let _guard = test_lock();
    expect_invalid_request(
        serde_json::json!({ "prompt": "привет", "settings": { "provider": "ollama" } }),
        AppState::for_tests().await,
    )
    .await;
}

// --- 5.3 Наложение настроек ---

#[test]
fn partial_settings_override_only_given_fields() {
    use crate::dto::ChatSettingsDto;
    use agentcore::config::{ChatSettings, Provider, ReasoningMode};

    let defaults = ChatSettings {
        model: Some("model-a".to_string()),
        reasoning: ReasoningMode::StepByStep,
        ..ChatSettings::default()
    };
    let dto: ChatSettingsDto =
        serde_json::from_value(serde_json::json!({ "temperature": 0.5 })).expect("настройки");
    let merged = dto.apply_to(defaults).expect("наложение");

    assert_eq!(merged.sampling.temperature, Some(0.5));
    assert_eq!(merged.model.as_deref(), Some("model-a"));
    assert_eq!(merged.reasoning, ReasoningMode::StepByStep);
    assert_eq!(merged.provider, Provider::Cloud);
}

// --- 5.4 Успешный вызов через конвейер ---

#[tokio::test]
async fn chat_returns_reply_from_provider() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ модели")))
        .mount(&server)
        .await;

    let state = state_with_provider(&server, &[]).await;
    let sent = send(state, post_chat(serde_json::json!({ "prompt": "привет" }))).await;

    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    assert_eq!(sent.body["content"], "ответ модели");
    assert_eq!(sent.body["model"], "model-a");
    assert_eq!(sent.body["usage"]["total_tokens"], 18);
    assert_eq!(sent.body["usage"]["reasoning_tokens"], 3);
    assert!(sent.body["timing"]["duration_ms"].is_number());
    assert_eq!(sent.body["policy"]["input"], serde_json::json!([]));
    assert_eq!(sent.body["policy"]["output"], serde_json::json!([]));
    assert!(sent.body["policy"]["judge"].is_null());
    assert_eq!(sent.body["request_id"], sent.request_id_header);
}

// --- 5.4b Лимит контекстного окна ---

#[tokio::test]
async fn no_configured_or_client_limit_reaches_provider() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ модели")))
        .mount(&server)
        .await;

    let state = state_with_provider(&server, &[]).await;
    let sent = send(state, post_chat(serde_json::json!({ "prompt": "привет" }))).await;

    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
}

#[tokio::test]
async fn client_limit_narrower_than_operator_default_is_accepted() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ модели")))
        .mount(&server)
        .await;

    let state = state_with_provider(&server, &[("AGENTD_MAX_CONTEXT_TOKENS", "1000")]).await;
    let sent = send(
        state,
        post_chat(serde_json::json!({
            "prompt": "привет",
            "settings": { "max_context_tokens": 500 }
        })),
    )
    .await;

    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
}

#[tokio::test]
async fn client_limit_wider_than_operator_default_is_rejected() {
    let _guard = test_lock();
    let state = state_with_provider(
        &MockServer::start().await,
        &[("AGENTD_MAX_CONTEXT_TOKENS", "1000")],
    )
    .await;
    let sent = send(
        state,
        post_chat(serde_json::json!({
            "prompt": "привет",
            "settings": { "max_context_tokens": 1001 }
        })),
    )
    .await;

    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "context_limit_invalid");
}

#[tokio::test]
async fn client_limit_without_operator_default_is_accepted_and_still_enforced() {
    // Клиентский лимит действует самостоятельно: отсутствие операторского
    // лимита не повод для отказа (specs/chat-context-limit, «Задан только
    // клиентский лимит»; design.md, решение 1).
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ модели")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[]).await;
    let sent = send(
        state.clone(),
        post_chat(serde_json::json!({
            "prompt": "привет",
            "settings": { "max_context_tokens": 500 }
        })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);

    let sent = send(
        state,
        post_chat(serde_json::json!({
            "prompt": "привет, как дела сегодня?",
            "settings": { "max_context_tokens": 1 }
        })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "context_limit_exceeded");
}

#[tokio::test]
async fn history_exceeding_effective_limit_is_rejected_without_calling_provider() {
    let _guard = test_lock();
    // Апстрим без смонтированных ожиданий: обращение к нему обвалило бы тест.
    let server = MockServer::start().await;
    let state = state_with_provider(&server, &[("AGENTD_MAX_CONTEXT_TOKENS", "1")]).await;
    let sent = send(
        state,
        post_chat(serde_json::json!({ "prompt": "привет, как дела сегодня?" })),
    )
    .await;

    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "context_limit_exceeded");
}

// --- 5.5 Служебные эндпоинты ---

#[tokio::test]
async fn models_lists_allowed_models() {
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_MODEL", "model-a"),
        ("AGENTD_ALLOWED_MODELS", "model-a,model-b"),
    ])
    .await;
    let sent = send(state, get("/v1/models")).await;
    assert_eq!(sent.status, StatusCode::OK);
    assert_eq!(
        sent.body["models"],
        serde_json::json!(["model-a", "model-b"])
    );
}

#[tokio::test]
async fn readyz_is_200_for_valid_config() {
    let _guard = test_lock();
    let sent = send(AppState::for_tests().await, get("/readyz")).await;
    assert_eq!(sent.status, StatusCode::OK);
}

#[tokio::test]
async fn readyz_is_503_for_invalid_config() {
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_UPSTREAM_BASE_URL", "не-адрес"),
    ])
    .await;
    let sent = send(state, get("/readyz")).await;
    assert_eq!(sent.status, StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn readyz_is_503_when_storage_fails_after_start() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    state.db.close().await;
    let sent = send(state.clone(), get("/readyz")).await;
    assert_eq!(sent.status, StatusCode::SERVICE_UNAVAILABLE);
    let healthz = send(state, get("/healthz")).await;
    assert_eq!(healthz.status, StatusCode::OK, "healthz не должен зависеть от базы");
}

// --- 5.6 Отображение ошибок ---

#[tokio::test]
async fn provider_error_maps_to_502_without_upstream_details() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    let upstream_host = server.address().to_string();
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_json(serde_json::json!({
            "error": { "message": format!("сбой на {upstream_host} с ключом sk-secret") }
        })))
        .mount(&server)
        .await;

    let state = state_with_provider(&server, &[]).await;
    let sent = send(state, post_chat(serde_json::json!({ "prompt": "привет" }))).await;

    assert_eq!(sent.status, StatusCode::BAD_GATEWAY, "тело: {}", sent.body);
    let raw = sent.body.to_string();
    assert!(!raw.contains(&upstream_host), "адрес апстрима утёк: {raw}");
    assert!(!raw.contains("sk-secret"), "ключ утёк: {raw}");
    assert_eq!(sent.body["error"]["code"], "upstream_error");
    assert_eq!(sent.body["error"]["request_id"], sent.request_id_header);
}

#[tokio::test]
async fn provider_timeout_maps_to_504() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(provider_reply("поздно"))
                .set_delay(std::time::Duration::from_secs(5)),
        )
        .mount(&server)
        .await;

    let state = state_with_provider(&server, &[("AGENTD_REQUEST_TIMEOUT_SECS", "1")]).await;
    let sent = send(state, post_chat(serde_json::json!({ "prompt": "привет" }))).await;

    assert_eq!(sent.status, StatusCode::GATEWAY_TIMEOUT, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "upstream_timeout");
}

// --- Служебные маршруты и слои ---

#[tokio::test]
async fn healthz_answers_200() {
    let _guard = test_lock();
    let sent = send(AppState::for_tests().await, get("/healthz")).await;
    assert_eq!(sent.status, StatusCode::OK);
}

fn post_chat_with_token(body: serde_json::Value, token: Option<&str>) -> Request<Body> {
    let mut builder = Request::builder()
        .method("POST")
        .uri("/v1/chat")
        .header("content-type", "application/json");
    if let Some(token) = token {
        builder = builder.header("authorization", format!("Bearer {token}"));
    }
    builder.body(Body::from(body.to_string())).expect("запрос")
}

async fn authenticated_state() -> AppState {
    AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_MODEL", "model-a"),
        ("AGENTD_CLIENT_TOKENS", "client-token-1,client-token-2"),
    ])
    .await
}

#[tokio::test]
async fn request_without_token_is_401() {
    let _guard = test_lock();
    let sent = send(
        authenticated_state().await,
        post_chat_with_token(serde_json::json!({ "prompt": "привет" }), None),
    )
    .await;
    assert_eq!(sent.status, StatusCode::UNAUTHORIZED);
    assert_eq!(sent.body["error"]["code"], "unauthorized");
    assert_eq!(sent.body["error"]["request_id"], sent.request_id_header);
}

#[tokio::test]
async fn request_with_foreign_token_is_401() {
    let _guard = test_lock();
    let sent = send(
        authenticated_state().await,
        post_chat_with_token(serde_json::json!({ "prompt": "привет" }), Some("чужой")),
    )
    .await;
    assert_eq!(sent.status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn healthz_needs_no_token() {
    let _guard = test_lock();
    let sent = send(authenticated_state().await, get("/healthz")).await;
    assert_eq!(sent.status, StatusCode::OK);
}

#[tokio::test]
async fn empty_token_list_serves_without_authentication() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ")))
        .mount(&server)
        .await;

    let state = state_with_provider(&server, &[]).await;
    assert!(state.config.client_tokens.is_empty());
    let sent = send(state, post_chat(serde_json::json!({ "prompt": "привет" }))).await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
}

#[tokio::test]
async fn oversized_body_is_413() {
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_MODEL", "model-a"),
        ("AGENTD_MAX_BODY_BYTES", "64"),
    ])
    .await;
    let body = serde_json::json!({ "prompt": "x".repeat(500) });
    let sent = send(state, post_chat(body)).await;
    assert_eq!(sent.status, StatusCode::PAYLOAD_TOO_LARGE, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "payload_too_large");
}

#[tokio::test]
async fn exceeded_concurrency_is_429() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(provider_reply("ответ"))
                .set_delay(std::time::Duration::from_secs(2)),
        )
        .mount(&server)
        .await;

    let state = state_with_provider(
        &server,
        &[
            ("AGENTD_MAX_CONCURRENCY", "1"),
            ("AGENTD_REQUEST_TIMEOUT_SECS", "10"),
        ],
    )
    .await;

    // Оба запроса идут через один и тот же собранный роутер: лимит
    // конкурентности живёт в его слое, а не в отдельном экземпляре.
    let app = router(state);
    let first = tokio::spawn({
        let app = app.clone();
        async move {
            app.oneshot(post_chat(serde_json::json!({ "prompt": "первый" })))
                .await
                .expect("ответ")
                .status()
        }
    });
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let second = app
        .oneshot(post_chat(serde_json::json!({ "prompt": "второй" })))
        .await
        .expect("ответ");

    assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(first.await.expect("задача"), StatusCode::OK);
}

#[tokio::test]
async fn request_id_matches_header_on_error() {
    let _guard = test_lock();
    let sent = send(
        AppState::for_tests().await,
        post_chat(serde_json::json!({ "messages": [] })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::BAD_REQUEST);
    assert!(!sent.request_id_header.is_empty());
    assert_eq!(sent.body["error"]["request_id"], sent.request_id_header);
}

// --- Управление чатами: создание, список, чтение, изменение, удаление ---

async fn create_chat(state: AppState, token: Option<&str>, body: serde_json::Value) -> Sent {
    send(state, request("POST", "/v1/chats", Some(body), token)).await
}

// 5.2 — отказ хранилища

#[tokio::test]
async fn storage_error_does_not_leak_details_and_has_request_id() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    state.db.close().await;
    let sent = create_chat(state, None, serde_json::json!({})).await;
    assert_eq!(sent.status, StatusCode::INTERNAL_SERVER_ERROR, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "storage_error");
    assert!(!sent.request_id_header.is_empty());
    assert_eq!(sent.body["error"]["request_id"], sent.request_id_header);
    let raw = sent.body.to_string();
    assert!(!raw.to_lowercase().contains("database"), "текст ошибки хранилища утёк: {raw}");
}

// 5.3 — создание чата

#[tokio::test]
async fn chat_is_created_with_defaults() {
    let _guard = test_lock();
    let sent = create_chat(AppState::for_tests().await, None, serde_json::json!({})).await;
    assert_eq!(sent.status, StatusCode::CREATED, "тело: {}", sent.body);
    assert_eq!(sent.body["title"], "Новый чат");
    assert_eq!(sent.body["message_count"], 0);
    assert!(sent.body["id"].is_string());
    assert!(sent.body["settings"].is_object());
}

#[tokio::test]
async fn blank_title_is_rejected() {
    let _guard = test_lock();
    let sent = create_chat(
        AppState::for_tests().await,
        None,
        serde_json::json!({ "title": "   " }),
    )
    .await;
    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "invalid_request");
}

#[tokio::test]
async fn chat_settings_are_validated_on_create() {
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_MODEL", "model-a"),
    ])
    .await;
    let sent = create_chat(
        state,
        None,
        serde_json::json!({ "settings": { "model": "model-x" } }),
    )
    .await;
    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "invalid_request");
}

// 5.4 — список чатов

#[tokio::test]
async fn list_chats_paginates_over_http() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    for i in 0..120 {
        let sent = create_chat(state.clone(), None, serde_json::json!({ "title": format!("Чат {i}") })).await;
        assert_eq!(sent.status, StatusCode::CREATED);
    }

    let mut seen = std::collections::HashSet::new();
    let mut cursor: Option<String> = None;
    let mut page_sizes = Vec::new();
    loop {
        let uri = match &cursor {
            Some(cursor) => format!("/v1/chats?limit=50&cursor={cursor}"),
            None => "/v1/chats?limit=50".to_string(),
        };
        let sent = send(state.clone(), request("GET", &uri, None, None)).await;
        assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
        let chats = sent.body["chats"].as_array().expect("список чатов");
        page_sizes.push(chats.len());
        for chat in chats {
            assert!(seen.insert(chat["id"].as_str().unwrap().to_string()));
        }
        let next = sent.body["next_cursor"].as_str().map(str::to_string);
        if next.is_none() {
            break;
        }
        cursor = next;
    }
    assert_eq!(seen.len(), 120);
    assert_eq!(page_sizes, vec![50, 50, 20]);
}

#[tokio::test]
async fn list_limit_over_cap_is_rejected() {
    let _guard = test_lock();
    let sent = send(
        AppState::for_tests().await,
        request("GET", "/v1/chats?limit=500", None, None),
    )
    .await;
    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "invalid_request");
}

#[tokio::test]
async fn unknown_cursor_is_rejected() {
    let _guard = test_lock();
    let sent = send(
        AppState::for_tests().await,
        request("GET", "/v1/chats?cursor=не-курсор", None, None),
    )
    .await;
    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "invalid_request");
}

// 5.5 — чтение чата

#[tokio::test]
async fn empty_chat_reads_with_no_messages() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap();

    let sent = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    assert_eq!(sent.body["messages"], serde_json::json!([]));
    assert!(sent.body["next_after"].is_null());
}

#[tokio::test]
async fn chat_messages_are_read_in_parts() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[]).await;

    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    for i in 0..3 {
        let sent = send(
            state.clone(),
            post_chat(serde_json::json!({ "chat_id": id, "prompt": format!("вопрос {i}") })),
        )
        .await;
        assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    }

    let first = send(state.clone(), request("GET", &format!("/v1/chats/{id}?limit=4"), None, None)).await;
    assert_eq!(first.status, StatusCode::OK, "тело: {}", first.body);
    let messages = first.body["messages"].as_array().expect("сообщения");
    assert_eq!(messages.len(), 4);
    assert_eq!(first.body["next_after"], 4);

    let after = first.body["next_after"].as_i64().unwrap();
    let second = send(
        state,
        request("GET", &format!("/v1/chats/{id}?after={after}"), None, None),
    )
    .await;
    let messages = second.body["messages"].as_array().expect("сообщения");
    assert_eq!(messages.len(), 2);
    assert!(second.body["next_after"].is_null());
}

// 5.6 — переименование и изменение настроек

#[tokio::test]
async fn chat_is_renamed() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let sent = send(
        state,
        request(
            "PATCH",
            &format!("/v1/chats/{id}"),
            Some(serde_json::json!({ "title": "Разбор логов" })),
            None,
        ),
    )
    .await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    assert_eq!(sent.body["title"], "Разбор логов");
}

#[tokio::test]
async fn patch_settings_persist_for_future_requests() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[]).await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let patched = send(
        state.clone(),
        request(
            "PATCH",
            &format!("/v1/chats/{id}"),
            Some(serde_json::json!({ "settings": { "temperature": 0.2 } })),
            None,
        ),
    )
    .await;
    assert_eq!(patched.status, StatusCode::OK, "тело: {}", patched.body);
    assert_eq!(patched.body["settings"]["sampling"]["temperature"], 0.2);

    let sent = send(
        state,
        post_chat(serde_json::json!({ "chat_id": id, "prompt": "привет" })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
}

#[tokio::test]
async fn patch_with_empty_body_is_rejected() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let sent = send(
        state,
        request("PATCH", &format!("/v1/chats/{id}"), Some(serde_json::json!({})), None),
    )
    .await;
    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "invalid_request");
}

#[tokio::test]
async fn patch_with_invalid_settings_does_not_change_chat() {
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_MODEL", "model-a"),
    ])
    .await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let sent = send(
        state.clone(),
        request(
            "PATCH",
            &format!("/v1/chats/{id}"),
            Some(serde_json::json!({ "settings": { "model": "model-x" } })),
            None,
        ),
    )
    .await;
    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);

    let loaded = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert_eq!(loaded.body["title"], "Новый чат");
}

// --- Настройки компактизации в контракте ---

#[tokio::test]
async fn patch_without_summary_field_keeps_stored_value_and_null_resets_it() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let patched = send(
        state.clone(),
        request(
            "PATCH",
            &format!("/v1/chats/{id}"),
            Some(serde_json::json!({ "settings": { "summary_keep_messages": 15 } })),
            None,
        ),
    )
    .await;
    assert_eq!(patched.status, StatusCode::OK, "тело: {}", patched.body);
    assert_eq!(patched.body["settings"]["summary_keep_messages"], 15);

    // Поле отсутствует — сохранённое значение остаётся.
    let renamed = send(
        state.clone(),
        request(
            "PATCH",
            &format!("/v1/chats/{id}"),
            Some(serde_json::json!({ "title": "Другой заголовок" })),
            None,
        ),
    )
    .await;
    assert_eq!(renamed.status, StatusCode::OK, "тело: {}", renamed.body);
    assert_eq!(renamed.body["settings"]["summary_keep_messages"], 15);

    // Явный null возвращает к операторскому умолчанию.
    let cleared = send(
        state.clone(),
        request(
            "PATCH",
            &format!("/v1/chats/{id}"),
            Some(serde_json::json!({ "settings": { "summary_keep_messages": null } })),
            None,
        ),
    )
    .await;
    assert_eq!(cleared.status, StatusCode::OK, "тело: {}", cleared.body);
    assert!(cleared.body["settings"]["summary_keep_messages"].is_null());
}

#[tokio::test]
async fn summary_keep_messages_above_operator_ceiling_is_rejected() {
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_SUMMARY_KEEP_MESSAGES", "20"),
    ])
    .await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let patched = send(
        state.clone(),
        request(
            "PATCH",
            &format!("/v1/chats/{id}"),
            Some(serde_json::json!({ "settings": { "summary_keep_messages": 40 } })),
            None,
        ),
    )
    .await;
    assert_eq!(patched.status, StatusCode::BAD_REQUEST, "тело: {}", patched.body);
    assert_eq!(patched.body["error"]["code"], "summary_settings_invalid");

    // Разовое переопределение той же границей.
    let sent = send(
        state,
        post_chat(serde_json::json!({
            "chat_id": id,
            "prompt": "привет",
            "settings": { "summary_keep_messages": 40 }
        })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "summary_settings_invalid");
}

#[tokio::test]
async fn summary_step_messages_below_operator_floor_is_rejected() {
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_SUMMARY_STEP_MESSAGES", "10"),
    ])
    .await;
    let created = create_chat(
        state.clone(),
        None,
        serde_json::json!({ "settings": { "summary_step_messages": 2 } }),
    )
    .await;
    assert_eq!(created.status, StatusCode::BAD_REQUEST, "тело: {}", created.body);
    assert_eq!(created.body["error"]["code"], "summary_settings_invalid");
}

#[tokio::test]
async fn zero_summary_keep_messages_is_rejected() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(
        state,
        None,
        serde_json::json!({ "settings": { "summary_keep_messages": 0 } }),
    )
    .await;
    assert_eq!(created.status, StatusCode::BAD_REQUEST, "тело: {}", created.body);
    assert_eq!(created.body["error"]["code"], "summary_settings_invalid");
}

#[tokio::test]
async fn chat_can_enable_summary_when_operator_default_is_disabled() {
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_SUMMARY_ENABLED", "false"),
    ])
    .await;
    let enabled = create_chat(
        state.clone(),
        None,
        serde_json::json!({ "settings": { "summary_enabled": true } }),
    )
    .await;
    assert_eq!(enabled.status, StatusCode::CREATED, "тело: {}", enabled.body);
    assert_eq!(enabled.body["settings"]["summary_enabled"], true);

    let default_chat = create_chat(state, None, serde_json::json!({})).await;
    assert!(default_chat.body["settings"]["summary_enabled"].is_null());
}

// --- Наблюдаемость: поля пересказа только для чтения ---

#[tokio::test]
async fn chat_without_summary_reports_null_summary_fields() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let loaded = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert!(loaded.body["summary"].is_null());
    assert!(loaded.body["summary_through_seq"].is_null());
}

#[tokio::test]
async fn patch_with_summary_field_is_rejected_as_read_only() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let sent = send(
        state.clone(),
        request(
            "PATCH",
            &format!("/v1/chats/{id}"),
            Some(serde_json::json!({ "settings": { "summary": "подделанный пересказ" } })),
            None,
        ),
    )
    .await;
    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);

    let loaded = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert!(loaded.body["summary"].is_null());
}

// 5.7 — удаление чата

#[tokio::test]
async fn delete_chat_then_second_delete_is_404() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let first = send(state.clone(), request("DELETE", &format!("/v1/chats/{id}"), None, None)).await;
    assert_eq!(first.status, StatusCode::NO_CONTENT, "тело: {}", first.body);
    assert!(first.body.is_null());

    let second = send(state, request("DELETE", &format!("/v1/chats/{id}"), None, None)).await;
    assert_eq!(second.status, StatusCode::NOT_FOUND);
    assert_eq!(second.body["error"]["code"], "chat_not_found");
}

// --- Диалог в чате (POST /v1/chat с chat_id) ---

// 6.1 — разбор тела

#[tokio::test]
async fn chat_id_with_messages_is_rejected() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let sent = send(
        state,
        post_chat(serde_json::json!({
            "chat_id": id,
            "messages": [{ "role": "user", "content": "привет" }]
        })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "invalid_request");
}

#[tokio::test]
async fn chat_id_without_prompt_is_rejected() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let sent = send(state, post_chat(serde_json::json!({ "chat_id": id }))).await;
    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "invalid_request");
}

// 6.2 — успешная запись обмена

#[tokio::test]
async fn chat_id_call_uses_stored_history_and_reports_chat_id_and_seq() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("первый ответ")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[]).await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let first = send(
        state.clone(),
        post_chat(serde_json::json!({ "chat_id": id, "prompt": "первый вопрос" })),
    )
    .await;
    assert_eq!(first.status, StatusCode::OK, "тело: {}", first.body);
    assert_eq!(first.body["chat_id"], id);
    assert_eq!(first.body["seq"], 2);

    server.reset().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("второй ответ")))
        .mount(&server)
        .await;

    let second = send(
        state.clone(),
        post_chat(serde_json::json!({ "chat_id": id, "prompt": "второй вопрос" })),
    )
    .await;
    assert_eq!(second.status, StatusCode::OK, "тело: {}", second.body);
    assert_eq!(second.body["seq"], 4);

    let loaded = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert_eq!(loaded.body["message_count"], 4);
    let messages = loaded.body["messages"].as_array().unwrap();
    assert_eq!(messages[0]["content"], "первый вопрос");
    assert_eq!(messages[2]["content"], "второй вопрос");
}

// 6.3 — ошибка провайдера не меняет чат

#[tokio::test]
async fn provider_error_with_chat_id_does_not_change_chat() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[]).await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();
    let before_updated_at = created.body["updated_at"].as_i64().unwrap();

    let sent = send(
        state.clone(),
        post_chat(serde_json::json!({ "chat_id": id, "prompt": "вопрос" })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::BAD_GATEWAY, "тело: {}", sent.body);

    let loaded = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert_eq!(loaded.body["message_count"], 0);
    assert_eq!(loaded.body["updated_at"], before_updated_at);
}

// 6.4 — разовое переопределение против постоянного изменения

#[tokio::test]
async fn settings_override_in_chat_call_does_not_persist() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[]).await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let sent = send(
        state.clone(),
        post_chat(serde_json::json!({
            "chat_id": id,
            "prompt": "вопрос",
            "settings": { "temperature": 0.9 }
        })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);

    let loaded = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert_ne!(loaded.body["settings"]["sampling"]["temperature"], 0.9);
}

// --- Лимит контекста сохраняется в настройках чата ---

#[tokio::test]
async fn patch_max_context_tokens_persists() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let patched = send(
        state.clone(),
        request(
            "PATCH",
            &format!("/v1/chats/{id}"),
            Some(serde_json::json!({ "settings": { "max_context_tokens": 4000 } })),
            None,
        ),
    )
    .await;
    assert_eq!(patched.status, StatusCode::OK, "тело: {}", patched.body);
    assert_eq!(patched.body["settings"]["max_context_tokens"], 4000);

    let loaded = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert_eq!(loaded.body["settings"]["max_context_tokens"], 4000);
}

#[tokio::test]
async fn patch_null_max_context_tokens_clears_stored_limit() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let patched = send(
        state.clone(),
        request(
            "PATCH",
            &format!("/v1/chats/{id}"),
            Some(serde_json::json!({ "settings": { "max_context_tokens": 4000 } })),
            None,
        ),
    )
    .await;
    assert_eq!(patched.status, StatusCode::OK, "тело: {}", patched.body);

    let cleared = send(
        state.clone(),
        request(
            "PATCH",
            &format!("/v1/chats/{id}"),
            Some(serde_json::json!({ "settings": { "max_context_tokens": null } })),
            None,
        ),
    )
    .await;
    assert_eq!(cleared.status, StatusCode::OK, "тело: {}", cleared.body);
    assert!(cleared.body["settings"]["max_context_tokens"].is_null());

    let loaded = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert!(loaded.body["settings"]["max_context_tokens"].is_null());
}

#[tokio::test]
async fn patch_without_max_context_tokens_field_keeps_stored_value() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let patched = send(
        state.clone(),
        request(
            "PATCH",
            &format!("/v1/chats/{id}"),
            Some(serde_json::json!({ "settings": { "max_context_tokens": 4000 } })),
            None,
        ),
    )
    .await;
    assert_eq!(patched.status, StatusCode::OK, "тело: {}", patched.body);

    let renamed = send(
        state.clone(),
        request(
            "PATCH",
            &format!("/v1/chats/{id}"),
            Some(serde_json::json!({ "settings": { "temperature": 0.3 } })),
            None,
        ),
    )
    .await;
    assert_eq!(renamed.status, StatusCode::OK, "тело: {}", renamed.body);

    let loaded = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert_eq!(loaded.body["settings"]["max_context_tokens"], 4000);
}

#[tokio::test]
async fn chat_call_without_override_uses_stored_chat_limit() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    // Никаких смонтированных ожиданий: запрос отклоняется до обращения к
    // апстриму, обращение к нему обвалило бы тест.
    let state = state_with_provider(&server, &[("AGENTD_MAX_CONTEXT_TOKENS", "1000")]).await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let patched = send(
        state.clone(),
        request(
            "PATCH",
            &format!("/v1/chats/{id}"),
            Some(serde_json::json!({ "settings": { "max_context_tokens": 1 } })),
            None,
        ),
    )
    .await;
    assert_eq!(patched.status, StatusCode::OK, "тело: {}", patched.body);

    let sent = send(
        state,
        post_chat(serde_json::json!({
            "chat_id": id,
            "prompt": "привет, как дела сегодня?"
        })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "context_limit_exceeded");
}

#[tokio::test]
async fn chat_call_override_does_not_change_stored_chat_limit() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[("AGENTD_MAX_CONTEXT_TOKENS", "1000")]).await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let patched = send(
        state.clone(),
        request(
            "PATCH",
            &format!("/v1/chats/{id}"),
            Some(serde_json::json!({ "settings": { "max_context_tokens": 500 } })),
            None,
        ),
    )
    .await;
    assert_eq!(patched.status, StatusCode::OK, "тело: {}", patched.body);

    let sent = send(
        state.clone(),
        post_chat(serde_json::json!({
            "chat_id": id,
            "prompt": "привет",
            "settings": { "max_context_tokens": 200 }
        })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);

    let loaded = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert_eq!(loaded.body["settings"]["max_context_tokens"], 500);
}

#[tokio::test]
async fn patch_max_context_tokens_above_operator_default_is_rejected_at_storage() {
    // Проверка лимита при сохранении срабатывает тем же кодом, что и при
    // вызове: отказ приходит на PATCH, а не на следующем POST /v1/chat
    // (specs/chat-context-limit, «Сохранение лимита шире операторского»).
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_MAX_CONTEXT_TOKENS", "1000"),
    ])
    .await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let patched = send(
        state.clone(),
        request(
            "PATCH",
            &format!("/v1/chats/{id}"),
            Some(serde_json::json!({ "settings": { "max_context_tokens": 2000 } })),
            None,
        ),
    )
    .await;
    assert_eq!(patched.status, StatusCode::BAD_REQUEST, "тело: {}", patched.body);
    assert_eq!(patched.body["error"]["code"], "context_limit_invalid");

    let loaded = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert!(loaded.body["settings"]["max_context_tokens"].is_null());
}

#[tokio::test]
async fn create_chat_with_ollama_provider_and_excessive_limit_is_rejected() {
    // Проверка лимита не обходится для чатов с локальным провайдером: он
    // относится к размеру истории, а не к способу вызова модели
    // (specs/chat-context-limit, «Сохранение лимита для чата с локальным
    // провайдером»).
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_MAX_CONTEXT_TOKENS", "4000"),
        ("AGENTD_OLLAMA_URL", "http://localhost:11434"),
    ])
    .await;
    let created = create_chat(
        state,
        None,
        serde_json::json!({ "settings": { "provider": "ollama", "max_context_tokens": 8000 } }),
    )
    .await;
    assert_eq!(created.status, StatusCode::BAD_REQUEST, "тело: {}", created.body);
    assert_eq!(created.body["error"]["code"], "context_limit_invalid");
}

// 6.5 — модель чата вне белого списка

#[tokio::test]
async fn chat_model_outside_allowed_list_is_rejected() {
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_MODEL", "model-a"),
        ("AGENTD_ALLOWED_MODELS", "model-a,model-b"),
    ])
    .await;
    let created = create_chat(
        state.clone(),
        None,
        serde_json::json!({ "settings": { "model": "model-b" } }),
    )
    .await;
    let id = created.body["id"].as_str().unwrap().to_string();

    // Модель чата больше не входит в белый список сервиса.
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_MODEL", "model-a"),
        ("AGENTD_ALLOWED_MODELS", "model-a"),
        ("AGENTD_DB_PATH", &state.config.db_path),
    ])
    .await;

    let sent = send(
        state.clone(),
        post_chat(serde_json::json!({ "chat_id": id, "prompt": "вопрос" })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);

    let loaded = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert_eq!(loaded.body["message_count"], 0);
}

// 6.6 — запрос без chat_id ведёт себя как раньше

#[tokio::test]
async fn request_without_chat_id_does_not_touch_storage() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[]).await;

    let sent = send(state.clone(), post_chat(serde_json::json!({ "prompt": "привет" }))).await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    assert!(sent.body["chat_id"].is_null());

    let list = send(state, request("GET", "/v1/chats", None, None)).await;
    assert_eq!(list.body["chats"], serde_json::json!([]));
}

// --- Владение и изоляция ---

// 7.1 — чат виден только создавшему токену

#[tokio::test]
async fn chat_is_visible_only_to_owner_token() {
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_MODEL", "model-a"),
        ("AGENTD_CLIENT_TOKENS", "token-a,token-b"),
    ])
    .await;
    let created = create_chat(state.clone(), Some("token-a"), serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let own_list = send(state.clone(), request("GET", "/v1/chats", None, Some("token-a"))).await;
    assert_eq!(own_list.body["chats"].as_array().unwrap().len(), 1);

    let other_list = send(state, request("GET", "/v1/chats", None, Some("token-b"))).await;
    assert_eq!(other_list.body["chats"], serde_json::json!([]));
    let _ = id;
}

// 7.2 — чужой чат неотличим от несуществующего для всех операций

#[tokio::test]
async fn foreign_chat_operations_are_404_and_do_not_call_model() {
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_MODEL", "model-a"),
        ("AGENTD_CLIENT_TOKENS", "token-a,token-b"),
    ])
    .await;
    let created = create_chat(state.clone(), Some("token-a"), serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let missing = send(
        state.clone(),
        request("GET", "/v1/chats/несуществующий-чат", None, Some("token-b")),
    )
    .await;
    let read = send(
        state.clone(),
        request("GET", &format!("/v1/chats/{id}"), None, Some("token-b")),
    )
    .await;
    assert_eq!(read.status, StatusCode::NOT_FOUND);
    assert_eq!(
        read.body["error"]["code"], missing.body["error"]["code"],
        "чужой чат должен отвечать как несуществующий"
    );
    assert_eq!(read.body["error"]["message"], missing.body["error"]["message"]);

    let patch = send(
        state.clone(),
        request(
            "PATCH",
            &format!("/v1/chats/{id}"),
            Some(serde_json::json!({ "title": "чужое" })),
            Some("token-b"),
        ),
    )
    .await;
    assert_eq!(patch.status, StatusCode::NOT_FOUND);
    assert_eq!(patch.body["error"]["code"], "chat_not_found");

    let delete = send(
        state.clone(),
        request("DELETE", &format!("/v1/chats/{id}"), None, Some("token-b")),
    )
    .await;
    assert_eq!(delete.status, StatusCode::NOT_FOUND);

    // Провайдер не настроен вовсе: если бы модель вызвалась, запрос
    // получил бы сетевую ошибку (502), а не 404.
    let chat_call = send(
        state,
        post_chat_with_token(
            serde_json::json!({ "chat_id": id, "prompt": "вопрос" }),
            Some("token-b"),
        ),
    )
    .await;
    assert_eq!(chat_call.status, StatusCode::NOT_FOUND);
    assert_eq!(chat_call.body["error"]["code"], "chat_not_found");
}

// 7.3 — идентификатор неверного вида

#[tokio::test]
async fn malformed_chat_id_is_404_not_400() {
    let _guard = test_lock();
    let sent = send(
        AppState::for_tests().await,
        request("GET", "/v1/chats/не-идентификатор", None, None),
    )
    .await;
    assert_eq!(sent.status, StatusCode::NOT_FOUND);
    assert_eq!(sent.body["error"]["code"], "chat_not_found");
}

// 7.4 — поведение при выключенной аутентификации

#[tokio::test]
async fn anonymous_clients_share_one_chat_list() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    create_chat(state.clone(), None, serde_json::json!({})).await;

    let first = send(state.clone(), request("GET", "/v1/chats", None, None)).await;
    let second = send(state, request("GET", "/v1/chats", None, None)).await;
    assert_eq!(first.body, second.body);
    assert_eq!(first.body["chats"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn enabling_authentication_hides_anonymous_chats() {
    let _guard = test_lock();
    let dir = tempfile::tempdir().expect("временный каталог");
    let db_path = dir.path().join("agentd.db");
    let db_path = db_path.to_str().unwrap();

    let anonymous_state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_MODEL", "model-a"),
        ("AGENTD_DB_PATH", db_path),
    ])
    .await;
    let created = create_chat(anonymous_state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();
    drop(anonymous_state);

    let authenticated_state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_MODEL", "model-a"),
        ("AGENTD_DB_PATH", db_path),
        ("AGENTD_CLIENT_TOKENS", "token-a"),
    ])
    .await;

    let list = send(
        authenticated_state.clone(),
        request("GET", "/v1/chats", None, Some("token-a")),
    )
    .await;
    assert_eq!(list.body["chats"], serde_json::json!([]));

    let read = send(
        authenticated_state,
        request("GET", &format!("/v1/chats/{id}"), None, Some("token-a")),
    )
    .await;
    assert_eq!(read.status, StatusCode::NOT_FOUND);
}

// --- Дозапись готовых сообщений (POST /v1/chats/{id}/messages) ---

fn append(chat_id: &str, body: serde_json::Value, token: Option<&str>) -> Request<Body> {
    request(
        "POST",
        &format!("/v1/chats/{chat_id}/messages"),
        Some(body),
        token,
    )
}

fn exchange_body() -> serde_json::Value {
    serde_json::json!({
        "messages": [
            { "role": "user", "content": "вопрос локальной модели" },
            {
                "role": "assistant",
                "content": "ответ локальной модели",
                "reasoning": "рассуждение",
                "model": "llama3",
                "usage": { "prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18, "reasoning_tokens": 3 },
                "timing": { "duration_ms": 120, "sent_at": 1000, "received_at": 1001 }
            }
        ]
    })
}

// 1.3 — дозапись не обращается к провайдеру

#[tokio::test]
async fn append_succeeds_without_reachable_provider() {
    let _guard = test_lock();
    // for_tests направляет провайдера на 127.0.0.1:1: доступного апстрима у
    // этого состояния нет вовсе, поэтому успех означает, что дозапись к
    // провайдеру не ходит.
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().expect("идентификатор чата").to_string();

    let sent = send(state, append(&id, exchange_body(), None)).await;
    assert_eq!(sent.status, StatusCode::CREATED, "тело: {}", sent.body);
    assert_eq!(sent.body["chat_id"], id);
    assert_eq!(sent.body["seqs"], serde_json::json!([1, 2]));
    assert!(sent.body["request_id"].is_string());
}

#[tokio::test]
async fn appended_text_is_not_filtered_by_pipeline() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().expect("идентификатор чата").to_string();
    let text = "  текст с краевыми пробелами и словом обход политики  ";

    let sent = send(
        state.clone(),
        append(
            &id,
            serde_json::json!({ "messages": [{ "role": "user", "content": text }] }),
            None,
        ),
    )
    .await;
    assert_eq!(sent.status, StatusCode::CREATED, "тело: {}", sent.body);

    let read = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert_eq!(read.body["messages"][0]["content"], text);
}

// 1.2 — валидация тела дозаписи

#[tokio::test]
async fn append_with_api_key_field_is_rejected() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().expect("идентификатор чата").to_string();

    let sent = send(
        state.clone(),
        append(
            &id,
            serde_json::json!({
                "messages": [{ "role": "user", "content": "вопрос", "api_key": "sk-secret" }]
            }),
            None,
        ),
    )
    .await;
    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "invalid_request");

    let read = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert_eq!(read.body["messages"], serde_json::json!([]));
}

// 1.4 — отказы дозаписи

#[tokio::test]
async fn append_with_empty_messages_is_rejected() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().expect("идентификатор чата").to_string();

    let sent = send(
        state,
        append(&id, serde_json::json!({ "messages": [] }), None),
    )
    .await;
    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "invalid_request");
}

#[tokio::test]
async fn append_with_unknown_role_is_rejected() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().expect("идентификатор чата").to_string();

    let sent = send(
        state,
        append(
            &id,
            serde_json::json!({ "messages": [{ "role": "system", "content": "вопрос" }] }),
            None,
        ),
    )
    .await;
    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "invalid_request");
}

#[tokio::test]
async fn append_to_foreign_and_missing_chat_look_identical() {
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_MODEL", "model-a"),
        ("AGENTD_CLIENT_TOKENS", "token-a,token-b"),
    ])
    .await;
    let created = create_chat(state.clone(), Some("token-a"), serde_json::json!({})).await;
    let id = created.body["id"].as_str().expect("идентификатор чата").to_string();

    let foreign = send(state.clone(), append(&id, exchange_body(), Some("token-b"))).await;
    let missing = send(
        state.clone(),
        append("00000000-0000-0000-0000-000000000000", exchange_body(), Some("token-b")),
    )
    .await;

    assert_eq!(foreign.status, StatusCode::NOT_FOUND, "тело: {}", foreign.body);
    assert_eq!(missing.status, StatusCode::NOT_FOUND, "тело: {}", missing.body);
    assert_eq!(foreign.body["error"]["code"], missing.body["error"]["code"]);
    assert_eq!(foreign.body["error"]["message"], missing.body["error"]["message"]);

    // История владельца от чужой попытки не изменилась.
    let read = send(
        state,
        request("GET", &format!("/v1/chats/{id}"), None, Some("token-a")),
    )
    .await;
    assert_eq!(read.body["messages"], serde_json::json!([]));
}

#[tokio::test]
async fn append_error_envelope_carries_request_id() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let sent = send(
        state,
        append("00000000-0000-0000-0000-000000000000", exchange_body(), None),
    )
    .await;

    assert_eq!(sent.status, StatusCode::NOT_FOUND, "тело: {}", sent.body);
    let envelope: ErrorEnvelope = serde_json::from_value(sent.body).expect("конверт");
    assert!(!sent.request_id_header.is_empty());
    assert_eq!(envelope.error.request_id, sent.request_id_header);
}

// 1.5 — дозаписанное читается и поднимает чат в списке

#[tokio::test]
async fn appended_messages_are_readable_with_telemetry_and_lift_chat() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let first = create_chat(state.clone(), None, serde_json::json!({ "title": "Первый" })).await;
    let second = create_chat(state.clone(), None, serde_json::json!({ "title": "Второй" })).await;
    let first_id = first.body["id"].as_str().expect("идентификатор чата").to_string();
    let second_id = second.body["id"].as_str().expect("идентификатор чата").to_string();

    // Время изменения хранится в секундах, поэтому оба чата, созданные в
    // одну секунду, различаются в списке только по идентификатору. Чтобы
    // подъём чата от дозаписи был наблюдаем, обе метки сдвигаются в прошлое.
    for (id, updated_at) in [(&first_id, 1000), (&second_id, 2000)] {
        sqlx::query("UPDATE chats SET updated_at = ? WHERE id = ?")
            .bind(updated_at)
            .bind(id)
            .execute(&state.db)
            .await
            .expect("подготовка времени изменения");
    }
    let before = send(state.clone(), request("GET", "/v1/chats", None, None)).await;
    assert_eq!(before.body["chats"][0]["id"], second_id);

    let appended = send(state.clone(), append(&first_id, exchange_body(), None)).await;
    assert_eq!(appended.status, StatusCode::CREATED, "тело: {}", appended.body);

    let read = send(
        state.clone(),
        request("GET", &format!("/v1/chats/{first_id}"), None, None),
    )
    .await;
    let messages = read.body["messages"].as_array().expect("сообщения");
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[0]["content"], "вопрос локальной модели");
    assert_eq!(messages[0]["seq"], 1);
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[1]["seq"], 2);
    assert_eq!(messages[1]["reasoning"], "рассуждение");
    assert_eq!(messages[1]["model"], "llama3");
    assert_eq!(messages[1]["usage"]["total_tokens"], 18);
    assert_eq!(messages[1]["timing"]["duration_ms"], 120);
    assert_eq!(read.body["message_count"], 2);

    let list = send(state, request("GET", "/v1/chats", None, None)).await;
    let chats = list.body["chats"].as_array().expect("список чатов");
    assert_eq!(chats[0]["id"], first_id, "дозапись поднимает чат в списке");
    assert_eq!(chats[1]["id"], second_id);
}

// 1.6 — приватность текстов дозаписи

#[test]
fn append_log_records_counts_without_message_text() {
    let _guard = test_lock();
    let capture = crate::telemetry::capture::Capture::default();
    let subscriber = tracing_subscriber::fmt()
        .json()
        .with_writer(capture.clone())
        .finish();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");

    // Перехват журнала ставится вокруг блокирующего вызова: подписчик
    // задаётся на текущий поток, а не на задачу.
    tracing::subscriber::with_default(subscriber, || {
        runtime.block_on(async {
            let state = AppState::for_tests().await;
            let created = create_chat(state.clone(), None, serde_json::json!({})).await;
            let id = created.body["id"].as_str().expect("идентификатор чата").to_string();
            let sent = send(state, append(&id, exchange_body(), None)).await;
            assert_eq!(sent.status, StatusCode::CREATED, "тело: {}", sent.body);
            id
        })
    });

    let output = capture.text();
    assert!(output.contains("сообщения дозаписаны в чат"), "нет записи: {output}");
    assert!(output.contains("\"messages\":2"), "нет числа сообщений: {output}");
    assert!(
        !output.contains("вопрос локальной модели") && !output.contains("ответ локальной модели"),
        "текст сообщений попал в журнал: {output}"
    );
}

// --- Настройки чата с локальным провайдером ---

// 1.8 — чат с провайдером ollama создаётся и меняется при ненастроенном Ollama

fn local_settings(model: &str) -> serde_json::Value {
    serde_json::json!({ "provider": "ollama", "model": model })
}

#[tokio::test]
async fn local_chat_is_stored_without_server_ollama() {
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_MODEL", "model-a"),
    ])
    .await;

    let created = create_chat(
        state.clone(),
        None,
        serde_json::json!({ "title": "Локальный", "settings": local_settings("llama3") }),
    )
    .await;
    assert_eq!(created.status, StatusCode::CREATED, "тело: {}", created.body);
    let id = created.body["id"].as_str().expect("идентификатор чата").to_string();
    assert_eq!(created.body["settings"]["provider"], "ollama");
    assert_eq!(created.body["settings"]["model"], "llama3");

    let read = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert_eq!(read.body["settings"]["provider"], "ollama");
    assert_eq!(read.body["settings"]["model"], "llama3");
}

#[tokio::test]
async fn local_chat_model_can_be_changed_without_allowlist() {
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_MODEL", "model-a"),
    ])
    .await;
    let created = create_chat(
        state.clone(),
        None,
        serde_json::json!({ "settings": local_settings("llama3") }),
    )
    .await;
    let id = created.body["id"].as_str().expect("идентификатор чата").to_string();

    let patched = send(
        state,
        request(
            "PATCH",
            &format!("/v1/chats/{id}"),
            Some(serde_json::json!({ "settings": local_settings("qwen2") })),
            None,
        ),
    )
    .await;
    assert_eq!(patched.status, StatusCode::OK, "тело: {}", patched.body);
    assert_eq!(patched.body["settings"]["model"], "qwen2");
}

// 1.9 — вызов модели в локальном чате остаётся невозможным

#[tokio::test]
async fn call_in_local_chat_is_rejected_without_server_ollama() {
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_MODEL", "model-a"),
    ])
    .await;
    let created = create_chat(
        state.clone(),
        None,
        serde_json::json!({ "settings": local_settings("llama3") }),
    )
    .await;
    let id = created.body["id"].as_str().expect("идентификатор чата").to_string();

    let sent = send(
        state.clone(),
        post_chat(serde_json::json!({ "chat_id": id, "prompt": "привет" })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "invalid_request");
    assert!(
        sent.body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("ollama"),
        "причина не названа: {}",
        sent.body
    );

    let read = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert_eq!(read.body["messages"], serde_json::json!([]));
}

#[tokio::test]
async fn cloud_chat_model_outside_allowlist_is_still_rejected() {
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_MODEL", "model-a"),
    ])
    .await;

    let sent = create_chat(
        state,
        None,
        serde_json::json!({ "settings": { "provider": "cloud", "model": "model-x" } }),
    )
    .await;
    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "invalid_request");
}

// 1.10 — настройки чата задаются целиком

#[tokio::test]
async fn explicit_null_clears_sampling_parameter() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(
        state.clone(),
        None,
        serde_json::json!({ "settings": { "temperature": 0.7, "top_p": 0.9 } }),
    )
    .await;
    let id = created.body["id"].as_str().expect("идентификатор чата").to_string();
    assert_eq!(created.body["settings"]["sampling"]["temperature"], 0.7);

    let patched = send(
        state,
        request(
            "PATCH",
            &format!("/v1/chats/{id}"),
            Some(serde_json::json!({ "settings": { "temperature": null } })),
            None,
        ),
    )
    .await;
    assert_eq!(patched.status, StatusCode::OK, "тело: {}", patched.body);
    assert!(
        patched.body["settings"]["sampling"]["temperature"].is_null(),
        "температура не сброшена: {}",
        patched.body
    );
    assert_eq!(
        patched.body["settings"]["sampling"]["top_p"], 0.9,
        "незаданное поле изменилось: {}",
        patched.body
    );
}

#[tokio::test]
async fn custom_response_mode_can_be_switched_off() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(
        state.clone(),
        None,
        serde_json::json!({
            "settings": { "response_format": { "description": "только JSON", "max_length": 500 } }
        }),
    )
    .await;
    let id = created.body["id"].as_str().expect("идентификатор чата").to_string();
    assert_eq!(created.body["settings"]["custom_response_mode"], true);

    let patched = send(
        state,
        request(
            "PATCH",
            &format!("/v1/chats/{id}"),
            Some(serde_json::json!({ "settings": { "custom_response_mode": false } })),
            None,
        ),
    )
    .await;
    assert_eq!(patched.status, StatusCode::OK, "тело: {}", patched.body);
    assert_eq!(patched.body["settings"]["custom_response_mode"], false);
    // Сам формат остаётся сохранённым: выключен режим, а не описание.
    assert_eq!(patched.body["settings"]["response_format"]["description"], "только JSON");
}

#[tokio::test]
async fn omitted_settings_fields_stay_unchanged() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(
        state.clone(),
        None,
        serde_json::json!({
            "settings": { "temperature": 0.4, "reasoning": "step-by-step", "experts": ["аналитик"] }
        }),
    )
    .await;
    let id = created.body["id"].as_str().expect("идентификатор чата").to_string();

    let patched = send(
        state,
        request(
            "PATCH",
            &format!("/v1/chats/{id}"),
            Some(serde_json::json!({ "settings": { "model": "test-model" } })),
            None,
        ),
    )
    .await;
    assert_eq!(patched.status, StatusCode::OK, "тело: {}", patched.body);
    assert_eq!(patched.body["settings"]["model"], "test-model");
    assert_eq!(patched.body["settings"]["sampling"]["temperature"], 0.4);
    assert_eq!(patched.body["settings"]["experts"], serde_json::json!(["аналитик"]));
}

// --- Компактизация истории: интеграционные тесты через wiremock ---

fn summary_chat_settings(keep: u32, step: u32) -> serde_json::Value {
    serde_json::json!({
        "settings": {
            "summary_enabled": true,
            "summary_keep_messages": keep,
            "summary_step_messages": step
        }
    })
}

// 12.1 — выключенная компактизация: провайдер видит всё, без обращений за пересказом

#[tokio::test]
async fn disabled_summary_sends_full_history_without_summary_calls() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ модели")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[]).await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();
    seed_exchanges(
        &state,
        &id,
        &[("вопрос 1", "ответ 1"), ("вопрос 2", "ответ 2"), ("вопрос 3", "ответ 3")],
    )
    .await;

    let sent = send(
        state,
        post_chat(serde_json::json!({ "chat_id": id, "prompt": "новый вопрос" })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    assert_eq!(sent.body["context"]["replaced_messages"], 0);
    assert_eq!(sent.body["context"]["summary_built"], false);

    let requests = server.received_requests().await.expect("запросы");
    assert_eq!(requests.len(), 1, "не должно быть отдельного обращения за пересказом");
    let body = String::from_utf8_lossy(&requests[0].body);
    assert!(body.contains("вопрос 1"), "полная история должна дойти до провайдера: {body}");
}

// 12.2 — порог достигнут: два обращения, тело второго содержит пересказ и не содержит вытесненных сообщений

#[tokio::test]
async fn reaching_step_builds_summary_and_sends_tail_without_evicted_messages() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(BodyContains(SUMMARY_CALL_MARKER))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("итог пересказа")))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(BodyLacks(SUMMARY_CALL_MARKER))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ модели")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[("AGENTD_SUMMARY_STEP_MESSAGES", "1")]).await;
    let created = create_chat(state.clone(), None, summary_chat_settings(4, 3)).await;
    let id = created.body["id"].as_str().unwrap().to_string();
    // 4 обмена = 8 сообщений; keep=4 оставляет хвост из последних 4 (2
    // обмена), в вытесненной части — 4 сообщения, порог шага (3) достигнут.
    seed_exchanges(
        &state,
        &id,
        &[
            ("вопрос 1", "ответ 1"),
            ("вопрос 2", "ответ 2"),
            ("вопрос 3", "ответ 3"),
            ("вопрос 4", "ответ 4"),
        ],
    )
    .await;

    let sent = send(
        state.clone(),
        post_chat(serde_json::json!({ "chat_id": id, "prompt": "новый вопрос" })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    assert_eq!(sent.body["content"], "ответ модели");
    assert_eq!(sent.body["context"]["replaced_messages"], 4);
    assert_eq!(sent.body["context"]["summary_built"], true);

    let requests = server.received_requests().await.expect("запросы");
    assert_eq!(requests.len(), 2, "ожидались ровно два обращения: пересказ и ответ");
    let main_call = requests
        .iter()
        .find(|r| !String::from_utf8_lossy(&r.body).contains(SUMMARY_CALL_MARKER))
        .expect("основной вызов");
    let body = String::from_utf8_lossy(&main_call.body);
    assert!(body.contains("Краткое содержание предыдущей части разговора"), "{body}");
    assert!(body.contains("итог пересказа"), "{body}");
    assert!(body.contains("вопрос 3"), "хвост из последних 4 сообщений должен дойти: {body}");
    assert!(!body.contains("вопрос 1"), "вытесненное сообщение не должно уйти провайдеру: {body}");
    assert!(!body.contains("вопрос 2"), "вытесненное сообщение не должно уйти провайдеру: {body}");

    let loaded = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert_eq!(loaded.body["summary"], "итог пересказа");
    assert_eq!(loaded.body["summary_through_seq"], 4);
}

// 12.3 — следующий запрос до новой ступени: одно обращение, подставлен прежний пересказ

#[tokio::test]
async fn next_request_below_step_reuses_stored_summary_with_one_call() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(BodyContains(SUMMARY_CALL_MARKER))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("итог пересказа")))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(BodyLacks(SUMMARY_CALL_MARKER))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ модели")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[("AGENTD_SUMMARY_STEP_MESSAGES", "1")]).await;
    let created = create_chat(state.clone(), None, summary_chat_settings(4, 3)).await;
    let id = created.body["id"].as_str().unwrap().to_string();
    seed_exchanges(
        &state,
        &id,
        &[
            ("вопрос 1", "ответ 1"),
            ("вопрос 2", "ответ 2"),
            ("вопрос 3", "ответ 3"),
            ("вопрос 4", "ответ 4"),
        ],
    )
    .await;
    // Первый запрос строит пересказ (через seq 4) и добавляет обмен seq 9/10.
    let first = send(
        state.clone(),
        post_chat(serde_json::json!({ "chat_id": id, "prompt": "новый вопрос" })),
    )
    .await;
    assert_eq!(first.status, StatusCode::OK, "тело: {}", first.body);

    let second = send(
        state.clone(),
        post_chat(serde_json::json!({ "chat_id": id, "prompt": "ещё вопрос" })),
    )
    .await;
    assert_eq!(second.status, StatusCode::OK, "тело: {}", second.body);
    // Вытесненных, но ещё не пересказанных сообщений после первого запроса —
    // 2 (seq 5 и 6), это меньше шага (3): пересказ не перестраивается.
    assert_eq!(second.body["context"]["summary_built"], false);

    let requests = server.received_requests().await.expect("запросы");
    assert_eq!(requests.len(), 3, "1 пересказ + 2 обычных ответа");
    let summary_calls = requests
        .iter()
        .filter(|r| String::from_utf8_lossy(&r.body).contains(SUMMARY_CALL_MARKER))
        .count();
    assert_eq!(summary_calls, 1, "пересказ не должен перестраиваться на втором запросе");

    let loaded = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert_eq!(loaded.body["summary_through_seq"], 4, "граница пересказа не должна была сдвинуться");
}

// 12.4 — ошибка провайдера на вызове пересказа не ломает пользовательский запрос

#[tokio::test]
async fn summary_call_failure_degrades_without_breaking_user_request() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(BodyContains(SUMMARY_CALL_MARKER))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(BodyLacks(SUMMARY_CALL_MARKER))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ модели")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[("AGENTD_SUMMARY_STEP_MESSAGES", "1")]).await;
    let created = create_chat(state.clone(), None, summary_chat_settings(4, 3)).await;
    let id = created.body["id"].as_str().unwrap().to_string();
    seed_exchanges(
        &state,
        &id,
        &[
            ("вопрос 1", "ответ 1"),
            ("вопрос 2", "ответ 2"),
            ("вопрос 3", "ответ 3"),
            ("вопрос 4", "ответ 4"),
        ],
    )
    .await;

    let sent = send(
        state.clone(),
        post_chat(serde_json::json!({ "chat_id": id, "prompt": "новый вопрос" })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::OK, "ошибка пересказа не должна ломать ответ: {}", sent.body);
    assert_eq!(sent.body["content"], "ответ модели");
    assert_eq!(sent.body["context"]["summary_built"], false);

    let loaded = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert!(loaded.body["summary"].is_null(), "сохранённый пересказ не должен был появиться");
}

// 12.5 — компактизация спасает от context_limit_exceeded, а компактизованный избыток всё равно отклоняется

#[tokio::test]
async fn compaction_avoids_context_limit_exceeded_when_full_history_would_not_fit() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(BodyContains(SUMMARY_CALL_MARKER))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("краткий итог")))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(BodyLacks(SUMMARY_CALL_MARKER))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ модели")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[("AGENTD_MAX_CONTEXT_TOKENS", "200"), ("AGENTD_SUMMARY_STEP_MESSAGES", "1")]).await;
    // Хвост (keep=2) — только последний обмен, чтобы компактизованная
    // история (пересказ + один короткий обмен + новое сообщение) уложилась
    // в лимит, даже когда полная история из четырёх длинных ответов — нет.
    let created = create_chat(state.clone(), None, summary_chat_settings(2, 1)).await;
    let id = created.body["id"].as_str().unwrap().to_string();
    let long_answer = "ответ ".repeat(200); // достаточно длинно, чтобы полная история превысила лимит в 200 токенов
    seed_exchanges(
        &state,
        &id,
        &[
            ("вопрос 1", &long_answer),
            ("вопрос 2", &long_answer),
            ("вопрос 3", &long_answer),
            ("вопрос 4", "короткий ответ"),
        ],
    )
    .await;

    let sent = send(
        state,
        post_chat(serde_json::json!({ "chat_id": id, "prompt": "новый вопрос" })),
    )
    .await;
    assert_eq!(
        sent.status,
        StatusCode::OK,
        "компактизация должна была уложить историю в лимит: {}",
        sent.body
    );
}

#[tokio::test]
async fn compacted_history_still_exceeding_limit_is_rejected_without_calling_provider_for_reply() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(BodyContains(SUMMARY_CALL_MARKER))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("краткий итог")))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(BodyLacks(SUMMARY_CALL_MARKER))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ модели")))
        .mount(&server)
        .await;
    // Лимит настолько мал, что даже пересказ с хвостом его превышают.
    let state = state_with_provider(&server, &[("AGENTD_MAX_CONTEXT_TOKENS", "1"), ("AGENTD_SUMMARY_STEP_MESSAGES", "1")]).await;
    let created = create_chat(state.clone(), None, summary_chat_settings(4, 3)).await;
    let id = created.body["id"].as_str().unwrap().to_string();
    seed_exchanges(
        &state,
        &id,
        &[
            ("вопрос 1", "ответ 1"),
            ("вопрос 2", "ответ 2"),
            ("вопрос 3", "ответ 3"),
            ("вопрос 4", "ответ 4"),
        ],
    )
    .await;

    let sent = send(
        state,
        post_chat(serde_json::json!({ "chat_id": id, "prompt": "новый вопрос" })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "context_limit_exceeded");

    let requests = server.received_requests().await.expect("запросы");
    assert!(
        requests.iter().all(|r| String::from_utf8_lossy(&r.body).contains(SUMMARY_CALL_MARKER)),
        "по пользовательскому сообщению провайдер не должен вызываться"
    );
}

// 12.6 — разовый вызов без chat_id не компактизуется

#[tokio::test]
async fn chat_without_chat_id_is_never_compacted() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ модели")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[]).await;

    let sent = send(
        state,
        post_chat(serde_json::json!({
            "prompt": "привет",
            "settings": { "summary_enabled": true }
        })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    assert!(sent.body.get("context").map(|c| c.is_null()).unwrap_or(true));

    let requests = server.received_requests().await.expect("запросы");
    assert_eq!(requests.len(), 1, "без chat_id обращение за пересказом не делается");
}
