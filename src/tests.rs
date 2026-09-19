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
    // `invariant-guard` — регистрируется по умолчанию (openspec/changes/
    // add-invariant-guardrails) и всегда проходит без служебного вызова к
    // модели, пока `AGENTD_INVARIANTS_PATH` не задан этому состоянию теста.
    assert_eq!(
        sent.body["policy"]["output"],
        serde_json::json!([{"stage": "invariant-guard", "action": "pass"}])
    );
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

// --- Режим отладки ---

/// Прогоняет один `/v1/chat` через провайдера на `wiremock` с заданными
/// переменными окружения и возвращает текст перехваченного журнала.
fn captured_chat_log(extra: &[(&str, &str)]) -> String {
    let capture = crate::telemetry::capture::Capture::default();
    let subscriber = tracing_subscriber::fmt()
        .json()
        .with_max_level(tracing::Level::DEBUG)
        .with_writer(capture.clone())
        .finish();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");

    // Подписчик задаётся на текущий поток, поэтому исполнение однопоточное:
    // иначе записи ушли бы мимо перехвата.
    tracing::subscriber::with_default(subscriber, || {
        runtime.block_on(async {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/chat/completions"))
                .respond_with(
                    ResponseTemplate::new(200).set_body_json(provider_reply("ответ модели")),
                )
                .mount(&server)
                .await;
            let state = state_with_provider(&server, extra).await;
            let sent = send(
                state,
                post_chat(serde_json::json!({ "prompt": "секретный вопрос" })),
            )
            .await;
            assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
        })
    });

    capture.text()
}

#[test]
fn debug_logs_api_and_upstream_bodies() {
    let _guard = test_lock();
    let output = captured_chat_log(&[("AGENTD_DEBUG", "true")]);

    assert!(output.contains("тело запроса"), "нет тела запроса: {output}");
    assert!(output.contains("секретный вопрос"), "нет текста запроса: {output}");
    assert!(output.contains("тело ответа"), "нет тела ответа: {output}");
    assert!(output.contains("запрос провайдеру"), "нет запроса к провайдеру: {output}");
    assert!(output.contains("ответ провайдера"), "нет ответа провайдера: {output}");
    assert!(output.contains("ответ модели"), "нет текста ответа модели: {output}");
}

/// Прогоняет `/v1/chat` в чате с назначенным встроенным профилем `teacher`
/// и возвращает текст перехваченного журнала — для проверки, что тексты
/// полей профиля журналируются только при `AGENTD_LOG_CONTENT=true`
/// (specs/user-profiles, «Тексты профиля маскируются в журнале»).
fn captured_chat_log_with_profile(extra: &[(&str, &str)]) -> String {
    let capture = crate::telemetry::capture::Capture::default();
    let subscriber = tracing_subscriber::fmt()
        .json()
        .with_max_level(tracing::Level::DEBUG)
        .with_writer(capture.clone())
        .finish();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");

    tracing::subscriber::with_default(subscriber, || {
        runtime.block_on(async {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/chat/completions"))
                .respond_with(
                    ResponseTemplate::new(200).set_body_json(provider_reply("ответ модели")),
                )
                .mount(&server)
                .await;
            let state = state_with_provider(&server, extra).await;
            let created = create_chat(
                state.clone(),
                None,
                serde_json::json!({ "settings": { "profile_id": "teacher" } }),
            )
            .await;
            let id = created.body["id"].as_str().expect("идентификатор чата").to_string();
            let sent = send(state, post_chat(serde_json::json!({ "chat_id": id, "prompt": "вопрос" }))).await;
            assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
        })
    });

    capture.text()
}

#[test]
fn profile_fields_are_not_logged_by_default() {
    let _guard = test_lock();
    let output = captured_chat_log_with_profile(&[]);

    assert!(output.contains("\"profile_id\":\"teacher\""), "нет идентификатора профиля: {output}");
    assert!(output.contains("\"chars\":"), "нет размера раздела: {output}");
    assert!(
        !output.contains("терпеливый преподаватель") && !output.contains("Простой язык"),
        "текст полей профиля попал в журнал: {output}"
    );
}

#[test]
fn profile_fields_are_logged_when_log_content_enabled() {
    let _guard = test_lock();
    let output = captured_chat_log_with_profile(&[("AGENTD_LOG_CONTENT", "true")]);

    assert!(output.contains("\"profile_id\":\"teacher\""), "нет идентификатора профиля: {output}");
    assert!(output.contains("терпеливый преподаватель"), "текст персоны профиля не попал в журнал: {output}");
}

#[test]
fn without_debug_bodies_are_not_logged() {
    let _guard = test_lock();
    let output = captured_chat_log(&[]);

    assert!(output.contains("обмен с моделью"), "нет сводки обмена: {output}");
    assert!(!output.contains("тело запроса"), "тело запроса попало в журнал: {output}");
    assert!(!output.contains("запрос провайдеру"), "обмен с провайдером в журнале: {output}");
    assert!(!output.contains("секретный вопрос"), "текст запроса в журнале: {output}");
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

// --- 4. Переключатель стратегий контекста ---

async fn create_chat_with_strategy(state: AppState, strategy: &str) -> String {
    let created = create_chat(
        state,
        None,
        serde_json::json!({ "settings": { "context_strategy": strategy } }),
    )
    .await;
    assert_eq!(created.status, StatusCode::CREATED, "тело: {}", created.body);
    created.body["id"].as_str().unwrap().to_string()
}

// --- 4.2 Одна точка выбора: не-summary стратегия вызывает провайдера один раз ---

#[tokio::test]
async fn non_summary_strategy_calls_provider_exactly_once() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ модели")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[]).await;
    let id = create_chat_with_strategy(state.clone(), "sliding_window").await;

    let sent = send(
        state,
        post_chat(serde_json::json!({ "chat_id": id, "prompt": "вопрос" })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    assert_eq!(sent.body["context"]["strategy"], "sliding_window");

    let requests = server.received_requests().await.expect("запросы");
    assert_eq!(requests.len(), 1, "стратегия без пересказа не делает вспомогательных вызовов");
}

// --- 4.3 Валидация стратегии на путях сохранения и вызова ---

#[tokio::test]
async fn unknown_context_strategy_is_rejected_on_create() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(
        state,
        None,
        serde_json::json!({ "settings": { "context_strategy": "magic" } }),
    )
    .await;
    assert_eq!(created.status, StatusCode::BAD_REQUEST, "тело: {}", created.body);
    assert_eq!(created.body["error"]["code"], "context_strategy_invalid");
}

#[tokio::test]
async fn unknown_context_strategy_is_rejected_on_patch() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let patched = send(
        state,
        request(
            "PATCH",
            &format!("/v1/chats/{id}"),
            Some(serde_json::json!({ "settings": { "context_strategy": "magic" } })),
            None,
        ),
    )
    .await;
    assert_eq!(patched.status, StatusCode::BAD_REQUEST, "тело: {}", patched.body);
    assert_eq!(patched.body["error"]["code"], "context_strategy_invalid");
}

#[tokio::test]
async fn disallowed_context_strategy_is_rejected_on_call() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    let state = state_with_provider(&server, &[("AGENTD_ALLOWED_CONTEXT_STRATEGIES", "summary")]).await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let sent = send(
        state,
        post_chat(serde_json::json!({
            "chat_id": id,
            "prompt": "вопрос",
            "settings": { "context_strategy": "facts" }
        })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "context_strategy_not_allowed");
}

// --- 4.3 Разовое переопределение не сохраняется ---

#[tokio::test]
async fn call_override_does_not_persist_context_strategy() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ модели")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[]).await;
    let id = create_chat_with_strategy(state.clone(), "facts").await;

    let sent = send(
        state.clone(),
        post_chat(serde_json::json!({
            "chat_id": id,
            "prompt": "вопрос",
            "settings": { "context_strategy": "sliding_window" }
        })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    assert_eq!(sent.body["context"]["strategy"], "sliding_window");

    let loaded = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert_eq!(loaded.body["settings"]["context_strategy"], "facts");
}

// --- 4.4 Блок context: поля чужой стратегии не выводятся ---

#[tokio::test]
async fn context_block_omits_fields_of_other_strategies() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ модели")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[]).await;
    let id = create_chat_with_strategy(state.clone(), "sliding_window").await;

    let sent = send(
        state,
        post_chat(serde_json::json!({ "chat_id": id, "prompt": "вопрос" })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    let context = &sent.body["context"];
    assert_eq!(context["strategy"], "sliding_window");
    assert!(context.get("facts_applied").is_none());
    assert!(context.get("branch_id").is_none());
    assert!(context.get("summary_built").is_none());
}

// --- 4.5 Лимит контекста считается по собранной истории ---

#[tokio::test]
async fn sliding_window_rescues_chat_from_context_limit_exceeded() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ок")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[("AGENTD_MAX_CONTEXT_TOKENS", "3000")]).await;
    let created = create_chat(
        state.clone(),
        None,
        serde_json::json!({ "settings": { "context_strategy": "sliding_window", "context_window_messages": 1 } }),
    )
    .await;
    assert_eq!(created.status, StatusCode::CREATED, "тело: {}", created.body);
    let id = created.body["id"].as_str().unwrap().to_string();

    // Каждое сообщение само под лимитом, но пять таких сообщений вместе — уже
    // не укладываются в 3000 токенов: окно размером 1 не даёт полной истории
    // накопиться в запросе, поэтому все пять запросов проходят.
    let long_message = "слово ".repeat(800);
    for _ in 0..5 {
        let sent = send(
            state.clone(),
            post_chat(serde_json::json!({ "chat_id": id, "prompt": long_message })),
        )
        .await;
        assert_eq!(
            sent.status,
            StatusCode::OK,
            "окно должно спасти запрос от context_limit_exceeded: {}",
            sent.body
        );
    }

    let loaded = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert_eq!(loaded.body["message_count"], 10, "вся история чата сохранена, несмотря на окно в запросах");
}

// --- 5.2 Границы context_window_messages ---

#[tokio::test]
async fn zero_context_window_messages_is_rejected() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(
        state,
        None,
        serde_json::json!({ "settings": { "context_window_messages": 0 } }),
    )
    .await;
    assert_eq!(created.status, StatusCode::BAD_REQUEST, "тело: {}", created.body);
    assert_eq!(created.body["error"]["code"], "context_window_invalid");
}

#[tokio::test]
async fn context_window_messages_above_operator_ceiling_is_rejected() {
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_CONTEXT_WINDOW_MESSAGES", "10"),
    ])
    .await;
    let created = create_chat(
        state,
        None,
        serde_json::json!({ "settings": { "context_window_messages": 11 } }),
    )
    .await;
    assert_eq!(created.status, StatusCode::BAD_REQUEST, "тело: {}", created.body);
    assert_eq!(created.body["error"]["code"], "context_window_invalid");
}

// --- 5.3 Сквозной запрос: окно 6 из 20 сохранённых сообщений ---

#[tokio::test]
async fn window_of_six_out_of_twenty_messages_reaches_provider_and_full_history_stays_readable() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ модели")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[]).await;
    let created = create_chat(
        state.clone(),
        None,
        serde_json::json!({ "settings": { "context_strategy": "sliding_window", "context_window_messages": 6 } }),
    )
    .await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let pairs: Vec<(&str, &str)> = (1..=10).map(|_| ("вопрос", "ответ")).collect();
    seed_exchanges(&state, &id, &pairs).await; // 20 сообщений

    let sent = send(
        state.clone(),
        post_chat(serde_json::json!({ "chat_id": id, "prompt": "новый вопрос" })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    assert_eq!(sent.body["context"]["sent_messages"], 6);
    assert_eq!(sent.body["context"]["dropped_messages"], 14);

    let requests = server.received_requests().await.expect("запросы");
    let last = requests.last().expect("запрос к провайдеру");
    let body: serde_json::Value = serde_json::from_slice(&last.body).expect("тело запроса");
    let messages = body["messages"].as_array().expect("messages");
    assert_eq!(messages.len(), 8, "системное сообщение, 6 сообщений окна плюс новое");

    let loaded = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert_eq!(loaded.body["message_count"], 22, "чтение чата возвращает всю сохранённую историю");
}

// --- 6.5 Эндпоинты фактов ---

fn facts_uri(chat_id: &str) -> String {
    format!("/v1/chats/{chat_id}/facts")
}

fn fact_uri(chat_id: &str, key: &str) -> String {
    format!("/v1/chats/{chat_id}/facts/{key}")
}

#[tokio::test]
async fn fact_can_be_set_read_and_deleted() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let set = send(
        state.clone(),
        request("PUT", &fact_uri(&id, "budget"), Some(serde_json::json!({ "value": "200000" })), None),
    )
    .await;
    assert_eq!(set.status, StatusCode::OK, "тело: {}", set.body);
    assert_eq!(set.body["value"], "200000");

    let listed = send(state.clone(), request("GET", &facts_uri(&id), None, None)).await;
    assert_eq!(listed.status, StatusCode::OK);
    let facts = listed.body["facts"].as_array().unwrap();
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0]["value"], "200000");

    let overwritten = send(
        state.clone(),
        request("PUT", &fact_uri(&id, "budget"), Some(serde_json::json!({ "value": "300000" })), None),
    )
    .await;
    assert_eq!(overwritten.body["value"], "300000");
    let listed = send(state.clone(), request("GET", &facts_uri(&id), None, None)).await;
    assert_eq!(listed.body["facts"].as_array().unwrap().len(), 1, "перезапись не создаёт вторую запись");

    let deleted = send(state.clone(), request("DELETE", &fact_uri(&id, "budget"), None, None)).await;
    assert_eq!(deleted.status, StatusCode::NO_CONTENT);
    let listed = send(state, request("GET", &facts_uri(&id), None, None)).await;
    assert!(listed.body["facts"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn deleting_missing_fact_is_404() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let deleted = send(state, request("DELETE", &fact_uri(&id, "missing"), None, None)).await;
    assert_eq!(deleted.status, StatusCode::NOT_FOUND, "тело: {}", deleted.body);
    assert_eq!(deleted.body["error"]["code"], "fact_not_found");
}

#[tokio::test]
async fn foreign_chat_facts_are_not_exposed() {
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_CLIENT_TOKENS", "owner-a,owner-b"),
    ])
    .await;
    let created = create_chat(state.clone(), Some("owner-a"), serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();
    send(
        state.clone(),
        request("PUT", &fact_uri(&id, "secret"), Some(serde_json::json!({ "value": "x" })), Some("owner-a")),
    )
    .await;

    let foreign = send(state, request("GET", &facts_uri(&id), None, Some("owner-b"))).await;
    assert_eq!(foreign.status, StatusCode::NOT_FOUND, "тело: {}", foreign.body);
}

#[tokio::test]
async fn manual_fact_over_limit_is_rejected() {
    let _guard = test_lock();
    let state = AppState::with_env(&[(API_KEY_VAR, "secret-key-value"), ("AGENTD_MAX_FACTS", "1")]).await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();
    send(
        state.clone(),
        request("PUT", &fact_uri(&id, "first"), Some(serde_json::json!({ "value": "a" })), None),
    )
    .await;

    let over_limit = send(
        state,
        request("PUT", &fact_uri(&id, "second"), Some(serde_json::json!({ "value": "b" })), None),
    )
    .await;
    assert_eq!(over_limit.status, StatusCode::BAD_REQUEST, "тело: {}", over_limit.body);
    assert_eq!(over_limit.body["error"]["code"], "facts_limit_exceeded");
}

// --- 6.3 Обновление фактов после обмена ---

#[tokio::test]
async fn facts_update_failure_does_not_break_main_response() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    // Первый вызов — основной ответ, второй (обновление фактов) — отказ.
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("основной ответ")))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[]).await;
    let created = create_chat(
        state.clone(),
        None,
        serde_json::json!({ "settings": { "context_strategy": "facts" } }),
    )
    .await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let sent = send(
        state.clone(),
        post_chat(serde_json::json!({ "chat_id": id, "prompt": "бюджет — 200000" })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::OK, "основной ответ не должен зависеть от обновления фактов: {}", sent.body);
    assert_eq!(sent.body["context"]["facts_updated"], false);

    let facts = send(state, request("GET", &facts_uri(&id), None, None)).await;
    assert!(facts.body["facts"].as_array().unwrap().is_empty(), "факты остаются прежними при отказе обновления");
}

// --- 4.4/4.5 Автоматическое название чата (specs/chat-title) ---

#[tokio::test]
async fn title_generation_saves_provider_title_after_first_exchange() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(BodyContains(crate::title::TITLE_UPDATE_MARKER))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("\"Бюджет проекта\".")))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(BodyLacks(crate::title::TITLE_UPDATE_MARKER))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ модели")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[]).await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();
    let owner = crate::store::ANONYMOUS_OWNER.to_string();
    let chat = crate::store::load_chat(&state.db, &owner, &id, &agentcore::config::ChatSettings::default())
        .await
        .expect("чат");

    crate::title::generate_and_save(
        state.clone(),
        id.clone(),
        owner,
        chat.settings.clone(),
        state.config.system_prompt.clone(),
        "бюджет на проект".to_string(),
    )
    .await;

    let loaded = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert_eq!(loaded.body["title"], "Бюджет проекта");
}

#[tokio::test]
async fn title_generation_failure_leaves_default_title() {
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
    let owner = crate::store::ANONYMOUS_OWNER.to_string();
    let chat = crate::store::load_chat(&state.db, &owner, &id, &agentcore::config::ChatSettings::default())
        .await
        .expect("чат");

    crate::title::generate_and_save(
        state.clone(),
        id.clone(),
        owner,
        chat.settings.clone(),
        state.config.system_prompt.clone(),
        "бюджет на проект".to_string(),
    )
    .await;

    let loaded = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert_eq!(loaded.body["title"], "Новый чат", "отказ генерации не должен менять название");
}

#[tokio::test]
async fn manual_rename_during_generation_is_kept() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("Название от модели")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[]).await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();
    let owner = crate::store::ANONYMOUS_OWNER.to_string();
    let chat = crate::store::load_chat(&state.db, &owner, &id, &agentcore::config::ChatSettings::default())
        .await
        .expect("чат");

    // Клиент переименовывает чат ДО того, как фоновая генерация успела
    // записать результат (specs/chat-title, «Ручное переименование во время
    // генерации»).
    send(
        state.clone(),
        request(
            "PATCH",
            &format!("/v1/chats/{id}"),
            Some(serde_json::json!({ "title": "Название клиента" })),
            None,
        ),
    )
    .await;

    crate::title::generate_and_save(
        state.clone(),
        id.clone(),
        owner,
        chat.settings.clone(),
        state.config.system_prompt.clone(),
        "бюджет на проект".to_string(),
    )
    .await;

    let loaded = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert_eq!(loaded.body["title"], "Название клиента", "название клиента не должно быть перезаписано");
}

#[tokio::test]
async fn auto_title_disabled_sends_no_title_request() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ модели")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[("AGENTD_AUTO_TITLE", "false")]).await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let sent = send(state.clone(), post_chat(serde_json::json!({ "chat_id": id, "prompt": "первый вопрос" }))).await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let requests = server.received_requests().await.expect("запросы");
    assert_eq!(requests.len(), 1, "выключенная генерация не делает второго вызова к провайдеру");

    let loaded = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert_eq!(loaded.body["title"], "Новый чат");
}

#[tokio::test]
async fn chat_created_with_own_title_skips_generation() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ модели")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[]).await;
    let created = create_chat(state.clone(), None, serde_json::json!({ "title": "Своё название" })).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let sent = send(state.clone(), post_chat(serde_json::json!({ "chat_id": id, "prompt": "первый вопрос" }))).await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let requests = server.received_requests().await.expect("запросы");
    assert_eq!(requests.len(), 1, "чат со своим названием не запускает генерацию");

    let loaded = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert_eq!(loaded.body["title"], "Своё название");
}

// --- 3.5/5.3 слоистая память: устойчивость маршрутизатора и что "memory_layers" больше не стратегия ---

#[tokio::test]
async fn memory_router_failure_does_not_break_main_response_or_change_memory() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    // Первый вызов — основной ответ, второй (маршрутизатор памяти) — отказ.
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("основной ответ")))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[]).await;
    let created = create_chat(
        state.clone(),
        None,
        serde_json::json!({ "settings": { "context_strategy": "sliding_window", "memory_layers_enabled": true } }),
    )
    .await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let sent = send(
        state.clone(),
        post_chat(serde_json::json!({ "chat_id": id, "prompt": "бюджет — 200000" })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::OK, "основной ответ не должен зависеть от маршрутизатора памяти: {}", sent.body);
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let owner = crate::store::ANONYMOUS_OWNER.to_string();
    let chat = crate::store::load_chat(&state.db, &owner, &id, &agentcore::config::ChatSettings::default())
        .await
        .expect("чат");
    let working = crate::store::load_working_memory(&state.db, &id, &chat.active_task_id)
        .await
        .expect("рабочая память");
    assert!(working.is_empty(), "отказ маршрутизатора не должен менять состояние памяти");
}

/// `memory_layers` — больше не значение `context_strategy`: запрос с ним
/// отклоняется как любое другое незнакомое имя стратегии, а не как
/// «стратегия не в списке разрешённых» (specs/memory-layers, «Стратегия
/// контекста memory_layers», REMOVED).
#[tokio::test]
async fn context_strategy_memory_layers_is_rejected_as_unknown_value() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    let state = state_with_provider(&server, &[("AGENTD_ALLOWED_CONTEXT_STRATEGIES", "summary")]).await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let sent = send(
        state,
        post_chat(serde_json::json!({
            "chat_id": id,
            "prompt": "вопрос",
            "settings": { "context_strategy": "memory_layers" }
        })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "context_strategy_invalid");
}

/// Слоистая память доступна поверх любой стратегии из
/// `AGENTD_ALLOWED_CONTEXT_STRATEGIES`, список которого больше не определяет
/// и не ограничивает её доступность отдельно (specs/memory-layers,
/// «Слоистая память включена независимо от списка разрешённых стратегий»).
#[tokio::test]
async fn memory_layers_enabled_works_over_a_strategy_allowed_by_the_operator_list() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[("AGENTD_ALLOWED_CONTEXT_STRATEGIES", "sliding_window")]).await;
    let created = create_chat(
        state.clone(),
        None,
        serde_json::json!({ "settings": { "context_strategy": "sliding_window", "memory_layers_enabled": true } }),
    )
    .await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let sent = send(state, post_chat(serde_json::json!({ "chat_id": id, "prompt": "вопрос" }))).await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    assert_eq!(sent.body["context"]["strategy"], "sliding_window");
    assert_eq!(sent.body["context"]["memory_long_term_entries"], 0);
}

/// Фоновый маршрутизатор памяти запускается по независимому переключателю
/// слоистой памяти, а не по действующей стратегии контекста
/// (decouple-memory-layers, решение 3): при одной и той же стратегии
/// `sliding_window` он вызывается при включённой памяти и не вызывается при
/// выключенной.
#[tokio::test]
async fn memory_router_runs_only_when_memory_layers_enabled_regardless_of_strategy() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("[]")))
        .mount(&server)
        .await;
    // Генерация названия чата — фоновый вызов той же модели после первого
    // обмена: выключаем, чтобы считать только запросы основного ответа и
    // маршрутизатора памяти.
    let state = state_with_provider(&server, &[("AGENTD_AUTO_TITLE", "false")]).await;

    let enabled = create_chat(
        state.clone(),
        None,
        serde_json::json!({ "settings": { "context_strategy": "sliding_window", "memory_layers_enabled": true } }),
    )
    .await;
    let enabled_id = enabled.body["id"].as_str().unwrap().to_string();
    let sent = send(state.clone(), post_chat(serde_json::json!({ "chat_id": enabled_id, "prompt": "вопрос" }))).await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let requests_after_enabled = server.received_requests().await.expect("запросы").len();
    assert_eq!(requests_after_enabled, 2, "маршрутизатор должен вызваться вторым запросом при включённой памяти");

    let disabled = create_chat(
        state.clone(),
        None,
        serde_json::json!({ "settings": { "context_strategy": "sliding_window", "memory_layers_enabled": false } }),
    )
    .await;
    let disabled_id = disabled.body["id"].as_str().unwrap().to_string();
    let sent = send(state.clone(), post_chat(serde_json::json!({ "chat_id": disabled_id, "prompt": "вопрос" }))).await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let requests_after_disabled = server.received_requests().await.expect("запросы").len();
    assert_eq!(
        requests_after_disabled,
        requests_after_enabled + 1,
        "маршрутизатор не должен вызываться при выключенной памяти"
    );
}

/// Слоистая память сочетается с каждой из четырёх оставшихся стратегий
/// контекста: ответ несёт поля стратегии и поля памяти одновременно
/// (specs/memory-layers, «Разбивка по слоям в блоке context ответа»).
#[tokio::test]
async fn memory_layers_combines_with_each_context_strategy() {
    for strategy in ["summary", "sliding_window", "facts", "branching"] {
        let _guard = test_lock();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ")))
            .mount(&server)
            .await;
        let state = state_with_provider(&server, &[]).await;
        let created = create_chat(
            state.clone(),
            None,
            serde_json::json!({ "settings": { "context_strategy": strategy, "memory_layers_enabled": true, "memory_router_enabled": false } }),
        )
        .await;
        let id = created.body["id"].as_str().unwrap().to_string();
        crate::store::set_long_term_memory(&state.db, crate::store::ANONYMOUS_OWNER, "decision", Some("auth"), "Clerk", "manual", None, 1)
            .await
            .expect("долговременная запись");

        let sent = send(state, post_chat(serde_json::json!({ "chat_id": id, "prompt": "вопрос" }))).await;
        assert_eq!(sent.status, StatusCode::OK, "стратегия {strategy}: тело {}", sent.body);
        assert_eq!(sent.body["context"]["strategy"], strategy, "стратегия {strategy}");
        assert_eq!(
            sent.body["context"]["memory_long_term_entries"], 1,
            "стратегия {strategy}: поля памяти должны присутствовать вместе с полями стратегии"
        );
    }
}

/// Факт (стратегия `facts`) и запись долговременной памяти (слоистая
/// память), сохранённые ранее, одновременно попадают в системное сообщение
/// следующего запроса, каждый в своём разделе (specs/memory-layers).
#[tokio::test]
async fn facts_and_memory_layers_sections_both_land_in_next_system_message() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[]).await;
    let created = create_chat(
        state.clone(),
        None,
        serde_json::json!({ "settings": { "context_strategy": "facts", "memory_layers_enabled": true, "memory_router_enabled": false } }),
    )
    .await;
    let id = created.body["id"].as_str().unwrap().to_string();
    crate::store::set_fact(&state.db, &id, "budget", "200000", 1).await.expect("факт сохранён");
    crate::store::set_long_term_memory(&state.db, crate::store::ANONYMOUS_OWNER, "decision", Some("auth"), "Clerk", "manual", None, 1)
        .await
        .expect("долговременная запись");

    let sent = send(state, post_chat(serde_json::json!({ "chat_id": id, "prompt": "вопрос" }))).await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);

    // Первый запрос — основной вызов модели (с системным сообщением);
    // второй — фоновое обновление фактов уже другим, однострочным промптом.
    let requests = server.received_requests().await.expect("запросы");
    let first = requests.first().expect("хотя бы один запрос");
    let body: serde_json::Value = first.body_json().expect("тело запроса — JSON");
    let system = body["messages"][0]["content"].as_str().expect("системное сообщение");
    assert!(system.contains("Факты чата:"), "раздел фактов отсутствует: {system}");
    assert!(system.contains("Долговременная память:"), "раздел памяти отсутствует: {system}");
}

// --- 6.2/6.3 Эндпоинты памяти (specs/memory-layers) ---

fn working_memory_uri(chat_id: &str) -> String {
    format!("/v1/chats/{chat_id}/memory/working")
}

fn delete_working_memory_uri(chat_id: &str, key: &str) -> String {
    format!("/v1/chats/{chat_id}/memory/working?key={key}")
}

fn finish_task_uri(chat_id: &str) -> String {
    format!("/v1/chats/{chat_id}/memory/working/finish-task")
}

fn delete_long_term_memory_uri(id: &str) -> String {
    format!("/v1/memory/long-term?id={id}")
}

#[tokio::test]
async fn working_memory_set_read_delete_over_http() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let set = send(
        state.clone(),
        request("POST", &working_memory_uri(&id), Some(serde_json::json!({ "key": "budget", "value": "200000" })), None),
    )
    .await;
    assert_eq!(set.status, StatusCode::OK, "тело: {}", set.body);
    assert_eq!(set.body["value"], "200000");
    assert_eq!(set.body["source"], "manual");

    let read = send(state.clone(), request("GET", &working_memory_uri(&id), None, None)).await;
    assert_eq!(read.body["entries"].as_array().unwrap().len(), 1);
    assert_eq!(read.body["entries"][0]["key"], "budget");

    let deleted = send(state.clone(), request("DELETE", &delete_working_memory_uri(&id, "budget"), None, None)).await;
    assert_eq!(deleted.status, StatusCode::NO_CONTENT);

    let read_after = send(state, request("GET", &working_memory_uri(&id), None, None)).await;
    assert!(read_after.body["entries"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn foreign_owner_cannot_read_or_write_working_memory() {
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_CLIENT_TOKENS", "owner-a,owner-b"),
    ])
    .await;
    let created = create_chat(state.clone(), Some("owner-a"), serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let foreign_read = send(state.clone(), request("GET", &working_memory_uri(&id), None, Some("owner-b"))).await;
    assert_eq!(foreign_read.status, StatusCode::NOT_FOUND, "тело: {}", foreign_read.body);

    let foreign_set = send(
        state,
        request(
            "POST",
            &working_memory_uri(&id),
            Some(serde_json::json!({ "key": "k", "value": "v" })),
            Some("owner-b"),
        ),
    )
    .await;
    assert_eq!(foreign_set.status, StatusCode::NOT_FOUND, "тело: {}", foreign_set.body);
}

#[tokio::test]
async fn manual_working_memory_write_overrides_earlier_automatic_operation_same_tact() {
    // specs/memory-layers, «Ручная запись переопределяет решение
    // маршрутизатора в том же такте»: ранняя автоматическая запись (source
    // router, меньший updated_at), затем ручная HTTP-запись (source manual,
    // больший updated_at, так как выполнена позже по времени) — итоговое
    // значение от ручной.
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();
    let owner = crate::store::ANONYMOUS_OWNER.to_string();
    let chat = crate::store::load_chat(&state.db, &owner, &id, &agentcore::config::ChatSettings::default())
        .await
        .expect("чат");

    crate::store::set_working_memory(&state.db, &id, &chat.active_task_id, "k", "от роутера", "router", 1)
        .await
        .expect("ранняя автоматическая запись");

    let set = send(
        state.clone(),
        request("POST", &working_memory_uri(&id), Some(serde_json::json!({ "key": "k", "value": "от клиента" })), None),
    )
    .await;
    assert_eq!(set.status, StatusCode::OK, "тело: {}", set.body);

    let read = send(state, request("GET", &working_memory_uri(&id), None, None)).await;
    let entries = read.body["entries"].as_array().unwrap();
    let entry = entries.iter().find(|e| e["key"] == "k").expect("запись k");
    assert_eq!(entry["value"], "от клиента");
    assert_eq!(entry["source"], "manual");
}

#[tokio::test]
async fn finish_task_over_http_transfers_carried_entries_and_rotates_task() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    send(
        state.clone(),
        request("POST", &working_memory_uri(&id), Some(serde_json::json!({ "key": "carried", "value": "значение" })), None),
    )
    .await;

    let finished = send(
        state.clone(),
        request("POST", &finish_task_uri(&id), Some(serde_json::json!({ "carry_forward_keys": ["carried"] })), None),
    )
    .await;
    assert_eq!(finished.status, StatusCode::OK, "тело: {}", finished.body);
    let transferred = finished.body["entries"].as_array().unwrap();
    assert_eq!(transferred.len(), 1);
    assert_eq!(transferred[0]["value"], "значение");

    let working_after = send(state, request("GET", &working_memory_uri(&id), None, None)).await;
    assert!(working_after.body["entries"].as_array().unwrap().is_empty(), "рабочая память прежней задачи очищена");
}

#[tokio::test]
async fn long_term_memory_set_read_delete_over_http() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;

    let set = send(
        state.clone(),
        request(
            "POST",
            "/v1/memory/long-term",
            Some(serde_json::json!({ "entry_type": "decision", "key": "auth_provider", "value": "Clerk" })),
            None,
        ),
    )
    .await;
    assert_eq!(set.status, StatusCode::OK, "тело: {}", set.body);
    let entry_id = set.body["id"].as_str().unwrap().to_string();

    let read = send(state.clone(), request("GET", "/v1/memory/long-term", None, None)).await;
    assert_eq!(read.body["entries"].as_array().unwrap().len(), 1);

    let deleted = send(state.clone(), request("DELETE", &delete_long_term_memory_uri(&entry_id), None, None)).await;
    assert_eq!(deleted.status, StatusCode::NO_CONTENT);

    let read_after = send(state, request("GET", "/v1/memory/long-term", None, None)).await;
    assert!(read_after.body["entries"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn foreign_owner_cannot_read_or_delete_long_term_memory() {
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_CLIENT_TOKENS", "owner-a,owner-b"),
    ])
    .await;
    let set = send(
        state.clone(),
        request(
            "POST",
            "/v1/memory/long-term",
            Some(serde_json::json!({ "entry_type": "profile", "key": "name", "value": "секрет" })),
            Some("owner-a"),
        ),
    )
    .await;
    let entry_id = set.body["id"].as_str().unwrap().to_string();

    let foreign_read = send(state.clone(), request("GET", "/v1/memory/long-term", None, Some("owner-b"))).await;
    assert!(foreign_read.body["entries"].as_array().unwrap().is_empty(), "чужая долговременная память не видна");

    let foreign_delete = send(
        state,
        request("DELETE", &delete_long_term_memory_uri(&entry_id), None, Some("owner-b")),
    )
    .await;
    assert_eq!(foreign_delete.status, StatusCode::NOT_FOUND, "тело: {}", foreign_delete.body);
}

// --- Профили (specs/user-profiles) ---

fn profile_uri(id: &str) -> String {
    format!("/v1/profiles/{id}")
}

#[tokio::test]
async fn owner_without_own_profiles_sees_exactly_three_built_in() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let list = send(state, request("GET", "/v1/profiles", None, None)).await;
    assert_eq!(list.status, StatusCode::OK, "тело: {}", list.body);
    let profiles = list.body["profiles"].as_array().unwrap();
    assert_eq!(profiles.len(), 3);
    assert!(profiles.iter().all(|p| p["built_in"] == true));
    let ids: Vec<&str> = profiles.iter().map(|p| p["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"teacher"));
    assert!(ids.contains(&"psychologist"));
    assert!(ids.contains(&"reviewer"));
}

#[tokio::test]
async fn created_profile_is_visible_in_list_with_non_built_in_flag() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = send(
        state.clone(),
        request(
            "POST",
            "/v1/profiles",
            Some(serde_json::json!({ "name": "Свой", "style": "кратко" })),
            None,
        ),
    )
    .await;
    assert_eq!(created.status, StatusCode::CREATED, "тело: {}", created.body);
    assert_eq!(created.body["built_in"], false);

    let list = send(state, request("GET", "/v1/profiles", None, None)).await;
    let profiles = list.body["profiles"].as_array().unwrap();
    assert_eq!(profiles.len(), 4);
    assert!(profiles.iter().any(|p| p["name"] == "Свой" && p["built_in"] == false));
}

#[tokio::test]
async fn profile_without_any_preference_is_rejected() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = send(
        state,
        request("POST", "/v1/profiles", Some(serde_json::json!({ "name": "Пустой" })), None),
    )
    .await;
    assert_eq!(created.status, StatusCode::BAD_REQUEST, "тело: {}", created.body);
    assert_eq!(created.body["error"]["code"], "profile_rejected");
}

#[tokio::test]
async fn patching_profile_changes_only_given_field() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = send(
        state.clone(),
        request(
            "POST",
            "/v1/profiles",
            Some(serde_json::json!({ "name": "Свой", "persona": "персона", "format": "списком" })),
            None,
        ),
    )
    .await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let patched = send(
        state,
        request("PATCH", &profile_uri(&id), Some(serde_json::json!({ "format": "прозой" })), None),
    )
    .await;
    assert_eq!(patched.status, StatusCode::OK, "тело: {}", patched.body);
    assert_eq!(patched.body["format"], "прозой");
    assert_eq!(patched.body["persona"], "персона", "остальные поля не затёрты");
    assert_eq!(patched.body["name"], "Свой");
}

#[tokio::test]
async fn deleted_profile_returns_404_and_leaves_list() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = send(
        state.clone(),
        request("POST", "/v1/profiles", Some(serde_json::json!({ "name": "Свой", "style": "кратко" })), None),
    )
    .await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let deleted = send(state.clone(), request("DELETE", &profile_uri(&id), None, None)).await;
    assert_eq!(deleted.status, StatusCode::NO_CONTENT);

    let read = send(state.clone(), request("GET", &profile_uri(&id), None, None)).await;
    assert_eq!(read.status, StatusCode::NOT_FOUND);

    let list = send(state, request("GET", "/v1/profiles", None, None)).await;
    assert_eq!(list.body["profiles"].as_array().unwrap().len(), 3);
}

#[tokio::test]
async fn built_in_profile_cannot_be_changed_or_deleted() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;

    let patched = send(
        state.clone(),
        request("PATCH", &profile_uri("teacher"), Some(serde_json::json!({ "style": "иначе" })), None),
    )
    .await;
    assert_eq!(patched.status, StatusCode::BAD_REQUEST, "тело: {}", patched.body);
    assert_eq!(patched.body["error"]["code"], "profile_rejected");

    let deleted = send(state.clone(), request("DELETE", &profile_uri("teacher"), None, None)).await;
    assert_eq!(deleted.status, StatusCode::BAD_REQUEST, "тело: {}", deleted.body);
    assert_eq!(deleted.body["error"]["code"], "profile_rejected");

    let read = send(state, request("GET", &profile_uri("teacher"), None, None)).await;
    assert_eq!(read.status, StatusCode::OK);
    assert_eq!(read.body["style"], "Простой язык, короткие предложения, примеры перед абстракцией.");
}

#[tokio::test]
async fn foreign_owner_cannot_read_or_modify_profile() {
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_CLIENT_TOKENS", "owner-a,owner-b"),
    ])
    .await;
    let created = send(
        state.clone(),
        request(
            "POST",
            "/v1/profiles",
            Some(serde_json::json!({ "name": "Секретный", "style": "кратко" })),
            Some("owner-a"),
        ),
    )
    .await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let foreign_list = send(state.clone(), request("GET", "/v1/profiles", None, Some("owner-b"))).await;
    let profiles = foreign_list.body["profiles"].as_array().unwrap();
    assert_eq!(profiles.len(), 3, "чужой профиль не виден в списке владельца Y");

    let foreign_read = send(state.clone(), request("GET", &profile_uri(&id), None, Some("owner-b"))).await;
    assert_eq!(foreign_read.status, StatusCode::NOT_FOUND);

    let foreign_delete = send(state, request("DELETE", &profile_uri(&id), None, Some("owner-b"))).await;
    assert_eq!(foreign_delete.status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn creating_profile_beyond_operator_limit_is_rejected() {
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_MAX_PROFILES", "1"),
    ])
    .await;
    let first = send(
        state.clone(),
        request("POST", "/v1/profiles", Some(serde_json::json!({ "name": "Первый", "style": "кратко" })), None),
    )
    .await;
    assert_eq!(first.status, StatusCode::CREATED, "тело: {}", first.body);

    let second = send(
        state.clone(),
        request("POST", "/v1/profiles", Some(serde_json::json!({ "name": "Второй", "style": "длинно" })), None),
    )
    .await;
    assert_eq!(second.status, StatusCode::BAD_REQUEST, "тело: {}", second.body);
    assert_eq!(second.body["error"]["code"], "profile_rejected");

    let list = send(state, request("GET", "/v1/profiles", None, None)).await;
    let own: Vec<_> = list.body["profiles"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|p| p["built_in"] == false)
        .collect();
    assert_eq!(own.len(), 1, "число профилей владельца не изменилось");
}

/// Неизвестный `profile_id` в `POST /v1/chat` отклоняется до обращения к
/// провайдеру: сервер-заглушка без настроенного мока подтверждает, что
/// запрос к провайдеру не уходит — иначе ответ был бы ошибкой провайдера, а
/// не `profile_invalid` (specs/user-profiles, «Несуществующий профиль в
/// запросе чата»).
#[tokio::test]
async fn unknown_profile_id_in_chat_call_is_rejected_before_calling_provider() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    let state = state_with_provider(&server, &[]).await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let sent = send(
        state,
        post_chat(serde_json::json!({
            "chat_id": id,
            "prompt": "вопрос",
            "settings": { "profile_id": "no-such-profile" }
        })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "profile_invalid");
}

#[tokio::test]
async fn foreign_profile_id_in_chat_call_is_rejected() {
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_CLIENT_TOKENS", "owner-a,owner-b"),
    ])
    .await;
    let created_profile = send(
        state.clone(),
        request(
            "POST",
            "/v1/profiles",
            Some(serde_json::json!({ "name": "Секретный", "style": "кратко" })),
            Some("owner-a"),
        ),
    )
    .await;
    let profile_id = created_profile.body["id"].as_str().unwrap().to_string();
    let chat = create_chat(state.clone(), Some("owner-b"), serde_json::json!({})).await;
    let chat_id = chat.body["id"].as_str().unwrap().to_string();

    let sent = send(
        state,
        request(
            "POST",
            "/v1/chat",
            Some(serde_json::json!({
                "chat_id": chat_id,
                "prompt": "вопрос",
                "settings": { "profile_id": profile_id }
            })),
            Some("owner-b"),
        ),
    )
    .await;
    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "profile_invalid");
}

#[tokio::test]
async fn unknown_profile_id_in_chat_settings_is_rejected_with_400() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(
        state.clone(),
        None,
        serde_json::json!({ "settings": { "profile_id": "no-such-profile" } }),
    )
    .await;
    assert_eq!(created.status, StatusCode::BAD_REQUEST, "тело: {}", created.body);
    assert_eq!(created.body["error"]["code"], "profile_invalid");
}

/// Один и тот же вопрос под тремя встроенными профилями даёт три разных
/// системных сообщения в телах запросов, пойманных `wiremock`
/// (specs/user-profiles, «Один вопрос под тремя профилями»).
#[tokio::test]
async fn same_question_under_three_profiles_yields_three_different_system_messages() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ модели")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[("AGENTD_AUTO_TITLE", "false")]).await;

    let mut bodies = Vec::new();
    for profile_id in ["teacher", "psychologist", "reviewer"] {
        let created = create_chat(
            state.clone(),
            None,
            serde_json::json!({ "settings": { "profile_id": profile_id } }),
        )
        .await;
        let id = created.body["id"].as_str().unwrap().to_string();
        let sent = send(
            state.clone(),
            post_chat(serde_json::json!({ "chat_id": id, "prompt": "один и тот же вопрос" })),
        )
        .await;
        assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
        bodies.push(profile_id);
    }

    let requests = server.received_requests().await.expect("запросы");
    let system_messages: Vec<String> = requests
        .iter()
        .map(|r| {
            let body: serde_json::Value = serde_json::from_slice(&r.body).expect("тело запроса провайдеру");
            body["messages"][0]["content"].as_str().unwrap().to_string()
        })
        .collect();
    assert_eq!(system_messages.len(), 3);
    assert_ne!(system_messages[0], system_messages[1]);
    assert_ne!(system_messages[1], system_messages[2]);
    assert_ne!(system_messages[0], system_messages[2]);
    assert!(system_messages[0].contains("терпеливый преподаватель"));
    assert!(system_messages[1].contains("внимательный собеседник"));
    assert!(system_messages[2].contains("строгий технический ревьюер"));
}

// --- 7.3 Эндпоинты веток ---

fn branches_uri(chat_id: &str) -> String {
    format!("/v1/chats/{chat_id}/branches")
}

#[tokio::test]
async fn branch_created_listed_and_activated_over_http() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();
    seed_exchanges(&state, &id, &[("вопрос", "ответ")]).await;

    let branches = send(state.clone(), request("GET", &branches_uri(&id), None, None)).await;
    assert_eq!(branches.status, StatusCode::OK, "тело: {}", branches.body);
    assert_eq!(branches.body["branches"].as_array().unwrap().len(), 1);

    let created_branch = send(
        state.clone(),
        request(
            "POST",
            &branches_uri(&id),
            Some(serde_json::json!({ "from_seq": 1, "name": "альтернатива" })),
            None,
        ),
    )
    .await;
    assert_eq!(created_branch.status, StatusCode::CREATED, "тело: {}", created_branch.body);
    let branch_id = created_branch.body["id"].as_str().unwrap().to_string();

    let activated = send(
        state.clone(),
        request("POST", &format!("{}/{branch_id}/activate", branches_uri(&id)), None, None),
    )
    .await;
    assert_eq!(activated.status, StatusCode::NO_CONTENT, "тело: {}", activated.body);

    let branches = send(state, request("GET", &branches_uri(&id), None, None)).await;
    let list = branches.body["branches"].as_array().unwrap();
    let active: Vec<_> = list.iter().filter(|b| b["active"] == true).collect();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0]["id"], branch_id);
}

#[tokio::test]
async fn branch_from_nonexistent_message_is_404_over_http() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let sent = send(
        state,
        request(
            "POST",
            &branches_uri(&id),
            Some(serde_json::json!({ "from_seq": 99, "name": "ветка" })),
            None,
        ),
    )
    .await;
    assert_eq!(sent.status, StatusCode::NOT_FOUND, "тело: {}", sent.body);
}

#[tokio::test]
async fn branch_operations_on_foreign_chat_are_404() {
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_CLIENT_TOKENS", "owner-a,owner-b"),
    ])
    .await;
    let created = create_chat(state.clone(), Some("owner-a"), serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();
    let appended = send(
        state.clone(),
        append(
            &id,
            serde_json::json!({
                "messages": [
                    { "role": "user", "content": "вопрос" },
                    { "role": "assistant", "content": "ответ" }
                ]
            }),
            Some("owner-a"),
        ),
    )
    .await;
    assert_eq!(appended.status, StatusCode::CREATED, "подготовка истории: {}", appended.body);

    let list = send(state.clone(), request("GET", &branches_uri(&id), None, Some("owner-b"))).await;
    assert_eq!(list.status, StatusCode::NOT_FOUND, "тело: {}", list.body);

    let create = send(
        state.clone(),
        request(
            "POST",
            &branches_uri(&id),
            Some(serde_json::json!({ "from_seq": 1, "name": "ветка" })),
            Some("owner-b"),
        ),
    )
    .await;
    assert_eq!(create.status, StatusCode::NOT_FOUND, "тело: {}", create.body);

    let activate = send(
        state,
        request("POST", &format!("{}/whatever/activate", branches_uri(&id)), None, Some("owner-b")),
    )
    .await;
    assert_eq!(activate.status, StatusCode::NOT_FOUND, "тело: {}", activate.body);
}

// --- 7.4 Чтение чата с указанием ветки ---

#[tokio::test]
async fn reading_explicit_branch_does_not_change_active_branch() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();
    seed_exchanges(&state, &id, &[("общий вопрос", "общий ответ")]).await;

    let root_branch_id = send(state.clone(), request("GET", &format!("/v1/chats/{id}"), None, None))
        .await
        .body["branch_id"]
        .as_str()
        .unwrap()
        .to_string();

    let created_branch = send(
        state.clone(),
        request(
            "POST",
            &branches_uri(&id),
            Some(serde_json::json!({ "from_seq": 2, "name": "вторая" })),
            None,
        ),
    )
    .await;
    let branch_id = created_branch.body["id"].as_str().unwrap().to_string();
    send(
        state.clone(),
        request("POST", &format!("{}/{branch_id}/activate", branches_uri(&id)), None, None),
    )
    .await;

    let explicit = send(
        state.clone(),
        request("GET", &format!("/v1/chats/{id}?branch={root_branch_id}"), None, None),
    )
    .await;
    assert_eq!(explicit.body["branch_id"], root_branch_id);

    let default_read = send(state, request("GET", &format!("/v1/chats/{id}"), None, None)).await;
    assert_eq!(
        default_read.body["branch_id"], branch_id,
        "явное указание ветки не переключает активную ветку чата"
    );
}

// --- 7.5 Независимость веток при сборке контекста ---

#[tokio::test]
async fn branching_strategy_keeps_sibling_branches_independent() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ модели")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[]).await;
    let created = create_chat(
        state.clone(),
        None,
        serde_json::json!({ "settings": { "context_strategy": "branching" } }),
    )
    .await;
    let id = created.body["id"].as_str().unwrap().to_string();
    seed_exchanges(&state, &id, &[("общий вопрос", "общий ответ")]).await;
    let root_branch_id = send(state.clone(), request("GET", &format!("/v1/chats/{id}"), None, None))
        .await
        .body["branch_id"]
        .as_str()
        .unwrap()
        .to_string();

    let branch_a = send(
        state.clone(),
        request("POST", &branches_uri(&id), Some(serde_json::json!({ "from_seq": 2, "name": "A" })), None),
    )
    .await;
    let branch_a_id = branch_a.body["id"].as_str().unwrap().to_string();
    send(
        state.clone(),
        request("POST", &format!("{}/{branch_a_id}/activate", branches_uri(&id)), None, None),
    )
    .await;
    send(
        state.clone(),
        post_chat(serde_json::json!({ "chat_id": id, "prompt": "только в ветке A" })),
    )
    .await;

    send(
        state.clone(),
        request("POST", &format!("{}/{root_branch_id}/activate", branches_uri(&id)), None, None),
    )
    .await;

    let branch_b = send(
        state.clone(),
        request("POST", &branches_uri(&id), Some(serde_json::json!({ "from_seq": 2, "name": "B" })), None),
    )
    .await;
    let branch_b_id = branch_b.body["id"].as_str().unwrap().to_string();
    send(
        state.clone(),
        request("POST", &format!("{}/{branch_b_id}/activate", branches_uri(&id)), None, None),
    )
    .await;
    server.reset().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ модели")))
        .mount(&server)
        .await;

    let sent = send(
        state,
        post_chat(serde_json::json!({ "chat_id": id, "prompt": "вопрос о ветке A" })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);

    let requests = server.received_requests().await.expect("запросы");
    let last = requests.last().expect("запрос к провайдеру");
    let body = String::from_utf8_lossy(&last.body);
    assert!(!body.contains("только в ветке A"), "сообщения соседней ветки не должны попасть в запрос: {body}");
}

// --- 9.1/9.2 Сравнение стратегий: сценарий сбора ТЗ против wiremock ---
//
// Сценарий (`tests/data/context-scenario.json`) — один и тот же для всех
// стратегий и для ручного прогона против живой модели (9.3), поэтому цифры
// в отчёте сравнимы (specs/context-strategy-comparison, «Один сценарий
// прогоняется на всех стратегиях»).

struct Scenario {
    messages: Vec<String>,
    control_answer_contains: Vec<String>,
}

fn load_scenario() -> Scenario {
    let raw = include_str!("../tests/data/context-scenario.json");
    let value: serde_json::Value = serde_json::from_str(raw).expect("сценарий разобран");
    Scenario {
        messages: value["messages"]
            .as_array()
            .expect("messages")
            .iter()
            .map(|m| m.as_str().expect("строка сообщения").to_string())
            .collect(),
        control_answer_contains: value["control_answer_contains"]
            .as_array()
            .expect("control_answer_contains")
            .iter()
            .map(|s| s.as_str().expect("строка варианта ответа").to_string())
            .collect(),
    }
}

/// Деталь контрольного вопроса присутствует в теле последнего запроса к
/// провайдеру — прокси для «модель могла бы ответить верно», не требующий
/// живой модели (specs/context-strategy-comparison, «Прогон измеряет...
/// удержание деталей»).
fn last_request_contains_detail(requests: &[wiremock::Request], scenario: &Scenario) -> bool {
    let last = requests.last().expect("хотя бы один запрос к провайдеру");
    let body = String::from_utf8_lossy(&last.body);
    scenario.control_answer_contains.iter().any(|needle| body.contains(needle))
}

#[tokio::test]
async fn scenario_on_summary_strategy_keeps_detail_via_summary_text() {
    let _guard = test_lock();
    let scenario = load_scenario();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(BodyContains(SUMMARY_CALL_MARKER))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply(
            "Пересказ: бюджет проекта 250000 рублей, дедлайн конец квартала, каналы — Telegram и email.",
        )))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(BodyLacks(SUMMARY_CALL_MARKER))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("принято, продолжаем")))
        .mount(&server)
        .await;
    let state = state_with_provider(
        &server,
        &[("AGENTD_SUMMARY_ENABLED", "true"), ("AGENTD_SUMMARY_KEEP_MESSAGES", "4"), ("AGENTD_SUMMARY_STEP_MESSAGES", "1")],
    )
    .await;
    let created = create_chat_with_strategy(state.clone(), "summary").await;

    for message in &scenario.messages {
        let sent = send(state.clone(), post_chat(serde_json::json!({ "chat_id": created, "prompt": message }))).await;
        assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    }

    let requests: Vec<_> = server
        .received_requests()
        .await
        .expect("запросы")
        .into_iter()
        .filter(|r| !String::from_utf8_lossy(&r.body).contains(SUMMARY_CALL_MARKER))
        .collect();
    assert!(
        last_request_contains_detail(&requests, &scenario),
        "пересказ должен донести деталь из первого сообщения до контрольного вопроса"
    );
}

#[tokio::test]
async fn scenario_on_sliding_window_strategy_loses_early_detail() {
    let _guard = test_lock();
    let scenario = load_scenario();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("принято, продолжаем")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[("AGENTD_CONTEXT_WINDOW_MESSAGES", "4")]).await;
    let created = create_chat_with_strategy(state.clone(), "sliding_window").await;

    for message in &scenario.messages {
        let sent = send(state.clone(), post_chat(serde_json::json!({ "chat_id": created, "prompt": message }))).await;
        assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    }

    let requests = server.received_requests().await.expect("запросы");
    assert!(
        !last_request_contains_detail(&requests, &scenario),
        "узкое окно без пересказа не может донести вытесненную деталь"
    );
}

#[tokio::test]
async fn scenario_on_facts_strategy_keeps_detail_via_fact() {
    let _guard = test_lock();
    let scenario = load_scenario();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(BodyContains(crate::facts::FACTS_UPDATE_MARKER))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply(
            r#"[{"op":"set","key":"budget","value":"250000 рублей"}]"#,
        )))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(BodyLacks(crate::facts::FACTS_UPDATE_MARKER))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("принято, продолжаем")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[("AGENTD_CONTEXT_WINDOW_MESSAGES", "4")]).await;
    let created = create_chat_with_strategy(state.clone(), "facts").await;

    for message in &scenario.messages {
        let sent = send(state.clone(), post_chat(serde_json::json!({ "chat_id": created, "prompt": message }))).await;
        assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    }

    let requests: Vec<_> = server
        .received_requests()
        .await
        .expect("запросы")
        .into_iter()
        .filter(|r| !String::from_utf8_lossy(&r.body).contains(crate::facts::FACTS_UPDATE_MARKER))
        .collect();
    assert!(
        last_request_contains_detail(&requests, &scenario),
        "устойчивый факт должен донести деталь из первого сообщения до контрольного вопроса"
    );
}

#[tokio::test]
async fn scenario_on_branching_strategy_checkpoints_and_switches() {
    let _guard = test_lock();
    let scenario = load_scenario();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("принято, продолжаем")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[]).await;
    let created = create_chat_with_strategy(state.clone(), "branching").await;

    // Первая половина сценария — общий ствол диалога (checkpoint).
    let midpoint = scenario.messages.len() / 2;
    for message in &scenario.messages[..midpoint] {
        let sent = send(state.clone(), post_chat(serde_json::json!({ "chat_id": created, "prompt": message }))).await;
        assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    }
    let checkpoint_seq = send(state.clone(), request("GET", &format!("/v1/chats/{created}"), None, None))
        .await
        .body["message_count"]
        .as_i64()
        .unwrap();
    let root_branch_id = send(state.clone(), request("GET", &format!("/v1/chats/{created}"), None, None))
        .await
        .body["branch_id"]
        .as_str()
        .unwrap()
        .to_string();

    let branch_a = send(
        state.clone(),
        request(
            "POST",
            &branches_uri(&created),
            Some(serde_json::json!({ "from_seq": checkpoint_seq, "name": "вариант A" })),
            None,
        ),
    )
    .await;
    let branch_a_id = branch_a.body["id"].as_str().unwrap().to_string();
    send(state.clone(), request("POST", &format!("{}/{branch_a_id}/activate", branches_uri(&created)), None, None)).await;
    for message in &scenario.messages[midpoint..] {
        let sent = send(state.clone(), post_chat(serde_json::json!({ "chat_id": created, "prompt": format!("[A] {message}") }))).await;
        assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    }

    send(state.clone(), request("POST", &format!("{}/{root_branch_id}/activate", branches_uri(&created)), None, None)).await;
    let branch_b = send(
        state.clone(),
        request(
            "POST",
            &branches_uri(&created),
            Some(serde_json::json!({ "from_seq": checkpoint_seq, "name": "вариант B" })),
            None,
        ),
    )
    .await;
    let branch_b_id = branch_b.body["id"].as_str().unwrap().to_string();
    send(state.clone(), request("POST", &format!("{}/{branch_b_id}/activate", branches_uri(&created)), None, None)).await;
    server.reset().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("принято, продолжаем")))
        .mount(&server)
        .await;
    for message in &scenario.messages[midpoint..] {
        let sent = send(state.clone(), post_chat(serde_json::json!({ "chat_id": created, "prompt": format!("[B] {message}") }))).await;
        assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    }

    let requests = server.received_requests().await.expect("запросы после переключения на B");
    let last = requests.last().expect("запрос к провайдеру");
    let body = String::from_utf8_lossy(&last.body);
    assert!(!body.contains("[A]"), "ветка B не должна видеть сообщения ветки A: {body}");
    assert!(body.contains("[B]"), "ветка B должна видеть собственные сообщения: {body}");
}

// --- Состояние задачи (specs/task-state) ---

fn task_uri(chat: &str) -> String {
    format!("/v1/chats/{chat}/task")
}

// --- 6.1 Эндпоинты ---

#[tokio::test]
async fn task_state_of_new_chat_is_planning_over_http() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let sent = send(state, get(&task_uri(&id))).await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    assert_eq!(sent.body["stage"], "planning");
    assert_eq!(sent.body["step"], "");
    assert_eq!(sent.body["paused"], false);
    assert!(sent.body["id"].as_str().is_some());
}

#[tokio::test]
async fn foreign_chat_task_state_is_unreachable() {
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "test-key-value"),
        ("AGENTD_UPSTREAM_BASE_URL", "http://127.0.0.1:1"),
        ("AGENTD_MODEL", "test-model"),
        ("AGENTD_CLIENT_TOKENS", "owner-a,owner-b"),
    ])
    .await;
    let created = create_chat(state.clone(), Some("owner-a"), serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let sent = send(state, request("GET", &task_uri(&id), None, Some("owner-b"))).await;
    assert_eq!(sent.status, StatusCode::NOT_FOUND, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "chat_not_found");
}

#[tokio::test]
async fn allowed_transition_is_applied_and_journaled() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let sent = send(
        state.clone(),
        request(
            "POST",
            &format!("{}/transition", task_uri(&id)),
            Some(serde_json::json!({ "stage": "clarification", "step": "правит парсер", "expected_action": "ждёт ответа модели" })),
            None,
        ),
    )
    .await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    assert_eq!(sent.body["stage"], "clarification");
    assert_eq!(sent.body["step"], "правит парсер");
    assert_eq!(sent.body["transitions"].as_array().unwrap().len(), 1);
    assert_eq!(sent.body["transitions"][0]["source"], "manual");
    assert!(sent.body["new_task_id"].is_null());
}

#[tokio::test]
async fn skipping_a_stage_over_http_is_rejected_with_request_id() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let sent = send(
        state.clone(),
        request("POST", &format!("{}/transition", task_uri(&id)), Some(serde_json::json!({ "stage": "validation" })), None),
    )
    .await;
    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "task_transition_invalid");
    assert!(!sent.body["error"]["request_id"].as_str().unwrap().is_empty());

    let after = send(state, get(&task_uri(&id))).await;
    assert_eq!(after.body["stage"], "planning", "состояние не должно меняться при отказе");
}

#[tokio::test]
async fn transition_to_done_carries_new_task_id_and_clears_working_memory() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();
    let owner = crate::store::ANONYMOUS_OWNER.to_string();
    let chat = crate::store::load_chat(&state.db, &owner, &id, &agentcore::config::ChatSettings::default())
        .await
        .expect("чат");
    crate::store::set_working_memory(&state.db, &id, &chat.active_task_id, "k", "v", "manual", 1)
        .await
        .expect("рабочая память");

    send(
        state.clone(),
        request("POST", &format!("{}/transition", task_uri(&id)), Some(serde_json::json!({ "stage": "clarification" })), None),
    )
    .await;
    send(
        state.clone(),
        request("POST", &format!("{}/transition", task_uri(&id)), Some(serde_json::json!({ "stage": "execution" })), None),
    )
    .await;
    send(
        state.clone(),
        request("POST", &format!("{}/transition", task_uri(&id)), Some(serde_json::json!({ "stage": "validation" })), None),
    )
    .await;
    let done = send(
        state.clone(),
        request("POST", &format!("{}/transition", task_uri(&id)), Some(serde_json::json!({ "stage": "done" })), None),
    )
    .await;
    assert_eq!(done.status, StatusCode::OK, "тело: {}", done.body);
    let new_task_id = done.body["new_task_id"].as_str().expect("новая задача").to_string();
    assert_ne!(new_task_id, chat.active_task_id);

    let after = send(state.clone(), get(&task_uri(&id))).await;
    assert_eq!(after.body["stage"], "planning");
    assert_eq!(after.body["id"], new_task_id);

    let working = crate::store::load_working_memory(&state.db, &id, &chat.active_task_id).await.expect("память");
    assert!(working.is_empty(), "рабочая память прежней задачи очищена переходом в done");
}

// --- 6.2 Журнал переходов ---

#[tokio::test]
async fn manual_transition_journal_entry_has_manual_source_and_new_task_journal_is_empty() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();
    send(
        state.clone(),
        request("POST", &format!("{}/transition", task_uri(&id)), Some(serde_json::json!({ "stage": "clarification" })), None),
    )
    .await;
    send(
        state.clone(),
        request("POST", &format!("{}/transition", task_uri(&id)), Some(serde_json::json!({ "stage": "execution" })), None),
    )
    .await;
    send(
        state.clone(),
        request("POST", &format!("{}/transition", task_uri(&id)), Some(serde_json::json!({ "stage": "validation" })), None),
    )
    .await;
    let done = send(
        state.clone(),
        request("POST", &format!("{}/transition", task_uri(&id)), Some(serde_json::json!({ "stage": "done" })), None),
    )
    .await;
    let new_task_id = done.body["new_task_id"].as_str().unwrap().to_string();

    let after = send(state, get(&task_uri(&id))).await;
    assert_eq!(after.body["id"], new_task_id);
    assert!(after.body["transitions"].as_array().unwrap().is_empty(), "журнал новой задачи пуст");
}

// --- Пауза и возобновление ---

#[tokio::test]
async fn pause_preserves_state_and_resume_keeps_stage() {
    let _guard = test_lock();
    let state = AppState::for_tests().await;
    let created = create_chat(state.clone(), None, serde_json::json!({})).await;
    let id = created.body["id"].as_str().unwrap().to_string();
    send(
        state.clone(),
        request("POST", &format!("{}/transition", task_uri(&id)), Some(serde_json::json!({ "stage": "clarification" })), None),
    )
    .await;
    send(
        state.clone(),
        request(
            "POST",
            &format!("{}/transition", task_uri(&id)),
            Some(serde_json::json!({ "stage": "execution", "step": "шаг", "expected_action": "действие" })),
            None,
        ),
    )
    .await;

    let paused = send(state.clone(), request("POST", &format!("{}/pause", task_uri(&id)), Some(serde_json::json!({})), None)).await;
    assert_eq!(paused.status, StatusCode::OK, "тело: {}", paused.body);
    assert_eq!(paused.body["paused"], true);
    assert_eq!(paused.body["stage"], "execution");
    assert_eq!(paused.body["step"], "шаг");

    // Явный переход на паузе отклоняется.
    let rejected = send(
        state.clone(),
        request("POST", &format!("{}/transition", task_uri(&id)), Some(serde_json::json!({ "stage": "validation" })), None),
    )
    .await;
    assert_eq!(rejected.status, StatusCode::BAD_REQUEST, "тело: {}", rejected.body);

    // Повторная пауза — не ошибка.
    let paused_again = send(state.clone(), request("POST", &format!("{}/pause", task_uri(&id)), Some(serde_json::json!({})), None)).await;
    assert_eq!(paused_again.status, StatusCode::OK);

    let resumed = send(state.clone(), request("POST", &format!("{}/resume", task_uri(&id)), Some(serde_json::json!({})), None)).await;
    assert_eq!(resumed.status, StatusCode::OK, "тело: {}", resumed.body);
    assert_eq!(resumed.body["paused"], false);
    assert_eq!(resumed.body["stage"], "execution");
}

// --- 5.2/5.3/5.4 Настройки чата, системное сообщение, блок context ---

#[tokio::test]
async fn task_state_section_is_present_when_enabled_and_absent_when_disabled() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[]).await;

    let enabled = create_chat(state.clone(), None, serde_json::json!({ "settings": { "task_state_enabled": true } })).await;
    let enabled_id = enabled.body["id"].as_str().unwrap().to_string();
    let sent = send(state.clone(), post_chat(serde_json::json!({ "chat_id": enabled_id, "prompt": "вопрос" }))).await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    assert_eq!(sent.body["context"]["task_stage"], "planning");
    assert_eq!(sent.body["context"]["task_paused"], false);
    let requests = server.received_requests().await.expect("запросы");
    let body = String::from_utf8_lossy(&requests[0].body);
    assert!(body.contains("Состояние задачи:"), "тело: {body}");

    server.reset().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ")))
        .mount(&server)
        .await;
    let disabled = create_chat(state.clone(), None, serde_json::json!({ "settings": { "task_state_enabled": false } })).await;
    let disabled_id = disabled.body["id"].as_str().unwrap().to_string();
    let sent = send(state.clone(), post_chat(serde_json::json!({ "chat_id": disabled_id, "prompt": "вопрос" }))).await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    assert!(sent.body["context"]["task_stage"].is_null());
    let requests = server.received_requests().await.expect("запросы");
    let body = String::from_utf8_lossy(&requests.last().unwrap().body);
    assert!(!body.contains("Состояние задачи:"), "тело: {body}");
}

#[tokio::test]
async fn client_enables_task_state_over_operator_default_and_disables_tracker() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ")))
        .mount(&server)
        .await;
    // Умолчание сервиса выключает и состояние задачи, и трекер.
    let state = state_with_provider(&server, &[("AGENTD_AUTO_TITLE", "false")]).await;
    assert!(!state.config.task_state_enabled);
    assert!(!state.config.task_state_auto_enabled);

    let created = create_chat(
        state.clone(),
        None,
        serde_json::json!({ "settings": { "task_state_enabled": true, "task_state_auto_enabled": true } }),
    )
    .await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let sent = send(state.clone(), post_chat(serde_json::json!({ "chat_id": id, "prompt": "вопрос" }))).await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    assert_eq!(sent.body["context"]["task_stage"], "planning", "клиент включил состояние задачи поверх умолчания");

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let requests_with_auto = server.received_requests().await.expect("запросы").len();
    assert_eq!(requests_with_auto, 2, "трекер должен вызваться вторым запросом при включённом автотрекере чата");

    let disabled_tracker = create_chat(
        state.clone(),
        None,
        serde_json::json!({ "settings": { "task_state_enabled": true, "task_state_auto_enabled": false } }),
    )
    .await;
    let id2 = disabled_tracker.body["id"].as_str().unwrap().to_string();
    let sent = send(state.clone(), post_chat(serde_json::json!({ "chat_id": id2, "prompt": "вопрос" }))).await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let requests_after = server.received_requests().await.expect("запросы").len();
    assert_eq!(requests_after, requests_with_auto + 1, "клиент выключил трекер поверх умолчания сервиса");
}

// --- 6.3 Трекер ---

#[tokio::test]
async fn tracker_applies_valid_proposal_and_next_response_reports_it() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(BodyContains(crate::task::TASK_TRACKER_MARKER))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply(
            r#"{"stage":"clarification","step":"задаёт вопросы","expected_action":"ждёт ответа пользователя","reason":"план собран, есть открытые вопросы"}"#,
        )))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(BodyLacks(crate::task::TASK_TRACKER_MARKER))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[("AGENTD_AUTO_TITLE", "false")]).await;
    let created = create_chat(
        state.clone(),
        None,
        serde_json::json!({ "settings": { "task_state_enabled": true, "task_state_auto_enabled": true } }),
    )
    .await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let first = send(state.clone(), post_chat(serde_json::json!({ "chat_id": id, "prompt": "начнём" }))).await;
    assert_eq!(first.status, StatusCode::OK, "тело: {}", first.body);
    assert_eq!(first.body["context"]["task_tracker_applied"], 0, "у первого ответа ещё нет прогона трекера");
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let second = send(state.clone(), post_chat(serde_json::json!({ "chat_id": id, "prompt": "продолжаем" }))).await;
    assert_eq!(second.status, StatusCode::OK, "тело: {}", second.body);
    assert_eq!(second.body["context"]["task_stage"], "clarification", "предложение трекера применилось к состоянию");
    assert_eq!(second.body["context"]["task_tracker_applied"], 1);
    assert_eq!(second.body["context"]["task_tracker_rejected"], 0);
}

#[tokio::test]
async fn tracker_applies_execution_only_after_explicit_user_confirmation_in_clarification() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(BodyContains(crate::task::TASK_TRACKER_MARKER))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply(
            r#"{"stage":"execution","step":"пишет код","expected_action":"ждёт ревью","reason":"пользователь подтвердил готовность"}"#,
        )))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(BodyLacks(crate::task::TASK_TRACKER_MARKER))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("да, начинай")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[("AGENTD_AUTO_TITLE", "false")]).await;
    let created = create_chat(
        state.clone(),
        None,
        serde_json::json!({ "settings": { "task_state_enabled": true, "task_state_auto_enabled": true } }),
    )
    .await;
    let id = created.body["id"].as_str().unwrap().to_string();
    send(
        state.clone(),
        request("POST", &format!("{}/transition", task_uri(&id)), Some(serde_json::json!({ "stage": "clarification" })), None),
    )
    .await;

    let first = send(state.clone(), post_chat(serde_json::json!({ "chat_id": id, "prompt": "готов начинать?" }))).await;
    assert_eq!(first.status, StatusCode::OK, "тело: {}", first.body);
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let second = send(state.clone(), post_chat(serde_json::json!({ "chat_id": id, "prompt": "продолжаем" }))).await;
    assert_eq!(second.status, StatusCode::OK, "тело: {}", second.body);
    assert_eq!(second.body["context"]["task_stage"], "execution", "переход в execution применился по автомату");
    assert_eq!(second.body["context"]["task_tracker_applied"], 1);
    assert_eq!(second.body["context"]["task_tracker_rejected"], 0);
}

#[tokio::test]
async fn tracker_never_applies_done_even_when_edge_is_valid() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(BodyContains(crate::task::TASK_TRACKER_MARKER))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply(
            r#"{"stage":"done","reason":"похоже, готово"}"#,
        )))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(BodyLacks(crate::task::TASK_TRACKER_MARKER))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[("AGENTD_AUTO_TITLE", "false")]).await;
    let created = create_chat(
        state.clone(),
        None,
        serde_json::json!({ "settings": { "task_state_enabled": true, "task_state_auto_enabled": true } }),
    )
    .await;
    let id = created.body["id"].as_str().unwrap().to_string();
    send(
        state.clone(),
        request("POST", &format!("{}/transition", task_uri(&id)), Some(serde_json::json!({ "stage": "clarification" })), None),
    )
    .await;
    send(
        state.clone(),
        request("POST", &format!("{}/transition", task_uri(&id)), Some(serde_json::json!({ "stage": "execution" })), None),
    )
    .await;
    send(
        state.clone(),
        request("POST", &format!("{}/transition", task_uri(&id)), Some(serde_json::json!({ "stage": "validation" })), None),
    )
    .await;

    let first = send(state.clone(), post_chat(serde_json::json!({ "chat_id": id, "prompt": "готово?" }))).await;
    assert_eq!(first.status, StatusCode::OK, "тело: {}", first.body);
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let second = send(state.clone(), post_chat(serde_json::json!({ "chat_id": id, "prompt": "и что дальше" }))).await;
    assert_eq!(second.status, StatusCode::OK, "тело: {}", second.body);
    assert_eq!(second.body["context"]["task_stage"], "validation", "предложение done не применяется трекером");
    assert_eq!(second.body["context"]["task_tracker_rejected"], 1);
}

#[tokio::test]
async fn tracker_does_not_run_while_task_is_paused() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[("AGENTD_AUTO_TITLE", "false")]).await;
    let created = create_chat(
        state.clone(),
        None,
        serde_json::json!({ "settings": { "task_state_enabled": true, "task_state_auto_enabled": true } }),
    )
    .await;
    let id = created.body["id"].as_str().unwrap().to_string();
    send(state.clone(), request("POST", &format!("{}/pause", task_uri(&id)), Some(serde_json::json!({})), None)).await;

    let sent = send(state.clone(), post_chat(serde_json::json!({ "chat_id": id, "prompt": "вопрос" }))).await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let requests = server.received_requests().await.expect("запросы").len();
    assert_eq!(requests, 1, "трекер не должен вызываться, пока задача на паузе");
}

#[tokio::test]
async fn tracker_failure_does_not_break_main_response() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("основной ответ")))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[("AGENTD_AUTO_TITLE", "false")]).await;
    let created = create_chat(
        state.clone(),
        None,
        serde_json::json!({ "settings": { "task_state_enabled": true, "task_state_auto_enabled": true } }),
    )
    .await;
    let id = created.body["id"].as_str().unwrap().to_string();

    let sent = send(state.clone(), post_chat(serde_json::json!({ "chat_id": id, "prompt": "вопрос" }))).await;
    assert_eq!(sent.status, StatusCode::OK, "основной ответ не должен зависеть от отказа трекера: {}", sent.body);
}

// --- 4.4/5.3 Бриф возобновления в системном сообщении ---

#[tokio::test]
async fn resumed_task_system_message_carries_prior_stage_step_and_resume_brief() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("принято")))
        .mount(&server)
        .await;
    let state = state_with_provider(&server, &[]).await;
    let created = create_chat(state.clone(), None, serde_json::json!({ "settings": { "task_state_enabled": true } })).await;
    let id = created.body["id"].as_str().unwrap().to_string();
    send(
        state.clone(),
        request("POST", &format!("{}/transition", task_uri(&id)), Some(serde_json::json!({ "stage": "clarification" })), None),
    )
    .await;
    send(
        state.clone(),
        request(
            "POST",
            &format!("{}/transition", task_uri(&id)),
            Some(serde_json::json!({ "stage": "execution", "step": "правит парсер", "expected_action": "ждёт ревью" })),
            None,
        ),
    )
    .await;
    send(state.clone(), request("POST", &format!("{}/pause", task_uri(&id)), Some(serde_json::json!({})), None)).await;
    send(state.clone(), request("POST", &format!("{}/resume", task_uri(&id)), Some(serde_json::json!({})), None)).await;

    let sent = send(
        state.clone(),
        post_chat(serde_json::json!({ "chat_id": id, "prompt": "продолжаем, без пересказа контекста" })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    assert_eq!(sent.body["context"]["task_stage"], "execution");
    assert_eq!(sent.body["context"]["task_step"], "правит парсер");

    let requests = server.received_requests().await.expect("запросы");
    let body = String::from_utf8_lossy(&requests[0].body);
    assert!(body.contains("execution"), "тело: {body}");
    assert!(body.contains("правит парсер"), "тело: {body}");
    assert!(body.contains("ждёт ревью"), "тело: {body}");
}


// --- Инварианты: `AGENTD_INVARIANTS_PATH`, отказ `invariant_violation`,
// отклонение клиентского поля `invariants`
// (openspec/changes/add-invariant-guardrails) ---

/// Маркер в теле служебного запроса `InvariantGuard` (см. приглашение в
/// `agentcore::invariants::InvariantGuard::prompt`): отличает его от
/// основного диалогового вызова на одном и том же `/chat/completions`.
const INVARIANT_GUARD_CALL_MARKER: &str = "ОТВЕТ МОДЕЛИ";

fn write_invariants_file(dir: &std::path::Path) -> String {
    let path = dir.join("invariants.toml");
    std::fs::write(
        &path,
        r#"
[[invariant]]
id = "no-client-side-secrets"
statement = "ключ провайдера принадлежит сервису, не клиенту"
category = "security"
"#,
    )
    .expect("запись файла инвариантов");
    path.to_str().expect("путь к файлу инвариантов").to_string()
}

#[tokio::test]
async fn invariant_violation_is_rejected_with_dedicated_code_and_request_id() {
    let _guard = test_lock();
    let dir = tempfile::tempdir().expect("временная директория");
    let invariants_path = write_invariants_file(dir.path());

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(BodyContains(INVARIANT_GUARD_CALL_MARKER))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply(
            r#"{"violated": true, "invariant_id": "no-client-side-secrets", "explanation": "предложил вынести ключ в тело запроса"}"#,
        )))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(BodyLacks(INVARIANT_GUARD_CALL_MARKER))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply(
            "вот твой ключ провайдера, положи его в тело запроса",
        )))
        .mount(&server)
        .await;

    let state = state_with_provider(&server, &[("AGENTD_INVARIANTS_PATH", &invariants_path)]).await;
    let sent = send(state, post_chat(serde_json::json!({ "prompt": "как передать ключ клиенту?" }))).await;

    assert_eq!(sent.status, StatusCode::UNPROCESSABLE_ENTITY, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "invariant_violation");
    assert!(
        sent.body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("no-client-side-secrets"),
        "тело: {}",
        sent.body
    );
    assert_eq!(sent.body["error"]["request_id"], sent.request_id_header);
    assert!(!sent.request_id_header.is_empty());
}

#[tokio::test]
async fn missing_invariants_path_behaves_like_before_the_change() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_reply("ответ модели")))
        .mount(&server)
        .await;

    // Без AGENTD_INVARIANTS_PATH: набор инвариантов пуст, InvariantGuard не
    // выполняет служебный вызов (spec.md, «Пустой источник»).
    let state = state_with_provider(&server, &[]).await;
    let sent = send(state, post_chat(serde_json::json!({ "prompt": "привет" }))).await;

    assert_eq!(sent.status, StatusCode::OK, "тело: {}", sent.body);
    assert_eq!(sent.body["content"], "ответ модели");
    assert_eq!(server.received_requests().await.expect("запросы").len(), 1);
}

#[tokio::test]
async fn client_supplied_invariants_field_is_rejected_before_calling_provider() {
    let _guard = test_lock();
    let server = MockServer::start().await;
    let state = state_with_provider(&server, &[]).await;

    let sent = send(
        state,
        post_chat(serde_json::json!({
            "prompt": "привет",
            "invariants": [{"id": "x", "statement": "y", "category": "z"}]
        })),
    )
    .await;

    assert_eq!(sent.status, StatusCode::BAD_REQUEST, "тело: {}", sent.body);
    assert_eq!(sent.body["error"]["code"], "invalid_request");
    assert_eq!(server.received_requests().await.expect("запросы").len(), 0);
}
