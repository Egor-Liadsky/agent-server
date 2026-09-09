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

/// Сервис, у которого апстрим — заданный `wiremock`.
pub async fn state_with_provider(server: &MockServer, extra: &[(&str, &str)]) -> AppState {
    let uri = server.uri();
    let mut pairs: Vec<(&str, &str)> = vec![
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_UPSTREAM_BASE_URL", uri.as_str()),
        ("AGENTD_MODEL", "model-a"),
    ];
    pairs.extend_from_slice(extra);
    AppState::with_env(&pairs)
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
    assert_eq!(settings.temperature, Some(0.3));
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
    };
    let value = serde_json::to_value(&response).expect("сериализация");
    assert_eq!(value["request_id"], "req-1");
    assert_eq!(value["content"], "ответ");
    assert!(value["reasoning"].is_null());
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
        AppState::for_tests(),
    )
    .await;
}

#[tokio::test]
async fn neither_prompt_nor_messages_is_rejected() {
    let _guard = test_lock();
    expect_invalid_request(serde_json::json!({ "prompt": "  " }), AppState::for_tests()).await;
}

#[tokio::test]
async fn api_key_in_body_is_rejected() {
    let _guard = test_lock();
    let envelope = expect_invalid_request(
        serde_json::json!({ "prompt": "привет", "settings": { "api_key": "sk-1" } }),
        AppState::for_tests(),
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
        ]),
    )
    .await;
}

#[tokio::test]
async fn ollama_without_url_is_rejected() {
    let _guard = test_lock();
    expect_invalid_request(
        serde_json::json!({ "prompt": "привет", "settings": { "provider": "ollama" } }),
        AppState::for_tests(),
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

// --- 5.5 Служебные эндпоинты ---

#[tokio::test]
async fn models_lists_allowed_models() {
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_MODEL", "model-a"),
        ("AGENTD_ALLOWED_MODELS", "model-a,model-b"),
    ]);
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
    let sent = send(AppState::for_tests(), get("/readyz")).await;
    assert_eq!(sent.status, StatusCode::OK);
}

#[tokio::test]
async fn readyz_is_503_for_invalid_config() {
    let _guard = test_lock();
    let state = AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_UPSTREAM_BASE_URL", "не-адрес"),
    ]);
    let sent = send(state, get("/readyz")).await;
    assert_eq!(sent.status, StatusCode::SERVICE_UNAVAILABLE);
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
    let sent = send(AppState::for_tests(), get("/healthz")).await;
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

fn authenticated_state() -> AppState {
    AppState::with_env(&[
        (API_KEY_VAR, "secret-key-value"),
        ("AGENTD_MODEL", "model-a"),
        ("AGENTD_CLIENT_TOKENS", "client-token-1,client-token-2"),
    ])
}

#[tokio::test]
async fn request_without_token_is_401() {
    let _guard = test_lock();
    let sent = send(
        authenticated_state(),
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
        authenticated_state(),
        post_chat_with_token(serde_json::json!({ "prompt": "привет" }), Some("чужой")),
    )
    .await;
    assert_eq!(sent.status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn healthz_needs_no_token() {
    let _guard = test_lock();
    let sent = send(authenticated_state(), get("/healthz")).await;
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
    ]);
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
        AppState::for_tests(),
        post_chat(serde_json::json!({ "messages": [] })),
    )
    .await;
    assert_eq!(sent.status, StatusCode::BAD_REQUEST);
    assert!(!sent.request_id_header.is_empty());
    assert_eq!(sent.body["error"]["request_id"], sent.request_id_header);
}
