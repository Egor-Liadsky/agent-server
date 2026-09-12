//! Слои, общие для всех маршрутов: идентификатор запроса, конверт ошибок,
//! аутентификация клиента.

use crate::error::ApiError;
use crate::state::AppState;
use crate::telemetry::{pretty, API_TARGET};
use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::{HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

pub const REQUEST_ID_HEADER: &str = "x-request-id";

/// Идентификатор запроса, общий для журнала, тела ответа и заголовка.
#[derive(Debug, Clone)]
pub struct RequestId(pub String);

/// Кто прислал запрос. Токен в журнал попадает только маскированным.
#[derive(Debug, Clone)]
pub struct ClientId(pub String);

pub const ANONYMOUS_CLIENT: &str = "anonymous";

/// Владелец чатов — необратимый отпечаток клиентского токена (или
/// `store::ANONYMOUS_OWNER` при выключенной аутентификации). В отличие от
/// `ClientId`, тут не маскированный текст для журнала, а значение колонки
/// `owner` в хранилище.
#[derive(Debug, Clone)]
pub struct Owner(pub String);

pub fn request_id_of(request: &Request) -> String {
    request
        .extensions()
        .get::<RequestId>()
        .map(|id| id.0.clone())
        .unwrap_or_default()
}

/// Присваивает идентификатор на входе и возвращает его в заголовке.
pub async fn assign_request_id(mut request: Request, next: Next) -> Response {
    let id = uuid::Uuid::new_v4().to_string();
    request.extensions_mut().insert(RequestId(id.clone()));
    let mut response = next.run(request).await;
    if let Ok(value) = HeaderValue::from_str(&id) {
        response.headers_mut().insert(REQUEST_ID_HEADER, value);
    }
    response
}

/// Приводит ответы, собранные слоями (413, 429, 400 от разбора тела),
/// к единому конверту ошибок.
pub async fn normalize_errors(request: Request, next: Next) -> Response {
    let request_id = request_id_of(&request);
    let response = next.run(request).await;
    let status = response.status();
    if status.is_success() {
        return response;
    }
    let is_json = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("application/json"));
    if is_json {
        return response;
    }
    envelope_for(status).with_request_id(request_id).into_response()
}

fn envelope_for(status: StatusCode) -> ApiError {
    match status {
        StatusCode::PAYLOAD_TOO_LARGE => ApiError::payload_too_large(),
        StatusCode::TOO_MANY_REQUESTS | StatusCode::SERVICE_UNAVAILABLE => ApiError::rate_limited(),
        StatusCode::UNAUTHORIZED => ApiError::unauthorized(),
        StatusCode::REQUEST_TIMEOUT | StatusCode::GATEWAY_TIMEOUT => ApiError::gateway_timeout(),
        status if status.is_client_error() => {
            ApiError::invalid_request("запрос не удалось разобрать")
        }
        _ => ApiError::internal(format!("слой вернул статус {status}")),
    }
}

/// Тела запроса и ответа в журнал отладки. Слой ставится только в режиме
/// отладки: он буферизует тела целиком, а вне отладки такой цены платить не
/// за что. Буферизация безопасна — ответы сервиса не потоковые, а размер
/// запроса уже ограничен `RequestBodyLimitLayer` снаружи.
pub async fn log_bodies(request: Request, next: Next) -> Response {
    let request_id = request_id_of(&request);
    let method = request.method().clone();
    let uri = request.uri().clone();

    let (parts, body) = request.into_parts();
    let bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(bytes) => bytes,
        Err(err) => return read_error(&err.to_string()).with_request_id(request_id).into_response(),
    };
    log_payload(&bytes, |event| {
        tracing::debug!(
            target: API_TARGET,
            %request_id,
            method = %method,
            uri = %uri,
            payload = %event,
            "тело запроса"
        );
    });

    let response = next.run(Request::from_parts(parts, Body::from(bytes))).await;

    let status = response.status();
    let (parts, body) = response.into_parts();
    let bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(bytes) => bytes,
        // Тело собственного ответа не прочиталось — журналу нечего показать,
        // но клиенту важнее получить ответ, поэтому статус сохраняется.
        Err(err) => {
            tracing::warn!(%request_id, error = %err, "тело ответа не прочитано для журнала");
            Bytes::new()
        }
    };
    log_payload(&bytes, |event| {
        tracing::debug!(
            target: API_TARGET,
            %request_id,
            status = status.as_u16(),
            payload = %event,
            "тело ответа"
        );
    });

    Response::from_parts(parts, Body::from(bytes))
}

/// Пустые тела не пишутся, JSON печатается с отступами, остальное — как есть.
fn log_payload(bytes: &Bytes, emit: impl FnOnce(&str)) {
    if bytes.is_empty() {
        return;
    }
    let text = String::from_utf8_lossy(bytes);
    match serde_json::from_str::<serde_json::Value>(&text) {
        Ok(value) => emit(&pretty(&value)),
        Err(_) => emit(&text),
    }
}

/// Чтение тела прервалось: превышение лимита отдаётся как 413, прочие
/// причины — как неразобранный запрос. Тип ошибки `axum` не различает их
/// иначе, чем по тексту вложенной причины.
fn read_error(message: &str) -> ApiError {
    if message.contains("length limit exceeded") {
        ApiError::payload_too_large()
    } else {
        ApiError::invalid_request("тело запроса не прочитано")
    }
}

/// Проверка клиентского токена. Пустой список токенов означает, что сервис
/// открыт: это допустимая, но явно предупреждаемая конфигурация.
pub async fn authenticate(
    State(state): State<AppState>,
    mut request: Request,
    next: Next,
) -> Response {
    if state.config.client_tokens.is_empty() {
        request
            .extensions_mut()
            .insert(ClientId(ANONYMOUS_CLIENT.to_string()));
        request
            .extensions_mut()
            .insert(Owner(crate::store::ANONYMOUS_OWNER.to_string()));
        return next.run(request).await;
    }
    let request_id = request_id_of(&request);
    let presented = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(|token| token.trim().to_string());

    let Some(presented) = presented else {
        return ApiError::unauthorized()
            .with_request_id(request_id)
            .into_response();
    };
    if !state
        .config
        .client_tokens
        .iter()
        .any(|known| constant_time_eq(known.as_bytes(), presented.as_bytes()))
    {
        return ApiError::unauthorized()
            .with_request_id(request_id)
            .into_response();
    }
    // Отпечаток считается с исходного токена: ClientId ниже хранит уже
    // маскированное значение только для журнала.
    request
        .extensions_mut()
        .insert(Owner(crate::store::owner_fingerprint(&presented)));
    request
        .extensions_mut()
        .insert(ClientId(crate::config::mask(&presented)));
    next.run(request).await
}

/// Сравнение за постоянное время: перебираются все байты обеих строк,
/// поэтому длительность не зависит от позиции первого несовпадения.
fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut difference = (left.len() ^ right.len()) as u8;
    let length = left.len().max(right.len());
    for index in 0..length {
        let a = left.get(index).copied().unwrap_or(0);
        let b = right.get(index).copied().unwrap_or(0);
        difference |= a ^ b;
    }
    difference == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_time_eq_matches_equality() {
        assert!(constant_time_eq(b"token", b"token"));
        assert!(!constant_time_eq(b"token", b"tokeN"));
        assert!(!constant_time_eq(b"token", b"token-long"));
        assert!(!constant_time_eq(b"", b"x"));
        assert!(constant_time_eq(b"", b""));
    }
}
