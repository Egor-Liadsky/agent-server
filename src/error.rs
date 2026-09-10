//! Единый конверт ошибок и отображение исходов обработки в коды состояния.

use agentcore::agent::AgentError;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEnvelope {
    pub error: ErrorBody,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBody {
    pub code: String,
    pub message: String,
    pub request_id: String,
}

#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: String,
    /// Полный текст для журнала: клиенту он не уходит.
    pub log_detail: Option<String>,
    pub request_id: Option<String>,
}

impl ApiError {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
            log_detail: None,
            request_id: None,
        }
    }

    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_request", message)
    }

    pub fn unauthorized() -> Self {
        Self::new(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "требуется корректный клиентский токен",
        )
    }

    pub fn payload_too_large() -> Self {
        Self::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "payload_too_large",
            "тело запроса превышает допустимый размер",
        )
    }

    pub fn policy_rejected(code: String, reason: String) -> Self {
        Self {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            code: "policy_rejected",
            message: reason,
            log_detail: Some(format!("код отказа политики: {code}")),
            request_id: None,
        }
        .with_policy_code(code)
    }

    /// Код отказа политики попадает клиенту вместе с причиной.
    fn with_policy_code(mut self, code: String) -> Self {
        self.message = format!("{}: {}", code, self.message);
        self
    }

    pub fn rate_limited() -> Self {
        Self::new(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limited",
            "превышен предел одновременной нагрузки, повторите позже",
        )
    }

    pub fn internal(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "internal_error",
            message: "внутренняя ошибка сервиса".to_string(),
            log_detail: Some(detail.into()),
            request_id: None,
        }
    }

    pub fn gateway_timeout() -> Self {
        Self::new(
            StatusCode::GATEWAY_TIMEOUT,
            "upstream_timeout",
            "обработка запроса не уложилась в отведённое время",
        )
    }

    pub fn not_ready() -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "not_ready",
            "конфигурация сервиса неполна",
        )
    }

    /// Чат не существует, недоступен клиенту или идентификатор неверного
    /// вида — все три исхода неотличимы (specs/chat-api, «Чужой чат
    /// неотличим от несуществующего»).
    pub fn chat_not_found() -> Self {
        Self::new(StatusCode::NOT_FOUND, "chat_not_found", "чат не найден")
    }

    /// Отказ хранилища. Клиенту уходит общее сообщение, подробности —
    /// только в журнал.
    pub fn storage_error(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "storage_error",
            message: "хранилище временно недоступно".to_string(),
            log_detail: Some(detail.into()),
            request_id: None,
        }
    }

    pub fn from_store_error(err: crate::store::StoreError) -> Self {
        match err {
            crate::store::StoreError::NotFound => Self::chat_not_found(),
            crate::store::StoreError::InvalidCursor => Self::invalid_request("неизвестный курсор"),
            crate::store::StoreError::Backend(err) => Self::storage_error(format!("{err:#}")),
        }
    }

    pub fn with_request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    /// Ошибка ядра в терминах контракта. Текст провайдера клиенту не уходит:
    /// он может содержать внутренний адрес или ключ.
    pub fn from_agent_error(err: &anyhow::Error) -> Self {
        match err.downcast_ref::<AgentError>() {
            Some(AgentError::Timeout { .. }) => Self {
                status: StatusCode::GATEWAY_TIMEOUT,
                code: "upstream_timeout",
                message: "провайдер модели не ответил в отведённое время".to_string(),
                log_detail: Some(format!("{err:#}")),
                request_id: None,
            },
            Some(AgentError::Provider { status, .. }) => {
                let mapped = if *status == 429 {
                    StatusCode::TOO_MANY_REQUESTS
                } else {
                    StatusCode::BAD_GATEWAY
                };
                let code = if *status == 429 {
                    "rate_limited"
                } else {
                    "upstream_error"
                };
                Self {
                    status: mapped,
                    code,
                    message: format!("провайдер модели вернул ошибку (код {status})"),
                    log_detail: Some(format!("{err:#}")),
                    request_id: None,
                }
            }
            Some(AgentError::Transport(_)) => Self {
                status: StatusCode::BAD_GATEWAY,
                code: "upstream_error",
                message: "не удалось обратиться к провайдеру модели".to_string(),
                log_detail: Some(format!("{err:#}")),
                request_id: None,
            },
            Some(AgentError::Decode(_)) => Self {
                status: StatusCode::BAD_GATEWAY,
                code: "upstream_error",
                message: "ответ провайдера модели не удалось разобрать".to_string(),
                log_detail: Some(format!("{err:#}")),
                request_id: None,
            },
            Some(AgentError::MissingApiKey { .. }) => Self::internal(format!("{err:#}")),
            // Ошибки клиентской стороны сервиса (отказ политики, лимит,
            // неверный запрос, отказ аутентификации) в вызове провайдера не
            // возникают: они появляются у потребителя контракта `/v1`.
            Some(AgentError::PolicyRejected { code, reason, .. }) => {
                Self::policy_rejected(code.clone(), reason.clone())
            }
            Some(AgentError::RateLimited { .. }) => Self::rate_limited(),
            Some(AgentError::InvalidRequest { message, .. }) => {
                Self::invalid_request(message.clone())
            }
            Some(AgentError::Unauthorized { .. }) => Self::internal(format!("{err:#}")),
            None => Self::internal(format!("{err:#}")),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let request_id = self.request_id.clone().unwrap_or_default();
        if let Some(detail) = &self.log_detail {
            tracing::warn!(
                request_id = %request_id,
                code = self.code,
                status = self.status.as_u16(),
                detail = %detail,
                "запрос завершился ошибкой"
            );
        }
        let body = ErrorEnvelope {
            error: ErrorBody {
                code: self.code.to_string(),
                message: self.message,
                request_id,
            },
        };
        (self.status, Json(body)).into_response()
    }
}
