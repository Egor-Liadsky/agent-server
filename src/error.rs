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

    /// Клиент передал `settings.max_context_tokens`, превышающее операторский
    /// лимит по умолчанию (или заданное при отсутствующем операторском
    /// лимите).
    pub fn context_limit_invalid(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "context_limit_invalid", message)
    }

    /// Оценка размера истории превышает эффективный лимит контекстного окна.
    pub fn context_limit_exceeded(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "context_limit_exceeded", message)
    }

    /// Клиентское значение настроек компактизации выходит за операторские
    /// границы: `summary_keep_messages` шире `AGENTD_SUMMARY_KEEP_MESSAGES`,
    /// `summary_step_messages` уже `AGENTD_SUMMARY_STEP_MESSAGES`, либо любое
    /// из значений — ноль (specs/context-summary, «Границы клиентских
    /// настроек компактизации»).
    pub fn summary_settings_invalid(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "summary_settings_invalid", message)
    }

    /// Неизвестное значение `context_strategy` (specs/context-strategies,
    /// «Стратегия контекста задаётся настройкой чата»).
    pub fn context_strategy_invalid(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "context_strategy_invalid", message)
    }

    /// Стратегия вне операторского списка разрешённых
    /// (specs/context-strategies, «Операторские умолчание и список
    /// разрешённых стратегий»).
    pub fn context_strategy_not_allowed(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            "context_strategy_not_allowed",
            message,
        )
    }

    /// `context_window_messages` — ноль либо превышает операторский потолок
    /// (specs/context-sliding-window, «Размер окна настраивается»).
    pub fn context_window_invalid(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "context_window_invalid", message)
    }

    /// Ручная правка факта превышает операторский потолок числа фактов
    /// (specs/context-facts, «Границы набора фактов»).
    pub fn facts_limit_exceeded(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "facts_limit_exceeded", message)
    }

    /// Факт или ветка, которых нет у чата — неотличимо от чужого чата
    /// (specs/context-facts, specs/chat-branching).
    pub fn fact_not_found() -> Self {
        Self::new(StatusCode::NOT_FOUND, "fact_not_found", "факт не найден")
    }

    pub fn branch_not_found() -> Self {
        Self::new(StatusCode::NOT_FOUND, "branch_not_found", "ветка не найдена")
    }

    /// Запись рабочей или долговременной памяти, которой нет — неотличимо
    /// от чужой (specs/memory-layers).
    pub fn memory_entry_not_found() -> Self {
        Self::new(StatusCode::NOT_FOUND, "memory_entry_not_found", "запись памяти не найдена")
    }

    /// Ручная правка памяти превышает операторский потолок длины ключа или
    /// значения (specs/memory-layers, «Операторские умолчания и лимиты памяти»).
    pub fn memory_limit_exceeded(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "memory_limit_exceeded", message)
    }

    /// Несуществующий или чужой профиль в `profile_id`
    /// (specs/user-profiles, «Неизвестный профиль отклоняется явной
    /// ошибкой»).
    pub fn profile_invalid(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "profile_invalid", message)
    }

    /// Профиль по идентификатору из `GET/PATCH/DELETE /v1/profiles/{id}` не
    /// найден или принадлежит другому владельцу — неотличимо
    /// (specs/user-profiles, «Чужой профиль не читается»).
    pub fn profile_not_found() -> Self {
        Self::new(StatusCode::NOT_FOUND, "profile_not_found", "профиль не найден")
    }

    /// Изменение или удаление встроенного профиля, создание профиля без
    /// единого непустого поля предпочтений, создание сверх
    /// `AGENTD_MAX_PROFILES` (specs/user-profiles).
    pub fn profile_rejected(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "profile_rejected", message)
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

#[cfg(test)]
mod tests {
    use super::*;

    async fn body_of(response: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .expect("тело ответа");
        serde_json::from_slice(&bytes).expect("JSON конверта ошибок")
    }

    #[tokio::test]
    async fn new_context_error_codes_produce_expected_envelope() {
        for (error, expected_status, expected_code) in [
            (
                ApiError::context_strategy_invalid("неизвестная стратегия"),
                StatusCode::BAD_REQUEST,
                "context_strategy_invalid",
            ),
            (
                ApiError::context_strategy_not_allowed("стратегия запрещена оператором"),
                StatusCode::BAD_REQUEST,
                "context_strategy_not_allowed",
            ),
            (
                ApiError::context_window_invalid("окно должно быть положительным"),
                StatusCode::BAD_REQUEST,
                "context_window_invalid",
            ),
            (
                ApiError::facts_limit_exceeded("превышен потолок фактов"),
                StatusCode::BAD_REQUEST,
                "facts_limit_exceeded",
            ),
        ] {
            let response = error.with_request_id("req-1").into_response();
            assert_eq!(response.status(), expected_status);
            let body = body_of(response).await;
            assert_eq!(body["error"]["code"], expected_code);
            assert_eq!(body["error"]["request_id"], "req-1");
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
