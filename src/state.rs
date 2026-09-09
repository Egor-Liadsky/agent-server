//! Общее состояние сервиса: конфигурация и единственный клиент провайдера.

use crate::config::AgentdConfig;
use agentcore::agent::{Agent, AgentReply, Message, OllamaAgent};
use agentcore::config::{ChatSettings, Config, Provider};
use agentcore::logging::ExchangeLog;
use agentupstream::UpstreamAgent;
use anyhow::Result;
use async_trait::async_trait;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Сколько раз сервис создавал клиента провайдера. Диагностика: клиент
/// создаётся один раз при старте, а не на каждый входящий запрос.
static AGENT_CREATIONS: AtomicUsize = AtomicUsize::new(0);

#[cfg_attr(not(test), allow(dead_code))]
pub fn agent_creations() -> usize {
    AGENT_CREATIONS.load(Ordering::SeqCst)
}

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<AgentdConfig>,
    /// Один клиент на весь процесс: соединения к провайдеру переиспользуются.
    pub agent: Arc<ServiceAgent>,
}

impl AppState {
    pub fn new(config: AgentdConfig) -> Result<Self> {
        let agent = build_agent(&config)?;
        Ok(Self {
            config: Arc::new(config),
            agent: Arc::new(agent),
        })
    }
}

/// Выбор провайдера по настройкам запроса. Диспетчеризация живёт в сервисе:
/// набор доступных провайдеров задаётся его зависимостями, а облачный код
/// лежит в отдельном крейте `agentupstream`.
pub struct ServiceAgent {
    cloud: UpstreamAgent,
    local: OllamaAgent,
}

#[async_trait]
impl Agent for ServiceAgent {
    async fn ask(&self, history: &[Message], settings: &ChatSettings) -> Result<AgentReply> {
        match settings.provider {
            Provider::Cloud => self.cloud.ask(history, settings).await,
            Provider::Ollama => self.local.ask(history, settings).await,
        }
    }
}

fn build_agent(config: &AgentdConfig) -> Result<ServiceAgent> {
    AGENT_CREATIONS.fetch_add(1, Ordering::SeqCst);
    // Журнал обмена в сервисе выключен: диалоги пишет structured logging,
    // а файлы JSONL внутри контейнера некуда складывать.
    // Ключ и адрес провайдера принадлежат сервису и берутся из его
    // переменных окружения, а не из пользовательского конфига клиента.
    let core_config = Config {
        model: Some(config.model.clone()),
        ..Config::default()
    };
    // Таймаут провайдера живёт на клиенте ядра: его истечение даёт
    // типизированный `AgentError::Timeout`, а не безымянное зависание.
    let log = Arc::new(ExchangeLog::disabled());
    Ok(ServiceAgent {
        cloud: UpstreamAgent::new(
            config.upstream_api_key.clone(),
            config.upstream_base_url.clone(),
            config.model.clone(),
            log.clone(),
        )
        .with_request_timeout(config.request_timeout)?,
        local: OllamaAgent::from_config(&core_config, log)?
            .with_request_timeout(config.request_timeout)?,
    })
}

/// Тесты, создающие состояние, выполняются по очереди: счётчик созданий
/// клиента общий на процесс, и параллельный тест сбил бы его измерение.
#[cfg(test)]
pub fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|err| err.into_inner())
}

#[cfg(test)]
impl AppState {
    pub fn for_tests() -> Self {
        Self::with_env(&[
            (crate::config::API_KEY_VAR, "test-key-value"),
            ("AGENTD_UPSTREAM_BASE_URL", "http://127.0.0.1:1"),
            ("AGENTD_MODEL", "test-model"),
        ])
    }

    /// Состояние по набору переменных окружения — так тест задаёт ровно то,
    /// что проверяет, не трогая окружение процесса.
    pub fn with_env(pairs: &[(&str, &str)]) -> Self {
        Self::new(crate::config::config_from(pairs).expect("конфигурация")).expect("состояние")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::router;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    async fn call_healthz(state: AppState) {
        router(state)
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .expect("запрос"),
            )
            .await
            .expect("ответ");
    }

    #[tokio::test]
    async fn agent_is_created_once_for_many_requests() {
        let _guard = test_lock();
        let before = agent_creations();
        let state = AppState::for_tests();
        let created = agent_creations() - before;

        call_healthz(state.clone()).await;
        call_healthz(state.clone()).await;

        assert_eq!(created, 1, "клиент провайдера создаётся один раз");
        assert_eq!(
            agent_creations() - before,
            1,
            "запросы не создают новых клиентов"
        );
        assert!(Arc::ptr_eq(&state.agent, &state.clone().agent));
    }
}
