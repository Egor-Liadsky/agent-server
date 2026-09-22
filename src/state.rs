//! Общее состояние сервиса: конфигурация и единственный клиент провайдера.

use crate::config::AgentdConfig;
use agentcore::agent::{Agent, AgentReply, Message, OllamaAgent, ToolSpec};
use agentcore::config::{ChatSettings, Config, Provider};
use agentcore::invariants::InvariantSet;
use agentcore::logging::ExchangeLog;
use agentupstream::UpstreamAgent;
use anyhow::Result;
use async_trait::async_trait;
use sqlx::SqlitePool;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// Сколько раз сервис создавал клиента провайдера. Диагностика: клиент
// создаётся один раз при старте, а не на каждый входящий запрос.
//
// Счётчик именно потоковый, а не общий на процесс: тесты идут параллельно,
// и каждый из них создаёт своё состояние. С общим счётчиком измерение
// теста сбивалось чужим созданием состояния между двумя чтениями
// (`left: 2, right: 1`) — тест проходил в одиночку и падал в полном
// прогоне. `build_agent` вызывается синхронно, до первой точки `await` в
// `AppState::new`, поэтому всегда исполняется на том же потоке, что и
// создающий состояние тест.
thread_local! {
    static AGENT_CREATIONS: AtomicUsize = const { AtomicUsize::new(0) };
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn agent_creations() -> usize {
    AGENT_CREATIONS.with(|counter| counter.load(Ordering::SeqCst))
}

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<AgentdConfig>,
    /// Один клиент на весь процесс: соединения к провайдеру переиспользуются.
    pub agent: Arc<ServiceAgent>,
    /// Активные инварианты: загружены один раз при старте из
    /// `AGENTD_INVARIANTS_PATH`, не приходят с запросом (design.md,
    /// «Отдельный тип `InvariantSet`»).
    pub invariants: Arc<InvariantSet>,
    pub db: SqlitePool,
    /// Счётчики последнего фонового прогона маршрутизатора памяти по чату
    /// (слоистая память) — маршрутизатор фоновый, поэтому его результат
    /// сообщается блоком `context` СЛЕДУЮЩЕГО ответа (design.md, решение 5).
    /// Не персистентно: перезапуск сервиса обнуляет счётчики, это тот же
    /// допустимый компромисс, что и однопроцессная SQLite (design.md,
    /// «Риски»).
    pub memory_route_outcomes: Arc<std::sync::Mutex<std::collections::HashMap<String, crate::memory::RouteOutcome>>>,
    /// Тот же приём, что `memory_route_outcomes`, для фонового трекера
    /// состояния задачи (specs/task-state, design.md решение 5).
    pub task_track_outcomes: Arc<std::sync::Mutex<std::collections::HashMap<String, crate::task::TrackOutcome>>>,
    /// Временный каталог тестовой базы. Держится здесь, чтобы не удалиться
    /// раньше последнего клона состояния; удаляется вместе с последним.
    #[cfg(test)]
    _test_db_dir: Option<Arc<tempfile::TempDir>>,
}

impl AppState {
    /// Открывает хранилище и собирает состояние. Непригодная база — фатальная
    /// ошибка старта, как и отсутствующий ключ провайдера.
    pub async fn new(config: AgentdConfig) -> Result<Self> {
        let agent = build_agent(&config)?;
        let invariants = match &config.invariants_path {
            Some(path) => InvariantSet::load(std::path::Path::new(path))?,
            None => InvariantSet::default(),
        };
        let db = crate::store::open_pool(
            &config.db_path,
            config.db_max_connections,
            config.db_busy_timeout_ms,
        )
        .await?;
        Ok(Self {
            config: Arc::new(config),
            agent: Arc::new(agent),
            invariants: Arc::new(invariants),
            db,
            memory_route_outcomes: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            task_track_outcomes: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            #[cfg(test)]
            _test_db_dir: None,
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

    async fn ask_with_tools(
        &self,
        history: &[Message],
        settings: &ChatSettings,
        tools: &[ToolSpec],
    ) -> Result<AgentReply> {
        match settings.provider {
            Provider::Cloud => self.cloud.ask_with_tools(history, settings, tools).await,
            Provider::Ollama => self.local.ask_with_tools(history, settings, tools).await,
        }
    }
}

fn build_agent(config: &AgentdConfig) -> Result<ServiceAgent> {
    AGENT_CREATIONS.with(|counter| counter.fetch_add(1, Ordering::SeqCst));
    // Ключ и адрес провайдера принадлежат сервису и берутся из его
    // переменных окружения, а не из пользовательского конфига клиента.
    let core_config = Config {
        model: Some(config.model.clone()),
        ..Config::default()
    };
    // Файлы JSONL внутри контейнера некуда складывать, поэтому в обычном
    // режиме журнал обмена выключен, а в дебаге сырые тела уходят приёмником
    // в structured logging сервиса.
    let log = Arc::new(if config.debug {
        ExchangeLog::to_sink(Arc::new(crate::telemetry::TracingExchangeSink))
    } else {
        ExchangeLog::disabled()
    });
    // Таймаут провайдера живёт на клиенте ядра: его истечение даёт
    // типизированный `AgentError::Timeout`, а не безымянное зависание.
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
    pub async fn for_tests() -> Self {
        Self::with_env(&[
            (crate::config::API_KEY_VAR, "test-key-value"),
            ("AGENTD_UPSTREAM_BASE_URL", "http://127.0.0.1:1"),
            ("AGENTD_MODEL", "test-model"),
        ])
        .await
    }

    /// Состояние по набору переменных окружения — так тест задаёт ровно то,
    /// что проверяет, не трогая окружение процесса. Каждый вызов открывает
    /// свою временную базу: тесты не делят файл и не оставляют его после
    /// себя (задача 8.3).
    pub async fn with_env(pairs: &[(&str, &str)]) -> Self {
        let dir = tempfile::tempdir().expect("временный каталог для базы теста");
        let db_path = dir.path().join("agentd.db");
        let mut owned: Vec<(String, String)> =
            pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        // Явно заданный в pairs AGENTD_DB_PATH (тест намеренно делит базу
        // между двумя состояниями) не переопределяется временным путём.
        if !owned.iter().any(|(key, _)| key == "AGENTD_DB_PATH") {
            owned.push((
                "AGENTD_DB_PATH".to_string(),
                db_path.to_str().expect("путь к тестовой базе").to_string(),
            ));
        }
        let map: std::collections::HashMap<String, String> = owned.into_iter().collect();
        let config = crate::config::AgentdConfig::from_source(&move |key| map.get(key).cloned())
            .expect("конфигурация");
        let mut state = Self::new(config).await.expect("состояние");
        // Каталог живёт, пока жив хотя бы один клон состояния, и удаляется
        // вместе с последним — тест не оставляет файл базы на диске.
        state._test_db_dir = Some(Arc::new(dir));
        state
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
        let state = AppState::for_tests().await;
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
