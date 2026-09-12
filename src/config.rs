//! Конфигурация сервиса. Читается только из переменных окружения:
//! пользовательский конфиг консольного клиента сервис не трогает.

use agentcore::config::ContextStrategy;
use anyhow::{bail, Context, Result};
use std::net::SocketAddr;
use std::time::Duration;

pub const DEFAULT_LISTEN_ADDR: &str = "0.0.0.0:8080";
pub const DEFAULT_BASE_URL: &str = "https://api.deepseek.com";
pub const DEFAULT_MODEL: &str = "deepseek-v4-flash";
pub const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 60;
pub const DEFAULT_MAX_BODY_BYTES: usize = 256 * 1024;
pub const DEFAULT_MAX_CONCURRENCY: usize = 16;
pub const DEFAULT_DB_PATH: &str = "agentd.db";
pub const DEFAULT_DB_MAX_CONNECTIONS: u32 = 5;
pub const DEFAULT_DB_BUSY_TIMEOUT_MS: u64 = 5000;
pub const DEFAULT_SUMMARY_KEEP_MESSAGES: u32 = 20;
pub const DEFAULT_SUMMARY_STEP_MESSAGES: u32 = 10;
pub const DEFAULT_SUMMARY_MAX_CHARS: u32 = 4000;
/// Умолчание стратегии контекста: существующее развёртывание без новых
/// переменных ведёт себя как до появления стратегий (specs/context-strategies).
pub const DEFAULT_CONTEXT_STRATEGY: ContextStrategy = ContextStrategy::Summary;
pub const DEFAULT_CONTEXT_WINDOW_MESSAGES: u32 = 10;
pub const DEFAULT_MAX_FACTS: u32 = 50;
pub const DEFAULT_FACT_VALUE_MAX_CHARS: u32 = 500;
pub const DEFAULT_MAX_BRANCH_DEPTH: u32 = 8;

pub const API_KEY_VAR: &str = "AGENTD_UPSTREAM_API_KEY";

/// Формат записей журнала сервиса.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    Json,
    Text,
}

/// Куда сервис пишет журнал.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogTarget {
    Stdout,
    Stderr,
}

#[derive(Debug, Clone)]
pub struct AgentdConfig {
    pub listen_addr: SocketAddr,
    /// Пустой список — аутентификация выключена, сервис открыт.
    pub client_tokens: Vec<String>,
    /// Ключ провайдера принадлежит сервису и никогда не приходит от клиента.
    pub upstream_api_key: String,
    pub upstream_base_url: String,
    pub model: String,
    /// Адрес локального Ollama. `None` — провайдер `ollama` запрещён.
    pub ollama_url: Option<String>,
    /// Модели, которые клиент вправе запросить. Всегда непустой список.
    pub allowed_models: Vec<String>,
    pub request_timeout: Duration,
    pub max_body_bytes: usize,
    pub max_concurrency: usize,
    pub log_format: LogFormat,
    pub log_target: LogTarget,
    /// Писать ли в журнал тексты промптов и ответов.
    pub log_content: bool,
    /// Режим отладки: подробный журнал в консоль и читаемый JSON обоих
    /// уровней обмена. Меняет только умолчания `AGENTD_LOG_*`, явно заданные
    /// значения сильнее.
    pub debug: bool,
    /// Путь к файлу базы SQLite.
    pub db_path: String,
    pub db_max_connections: u32,
    pub db_busy_timeout_ms: u64,
    /// Операторский лимит контекстного окна по умолчанию, в токенах.
    /// `None` — проверка размера истории не выполняется.
    pub max_context_tokens: Option<u32>,
    /// Компактизация истории включена по умолчанию для новых запросов чата.
    /// Чат может включить или выключить её своим значением
    /// `settings.summary_enabled` (specs/context-summary, «Операторские
    /// настройки компактизации»).
    pub summary_enabled: bool,
    /// Потолок дословного хвоста компактизации: клиент может только сузить.
    pub summary_keep_messages: u32,
    /// Нижняя граница шага пересказа: клиент может только увеличить.
    pub summary_step_messages: u32,
    /// Верхняя граница длины пересказа в символах. Операторское значение,
    /// клиенту не передаётся.
    pub summary_max_chars: u32,
    /// Модель для построения пересказа. `None` — используется модель чата.
    pub summary_model: Option<String>,
    /// Стратегия контекста по умолчанию для чатов без явного значения.
    pub context_strategy: ContextStrategy,
    /// Стратегии, которые клиент вправе запросить. Пустой список — заданной
    /// операторской переменной не было, разрешены все.
    pub allowed_context_strategies: Vec<ContextStrategy>,
    /// Операторское умолчание размера окна для `sliding_window` и `facts`.
    pub context_window_messages: u32,
    /// Потолок числа фактов чата.
    pub max_facts: u32,
    /// Потолок длины значения одного факта в символах.
    pub fact_value_max_chars: u32,
    /// Модель для обновления фактов. `None` — используется модель чата.
    pub facts_model: Option<String>,
    /// Потолок длины цепочки родителей при сборке истории ветки.
    pub max_branch_depth: u32,
}

impl AgentdConfig {
    /// Конфигурация из окружения процесса.
    pub fn from_env() -> Result<Self> {
        Self::from_source(&|key| std::env::var(key).ok())
    }

    /// Конфигурация из произвольного источника: так её можно проверить
    /// тестом, не трогая глобальное окружение процесса.
    pub fn from_source(source: &dyn Fn(&str) -> Option<String>) -> Result<Self> {
        let get = |key: &str| -> Option<String> {
            source(key).map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
        };

        let upstream_api_key = get(API_KEY_VAR)
            .with_context(|| format!("не задана переменная окружения {API_KEY_VAR}"))?;

        let listen_addr = match get("AGENTD_LISTEN_ADDR") {
            Some(value) => value
                .parse()
                .with_context(|| format!("не удалось разобрать AGENTD_LISTEN_ADDR: {value}"))?,
            None => match get("PORT") {
                Some(port) => format!("0.0.0.0:{port}")
                    .parse()
                    .with_context(|| format!("не удалось разобрать PORT: {port}"))?,
                None => DEFAULT_LISTEN_ADDR.parse().expect("адрес по умолчанию"),
            },
        };

        // Читается до остальных полей журнала: дебаг сдвигает их умолчания.
        let debug = parse_bool_named(get("AGENTD_DEBUG"), "AGENTD_DEBUG")?;

        let model = get("AGENTD_MODEL").unwrap_or_else(|| DEFAULT_MODEL.to_string());
        // Пустой список означает «разрешена только модель по умолчанию»:
        // иначе клиент направлял бы трафик на любую модель провайдера.
        let allowed_models = match get("AGENTD_ALLOWED_MODELS") {
            Some(value) => split_list(&value),
            None => Vec::new(),
        };
        let allowed_models = if allowed_models.is_empty() {
            vec![model.clone()]
        } else {
            allowed_models
        };

        Ok(Self {
            listen_addr,
            client_tokens: get("AGENTD_CLIENT_TOKENS")
                .map(|value| split_list(&value))
                .unwrap_or_default(),
            upstream_api_key,
            upstream_base_url: get("AGENTD_UPSTREAM_BASE_URL")
                .unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            model,
            ollama_url: get("AGENTD_OLLAMA_URL"),
            allowed_models,
            request_timeout: Duration::from_secs(parse_number(
                get("AGENTD_REQUEST_TIMEOUT_SECS"),
                "AGENTD_REQUEST_TIMEOUT_SECS",
                DEFAULT_REQUEST_TIMEOUT_SECS,
            )?),
            max_body_bytes: parse_number(
                get("AGENTD_MAX_BODY_BYTES"),
                "AGENTD_MAX_BODY_BYTES",
                DEFAULT_MAX_BODY_BYTES,
            )?,
            max_concurrency: parse_number(
                get("AGENTD_MAX_CONCURRENCY"),
                "AGENTD_MAX_CONCURRENCY",
                DEFAULT_MAX_CONCURRENCY,
            )?,
            // В дебаге умолчание — текстовый формат: JSON-строки в консоли
            // читать неудобно, а машинный разбор при отладке не нужен.
            log_format: match get("AGENTD_LOG_FORMAT").as_deref() {
                None if debug => LogFormat::Text,
                None | Some("json") => LogFormat::Json,
                Some("text") => LogFormat::Text,
                Some(other) => bail!("AGENTD_LOG_FORMAT должен быть json или text, задано: {other}"),
            },
            log_target: match get("AGENTD_LOG_TARGET").as_deref() {
                None | Some("stdout") => LogTarget::Stdout,
                Some("stderr") => LogTarget::Stderr,
                Some(other) => {
                    bail!("AGENTD_LOG_TARGET должен быть stdout или stderr, задано: {other}")
                }
            },
            // Отладка без текстов промптов и ответов бессмысленна, поэтому в
            // дебаге содержимое пишется, пока переменная не запретит явно.
            log_content: match get("AGENTD_LOG_CONTENT") {
                None => debug,
                value => parse_bool(value)?,
            },
            debug,
            db_path: get("AGENTD_DB_PATH").unwrap_or_else(|| DEFAULT_DB_PATH.to_string()),
            db_max_connections: parse_nonzero(
                get("AGENTD_DB_MAX_CONNECTIONS"),
                "AGENTD_DB_MAX_CONNECTIONS",
                DEFAULT_DB_MAX_CONNECTIONS,
            )?,
            db_busy_timeout_ms: parse_nonzero(
                get("AGENTD_DB_BUSY_TIMEOUT_MS"),
                "AGENTD_DB_BUSY_TIMEOUT_MS",
                DEFAULT_DB_BUSY_TIMEOUT_MS,
            )?,
            max_context_tokens: parse_optional_positive(
                get("AGENTD_MAX_CONTEXT_TOKENS"),
                "AGENTD_MAX_CONTEXT_TOKENS",
            )?,
            summary_enabled: parse_bool_named(get("AGENTD_SUMMARY_ENABLED"), "AGENTD_SUMMARY_ENABLED")?,
            summary_keep_messages: parse_nonzero(
                get("AGENTD_SUMMARY_KEEP_MESSAGES"),
                "AGENTD_SUMMARY_KEEP_MESSAGES",
                DEFAULT_SUMMARY_KEEP_MESSAGES,
            )?,
            summary_step_messages: parse_nonzero(
                get("AGENTD_SUMMARY_STEP_MESSAGES"),
                "AGENTD_SUMMARY_STEP_MESSAGES",
                DEFAULT_SUMMARY_STEP_MESSAGES,
            )?,
            summary_max_chars: parse_nonzero(
                get("AGENTD_SUMMARY_MAX_CHARS"),
                "AGENTD_SUMMARY_MAX_CHARS",
                DEFAULT_SUMMARY_MAX_CHARS,
            )?,
            summary_model: get("AGENTD_SUMMARY_MODEL"),
            context_strategy: match get("AGENTD_CONTEXT_STRATEGY") {
                None => DEFAULT_CONTEXT_STRATEGY,
                Some(value) => ContextStrategy::parse(&value).ok_or_else(|| {
                    anyhow::anyhow!("AGENTD_CONTEXT_STRATEGY должен быть именем стратегии, задано: {value}")
                })?,
            },
            allowed_context_strategies: match get("AGENTD_ALLOWED_CONTEXT_STRATEGIES") {
                None => Vec::new(),
                Some(value) => split_list(&value)
                    .into_iter()
                    .map(|name| {
                        ContextStrategy::parse(&name).ok_or_else(|| {
                            anyhow::anyhow!(
                                "AGENTD_ALLOWED_CONTEXT_STRATEGIES содержит неизвестную стратегию: {name}"
                            )
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            },
            context_window_messages: parse_nonzero(
                get("AGENTD_CONTEXT_WINDOW_MESSAGES"),
                "AGENTD_CONTEXT_WINDOW_MESSAGES",
                DEFAULT_CONTEXT_WINDOW_MESSAGES,
            )?,
            max_facts: parse_nonzero(get("AGENTD_MAX_FACTS"), "AGENTD_MAX_FACTS", DEFAULT_MAX_FACTS)?,
            fact_value_max_chars: parse_nonzero(
                get("AGENTD_FACT_VALUE_MAX_CHARS"),
                "AGENTD_FACT_VALUE_MAX_CHARS",
                DEFAULT_FACT_VALUE_MAX_CHARS,
            )?,
            facts_model: get("AGENTD_FACTS_MODEL"),
            max_branch_depth: parse_nonzero(
                get("AGENTD_MAX_BRANCH_DEPTH"),
                "AGENTD_MAX_BRANCH_DEPTH",
                DEFAULT_MAX_BRANCH_DEPTH,
            )?,
        })
    }

    /// Стратегия разрешена, если список операторских ограничений пуст
    /// (умолчание — разрешены все) либо явно её называет.
    pub fn is_context_strategy_allowed(&self, strategy: ContextStrategy) -> bool {
        self.allowed_context_strategies.is_empty()
            || self.allowed_context_strategies.contains(&strategy)
    }

    pub fn is_model_allowed(&self, model: &str) -> bool {
        self.allowed_models.iter().any(|allowed| allowed == model)
    }

    /// Готовность: ключ задан и базовый адрес разбирается.
    pub fn is_ready(&self) -> bool {
        !self.upstream_api_key.trim().is_empty()
            && reqwest_url_is_valid(&self.upstream_base_url)
            && !self.allowed_models.is_empty()
    }

    /// Ключ в диагностическом выводе — только маскированным.
    pub fn masked_api_key(&self) -> String {
        mask(&self.upstream_api_key)
    }
}

fn reqwest_url_is_valid(url: &str) -> bool {
    let url = url.trim();
    !url.is_empty() && (url.starts_with("http://") || url.starts_with("https://"))
}

fn split_list(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(|item| item.trim().to_string())
        .filter(|item| !item.is_empty())
        .collect()
}

fn parse_number<T: std::str::FromStr>(value: Option<String>, name: &str, default: T) -> Result<T> {
    match value {
        None => Ok(default),
        Some(value) => value
            .parse()
            .map_err(|_| anyhow::anyhow!("не удалось разобрать {name}: {value}")),
    }
}

/// Как `parse_number`, но нулевое значение тоже считается ошибкой:
/// нулевой пул соединений или нулевой `busy_timeout` не имеют смысла.
fn parse_nonzero<T>(value: Option<String>, name: &str, default: T) -> Result<T>
where
    T: std::str::FromStr + PartialEq + Default,
{
    let parsed = parse_number(value, name, default)?;
    if parsed == T::default() {
        bail!("{name} не может быть нулём");
    }
    Ok(parsed)
}

/// Необязательное положительное целое: не задано — `None`, задано, но не
/// положительное целое — фатальная ошибка старта, как для прочих числовых
/// переменных `AGENTD_*`.
fn parse_optional_positive(value: Option<String>, name: &str) -> Result<Option<u32>> {
    match value {
        None => Ok(None),
        Some(value) => {
            let parsed: u32 = value
                .parse()
                .map_err(|_| anyhow::anyhow!("не удалось разобрать {name}: {value}"))?;
            if parsed == 0 {
                bail!("{name} не может быть нулём");
            }
            Ok(Some(parsed))
        }
    }
}

fn parse_bool_named(value: Option<String>, name: &str) -> Result<bool> {
    match value.as_deref() {
        None => Ok(false),
        Some("1" | "true" | "yes" | "on") => Ok(true),
        Some("0" | "false" | "no" | "off") => Ok(false),
        Some(other) => bail!("{name} должен быть булевым значением, задано: {other}"),
    }
}

fn parse_bool(value: Option<String>) -> Result<bool> {
    parse_bool_named(value, "AGENTD_LOG_CONTENT")
}

/// Маскированное представление секрета: не более четырёх первых и четырёх
/// последних символов.
pub fn mask(secret: &str) -> String {
    let chars: Vec<char> = secret.chars().collect();
    match chars.len() {
        0 => "<не задан>".to_string(),
        n if n <= 8 => "*".repeat(n),
        n => {
            let head: String = chars[..4].iter().collect();
            let tail: String = chars[n - 4..].iter().collect();
            format!("{head}***{tail}")
        }
    }
}

/// Источник переменных для тестов.
#[cfg(test)]
pub fn source_from(pairs: &[(&str, &str)]) -> std::collections::HashMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
        .collect()
}

#[cfg(test)]
pub fn config_from(pairs: &[(&str, &str)]) -> Result<AgentdConfig> {
    let map = source_from(pairs);
    AgentdConfig::from_source(&move |key| map.get(key).cloned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_shifts_log_defaults() {
        let config = config_from(&[(API_KEY_VAR, "secret-key-value"), ("AGENTD_DEBUG", "true")])
            .expect("конфигурация");
        assert!(config.debug);
        assert_eq!(config.log_format, LogFormat::Text);
        assert!(config.log_content);
    }

    #[test]
    fn explicit_log_settings_win_over_debug() {
        let config = config_from(&[
            (API_KEY_VAR, "secret-key-value"),
            ("AGENTD_DEBUG", "true"),
            ("AGENTD_LOG_FORMAT", "json"),
            ("AGENTD_LOG_CONTENT", "false"),
        ])
        .expect("конфигурация");
        assert!(config.debug);
        assert_eq!(config.log_format, LogFormat::Json);
        assert!(!config.log_content);
    }

    #[test]
    fn defaults_are_applied() {
        let config = config_from(&[(API_KEY_VAR, "secret-key-value")]).expect("конфигурация");
        assert_eq!(config.listen_addr.to_string(), DEFAULT_LISTEN_ADDR);
        assert_eq!(config.upstream_base_url, DEFAULT_BASE_URL);
        assert_eq!(config.model, DEFAULT_MODEL);
        assert!(config.client_tokens.is_empty());
        assert!(config.ollama_url.is_none());
        assert_eq!(config.request_timeout.as_secs(), DEFAULT_REQUEST_TIMEOUT_SECS);
        assert_eq!(config.max_body_bytes, DEFAULT_MAX_BODY_BYTES);
        assert_eq!(config.max_concurrency, DEFAULT_MAX_CONCURRENCY);
        assert_eq!(config.log_format, LogFormat::Json);
        assert_eq!(config.log_target, LogTarget::Stdout);
        assert!(!config.log_content);
        assert!(!config.debug);
        assert_eq!(config.db_path, DEFAULT_DB_PATH);
        assert_eq!(config.db_max_connections, DEFAULT_DB_MAX_CONNECTIONS);
        assert_eq!(config.db_busy_timeout_ms, DEFAULT_DB_BUSY_TIMEOUT_MS);
        assert_eq!(config.max_context_tokens, None);
        assert!(!config.summary_enabled);
        assert_eq!(config.summary_keep_messages, DEFAULT_SUMMARY_KEEP_MESSAGES);
        assert_eq!(config.summary_step_messages, DEFAULT_SUMMARY_STEP_MESSAGES);
        assert_eq!(config.summary_max_chars, DEFAULT_SUMMARY_MAX_CHARS);
        assert_eq!(config.summary_model, None);
        assert_eq!(config.context_strategy, DEFAULT_CONTEXT_STRATEGY);
    }

    #[test]
    fn summary_variables_are_applied() {
        let config = config_from(&[
            (API_KEY_VAR, "secret-key-value"),
            ("AGENTD_SUMMARY_ENABLED", "true"),
            ("AGENTD_SUMMARY_KEEP_MESSAGES", "30"),
            ("AGENTD_SUMMARY_STEP_MESSAGES", "5"),
            ("AGENTD_SUMMARY_MAX_CHARS", "1000"),
            ("AGENTD_SUMMARY_MODEL", "summary-model"),
        ])
        .expect("конфигурация");
        assert!(config.summary_enabled);
        assert_eq!(config.summary_keep_messages, 30);
        assert_eq!(config.summary_step_messages, 5);
        assert_eq!(config.summary_max_chars, 1000);
        assert_eq!(config.summary_model.as_deref(), Some("summary-model"));
    }

    #[test]
    fn non_numeric_summary_keep_messages_is_an_error() {
        let err = config_from(&[
            (API_KEY_VAR, "secret-key-value"),
            ("AGENTD_SUMMARY_KEEP_MESSAGES", "много"),
        ])
        .expect_err("ожидалась ошибка");
        assert!(format!("{err}").contains("AGENTD_SUMMARY_KEEP_MESSAGES"));
    }

    #[test]
    fn zero_summary_keep_messages_is_an_error() {
        let err = config_from(&[
            (API_KEY_VAR, "secret-key-value"),
            ("AGENTD_SUMMARY_KEEP_MESSAGES", "0"),
        ])
        .expect_err("ожидалась ошибка");
        assert!(format!("{err}").contains("AGENTD_SUMMARY_KEEP_MESSAGES"));
    }

    #[test]
    fn non_numeric_summary_step_messages_is_an_error() {
        let err = config_from(&[
            (API_KEY_VAR, "secret-key-value"),
            ("AGENTD_SUMMARY_STEP_MESSAGES", "много"),
        ])
        .expect_err("ожидалась ошибка");
        assert!(format!("{err}").contains("AGENTD_SUMMARY_STEP_MESSAGES"));
    }

    #[test]
    fn zero_summary_step_messages_is_an_error() {
        let err = config_from(&[
            (API_KEY_VAR, "secret-key-value"),
            ("AGENTD_SUMMARY_STEP_MESSAGES", "0"),
        ])
        .expect_err("ожидалась ошибка");
        assert!(format!("{err}").contains("AGENTD_SUMMARY_STEP_MESSAGES"));
    }

    #[test]
    fn non_numeric_summary_max_chars_is_an_error() {
        let err = config_from(&[
            (API_KEY_VAR, "secret-key-value"),
            ("AGENTD_SUMMARY_MAX_CHARS", "много"),
        ])
        .expect_err("ожидалась ошибка");
        assert!(format!("{err}").contains("AGENTD_SUMMARY_MAX_CHARS"));
    }

    #[test]
    fn zero_summary_max_chars_is_an_error() {
        let err = config_from(&[
            (API_KEY_VAR, "secret-key-value"),
            ("AGENTD_SUMMARY_MAX_CHARS", "0"),
        ])
        .expect_err("ожидалась ошибка");
        assert!(format!("{err}").contains("AGENTD_SUMMARY_MAX_CHARS"));
    }

    #[test]
    fn invalid_summary_enabled_is_an_error() {
        let err = config_from(&[
            (API_KEY_VAR, "secret-key-value"),
            ("AGENTD_SUMMARY_ENABLED", "может быть"),
        ])
        .expect_err("ожидалась ошибка");
        assert!(format!("{err}").contains("AGENTD_SUMMARY_ENABLED"));
    }

    #[test]
    fn max_context_tokens_is_applied() {
        let config = config_from(&[
            (API_KEY_VAR, "secret-key-value"),
            ("AGENTD_MAX_CONTEXT_TOKENS", "8000"),
        ])
        .expect("конфигурация");
        assert_eq!(config.max_context_tokens, Some(8000));
    }

    #[test]
    fn non_numeric_max_context_tokens_is_an_error() {
        let err = config_from(&[
            (API_KEY_VAR, "secret-key-value"),
            ("AGENTD_MAX_CONTEXT_TOKENS", "много"),
        ])
        .expect_err("ожидалась ошибка");
        assert!(format!("{err}").contains("AGENTD_MAX_CONTEXT_TOKENS"));
    }

    #[test]
    fn zero_max_context_tokens_is_an_error() {
        let err = config_from(&[
            (API_KEY_VAR, "secret-key-value"),
            ("AGENTD_MAX_CONTEXT_TOKENS", "0"),
        ])
        .expect_err("ожидалась ошибка");
        assert!(format!("{err}").contains("AGENTD_MAX_CONTEXT_TOKENS"));
    }

    #[test]
    fn db_variables_are_applied() {
        let config = config_from(&[
            (API_KEY_VAR, "secret-key-value"),
            ("AGENTD_DB_PATH", "/tmp/custom.db"),
            ("AGENTD_DB_MAX_CONNECTIONS", "3"),
            ("AGENTD_DB_BUSY_TIMEOUT_MS", "1000"),
        ])
        .expect("конфигурация");
        assert_eq!(config.db_path, "/tmp/custom.db");
        assert_eq!(config.db_max_connections, 3);
        assert_eq!(config.db_busy_timeout_ms, 1000);
    }

    #[test]
    fn non_numeric_db_max_connections_is_an_error() {
        let err = config_from(&[
            (API_KEY_VAR, "secret-key-value"),
            ("AGENTD_DB_MAX_CONNECTIONS", "много"),
        ])
        .expect_err("ожидалась ошибка");
        assert!(format!("{err}").contains("AGENTD_DB_MAX_CONNECTIONS"));
    }

    #[test]
    fn zero_db_max_connections_is_an_error() {
        let err = config_from(&[
            (API_KEY_VAR, "secret-key-value"),
            ("AGENTD_DB_MAX_CONNECTIONS", "0"),
        ])
        .expect_err("ожидалась ошибка");
        assert!(format!("{err}").contains("AGENTD_DB_MAX_CONNECTIONS"));
    }

    #[test]
    fn zero_db_busy_timeout_is_an_error() {
        let err = config_from(&[
            (API_KEY_VAR, "secret-key-value"),
            ("AGENTD_DB_BUSY_TIMEOUT_MS", "0"),
        ])
        .expect_err("ожидалась ошибка");
        assert!(format!("{err}").contains("AGENTD_DB_BUSY_TIMEOUT_MS"));
    }

    #[test]
    fn port_variable_sets_listen_addr() {
        let config =
            config_from(&[(API_KEY_VAR, "secret-key-value"), ("PORT", "9000")]).expect("конфигурация");
        assert_eq!(config.listen_addr.to_string(), "0.0.0.0:9000");
    }

    #[test]
    fn listen_addr_wins_over_port() {
        let config = config_from(&[
            (API_KEY_VAR, "secret-key-value"),
            ("PORT", "9000"),
            ("AGENTD_LISTEN_ADDR", "127.0.0.1:7000"),
        ])
        .expect("конфигурация");
        assert_eq!(config.listen_addr.to_string(), "127.0.0.1:7000");
    }

    #[test]
    fn empty_allowed_models_means_only_default_model() {
        let config = config_from(&[
            (API_KEY_VAR, "secret-key-value"),
            ("AGENTD_MODEL", "model-a"),
            ("AGENTD_ALLOWED_MODELS", ""),
        ])
        .expect("конфигурация");
        assert_eq!(config.allowed_models, vec!["model-a".to_string()]);
        assert!(config.is_model_allowed("model-a"));
        assert!(!config.is_model_allowed("model-b"));
    }

    #[test]
    fn allowed_models_are_split_by_comma() {
        let config = config_from(&[
            (API_KEY_VAR, "secret-key-value"),
            ("AGENTD_ALLOWED_MODELS", "model-a, model-b ,"),
        ])
        .expect("конфигурация");
        assert_eq!(
            config.allowed_models,
            vec!["model-a".to_string(), "model-b".to_string()]
        );
    }

    #[test]
    fn missing_api_key_is_an_error_naming_the_variable() {
        let err = config_from(&[]).expect_err("ожидалась ошибка");
        assert!(format!("{err}").contains(API_KEY_VAR), "получено: {err}");
    }

    #[test]
    fn context_strategy_defaults_to_summary() {
        let config = config_from(&[(API_KEY_VAR, "secret-key-value")]).expect("конфигурация");
        assert_eq!(config.context_strategy, ContextStrategy::Summary);
        assert!(config.allowed_context_strategies.is_empty());
        assert_eq!(config.context_window_messages, DEFAULT_CONTEXT_WINDOW_MESSAGES);
        assert_eq!(config.max_facts, DEFAULT_MAX_FACTS);
        assert_eq!(config.fact_value_max_chars, DEFAULT_FACT_VALUE_MAX_CHARS);
        assert_eq!(config.max_branch_depth, DEFAULT_MAX_BRANCH_DEPTH);
        assert!(config.is_context_strategy_allowed(ContextStrategy::Branching));
    }

    #[test]
    fn context_strategy_variables_are_applied() {
        let config = config_from(&[
            (API_KEY_VAR, "secret-key-value"),
            ("AGENTD_CONTEXT_STRATEGY", "sliding_window"),
            ("AGENTD_ALLOWED_CONTEXT_STRATEGIES", "summary, sliding_window"),
            ("AGENTD_CONTEXT_WINDOW_MESSAGES", "6"),
            ("AGENTD_MAX_FACTS", "20"),
            ("AGENTD_FACT_VALUE_MAX_CHARS", "200"),
            ("AGENTD_FACTS_MODEL", "facts-model"),
            ("AGENTD_MAX_BRANCH_DEPTH", "3"),
        ])
        .expect("конфигурация");
        assert_eq!(config.context_strategy, ContextStrategy::SlidingWindow);
        assert!(config.is_context_strategy_allowed(ContextStrategy::SlidingWindow));
        assert!(!config.is_context_strategy_allowed(ContextStrategy::Facts));
        assert_eq!(config.context_window_messages, 6);
        assert_eq!(config.max_facts, 20);
        assert_eq!(config.fact_value_max_chars, 200);
        assert_eq!(config.facts_model.as_deref(), Some("facts-model"));
        assert_eq!(config.max_branch_depth, 3);
    }

    #[test]
    fn unknown_context_strategy_default_fails_startup() {
        let err = config_from(&[
            (API_KEY_VAR, "secret-key-value"),
            ("AGENTD_CONTEXT_STRATEGY", "magic"),
        ])
        .expect_err("ожидалась ошибка");
        assert!(format!("{err}").contains("AGENTD_CONTEXT_STRATEGY"));
    }

    #[test]
    fn unknown_allowed_context_strategy_fails_startup() {
        let err = config_from(&[
            (API_KEY_VAR, "secret-key-value"),
            ("AGENTD_ALLOWED_CONTEXT_STRATEGIES", "summary,magic"),
        ])
        .expect_err("ожидалась ошибка");
        assert!(format!("{err}").contains("AGENTD_ALLOWED_CONTEXT_STRATEGIES"));
    }

    #[test]
    fn zero_context_window_messages_is_an_error() {
        let err = config_from(&[
            (API_KEY_VAR, "secret-key-value"),
            ("AGENTD_CONTEXT_WINDOW_MESSAGES", "0"),
        ])
        .expect_err("ожидалась ошибка");
        assert!(format!("{err}").contains("AGENTD_CONTEXT_WINDOW_MESSAGES"));
    }

    #[test]
    fn api_key_is_masked() {
        let config = config_from(&[(API_KEY_VAR, "sk-1234567890abcd")]).expect("конфигурация");
        let masked = config.masked_api_key();
        assert_eq!(masked, "sk-1***abcd");
        assert!(!masked.contains("234567890"));
    }
}
