//! Журнал сервиса: настройка подписчика и запись об одном обмене.

use crate::config::{AgentdConfig, LogFormat, LogTarget};
use agentcore::logging::{ExchangeKind, ExchangeSink};
use tracing_subscriber::fmt::writer::BoxMakeWriter;
use tracing_subscriber::EnvFilter;

/// Цели записей отладки: по ним фильтр `RUST_LOG` включает или гасит
/// отдельно обмен с клиентом и обмен с провайдером.
pub const API_TARGET: &str = "agentd::api";
pub const UPSTREAM_TARGET: &str = "agentd::upstream";

/// Человекочитаемый JSON для отладочных записей. При сбое сериализации
/// возвращается компактная запись: диагностика не должна падать.
pub fn pretty(value: &serde_json::Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}

/// Приёмник обмена с провайдером: сырые тела запроса и ответа уходят в журнал
/// сервиса. Файлов JSONL внутри контейнера нет, поэтому вместо директории
/// ядру отдаётся этот приёмник.
pub struct TracingExchangeSink;

impl ExchangeSink for TracingExchangeSink {
    fn record(&self, kind: ExchangeKind, entry: &serde_json::Value) {
        let direction = match kind {
            ExchangeKind::Request => "запрос провайдеру",
            ExchangeKind::Response => "ответ провайдера",
        };
        tracing::debug!(
            target: UPSTREAM_TARGET,
            id = entry.get("id").and_then(|v| v.as_str()).unwrap_or_default(),
            payload = %pretty(entry),
            "{direction}"
        );
    }
}

/// Что попадает в журнал по каждому запросу. Тексты промпта и ответа
/// заполняются только при включённом признаке записи содержимого.
#[derive(Debug, Default)]
pub struct ExchangeRecord<'a> {
    pub request_id: &'a str,
    pub client: &'a str,
    pub model: &'a str,
    pub duration_ms: u64,
    pub status: u16,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
    pub prompt: Option<&'a str>,
    pub response: Option<&'a str>,
}

pub fn log_exchange(record: ExchangeRecord<'_>) {
    tracing::info!(
        request_id = record.request_id,
        client = record.client,
        model = record.model,
        duration_ms = record.duration_ms,
        status = record.status,
        prompt_tokens = record.prompt_tokens,
        completion_tokens = record.completion_tokens,
        total_tokens = record.total_tokens,
        prompt = record.prompt,
        response = record.response,
        "обмен с моделью"
    );
}

/// Подписчик по конфигурации сервиса. Ключ провайдера в стартовой записи —
/// только маскированный.
///
/// Текстовый формат получает цвет и компактную раскладку (для человека в
/// терминале); JSON остаётся машинным без косметики, чтобы не ломать
/// парсинг агрегаторами логов.
pub fn init(config: &AgentdConfig) {
    // `RUST_LOG` сильнее дебага: точечную настройку уровней он не отменяет.
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(default_filter(config.debug)));
    let writer = match config.log_target {
        LogTarget::Stdout => BoxMakeWriter::new(std::io::stdout),
        LogTarget::Stderr => BoxMakeWriter::new(std::io::stderr),
    };
    let builder = tracing_subscriber::fmt().with_env_filter(filter).with_writer(writer);
    match (config.log_format, config.debug) {
        (LogFormat::Json, _) => builder.json().init(),
        // В отладке компактная раскладка не годится: она повторяет поля
        // спана в хвосте каждой записи, и многострочный JSON тонет между
        // ними. Полный формат печатает их один раз в заголовке спана.
        (LogFormat::Text, true) => builder.with_target(false).init(),
        (LogFormat::Text, false) => builder.compact().with_target(false).init(),
    }

    print_banner(config);

    tracing::info!(
        listen_addr = %config.listen_addr,
        model = %config.model,
        allowed_models = %config.allowed_models.join(", "),
        api_key = %config.masked_api_key(),
        log_content = config.log_content,
        "конфигурация сервиса"
    );
    if config.client_tokens.is_empty() {
        tracing::warn!(
            "список клиентских токенов пуст: сервис принимает запросы без аутентификации"
        );
    }
    if config.debug {
        tracing::warn!(
            "режим отладки: в журнал попадают тела запросов и ответов, включая тексты \
             промптов и ответов модели — не включать на рабочем стенде"
        );
    }
}

/// Фильтр по умолчанию. В дебаге поднимается уровень сервиса и HTTP-слоя, но
/// шумные библиотеки остаются на своих уровнях: иначе полезные записи тонут
/// в трассировке соединений и SQL.
fn default_filter(debug: bool) -> &'static str {
    if debug {
        "debug,hyper=info,hyper_util=info,reqwest=info,sqlx=warn,h2=info"
    } else {
        "info"
    }
}

const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const CYAN: &str = "\x1b[36m";
const YELLOW: &str = "\x1b[33m";
const GREEN: &str = "\x1b[32m";
const RESET: &str = "\x1b[0m";

/// Стартовый баннер для человека в терминале. Пишется отдельно от журнала
/// (`eprintln`, не `tracing`), чтобы не мешать разбору JSON-строк и не
/// зависеть от уровня фильтра. Цвет включается только на TTY.
fn print_banner(config: &AgentdConfig) {
    let color = std::io::IsTerminal::is_terminal(&std::io::stderr());
    let paint = |code: &str, text: &str| -> String {
        if color {
            format!("{code}{text}{RESET}")
        } else {
            text.to_string()
        }
    };

    let auth = if config.client_tokens.is_empty() {
        paint(YELLOW, "выключена")
    } else {
        paint(GREEN, "включена")
    };

    eprintln!();
    eprintln!(
        "  {} {}",
        paint(BOLD, "agentd"),
        paint(DIM, concat!("v", env!("CARGO_PKG_VERSION")))
    );
    eprintln!("  {} {}", paint(DIM, "адрес:"), paint(CYAN, &config.listen_addr.to_string()));
    eprintln!("  {} {}", paint(DIM, "модель:"), paint(CYAN, &config.model));
    eprintln!("  {} {auth}", paint(DIM, "аутентификация:"));
    eprintln!(
        "  {} {}/{}",
        paint(DIM, "журнал:"),
        format!("{:?}", config.log_format).to_lowercase(),
        format!("{:?}", config.log_target).to_lowercase()
    );
    if config.debug {
        eprintln!("  {} {}", paint(DIM, "режим:"), paint(YELLOW, "отладка"));
    }
    eprintln!();
}

/// Перехват журнала для тестов: подписчик пишет в буфер, а тест читает из
/// него текст записи. Живёт здесь, а не в модуле тестов, потому что тем же
/// перехватом проверяются записи обработчиков (`src/tests.rs`).
#[cfg(test)]
pub mod capture {
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Default)]
    pub struct Capture(pub Arc<Mutex<Vec<u8>>>);

    impl Capture {
        pub fn text(&self) -> String {
            let bytes = self.0.lock().expect("буфер").clone();
            String::from_utf8(bytes).expect("текст журнала")
        }
    }

    impl Write for Capture {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().expect("буфер").extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Capture {
        type Writer = Capture;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::capture::Capture;

    fn captured(record: ExchangeRecord<'_>) -> String {
        let capture = Capture::default();
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(capture.clone())
            .finish();
        tracing::subscriber::with_default(subscriber, || log_exchange(record));
        capture.text()
    }

    fn record() -> ExchangeRecord<'static> {
        ExchangeRecord {
            request_id: "req-1",
            client: "client-1",
            model: "model-a",
            duration_ms: 12,
            status: 200,
            prompt_tokens: Some(11),
            completion_tokens: Some(7),
            total_tokens: Some(18),
            prompt: None,
            response: None,
        }
    }

    #[test]
    fn telemetry_is_logged_without_content() {
        let output = captured(record());
        assert!(output.contains("req-1"), "нет идентификатора: {output}");
        assert!(output.contains("client-1"));
        assert!(output.contains("model-a"));
        assert!(output.contains("\"duration_ms\":12"));
        assert!(output.contains("\"status\":200"));
        assert!(output.contains("\"total_tokens\":18"));
        assert!(!output.contains("секретный промпт"));
    }

    #[test]
    fn content_is_logged_when_enabled() {
        let output = captured(ExchangeRecord {
            prompt: Some("секретный промпт"),
            response: Some("ответ модели"),
            ..record()
        });
        assert!(output.contains("секретный промпт"), "нет промпта: {output}");
        assert!(output.contains("ответ модели"));
    }
}
