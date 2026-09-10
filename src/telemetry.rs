//! Журнал сервиса: настройка подписчика и запись об одном обмене.

use crate::config::{AgentdConfig, LogFormat, LogTarget};
use tracing_subscriber::EnvFilter;

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
pub fn init(config: &AgentdConfig) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    match (config.log_format, config.log_target) {
        (LogFormat::Json, LogTarget::Stdout) => tracing_subscriber::fmt()
            .json()
            .with_env_filter(filter)
            .init(),
        (LogFormat::Json, LogTarget::Stderr) => tracing_subscriber::fmt()
            .json()
            .with_writer(std::io::stderr)
            .with_env_filter(filter)
            .init(),
        (LogFormat::Text, LogTarget::Stdout) => {
            tracing_subscriber::fmt().with_env_filter(filter).init()
        }
        (LogFormat::Text, LogTarget::Stderr) => tracing_subscriber::fmt()
            .with_writer(std::io::stderr)
            .with_env_filter(filter)
            .init(),
    }

    tracing::info!(
        listen_addr = %config.listen_addr,
        model = %config.model,
        allowed_models = ?config.allowed_models,
        api_key = %config.masked_api_key(),
        log_content = config.log_content,
        "конфигурация сервиса"
    );
    if config.client_tokens.is_empty() {
        tracing::warn!(
            "список клиентских токенов пуст: сервис принимает запросы без аутентификации"
        );
    }
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
