mod app;
mod config;
mod dto;
mod error;
mod middleware;
mod state;
mod store;
mod telemetry;

#[cfg(test)]
mod tests;

use crate::config::AgentdConfig;
use crate::state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Ключ провайдера обязателен: без него сервис не стартует, а не отказывает
    // на первом запросе.
    let config = match AgentdConfig::from_env() {
        Ok(config) => config,
        Err(err) => {
            eprintln!("agentd: {err:#}");
            std::process::exit(1);
        }
    };
    telemetry::init(&config);

    let listen_addr = config.listen_addr;
    let state = match AppState::new(config).await {
        Ok(state) => state,
        Err(err) => {
            eprintln!("agentd: {err:#}");
            std::process::exit(1);
        }
    };

    let listener = tokio::net::TcpListener::bind(listen_addr).await?;
    tracing::info!(%listen_addr, "сервис принимает запросы");
    axum::serve(listener, app::router(state))
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    tracing::info!("сервис остановлен");
    Ok(())
}

/// Завершение по `SIGTERM` и `SIGINT`: новые соединения не принимаются,
/// уже принятые запросы доводятся до ответа.
async fn shutdown_signal() {
    let interrupt = async {
        tokio::signal::ctrl_c().await.expect("обработчик SIGINT");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("обработчик SIGTERM")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = interrupt => tracing::info!("получен SIGINT, завершаем работу"),
        _ = terminate => tracing::info!("получен SIGTERM, завершаем работу"),
    }
}
