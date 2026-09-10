# Сборка. Ядро `agentcore` тянется git-зависимостью, поэтому контекст сборки
# не выходит за пределы репозитория сервиса.
FROM rust:1.97-slim-bookworm AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config libssl-dev git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY migrations ./migrations
COPY src ./src
RUN cargo build --release --locked

# Рантайм без инструментов сборки.
FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --uid 10001 agentd \
    && mkdir -p /var/lib/agentd \
    && chown agentd:agentd /var/lib/agentd

COPY --from=builder /build/target/release/agentd /usr/local/bin/agentd

USER agentd
ENV PORT=8080
# Файл базы и его WAL-спутники (-wal, -shm) живут в томе, а не в слое
# образа: без тома они исчезали бы вместе с пересозданным контейнером
# (design.md, решение 11).
ENV AGENTD_DB_PATH=/var/lib/agentd/agentd.db
VOLUME /var/lib/agentd
EXPOSE 8080

# Проверка живости: порт берётся из того же окружения, что и у сервиса.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/healthz" || exit 1

ENTRYPOINT ["/usr/local/bin/agentd"]
