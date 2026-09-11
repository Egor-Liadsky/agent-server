## 1. DTO: `max_context_tokens` через `double_option`

- [x] 1.1 Заменить `pub max_context_tokens: Option<u32>` на
      `#[serde(default, deserialize_with = "double_option")]
      pub max_context_tokens: Option<Option<u32>>` в `ChatSettingsDto`
      (`src/dto.rs`), обновить комментарий над полем: значение теперь
      участвует и в сохранении настроек чата, не только в разовом вызове
- [x] 1.2 В `ChatSettingsDto::apply_to` добавить перенос поля в
      `defaults.max_context_tokens` тем же способом, что параметры
      сэмплирования (`self.max_context_tokens.unwrap_or(defaults.max_context_tokens)`)

## 2. Слияние лимита при вызове модели

- [x] 2.1 В `app.rs` убрать вычисление `client_context_limit` из сырого
      `request.settings` до `merge_settings` (оба места: без `chat_id` и с
      `chat_id`); эффективный лимит `effective_context_limit` теперь
      получает `settings.max_context_tokens` — поле уже слитого результата
      `merge_settings` (после `apply_to`)
- [x] 2.2 Проверить, что порядок вычислений не ломает поведение при ошибке:
      `check_context_limit`/`effective_context_limit` вызываются после
      `merge_settings`, как и получение модели (`settings.model`) — не
      раньше

## 3. Тесты

- [x] 3.1 Тест: `PATCH /v1/chats/{id}` с `settings.max_context_tokens: N`
      сохраняет значение — последующий `GET /v1/chats/{id}` возвращает его
      в составе настроек
- [x] 3.2 Тест: `PATCH /v1/chats/{id}` с `settings.max_context_tokens: null`
      снимает ранее сохранённое значение
- [x] 3.3 Тест: `PATCH /v1/chats/{id}` без поля `max_context_tokens` не
      меняет ранее сохранённое значение (обновляется другое поле,
      например `model`)
- [x] 3.4 Тест: `POST /v1/chat` с `chat_id` без `settings.max_context_tokens`
      в запросе использует сохранённый лимит чата как эффективный (история
      чуть больше сохранённого лимита отклоняется `context_limit_exceeded`)
- [x] 3.5 Тест: `POST /v1/chat` с `chat_id` и заданным
      `settings.max_context_tokens` переопределяет сохранённый лимит чата
      для этого вызова, не меняя сохранённое значение (последующий
      `GET /v1/chats/{id}` показывает прежнее)
- [x] 3.6 Тест: сохранённый в чате лимит больше операторского даёт `400
      context_limit_invalid` при `POST /v1/chat` для этого чата (лимит не
      проверяется при `PATCH`, см. design.md Non-Goals)
- [x] 3.7 Существующие тесты `effective_context_limit`/`context-limit` в
      `src/app.rs`/`src/tests.rs` обновить под новую сигнатуру источника
      клиентского значения там, где они опирались на прежнее одноуровневое
      поведение

## 4. Документация и проверка

- [x] 4.1 Обновить README: `PATCH /v1/chats/{id}` — `max_context_tokens`
      сохраняется в настройках чата (число задаёт, `null` снимает,
      отсутствие поля не трогает); `POST /v1/chat` — поле остаётся разовым
      переопределением поверх сохранённого лимита чата
- [x] 4.2 Прогнать `cargo test` в `agent-sever` целиком (локально —
      с активным `[patch]` на рабочую копию `agent-cli/crates/core`, пока
      клиентское изменение `scope-context-limit-to-chat` не запушено в
      `main`; после пуша — снять `[patch]`, `cargo generate-lockfile`,
      `cargo test --locked`, см.
      `agent-cli/openspec/changes/scope-context-limit-to-chat/design.md`,
      Migration Plan)
