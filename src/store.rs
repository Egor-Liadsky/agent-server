//! Слой хранилища: чаты и сообщения в SQLite. Модуль не знает ни про axum,
//! ни про `ApiError` — отображение ошибок хранилища в HTTP живёт в
//! `src/error.rs`.

use agentcore::agent::{Message, MessageMeta, Role};
use agentcore::config::ChatSettings;
use base64::Engine;
use sha2::{Digest, Sha256};
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous};
use sqlx::{Row, SqlitePool};
use std::path::Path;
use std::str::FromStr;
use std::time::Duration;

/// Владелец анонимных чатов при выключенной аутентификации.
pub const ANONYMOUS_OWNER: &str = "anonymous";

/// Отпечаток клиентского токена, кладущийся в колонку `owner`. Токен сам по
/// себе в хранилище не попадает ни в каком виде.
pub fn owner_fingerprint(token: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"agentd-client-token:v1:");
    hasher.update(token.as_bytes());
    hex::encode(hasher.finalize())
}

/// Типизированная ошибка хранилища.
#[derive(Debug)]
pub enum StoreError {
    /// Чат не существует, или существует, но принадлежит другому владельцу:
    /// с точки зрения вызывающего кода это один и тот же исход.
    NotFound,
    /// Курсор постраничной выдачи не разобран: ошибка запроса клиента, а не
    /// хранилища.
    InvalidCursor,
    /// Отказ хранилища. Полный текст — только для журнала.
    Backend(anyhow::Error),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::NotFound => write!(f, "чат не найден"),
            StoreError::InvalidCursor => write!(f, "неизвестный курсор"),
            StoreError::Backend(err) => write!(f, "{err:#}"),
        }
    }
}

impl std::error::Error for StoreError {}

impl From<sqlx::Error> for StoreError {
    fn from(err: sqlx::Error) -> Self {
        match err {
            sqlx::Error::RowNotFound => StoreError::NotFound,
            other => StoreError::Backend(other.into()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Chat {
    pub id: String,
    pub title: String,
    pub settings: ChatSettings,
    pub created_at: i64,
    pub updated_at: i64,
    pub message_count: i64,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub seq: i64,
    pub role: Role,
    pub content: String,
    pub reasoning: Option<String>,
    pub meta: Option<MessageMeta>,
    pub created_at: i64,
}

/// Новое сообщение перед записью: без `id` и `seq` — их назначает
/// хранилище внутри транзакции.
#[derive(Debug, Clone)]
pub struct NewMessage {
    pub role: Role,
    pub content: String,
    pub reasoning: Option<String>,
    pub meta: Option<MessageMeta>,
}

impl NewMessage {
    pub fn from_message(message: Message) -> Self {
        Self {
            role: message.role,
            content: message.content,
            reasoning: message.reasoning,
            meta: message.meta,
        }
    }
}

fn role_to_str(role: Role) -> &'static str {
    match role {
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

fn role_from_str(value: &str) -> Result<Role, StoreError> {
    match value {
        "user" => Ok(Role::User),
        "assistant" => Ok(Role::Assistant),
        other => Err(StoreError::Backend(anyhow::anyhow!(
            "неизвестная роль сообщения в хранилище: {other}"
        ))),
    }
}

fn now_secs() -> i64 {
    agentcore::agent::now_secs()
}

/// Открывает пул соединений и приводит схему к актуальной. Каталог базы
/// создаётся, если его ещё нет.
pub async fn open_pool(
    db_path: &str,
    max_connections: u32,
    busy_timeout_ms: u64,
) -> anyhow::Result<SqlitePool> {
    if let Some(parent) = Path::new(db_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|err| anyhow::anyhow!("не удалось создать каталог базы {db_path}: {err}"))?;
        }
    }

    let options = SqliteConnectOptions::from_str(&format!("sqlite:{db_path}"))
        .map_err(|err| anyhow::anyhow!("не удалось разобрать путь к базе {db_path}: {err}"))?
        .create_if_missing(true)
        .journal_mode(SqliteJournalMode::Wal)
        .synchronous(SqliteSynchronous::Normal)
        .busy_timeout(Duration::from_millis(busy_timeout_ms))
        // Признак соединения, не базы: без него каскадное удаление молча
        // не работает (design.md, решение 2).
        .foreign_keys(true);

    let pool = SqlitePoolOptions::new()
        .max_connections(max_connections)
        .connect_with(options)
        .await
        .map_err(|err| anyhow::anyhow!("не удалось открыть базу {db_path}: {err}"))?;

    sqlx::migrate!()
        .run(&pool)
        .await
        .map_err(|err| anyhow::anyhow!("не удалось применить миграции к базе {db_path}: {err}"))?;

    Ok(pool)
}

/// Проверка пригодности хранилища для `GET /readyz`.
pub async fn ping(pool: &SqlitePool) -> Result<(), StoreError> {
    sqlx::query("SELECT 1").fetch_one(pool).await?;
    Ok(())
}

fn parse_settings(raw: &str, chat_id: &str, defaults: &ChatSettings) -> ChatSettings {
    match serde_json::from_str(raw) {
        Ok(settings) => settings,
        Err(err) => {
            // Несовместимое изменение ChatSettings в ядре не должно делать
            // чат недоступным (design.md, риск в конце документа).
            tracing::warn!(
                chat_id,
                error = %err,
                "не удалось разобрать сохранённые настройки чата, использованы умолчания сервиса"
            );
            defaults.clone()
        }
    }
}

fn chat_from_row(row: &sqlx::sqlite::SqliteRow, defaults: &ChatSettings) -> Result<Chat, StoreError> {
    let id: String = row.try_get("id")?;
    let settings_raw: String = row.try_get("settings")?;
    let settings = parse_settings(&settings_raw, &id, defaults);
    Ok(Chat {
        title: row.try_get("title")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
        message_count: row.try_get::<i64, _>("message_count").unwrap_or(0),
        id,
        settings,
    })
}

pub async fn create_chat(
    pool: &SqlitePool,
    owner: &str,
    title: &str,
    settings: &ChatSettings,
) -> Result<Chat, StoreError> {
    let id = uuid::Uuid::new_v4().to_string();
    let now = now_secs();
    let settings_json = serde_json::to_string(settings)
        .map_err(|err| StoreError::Backend(anyhow::anyhow!("не удалось сериализовать настройки: {err}")))?;

    sqlx::query(
        "INSERT INTO chats (id, owner, title, settings, created_at, updated_at) \
         VALUES (?, ?, ?, ?, ?, ?)",
    )
    .bind(&id)
    .bind(owner)
    .bind(title)
    .bind(&settings_json)
    .bind(now)
    .bind(now)
    .execute(pool)
    .await?;

    Ok(Chat {
        id,
        title: title.to_string(),
        settings: settings.clone(),
        created_at: now,
        updated_at: now,
        message_count: 0,
    })
}

/// Чат по владельцу и идентификатору. Владелец входит в условие выборки:
/// чужой чат и несуществующий неотличимы (design.md, решение 7).
pub async fn load_chat(pool: &SqlitePool, owner: &str, id: &str, defaults: &ChatSettings) -> Result<Chat, StoreError> {
    let row = sqlx::query(
        "SELECT c.id, c.owner, c.title, c.settings, c.created_at, c.updated_at, \
                (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id) AS message_count \
         FROM chats c WHERE c.id = ? AND c.owner = ?",
    )
    .bind(id)
    .bind(owner)
    .fetch_optional(pool)
    .await?
    .ok_or(StoreError::NotFound)?;

    chat_from_row(&row, defaults)
}

pub async fn update_chat(
    pool: &SqlitePool,
    owner: &str,
    id: &str,
    title: Option<&str>,
    settings: Option<&ChatSettings>,
    defaults: &ChatSettings,
) -> Result<Chat, StoreError> {
    let mut tx = pool.begin().await?;

    let existing = sqlx::query("SELECT settings FROM chats WHERE id = ? AND owner = ?")
        .bind(id)
        .bind(owner)
        .fetch_optional(&mut *tx)
        .await?
        .ok_or(StoreError::NotFound)?;

    let existing_raw: String = existing.try_get("settings")?;
    let existing_settings = parse_settings(&existing_raw, id, defaults);
    let new_settings = settings.cloned().unwrap_or(existing_settings);
    let settings_json = serde_json::to_string(&new_settings)
        .map_err(|err| StoreError::Backend(anyhow::anyhow!("не удалось сериализовать настройки: {err}")))?;
    let now = now_secs();

    if let Some(title) = title {
        sqlx::query("UPDATE chats SET title = ?, settings = ?, updated_at = ? WHERE id = ? AND owner = ?")
            .bind(title)
            .bind(&settings_json)
            .bind(now)
            .bind(id)
            .bind(owner)
            .execute(&mut *tx)
            .await?;
    } else {
        sqlx::query("UPDATE chats SET settings = ?, updated_at = ? WHERE id = ? AND owner = ?")
            .bind(&settings_json)
            .bind(now)
            .bind(id)
            .bind(owner)
            .execute(&mut *tx)
            .await?;
    }

    tx.commit().await?;
    load_chat(pool, owner, id, defaults).await
}

pub async fn delete_chat(pool: &SqlitePool, owner: &str, id: &str) -> Result<(), StoreError> {
    let result = sqlx::query("DELETE FROM chats WHERE id = ? AND owner = ?")
        .bind(id)
        .bind(owner)
        .execute(pool)
        .await?;
    if result.rows_affected() == 0 {
        return Err(StoreError::NotFound);
    }
    Ok(())
}

/// Непрозрачный курсор постраничной выдачи списка чатов: base64url от
/// `updated_at:id`.
fn encode_cursor(updated_at: i64, id: &str) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(format!("{updated_at}:{id}"))
}

fn decode_cursor(cursor: &str) -> Option<(i64, String)> {
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(cursor)
        .ok()?;
    let text = String::from_utf8(bytes).ok()?;
    let (updated_at, id) = text.split_once(':')?;
    Some((updated_at.parse().ok()?, id.to_string()))
}

pub struct ChatPage {
    pub chats: Vec<Chat>,
    pub next_cursor: Option<String>,
}

/// Список чатов владельца, упорядоченный `(updated_at DESC, id DESC)` —
/// тот же порядок, что и у ключа постраничной выдачи (design.md, решение 8).
pub async fn list_chats(
    pool: &SqlitePool,
    owner: &str,
    limit: u32,
    cursor: Option<&str>,
    defaults: &ChatSettings,
) -> Result<ChatPage, StoreError> {
    let after = match cursor {
        Some(cursor) => Some(decode_cursor(cursor).ok_or(StoreError::InvalidCursor)?),
        None => None,
    };

    // Запрашивается на одну строку больше лимита: если она есть, значит
    // страница не последняя, и по ней строится next_cursor.
    let fetch_limit = i64::from(limit) + 1;
    let rows = match &after {
        Some((updated_at, id)) => {
            sqlx::query(
                "SELECT c.id, c.owner, c.title, c.settings, c.created_at, c.updated_at, \
                        (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id) AS message_count \
                 FROM chats c WHERE c.owner = ? \
                   AND (c.updated_at < ? OR (c.updated_at = ? AND c.id < ?)) \
                 ORDER BY c.updated_at DESC, c.id DESC LIMIT ?",
            )
            .bind(owner)
            .bind(updated_at)
            .bind(updated_at)
            .bind(id)
            .bind(fetch_limit)
            .fetch_all(pool)
            .await?
        }
        None => {
            sqlx::query(
                "SELECT c.id, c.owner, c.title, c.settings, c.created_at, c.updated_at, \
                        (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id) AS message_count \
                 FROM chats c WHERE c.owner = ? \
                 ORDER BY c.updated_at DESC, c.id DESC LIMIT ?",
            )
            .bind(owner)
            .bind(fetch_limit)
            .fetch_all(pool)
            .await?
        }
    };

    let mut chats: Vec<Chat> = rows
        .iter()
        .map(|row| chat_from_row(row, defaults))
        .collect::<Result<_, _>>()?;

    let next_cursor = if chats.len() > limit as usize {
        chats.truncate(limit as usize);
        chats
            .last()
            .map(|chat| encode_cursor(chat.updated_at, &chat.id))
    } else {
        None
    };

    Ok(ChatPage { chats, next_cursor })
}

pub struct MessagePage {
    pub messages: Vec<ChatMessage>,
    pub next_after: Option<i64>,
}

/// Сообщения чата, отдаваемые по возрастанию `seq`. Ответственность за
/// проверку владения чатом лежит на вызывающем коде.
pub async fn load_messages(
    pool: &SqlitePool,
    chat_id: &str,
    after: i64,
    limit: u32,
) -> Result<MessagePage, StoreError> {
    let fetch_limit = i64::from(limit) + 1;
    let rows = sqlx::query(
        "SELECT seq, role, content, reasoning, meta, created_at FROM messages \
         WHERE chat_id = ? AND seq > ? ORDER BY seq ASC LIMIT ?",
    )
    .bind(chat_id)
    .bind(after)
    .bind(fetch_limit)
    .fetch_all(pool)
    .await?;

    let mut messages = Vec::with_capacity(rows.len());
    for row in &rows {
        let role: String = row.try_get("role")?;
        let meta: Option<String> = row.try_get("meta")?;
        let meta = meta
            .map(|raw| serde_json::from_str(&raw))
            .transpose()
            .map_err(|err| StoreError::Backend(anyhow::anyhow!("не удалось разобрать телеметрию сообщения: {err}")))?;
        messages.push(ChatMessage {
            seq: row.try_get("seq")?,
            role: role_from_str(&role)?,
            content: row.try_get("content")?,
            reasoning: row.try_get("reasoning")?,
            meta,
            created_at: row.try_get("created_at")?,
        });
    }

    let next_after = if messages.len() > limit as usize {
        messages.truncate(limit as usize);
        messages.last().map(|message| message.seq)
    } else {
        None
    };

    Ok(MessagePage { messages, next_after })
}

/// Записывает обмен (сообщение пользователя и ответ модели) одной
/// транзакцией `BEGIN IMMEDIATE`: либо оба сообщения появляются в чате,
/// либо чат не меняется (design.md, решения 2 и 9).
pub async fn append_exchange(
    pool: &SqlitePool,
    owner: &str,
    chat_id: &str,
    user_message: NewMessage,
    assistant_message: NewMessage,
) -> Result<(ChatMessage, ChatMessage), StoreError> {
    let mut rows = append_messages(pool, owner, chat_id, vec![user_message, assistant_message]).await?;
    let assistant_row = rows.pop().expect("две записанные реплики обмена");
    let user_row = rows.pop().expect("две записанные реплики обмена");
    Ok((user_row, assistant_row))
}

/// Дописывает готовые сообщения в конец чата одной транзакцией
/// `BEGIN IMMEDIATE`: либо в чате появляются все сообщения списка, либо ни
/// одно. Номера `seq` назначает хранилище, порядок сохраняется тот же, в
/// котором сообщения переданы. Провайдер здесь не участвует: содержимое
/// пришло готовым (specs/chat-message-api).
pub async fn append_messages(
    pool: &SqlitePool,
    owner: &str,
    chat_id: &str,
    messages: Vec<NewMessage>,
) -> Result<Vec<ChatMessage>, StoreError> {
    if messages.is_empty() {
        return Err(StoreError::Backend(anyhow::anyhow!(
            "дозапись пустого списка сообщений: вызывающий код обязан отклонить такой запрос раньше"
        )));
    }

    let mut tx = pool.begin_with("BEGIN IMMEDIATE").await?;

    let exists = sqlx::query("SELECT 1 FROM chats WHERE id = ? AND owner = ?")
        .bind(chat_id)
        .bind(owner)
        .fetch_optional(&mut *tx)
        .await?;
    if exists.is_none() {
        tx.rollback().await?;
        return Err(StoreError::NotFound);
    }

    let max_seq: i64 = sqlx::query_scalar("SELECT COALESCE(MAX(seq), 0) FROM messages WHERE chat_id = ?")
        .bind(chat_id)
        .fetch_one(&mut *tx)
        .await?;

    let now = now_secs();
    let mut rows = Vec::with_capacity(messages.len());
    for (offset, message) in messages.iter().enumerate() {
        let seq = max_seq + 1 + offset as i64;
        rows.push(insert_message(&mut *tx, chat_id, seq, message, now).await?);
    }

    sqlx::query("UPDATE chats SET updated_at = ? WHERE id = ?")
        .bind(now)
        .bind(chat_id)
        .execute(&mut *tx)
        .await?;

    tx.commit().await?;
    Ok(rows)
}

async fn insert_message(
    tx: &mut sqlx::SqliteConnection,
    chat_id: &str,
    seq: i64,
    message: &NewMessage,
    now: i64,
) -> Result<ChatMessage, StoreError> {
    let id = uuid::Uuid::new_v4().to_string();
    let meta_json = message
        .meta
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .map_err(|err| StoreError::Backend(anyhow::anyhow!("не удалось сериализовать телеметрию: {err}")))?;

    sqlx::query(
        "INSERT INTO messages (id, chat_id, seq, role, content, reasoning, meta, created_at) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(&id)
    .bind(chat_id)
    .bind(seq)
    .bind(role_to_str(message.role))
    .bind(&message.content)
    .bind(&message.reasoning)
    .bind(&meta_json)
    .bind(now)
    .execute(&mut *tx)
    .await?;

    Ok(ChatMessage {
        seq,
        role: message.role,
        content: message.content.clone(),
        reasoning: message.reasoning.clone(),
        meta: message.meta.clone(),
        created_at: now,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn temp_pool() -> (tempfile::TempDir, SqlitePool) {
        let dir = tempfile::tempdir().expect("временный каталог");
        let path = dir.path().join("agentd.db");
        let pool = open_pool(path.to_str().expect("путь"), 5, 5000)
            .await
            .expect("пул");
        (dir, pool)
    }

    fn new_message(role: Role, content: &str) -> NewMessage {
        NewMessage {
            role,
            content: content.to_string(),
            reasoning: None,
            meta: None,
        }
    }

    // --- 2.2 Открытие пула ---

    #[tokio::test]
    async fn empty_directory_creates_database_and_schema() {
        let (_dir, pool) = temp_pool().await;
        ping(&pool).await.expect("проверка живости хранилища");
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM chats")
            .fetch_one(&pool)
            .await
            .expect("таблица chats существует");
        assert_eq!(count, 0);
    }

    // --- 4.2 Непригодное хранилище ---

    #[tokio::test]
    async fn unwritable_path_fails_with_path_in_message() {
        let dir = tempfile::tempdir().expect("временный каталог");
        // Каталог, недоступный на запись: подкаталог базы в нём создать
        // нельзя, поэтому open_pool должен провалиться с путём в сообщении.
        let readonly_dir = dir.path().join("readonly");
        std::fs::create_dir(&readonly_dir).expect("подкаталог");
        let mut perms = std::fs::metadata(&readonly_dir).expect("права").permissions();
        perms.set_readonly(true);
        std::fs::set_permissions(&readonly_dir, perms.clone()).expect("выставление прав");

        let db_path = readonly_dir.join("nested").join("agentd.db");
        let db_path = db_path.to_str().expect("путь").to_string();

        let err = open_pool(&db_path, 5, 5000).await.expect_err("ожидалась ошибка");
        assert!(format!("{err:#}").contains(&db_path), "путь не найден в ошибке: {err:#}");

        perms.set_readonly(false);
        std::fs::set_permissions(&readonly_dir, perms).expect("восстановление прав");
    }

    // --- 2.3 Идемпотентность ---

    #[tokio::test]
    async fn reopening_same_database_is_idempotent() {
        let dir = tempfile::tempdir().expect("временный каталог");
        let path = dir.path().join("agentd.db");
        let path = path.to_str().expect("путь");

        let pool = open_pool(path, 5, 5000).await.expect("первый пул");
        let defaults = ChatSettings::default();
        let chat = create_chat(&pool, "owner-1", "Чат", &defaults).await.expect("чат");
        pool.close().await;

        let pool = open_pool(path, 5, 5000).await.expect("повторное открытие");
        let loaded = load_chat(&pool, "owner-1", &chat.id, &defaults)
            .await
            .expect("чат сохранился");
        assert_eq!(loaded.title, "Чат");
    }

    // --- 2.4 Каскадное удаление ---

    #[tokio::test]
    async fn cascade_delete_removes_messages() {
        let (_dir, pool) = temp_pool().await;
        let defaults = ChatSettings::default();
        let chat = create_chat(&pool, "owner-1", "Чат", &defaults).await.expect("чат");
        append_exchange(
            &pool,
            "owner-1",
            &chat.id,
            new_message(Role::User, "вопрос"),
            new_message(Role::Assistant, "ответ"),
        )
        .await
        .expect("обмен записан");

        delete_chat(&pool, "owner-1", &chat.id).await.expect("удаление");

        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM messages WHERE chat_id = ?")
            .bind(&chat.id)
            .fetch_one(&pool)
            .await
            .expect("подсчёт сообщений");
        assert_eq!(count, 0, "foreign_keys=ON должен каскадно удалить сообщения");
    }

    // --- 3.2 CRUD чата ---

    #[tokio::test]
    async fn chat_create_load_update_delete() {
        let (_dir, pool) = temp_pool().await;
        let defaults = ChatSettings::default();
        let chat = create_chat(&pool, "owner-1", "Новый чат", &defaults).await.expect("создание");
        assert_eq!(chat.message_count, 0);

        let loaded = load_chat(&pool, "owner-1", &chat.id, &defaults).await.expect("чтение");
        assert_eq!(loaded.title, "Новый чат");

        let mut settings = defaults.clone();
        settings.sampling.temperature = Some(0.2);
        let updated = update_chat(&pool, "owner-1", &chat.id, Some("Переименован"), Some(&settings), &defaults)
            .await
            .expect("изменение");
        assert_eq!(updated.title, "Переименован");
        assert_eq!(updated.settings.sampling.temperature, Some(0.2));
        assert!(updated.updated_at >= chat.updated_at);

        delete_chat(&pool, "owner-1", &chat.id).await.expect("удаление");
        let missing = load_chat(&pool, "owner-1", &chat.id, &defaults).await;
        assert!(matches!(missing, Err(StoreError::NotFound)));
    }

    // --- 3.3 Постраничный список ---

    #[tokio::test]
    async fn list_chats_paginates_without_gaps_or_duplicates() {
        let (_dir, pool) = temp_pool().await;
        let defaults = ChatSettings::default();
        for i in 0..25 {
            let chat = create_chat(&pool, "owner-1", &format!("Чат {i}"), &defaults)
                .await
                .expect("создание");
            // Одинаковый updated_at у части чатов: проверяем, что вторичный
            // ключ (id) не даёт страницам пересекаться.
            sqlx::query("UPDATE chats SET updated_at = ? WHERE id = ?")
                .bind(1000_i64 + (i / 5))
                .bind(&chat.id)
                .execute(&pool)
                .await
                .expect("выставление updated_at");
        }

        let mut seen = std::collections::HashSet::new();
        let mut cursor: Option<String> = None;
        let mut pages = 0;
        loop {
            let page = list_chats(&pool, "owner-1", 10, cursor.as_deref(), &defaults)
                .await
                .expect("страница");
            pages += 1;
            for chat in &page.chats {
                assert!(seen.insert(chat.id.clone()), "чат {} повторился", chat.id);
            }
            if page.next_cursor.is_none() {
                break;
            }
            cursor = page.next_cursor;
            assert!(pages <= 5, "постраничная выдача не завершилась");
        }
        assert_eq!(seen.len(), 25);
        assert_eq!(pages, 3);
    }

    // --- 3.4 Сообщения по частям ---

    #[tokio::test]
    async fn load_messages_paginates_by_seq() {
        let (_dir, pool) = temp_pool().await;
        let defaults = ChatSettings::default();
        let chat = create_chat(&pool, "owner-1", "Чат", &defaults).await.expect("чат");
        for i in 0..5 {
            append_exchange(
                &pool,
                "owner-1",
                &chat.id,
                new_message(Role::User, &format!("вопрос {i}")),
                new_message(Role::Assistant, &format!("ответ {i}")),
            )
            .await
            .expect("обмен");
        }

        let first = load_messages(&pool, &chat.id, 0, 4).await.expect("первая часть");
        assert_eq!(first.messages.len(), 4);
        assert_eq!(first.messages.iter().map(|m| m.seq).collect::<Vec<_>>(), vec![1, 2, 3, 4]);
        assert_eq!(first.next_after, Some(4));

        let second = load_messages(&pool, &chat.id, 4, 20)
            .await
            .expect("остаток");
        assert_eq!(second.messages.len(), 6);
        assert_eq!(second.next_after, None);
    }

    // --- 3.5 Запись обмена ---

    #[tokio::test]
    async fn append_exchange_assigns_sequential_numbers_and_round_trips_telemetry() {
        let (_dir, pool) = temp_pool().await;
        let defaults = ChatSettings::default();
        let chat = create_chat(&pool, "owner-1", "Чат", &defaults).await.expect("чат");

        let meta = MessageMeta {
            prompt_tokens: Some(11),
            completion_tokens: Some(7),
            total_tokens: Some(18),
            reasoning_tokens: Some(3),
            duration_ms: Some(120),
            sent_at: Some(1_000),
            received_at: Some(1_001),
            model: Some("model-a".to_string()),
        };
        let (user, assistant) = append_exchange(
            &pool,
            "owner-1",
            &chat.id,
            new_message(Role::User, "вопрос"),
            NewMessage {
                role: Role::Assistant,
                content: "ответ".to_string(),
                reasoning: Some("рассуждение".to_string()),
                meta: Some(meta.clone()),
            },
        )
        .await
        .expect("обмен");

        assert_eq!(user.seq, 1);
        assert_eq!(assistant.seq, 2);

        let loaded = load_messages(&pool, &chat.id, 0, 10).await.expect("чтение");
        assert_eq!(loaded.messages.len(), 2);
        let stored_assistant = &loaded.messages[1];
        assert_eq!(stored_assistant.reasoning.as_deref(), Some("рассуждение"));
        assert_eq!(stored_assistant.meta.as_ref().unwrap().total_tokens, Some(18));
    }

    // --- 1.1 Дозапись готовых сообщений ---

    #[tokio::test]
    async fn append_messages_keeps_request_order_and_lifts_chat() {
        let (_dir, pool) = temp_pool().await;
        let defaults = ChatSettings::default();
        let chat = create_chat(&pool, "owner-1", "Чат", &defaults).await.expect("чат");
        // Заведомо старое время изменения: иначе рост updated_at не отличить
        // от исходного значения — обе метки в секундах и совпадут.
        sqlx::query("UPDATE chats SET updated_at = 1000 WHERE id = ?")
            .bind(&chat.id)
            .execute(&pool)
            .await
            .expect("подготовка времени изменения");

        let meta = MessageMeta {
            completion_tokens: Some(5),
            total_tokens: Some(9),
            duration_ms: Some(42),
            model: Some("model-local".to_string()),
            ..MessageMeta::default()
        };
        let rows = append_messages(
            &pool,
            "owner-1",
            &chat.id,
            vec![
                new_message(Role::User, "вопрос"),
                NewMessage {
                    role: Role::Assistant,
                    content: "ответ".to_string(),
                    reasoning: Some("рассуждение".to_string()),
                    meta: Some(meta),
                },
            ],
        )
        .await
        .expect("дозапись");

        assert_eq!(rows.iter().map(|row| row.seq).collect::<Vec<_>>(), vec![1, 2]);

        let loaded = load_messages(&pool, &chat.id, 0, 10).await.expect("чтение");
        assert_eq!(loaded.messages[0].content, "вопрос");
        assert_eq!(loaded.messages[1].content, "ответ");
        assert_eq!(loaded.messages[1].meta.as_ref().unwrap().duration_ms, Some(42));

        let reloaded = load_chat(&pool, "owner-1", &chat.id, &defaults).await.expect("чат");
        assert_eq!(reloaded.message_count, 2);
        assert!(reloaded.updated_at > 1000);
    }

    #[tokio::test]
    async fn append_messages_after_existing_history_continues_numbering() {
        let (_dir, pool) = temp_pool().await;
        let defaults = ChatSettings::default();
        let chat = create_chat(&pool, "owner-1", "Чат", &defaults).await.expect("чат");
        append_messages(&pool, "owner-1", &chat.id, vec![new_message(Role::User, "первое")])
            .await
            .expect("первая дозапись");

        let rows = append_messages(
            &pool,
            "owner-1",
            &chat.id,
            vec![
                new_message(Role::Assistant, "второе"),
                new_message(Role::User, "третье"),
            ],
        )
        .await
        .expect("вторая дозапись");

        assert_eq!(rows.iter().map(|row| row.seq).collect::<Vec<_>>(), vec![2, 3]);
    }

    #[tokio::test]
    async fn append_messages_rolls_back_when_second_insert_fails() {
        let (_dir, pool) = temp_pool().await;
        let defaults = ChatSettings::default();
        let chat = create_chat(&pool, "owner-1", "Чат", &defaults).await.expect("чат");
        // Отказ хранилища ровно на втором сообщении списка: триггер живёт
        // только в тестовой базе и ломает вставку, когда в чате уже есть
        // одно сообщение этой же транзакции.
        sqlx::query(
            "CREATE TRIGGER fail_on_second_message BEFORE INSERT ON messages \
             WHEN (SELECT COUNT(*) FROM messages WHERE chat_id = NEW.chat_id) = 1 \
             BEGIN SELECT RAISE(ABORT, 'сбой на втором сообщении'); END",
        )
        .execute(&pool)
        .await
        .expect("создание триггера");

        let result = append_messages(
            &pool,
            "owner-1",
            &chat.id,
            vec![
                new_message(Role::User, "вопрос"),
                new_message(Role::Assistant, "ответ"),
            ],
        )
        .await;

        assert!(matches!(result, Err(StoreError::Backend(_))));

        let loaded = load_messages(&pool, &chat.id, 0, 10).await.expect("чтение");
        assert!(
            loaded.messages.is_empty(),
            "после отказа на втором сообщении в чате не должно остаться ни одного"
        );
    }

    #[tokio::test]
    async fn append_messages_to_foreign_chat_is_not_found() {
        let (_dir, pool) = temp_pool().await;
        let defaults = ChatSettings::default();
        let chat = create_chat(&pool, "owner-1", "Чат", &defaults).await.expect("чат");

        let result = append_messages(
            &pool,
            "owner-2",
            &chat.id,
            vec![new_message(Role::User, "вопрос")],
        )
        .await;

        assert!(matches!(result, Err(StoreError::NotFound)));
        let loaded = load_messages(&pool, &chat.id, 0, 10).await.expect("чтение");
        assert!(loaded.messages.is_empty());
    }

    // --- 3.6 Конкурентная запись ---

    #[tokio::test]
    async fn concurrent_append_exchange_gives_sequential_numbers() {
        let (_dir, pool) = temp_pool().await;
        let defaults = ChatSettings::default();
        let chat = create_chat(&pool, "owner-1", "Чат", &defaults).await.expect("чат");

        let pool_a = pool.clone();
        let pool_b = pool.clone();
        let chat_id_a = chat.id.clone();
        let chat_id_b = chat.id.clone();

        let a = tokio::spawn(async move {
            append_exchange(
                &pool_a,
                "owner-1",
                &chat_id_a,
                new_message(Role::User, "вопрос A"),
                new_message(Role::Assistant, "ответ A"),
            )
            .await
        });
        let b = tokio::spawn(async move {
            append_exchange(
                &pool_b,
                "owner-1",
                &chat_id_b,
                new_message(Role::User, "вопрос B"),
                new_message(Role::Assistant, "ответ B"),
            )
            .await
        });

        a.await.expect("задача A").expect("обмен A");
        b.await.expect("задача B").expect("обмен B");

        let page = load_messages(&pool, &chat.id, 0, 10).await.expect("чтение");
        let mut seqs: Vec<i64> = page.messages.iter().map(|m| m.seq).collect();
        seqs.sort_unstable();
        assert_eq!(seqs, vec![1, 2, 3, 4]);
    }

    // --- 3.7 Отпечаток владельца ---

    #[test]
    fn owner_fingerprint_is_stable_and_distinct() {
        let a = owner_fingerprint("token-a");
        let a_again = owner_fingerprint("token-a");
        let b = owner_fingerprint("token-b");
        assert_eq!(a, a_again);
        assert_ne!(a, b);
        assert_ne!(a, ANONYMOUS_OWNER);
    }
}
