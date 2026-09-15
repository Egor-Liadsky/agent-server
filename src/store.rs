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
    /// Ветка, активная для новых сообщений и умалчиваемого чтения истории.
    pub active_branch: String,
    /// Задача, к которой привязана рабочая память чата (design.md, решение 2).
    /// По умолчанию — идентификатор самого чата.
    pub active_task_id: String,
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
        // Системные сообщения не записываются: они синтезируются на каждый
        // запрос (design.md, решение 4). Ветка нужна только для
        // исчерпывающего match.
        Role::System => "system",
    }
}

/// `"system"` разбирается наравне с `"user"`/`"assistant"`, хотя
/// системные сообщения хранилище никогда не пишет: чтение не должно
/// падать, если строка с такой ролью всё же появится в базе
/// (design.md, решение 4, «чтобы чтение не падало на данных из
/// будущего»).
fn role_from_str(value: &str) -> Result<Role, StoreError> {
    match value {
        "user" => Ok(Role::User),
        "assistant" => Ok(Role::Assistant),
        "system" => Ok(Role::System),
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
        active_branch: row.try_get("active_branch")?,
        active_task_id: row.try_get("active_task_id")?,
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

    let mut tx = pool.begin().await?;

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
    .execute(&mut *tx)
    .await?;

    // Каждый чат сразу получает корневую ветку и её же — активной
    // (specs/chat-branching, «У чата есть ветки и активная ветка»).
    let root_branch = insert_root_branch(&mut tx, &id, now).await?;
    // Задача рабочей памяти по умолчанию — сам чат (design.md, решение 2):
    // рабочая память доступна без обязательного явного заведения задачи.
    sqlx::query("UPDATE chats SET active_branch = ?, active_task_id = ? WHERE id = ?")
        .bind(&root_branch)
        .bind(&id)
        .bind(&id)
        .execute(&mut *tx)
        .await?;

    tx.commit().await?;

    Ok(Chat {
        id: id.clone(),
        title: title.to_string(),
        settings: settings.clone(),
        created_at: now,
        updated_at: now,
        message_count: 0,
        active_branch: root_branch,
        active_task_id: id,
    })
}

async fn insert_root_branch(
    tx: &mut sqlx::SqliteConnection,
    chat_id: &str,
    now: i64,
) -> Result<String, StoreError> {
    let branch_id = uuid::Uuid::new_v4().to_string();
    sqlx::query(
        "INSERT INTO chat_branches (id, chat_id, parent_id, fork_seq, name, created_at) \
         VALUES (?, ?, NULL, NULL, 'root', ?)",
    )
    .bind(&branch_id)
    .bind(chat_id)
    .bind(now)
    .execute(&mut *tx)
    .await?;
    Ok(branch_id)
}

/// Чат по владельцу и идентификатору. Владелец входит в условие выборки:
/// чужой чат и несуществующий неотличимы (design.md, решение 7).
pub async fn load_chat(pool: &SqlitePool, owner: &str, id: &str, defaults: &ChatSettings) -> Result<Chat, StoreError> {
    let row = sqlx::query(
        "SELECT c.id, c.owner, c.title, c.settings, c.created_at, c.updated_at, c.active_branch, c.active_task_id, \
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

/// Обновляет название чата, только если оно всё ещё равно `expected` —
/// закрывает гонку с ручным `PATCH /v1/chats/{id}` одним запросом: если
/// клиент успел переименовать чат, условие не выполнится и сгенерированное
/// название будет отброшено (specs/chat-title, design.md, решение 6).
/// Возвращает `true`, если строка действительно обновилась.
pub async fn set_title_if_default(
    pool: &SqlitePool,
    owner: &str,
    id: &str,
    title: &str,
    expected: &str,
) -> Result<bool, StoreError> {
    let now = now_secs();
    let result = sqlx::query(
        "UPDATE chats SET title = ?, updated_at = ? WHERE id = ? AND owner = ? AND title = ?",
    )
    .bind(title)
    .bind(now)
    .bind(id)
    .bind(owner)
    .bind(expected)
    .execute(pool)
    .await?;
    Ok(result.rows_affected() > 0)
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
                "SELECT c.id, c.owner, c.title, c.settings, c.created_at, c.updated_at, c.active_branch, c.active_task_id, \
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
                "SELECT c.id, c.owner, c.title, c.settings, c.created_at, c.updated_at, c.active_branch, c.active_task_id, \
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
    /// Ветка, чья история отдана.
    pub branch_id: String,
}

/// Ветка чата с числом собственных сообщений (специфика хранилища —
/// `store::Branch` отличается от одноимённого типа в `agentclient`).
#[derive(Debug, Clone)]
pub struct Branch {
    pub id: String,
    pub parent_id: Option<String>,
    pub fork_seq: Option<i64>,
    pub name: String,
    pub message_count: i64,
    pub active: bool,
}

/// Одна ветка цепочки родителей — без числа сообщений: используется только
/// для сборки истории, а не для показа списка веток.
struct BranchLink {
    id: String,
    parent_id: Option<String>,
    fork_seq: Option<i64>,
}

async fn load_branch_link(
    pool: &SqlitePool,
    chat_id: &str,
    branch_id: &str,
) -> Result<BranchLink, StoreError> {
    let row = sqlx::query("SELECT id, parent_id, fork_seq FROM chat_branches WHERE id = ? AND chat_id = ?")
        .bind(branch_id)
        .bind(chat_id)
        .fetch_optional(pool)
        .await?
        .ok_or(StoreError::NotFound)?;
    Ok(BranchLink {
        id: row.try_get("id")?,
        parent_id: row.try_get("parent_id")?,
        fork_seq: row.try_get("fork_seq")?,
    })
}

/// Цепочка веток от корня до `branch_id` включительно. Ограничена
/// `max_depth`: длиннее — фатальная ошибка хранилища, а не бесконечный
/// подъём по `parent_id` (specs/context-strategies, design.md решение 4).
async fn branch_chain(
    pool: &SqlitePool,
    chat_id: &str,
    branch_id: &str,
    max_depth: u32,
) -> Result<Vec<BranchLink>, StoreError> {
    let mut chain = vec![load_branch_link(pool, chat_id, branch_id).await?];
    loop {
        let parent_id = match &chain.last().expect("цепочка не пуста").parent_id {
            Some(id) => id.clone(),
            None => break,
        };
        if chain.len() as u32 >= max_depth {
            return Err(StoreError::Backend(anyhow::anyhow!(
                "цепочка веток чата {chat_id} длиннее операторского потолка {max_depth}"
            )));
        }
        chain.push(load_branch_link(pool, chat_id, &parent_id).await?);
    }
    chain.reverse();
    Ok(chain)
}

async fn branch_messages_upto(
    pool: &SqlitePool,
    branch_id: &str,
    upto_seq: Option<i64>,
) -> Result<Vec<ChatMessage>, StoreError> {
    let rows = match upto_seq {
        Some(upto) => {
            sqlx::query(
                "SELECT seq, role, content, reasoning, meta, created_at FROM messages \
                 WHERE branch_id = ? AND seq <= ? ORDER BY seq ASC",
            )
            .bind(branch_id)
            .bind(upto)
            .fetch_all(pool)
            .await?
        }
        None => {
            sqlx::query(
                "SELECT seq, role, content, reasoning, meta, created_at FROM messages \
                 WHERE branch_id = ? ORDER BY seq ASC",
            )
            .bind(branch_id)
            .fetch_all(pool)
            .await?
        }
    };
    rows_to_messages(&rows)
}

fn rows_to_messages(rows: &[sqlx::sqlite::SqliteRow]) -> Result<Vec<ChatMessage>, StoreError> {
    let mut messages = Vec::with_capacity(rows.len());
    for row in rows {
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
    Ok(messages)
}

/// История ветки: сообщения родительских веток до их точек ветвления, затем
/// собственные сообщения ветки — без пагинации (specs/chat-branching,
/// «История собирается по цепочке ветки»). Используется и для чтения чата
/// клиентом, и для сборки контекста стратегией `branching`.
pub async fn load_branch_history(
    pool: &SqlitePool,
    chat_id: &str,
    branch_id: &str,
    max_depth: u32,
) -> Result<Vec<ChatMessage>, StoreError> {
    let chain = branch_chain(pool, chat_id, branch_id, max_depth).await?;
    let mut messages = Vec::new();
    for (index, link) in chain.iter().enumerate() {
        let upto = chain.get(index + 1).and_then(|child| child.fork_seq);
        // Только последнее звено (сама ветка) не режется по fork_seq
        // потомка — им и заканчивается цепочка.
        let upto = if index + 1 == chain.len() { None } else { upto };
        messages.extend(branch_messages_upto(pool, &link.id, upto).await?);
    }
    Ok(messages)
}

/// Сообщения чата, отдаваемые по возрастанию `seq`. `branch_id` — явно
/// запрошенная ветка, `None` — активная ветка чата
/// (specs/chat-branching, «Чтение истории учитывает ветку»). Ветка без
/// родителей (обычный случай — единственная ветка чата) читается постранично
/// как раньше; ветка с предками отдаётся целиком одной страницей — сценарии
/// ветвления короткие, а постраничный курсор через разные пространства
/// `seq` родителя и потомка усложнил бы контракт без практической пользы
/// (design.md, «Non-Goals»).
pub async fn load_messages(
    pool: &SqlitePool,
    chat_id: &str,
    after: i64,
    limit: u32,
    branch_id: Option<&str>,
    max_branch_depth: u32,
) -> Result<MessagePage, StoreError> {
    let target_branch = match branch_id {
        Some(id) => id.to_string(),
        None => active_branch_of(pool, chat_id).await?,
    };
    let link = load_branch_link(pool, chat_id, &target_branch).await?;

    if link.parent_id.is_none() {
        // Ветка без родителей: обычная постраничная выдача по seq, как до
        // появления веток.
        let fetch_limit = i64::from(limit) + 1;
        let rows = sqlx::query(
            "SELECT seq, role, content, reasoning, meta, created_at FROM messages \
             WHERE branch_id = ? AND seq > ? ORDER BY seq ASC LIMIT ?",
        )
        .bind(&target_branch)
        .bind(after)
        .bind(fetch_limit)
        .fetch_all(pool)
        .await?;

        let mut messages = rows_to_messages(&rows)?;
        let next_after = if messages.len() > limit as usize {
            messages.truncate(limit as usize);
            messages.last().map(|message| message.seq)
        } else {
            None
        };

        return Ok(MessagePage {
            messages,
            next_after,
            branch_id: target_branch,
        });
    }

    let messages = load_branch_history(pool, chat_id, &target_branch, max_branch_depth).await?;
    Ok(MessagePage {
        messages,
        next_after: None,
        branch_id: target_branch,
    })
}

async fn active_branch_of(pool: &SqlitePool, chat_id: &str) -> Result<String, StoreError> {
    sqlx::query_scalar("SELECT active_branch FROM chats WHERE id = ?")
        .bind(chat_id)
        .fetch_optional(pool)
        .await?
        .ok_or(StoreError::NotFound)
}

/// Ветки чата с числом собственных сообщений и признаком активной
/// (specs/chat-branching, «Ветки перечисляются и переключаются»).
pub async fn list_branches(pool: &SqlitePool, owner: &str, chat_id: &str) -> Result<Vec<Branch>, StoreError> {
    let active_branch: Option<String> =
        sqlx::query_scalar("SELECT active_branch FROM chats WHERE id = ? AND owner = ?")
            .bind(chat_id)
            .bind(owner)
            .fetch_optional(pool)
            .await?;
    let active_branch = active_branch.ok_or(StoreError::NotFound)?;

    let rows = sqlx::query(
        "SELECT b.id, b.chat_id, b.parent_id, b.fork_seq, b.name, b.created_at, \
                (SELECT COUNT(*) FROM messages m WHERE m.branch_id = b.id) AS message_count \
         FROM chat_branches b WHERE b.chat_id = ? ORDER BY b.created_at ASC",
    )
    .bind(chat_id)
    .fetch_all(pool)
    .await?;

    rows.iter()
        .map(|row| {
            let id: String = row.try_get("id")?;
            Ok(Branch {
                active: id == active_branch,
                id,
                parent_id: row.try_get("parent_id")?,
                fork_seq: row.try_get("fork_seq")?,
                name: row.try_get("name")?,
                message_count: row.try_get("message_count")?,
            })
        })
        .collect()
}

/// Ветка от указанного сообщения активной ветки чата — точки ветвления.
/// Сообщение, которого нет в активной ветке, — `StoreError::NotFound`
/// (specs/chat-branching, «Ветка создаётся от выбранного сообщения»).
pub async fn create_branch(
    pool: &SqlitePool,
    owner: &str,
    chat_id: &str,
    from_seq: i64,
    name: &str,
) -> Result<Branch, StoreError> {
    let mut tx = pool.begin().await?;

    let active_branch: Option<String> =
        sqlx::query_scalar("SELECT active_branch FROM chats WHERE id = ? AND owner = ?")
            .bind(chat_id)
            .bind(owner)
            .fetch_optional(&mut *tx)
            .await?;
    let active_branch = match active_branch {
        Some(branch) => branch,
        None => {
            tx.rollback().await?;
            return Err(StoreError::NotFound);
        }
    };

    let fork_point_exists: Option<i64> = sqlx::query_scalar(
        "SELECT 1 FROM messages WHERE branch_id = ? AND seq = ?",
    )
    .bind(&active_branch)
    .bind(from_seq)
    .fetch_optional(&mut *tx)
    .await?;
    if fork_point_exists.is_none() {
        tx.rollback().await?;
        return Err(StoreError::NotFound);
    }

    let id = uuid::Uuid::new_v4().to_string();
    let now = now_secs();
    sqlx::query(
        "INSERT INTO chat_branches (id, chat_id, parent_id, fork_seq, name, created_at) \
         VALUES (?, ?, ?, ?, ?, ?)",
    )
    .bind(&id)
    .bind(chat_id)
    .bind(&active_branch)
    .bind(from_seq)
    .bind(name)
    .bind(now)
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;

    Ok(Branch {
        id,
        parent_id: Some(active_branch),
        fork_seq: Some(from_seq),
        name: name.to_string(),
        message_count: 0,
        active: false,
    })
}

/// Переключает активную ветку чата. Ветка чужого чата или отсутствующая
/// ветка — `StoreError::NotFound`.
pub async fn activate_branch(
    pool: &SqlitePool,
    owner: &str,
    chat_id: &str,
    branch_id: &str,
) -> Result<(), StoreError> {
    let owns_chat: Option<i64> = sqlx::query_scalar("SELECT 1 FROM chats WHERE id = ? AND owner = ?")
        .bind(chat_id)
        .bind(owner)
        .fetch_optional(pool)
        .await?;
    if owns_chat.is_none() {
        return Err(StoreError::NotFound);
    }
    let branch_exists: Option<i64> =
        sqlx::query_scalar("SELECT 1 FROM chat_branches WHERE id = ? AND chat_id = ?")
            .bind(branch_id)
            .bind(chat_id)
            .fetch_optional(pool)
            .await?;
    if branch_exists.is_none() {
        return Err(StoreError::NotFound);
    }

    sqlx::query("UPDATE chats SET active_branch = ?, updated_at = ? WHERE id = ?")
        .bind(branch_id)
        .bind(now_secs())
        .bind(chat_id)
        .execute(pool)
        .await?;
    Ok(())
}

/// Факт чата — пара «ключ-значение» стратегии `facts`
/// (specs/context-facts, «Факты хранятся отдельно от сообщений»).
#[derive(Debug, Clone)]
pub struct Fact {
    pub key: String,
    pub value: String,
    pub through_seq: i64,
    pub updated_at: i64,
}

/// Все факты чата, в порядке ключа — детерминированный порядок важен для
/// сборки блока фактов в промпте (specs/context-facts).
pub async fn load_facts(pool: &SqlitePool, chat_id: &str) -> Result<Vec<Fact>, StoreError> {
    let rows = sqlx::query(
        "SELECT key, value, through_seq, updated_at FROM chat_facts WHERE chat_id = ? ORDER BY key ASC",
    )
    .bind(chat_id)
    .fetch_all(pool)
    .await?;
    rows.iter()
        .map(|row| {
            Ok(Fact {
                key: row.try_get("key")?,
                value: row.try_get("value")?,
                through_seq: row.try_get("through_seq")?,
                updated_at: row.try_get("updated_at")?,
            })
        })
        .collect()
}

/// Задаёт значение факта, заменяя прежнее у существующего ключа
/// (specs/context-facts, «Факты читаются и правятся вручную»).
pub async fn set_fact(
    pool: &SqlitePool,
    chat_id: &str,
    key: &str,
    value: &str,
    through_seq: i64,
) -> Result<Fact, StoreError> {
    let now = now_secs();
    sqlx::query(
        "INSERT INTO chat_facts (chat_id, key, value, through_seq, updated_at) \
         VALUES (?, ?, ?, ?, ?) \
         ON CONFLICT (chat_id, key) DO UPDATE SET \
             value = excluded.value, through_seq = excluded.through_seq, updated_at = excluded.updated_at",
    )
    .bind(chat_id)
    .bind(key)
    .bind(value)
    .bind(through_seq)
    .bind(now)
    .execute(pool)
    .await?;
    Ok(Fact {
        key: key.to_string(),
        value: value.to_string(),
        through_seq,
        updated_at: now,
    })
}

/// Удаляет факт по ключу. Отсутствующий ключ — `StoreError::NotFound`.
pub async fn delete_fact(pool: &SqlitePool, chat_id: &str, key: &str) -> Result<(), StoreError> {
    let result = sqlx::query("DELETE FROM chat_facts WHERE chat_id = ? AND key = ?")
        .bind(chat_id)
        .bind(key)
        .execute(pool)
        .await?;
    if result.rows_affected() == 0 {
        return Err(StoreError::NotFound);
    }
    Ok(())
}

// --- Память: рабочая (чат + задача) и долговременная (владелец) ---
// (design.md решения 1, 2, 4)

#[derive(Debug, Clone)]
pub struct WorkingMemoryEntry {
    pub key: String,
    pub value: String,
    pub source: String,
    pub updated_at: i64,
}

/// Записи рабочей памяти активной задачи чата, в порядке ключа —
/// детерминированный порядок важен для сборки раздела в системном
/// сообщении (specs/memory-layers).
pub async fn load_working_memory(
    pool: &SqlitePool,
    chat_id: &str,
    task_id: &str,
) -> Result<Vec<WorkingMemoryEntry>, StoreError> {
    let rows = sqlx::query(
        "SELECT key, value, source, updated_at FROM chat_working_memory \
         WHERE chat_id = ? AND task_id = ? ORDER BY key ASC",
    )
    .bind(chat_id)
    .bind(task_id)
    .fetch_all(pool)
    .await?;
    rows.iter()
        .map(|row| {
            Ok(WorkingMemoryEntry {
                key: row.try_get("key")?,
                value: row.try_get("value")?,
                source: row.try_get("source")?,
                updated_at: row.try_get("updated_at")?,
            })
        })
        .collect()
}

/// Задаёт значение записи рабочей памяти. Побеждает более позднее
/// `updated_at`, независимо от источника (design.md, решение 4): при
/// конфликте по `(chat_id, task_id, key)` обновление применяется, только
/// если `updated_at` не раньше уже сохранённого. Возвращает `true`, если
/// запись действительно применилась.
pub async fn set_working_memory(
    pool: &SqlitePool,
    chat_id: &str,
    task_id: &str,
    key: &str,
    value: &str,
    source: &str,
    updated_at: i64,
) -> Result<bool, StoreError> {
    let result = sqlx::query(
        "INSERT INTO chat_working_memory (id, chat_id, task_id, key, value, source, updated_at) \
         VALUES (?, ?, ?, ?, ?, ?, ?) \
         ON CONFLICT (chat_id, task_id, key) DO UPDATE SET \
             value = excluded.value, source = excluded.source, updated_at = excluded.updated_at \
         WHERE excluded.updated_at >= chat_working_memory.updated_at",
    )
    .bind(uuid::Uuid::new_v4().to_string())
    .bind(chat_id)
    .bind(task_id)
    .bind(key)
    .bind(value)
    .bind(source)
    .bind(updated_at)
    .execute(pool)
    .await?;
    Ok(result.rows_affected() > 0)
}

pub async fn delete_working_memory(
    pool: &SqlitePool,
    chat_id: &str,
    task_id: &str,
    key: &str,
) -> Result<(), StoreError> {
    let result = sqlx::query("DELETE FROM chat_working_memory WHERE chat_id = ? AND task_id = ? AND key = ?")
        .bind(chat_id)
        .bind(task_id)
        .bind(key)
        .execute(pool)
        .await?;
    if result.rows_affected() == 0 {
        return Err(StoreError::NotFound);
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct LongTermMemoryEntry {
    pub id: String,
    pub entry_type: String,
    pub key: Option<String>,
    pub value: String,
    pub source: String,
    pub source_chat_id: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

/// Записи долговременной памяти владельца, самые свежие первыми —
/// стратегия `memory_layers` вытесняет по `updated_at` (design.md,
/// решение 5), поэтому порядок уже готов под усечение по лимиту вызывающим
/// кодом.
pub async fn load_long_term_memory(
    pool: &SqlitePool,
    owner: &str,
    limit: u32,
) -> Result<Vec<LongTermMemoryEntry>, StoreError> {
    let rows = sqlx::query(
        "SELECT id, entry_type, key, value, source, source_chat_id, created_at, updated_at \
         FROM owner_long_term_memory WHERE owner = ? ORDER BY updated_at DESC LIMIT ?",
    )
    .bind(owner)
    .bind(limit)
    .fetch_all(pool)
    .await?;
    rows.iter()
        .map(|row| {
            Ok(LongTermMemoryEntry {
                id: row.try_get("id")?,
                entry_type: row.try_get("entry_type")?,
                key: row.try_get("key")?,
                value: row.try_get("value")?,
                source: row.try_get("source")?,
                source_chat_id: row.try_get("source_chat_id")?,
                created_at: row.try_get("created_at")?,
                updated_at: row.try_get("updated_at")?,
            })
        })
        .collect()
}

/// Задаёт запись долговременной памяти. С ключом — тот же приём
/// приоритета «побеждает более позднее `updated_at`», что и у рабочей
/// памяти (design.md, решение 4), конфликт — по `(owner, entry_type, key)`.
/// Без ключа (`key: None`, свободная заметка) — всегда новая запись: индекс
/// уникальности на такие строки не распространяется. Возвращает
/// `(id, применилась ли запись)`.
pub async fn set_long_term_memory(
    pool: &SqlitePool,
    owner: &str,
    entry_type: &str,
    key: Option<&str>,
    value: &str,
    source: &str,
    source_chat_id: Option<&str>,
    updated_at: i64,
) -> Result<(String, bool), StoreError> {
    match key {
        Some(key) => {
            let id = uuid::Uuid::new_v4().to_string();
            let result = sqlx::query(
                "INSERT INTO owner_long_term_memory \
                     (id, owner, entry_type, key, value, source, source_chat_id, created_at, updated_at) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) \
                 ON CONFLICT (owner, entry_type, key) WHERE key IS NOT NULL DO UPDATE SET \
                     value = excluded.value, source = excluded.source, \
                     source_chat_id = excluded.source_chat_id, updated_at = excluded.updated_at \
                 WHERE excluded.updated_at >= owner_long_term_memory.updated_at",
            )
            .bind(&id)
            .bind(owner)
            .bind(entry_type)
            .bind(key)
            .bind(value)
            .bind(source)
            .bind(source_chat_id)
            .bind(updated_at)
            .bind(updated_at)
            .execute(pool)
            .await?;
            if result.rows_affected() == 0 {
                // Конфликт проигран прежней записи — вернуть её существующий id,
                // а не свежесгенерированный, которым ничего не записано.
                let existing_id: String = sqlx::query_scalar(
                    "SELECT id FROM owner_long_term_memory WHERE owner = ? AND entry_type = ? AND key = ?",
                )
                .bind(owner)
                .bind(entry_type)
                .bind(key)
                .fetch_one(pool)
                .await?;
                return Ok((existing_id, false));
            }
            Ok((id, true))
        }
        None => {
            let id = uuid::Uuid::new_v4().to_string();
            sqlx::query(
                "INSERT INTO owner_long_term_memory \
                     (id, owner, entry_type, key, value, source, source_chat_id, created_at, updated_at) \
                 VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?)",
            )
            .bind(&id)
            .bind(owner)
            .bind(entry_type)
            .bind(value)
            .bind(source)
            .bind(source_chat_id)
            .bind(updated_at)
            .bind(updated_at)
            .execute(pool)
            .await?;
            Ok((id, true))
        }
    }
}

pub async fn delete_long_term_memory(pool: &SqlitePool, owner: &str, id: &str) -> Result<(), StoreError> {
    let result = sqlx::query("DELETE FROM owner_long_term_memory WHERE owner = ? AND id = ?")
        .bind(owner)
        .bind(id)
        .execute(pool)
        .await?;
    if result.rows_affected() == 0 {
        return Err(StoreError::NotFound);
    }
    Ok(())
}

/// Удаляет запись долговременной памяти по ключу — путь маршрутизатора,
/// операция `delete` которого не знает `entry_type`/`id` записи, только
/// `key` (specs/memory-layers).
pub async fn delete_long_term_memory_by_key(pool: &SqlitePool, owner: &str, key: &str) -> Result<(), StoreError> {
    let result = sqlx::query("DELETE FROM owner_long_term_memory WHERE owner = ? AND key = ?")
        .bind(owner)
        .bind(key)
        .execute(pool)
        .await?;
    if result.rows_affected() == 0 {
        return Err(StoreError::NotFound);
    }
    Ok(())
}

/// Завершает активную задачу чата: записи, отмеченные `carry_forward_keys`,
/// переносятся в долговременную память ДО удаления рабочей памяти прежней
/// задачи (specs/memory-layers, «Перенос отмеченной записи в долговременную
/// память при завершении задачи»), затем чат получает новую случайную
/// задачу. Возвращает перенесённые записи.
pub async fn finish_task(
    pool: &SqlitePool,
    owner: &str,
    chat_id: &str,
    carry_forward_keys: &[String],
) -> Result<Vec<LongTermMemoryEntry>, StoreError> {
    let mut tx = pool.begin().await?;
    let task_id: String = sqlx::query_scalar("SELECT active_task_id FROM chats WHERE id = ? AND owner = ?")
        .bind(chat_id)
        .bind(owner)
        .fetch_optional(&mut *tx)
        .await?
        .ok_or(StoreError::NotFound)?;

    let now = now_secs();
    let mut transferred = Vec::new();
    for key in carry_forward_keys {
        let row = sqlx::query(
            "SELECT value FROM chat_working_memory WHERE chat_id = ? AND task_id = ? AND key = ?",
        )
        .bind(chat_id)
        .bind(&task_id)
        .bind(key)
        .fetch_optional(&mut *tx)
        .await?;
        let Some(row) = row else { continue };
        let value: String = row.try_get("value")?;
        let id = uuid::Uuid::new_v4().to_string();
        sqlx::query(
            "INSERT INTO owner_long_term_memory \
                 (id, owner, entry_type, key, value, source, source_chat_id, created_at, updated_at) \
             VALUES (?, ?, 'knowledge', ?, ?, 'router', ?, ?, ?) \
             ON CONFLICT (owner, entry_type, key) WHERE key IS NOT NULL DO UPDATE SET \
                 value = excluded.value, source = excluded.source, \
                 source_chat_id = excluded.source_chat_id, updated_at = excluded.updated_at",
        )
        .bind(&id)
        .bind(owner)
        .bind(key)
        .bind(&value)
        .bind(chat_id)
        .bind(now)
        .bind(now)
        .execute(&mut *tx)
        .await?;
        transferred.push(LongTermMemoryEntry {
            id,
            entry_type: "knowledge".to_string(),
            key: Some(key.clone()),
            value,
            source: "router".to_string(),
            source_chat_id: Some(chat_id.to_string()),
            created_at: now,
            updated_at: now,
        });
    }

    sqlx::query("DELETE FROM chat_working_memory WHERE chat_id = ? AND task_id = ?")
        .bind(chat_id)
        .bind(&task_id)
        .execute(&mut *tx)
        .await?;

    let new_task_id = uuid::Uuid::new_v4().to_string();
    sqlx::query("UPDATE chats SET active_task_id = ? WHERE id = ?")
        .bind(&new_task_id)
        .bind(chat_id)
        .execute(&mut *tx)
        .await?;

    tx.commit().await?;
    Ok(transferred)
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

    let active_branch: Option<String> = sqlx::query_scalar(
        "SELECT active_branch FROM chats WHERE id = ? AND owner = ?",
    )
    .bind(chat_id)
    .bind(owner)
    .fetch_optional(&mut *tx)
    .await?;
    let active_branch = match active_branch {
        Some(branch) => branch,
        None => {
            tx.rollback().await?;
            return Err(StoreError::NotFound);
        }
    };

    // seq растёт внутри ветки, а не всего чата: у новой ветки собственная
    // нумерация с единицы (design.md, решение 3).
    let max_seq: i64 = sqlx::query_scalar("SELECT COALESCE(MAX(seq), 0) FROM messages WHERE branch_id = ?")
        .bind(&active_branch)
        .fetch_one(&mut *tx)
        .await?;

    let now = now_secs();
    let mut rows = Vec::with_capacity(messages.len());
    for (offset, message) in messages.iter().enumerate() {
        let seq = max_seq + 1 + offset as i64;
        rows.push(insert_message(&mut *tx, chat_id, &active_branch, seq, message, now).await?);
    }

    sqlx::query("UPDATE chats SET updated_at = ? WHERE id = ?")
        .bind(now)
        .bind(chat_id)
        .execute(&mut *tx)
        .await?;

    tx.commit().await?;
    Ok(rows)
}

/// Пересказ вытесненной части истории чата, хранимый отдельно от сообщений
/// (design.md, решение 4).
#[derive(Debug, Clone)]
pub struct ChatSummary {
    pub summary: String,
    pub through_seq: i64,
}

/// Пересказ чата, если он уже построен.
pub async fn load_summary(pool: &SqlitePool, chat_id: &str) -> Result<Option<ChatSummary>, StoreError> {
    let row = sqlx::query("SELECT summary, through_seq FROM chat_summaries WHERE chat_id = ?")
        .bind(chat_id)
        .fetch_optional(pool)
        .await?;

    Ok(match row {
        Some(row) => Some(ChatSummary {
            summary: row.try_get("summary")?,
            through_seq: row.try_get("through_seq")?,
        }),
        None => None,
    })
}

/// Сохраняет пересказ чата, заменяя прежний. `through_seq` монотонно не
/// убывает: конкурентная запись худшим исходом даёт лишний вызов модели, а
/// не откат границы назад (design.md, риск «Параллельные запросы»).
pub async fn save_summary(
    pool: &SqlitePool,
    chat_id: &str,
    summary: &str,
    through_seq: i64,
) -> Result<(), StoreError> {
    let now = now_secs();
    sqlx::query(
        "INSERT INTO chat_summaries (chat_id, summary, through_seq, updated_at) \
         VALUES (?, ?, ?, ?) \
         ON CONFLICT (chat_id) DO UPDATE SET \
             summary = excluded.summary, \
             through_seq = excluded.through_seq, \
             updated_at = excluded.updated_at \
         WHERE excluded.through_seq >= chat_summaries.through_seq",
    )
    .bind(chat_id)
    .bind(summary)
    .bind(through_seq)
    .bind(now)
    .execute(pool)
    .await?;
    Ok(())
}

async fn insert_message(
    tx: &mut sqlx::SqliteConnection,
    chat_id: &str,
    branch_id: &str,
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
        "INSERT INTO messages (id, chat_id, branch_id, seq, role, content, reasoning, meta, created_at) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(&id)
    .bind(chat_id)
    .bind(branch_id)
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

    // --- Рабочая память: CRUD и очистка при смене задачи ---

    #[tokio::test]
    async fn working_memory_set_load_delete_round_trip() {
        let (_dir, pool) = temp_pool().await;
        let chat = create_chat(&pool, "owner-1", "Чат", &ChatSettings::default()).await.expect("чат");

        let applied = set_working_memory(&pool, &chat.id, &chat.active_task_id, "budget", "200000", "manual", 10)
            .await
            .expect("запись");
        assert!(applied);

        let entries = load_working_memory(&pool, &chat.id, &chat.active_task_id).await.expect("чтение");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].value, "200000");
        assert_eq!(entries[0].source, "manual");

        delete_working_memory(&pool, &chat.id, &chat.active_task_id, "budget").await.expect("удаление");
        assert!(load_working_memory(&pool, &chat.id, &chat.active_task_id).await.expect("чтение").is_empty());

        let err = delete_working_memory(&pool, &chat.id, &chat.active_task_id, "budget")
            .await
            .expect_err("ключа уже нет");
        assert!(matches!(err, StoreError::NotFound));
    }

    #[tokio::test]
    async fn working_memory_is_isolated_per_chat() {
        let (_dir, pool) = temp_pool().await;
        let chat_a = create_chat(&pool, "owner-1", "Чат A", &ChatSettings::default()).await.expect("чат A");
        let chat_b = create_chat(&pool, "owner-1", "Чат B", &ChatSettings::default()).await.expect("чат B");

        set_working_memory(&pool, &chat_a.id, &chat_a.active_task_id, "budget", "100", "manual", 1)
            .await
            .expect("запись в чате A");

        let entries_b = load_working_memory(&pool, &chat_b.id, &chat_b.active_task_id).await.expect("чтение B");
        assert!(entries_b.is_empty(), "рабочая память не видна другому чату того же владельца");
    }

    #[tokio::test]
    async fn later_update_wins_regardless_of_source() {
        let (_dir, pool) = temp_pool().await;
        let chat = create_chat(&pool, "owner-1", "Чат", &ChatSettings::default()).await.expect("чат");

        set_working_memory(&pool, &chat.id, &chat.active_task_id, "k", "от роутера", "router", 5)
            .await
            .expect("ранняя автоматическая запись");
        let applied =
            set_working_memory(&pool, &chat.id, &chat.active_task_id, "k", "от клиента", "manual", 10)
                .await
                .expect("поздняя ручная запись");
        assert!(applied);
        let entries = load_working_memory(&pool, &chat.id, &chat.active_task_id).await.expect("чтение");
        assert_eq!(entries[0].value, "от клиента");

        // Обратный порядок по времени: ручная запись раньше, автоматическая позже.
        let applied =
            set_working_memory(&pool, &chat.id, &chat.active_task_id, "k2", "от клиента", "manual", 10)
                .await
                .expect("ранняя ручная запись");
        assert!(applied);
        let applied =
            set_working_memory(&pool, &chat.id, &chat.active_task_id, "k2", "от роутера", "router", 5)
                .await
                .expect("вызов не должен упасть");
        assert!(!applied, "более ранняя операция не должна применяться поверх более поздней");
        let entries = load_working_memory(&pool, &chat.id, &chat.active_task_id).await.expect("чтение");
        let k2 = entries.iter().find(|e| e.key == "k2").expect("запись k2");
        assert_eq!(k2.value, "от клиента");
    }

    #[tokio::test]
    async fn finish_task_transfers_carried_entries_before_clearing_and_rotates_task() {
        let (_dir, pool) = temp_pool().await;
        let chat = create_chat(&pool, "owner-1", "Чат", &ChatSettings::default()).await.expect("чат");
        let old_task_id = chat.active_task_id.clone();

        set_working_memory(&pool, &chat.id, &old_task_id, "carried", "значение", "manual", 1)
            .await
            .expect("запись для переноса");
        set_working_memory(&pool, &chat.id, &old_task_id, "dropped", "потеряется", "manual", 1)
            .await
            .expect("запись без переноса");

        let transferred = finish_task(&pool, "owner-1", &chat.id, &["carried".to_string()])
            .await
            .expect("завершение задачи");
        assert_eq!(transferred.len(), 1);
        assert_eq!(transferred[0].value, "значение");

        let long_term = load_long_term_memory(&pool, "owner-1", 50).await.expect("долговременная память");
        assert!(long_term.iter().any(|e| e.key.as_deref() == Some("carried") && e.value == "значение"));

        let reloaded = load_chat(&pool, "owner-1", &chat.id, &ChatSettings::default()).await.expect("чат");
        assert_ne!(reloaded.active_task_id, old_task_id, "у чата новая активная задача");

        let old_task_entries = load_working_memory(&pool, &chat.id, &old_task_id).await.expect("чтение прежней задачи");
        assert!(old_task_entries.is_empty(), "рабочая память прежней задачи очищена");
        let new_task_entries = load_working_memory(&pool, &chat.id, &reloaded.active_task_id)
            .await
            .expect("чтение новой задачи");
        assert!(new_task_entries.is_empty(), "новая задача начинает с пустого набора");
    }

    // --- Долговременная память: CRUD и изоляция по владельцу ---

    #[tokio::test]
    async fn long_term_memory_set_load_delete_round_trip() {
        let (_dir, pool) = temp_pool().await;
        let (id, applied) =
            set_long_term_memory(&pool, "owner-1", "decision", Some("auth_provider"), "Clerk", "manual", None, 1)
                .await
                .expect("запись");
        assert!(applied);

        let entries = load_long_term_memory(&pool, "owner-1", 50).await.expect("чтение");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].value, "Clerk");

        delete_long_term_memory(&pool, "owner-1", &id).await.expect("удаление");
        assert!(load_long_term_memory(&pool, "owner-1", 50).await.expect("чтение").is_empty());

        let err = delete_long_term_memory(&pool, "owner-1", &id).await.expect_err("записи уже нет");
        assert!(matches!(err, StoreError::NotFound));
    }

    #[tokio::test]
    async fn long_term_memory_is_isolated_per_owner() {
        let (_dir, pool) = temp_pool().await;
        set_long_term_memory(&pool, "owner-x", "profile", Some("name"), "секрет X", "manual", None, 1)
            .await
            .expect("запись владельца X");

        let entries_y = load_long_term_memory(&pool, "owner-y", 50).await.expect("чтение владельца Y");
        assert!(entries_y.is_empty(), "долговременная память не видна другому владельцу");
    }

    #[tokio::test]
    async fn long_term_memory_without_key_never_upserts() {
        let (_dir, pool) = temp_pool().await;
        set_long_term_memory(&pool, "owner-1", "knowledge", None, "заметка один", "manual", None, 1)
            .await
            .expect("заметка 1");
        set_long_term_memory(&pool, "owner-1", "knowledge", None, "заметка два", "manual", None, 2)
            .await
            .expect("заметка 2");

        let entries = load_long_term_memory(&pool, "owner-1", 50).await.expect("чтение");
        assert_eq!(entries.len(), 2, "записи без ключа не схлопываются в одну");
    }

    // --- Условное обновление названия чата (specs/chat-title, design.md, решение 6) ---

    #[tokio::test]
    async fn set_title_if_default_updates_when_title_still_matches() {
        let (_dir, pool) = temp_pool().await;
        let chat = create_chat(&pool, "owner-1", "Новый чат", &ChatSettings::default())
            .await
            .expect("чат");
        let updated = set_title_if_default(&pool, "owner-1", &chat.id, "Название от модели", "Новый чат")
            .await
            .expect("обновление названия");
        assert!(updated);
        let loaded = load_chat(&pool, "owner-1", &chat.id, &ChatSettings::default())
            .await
            .expect("чат");
        assert_eq!(loaded.title, "Название от модели");
    }

    #[tokio::test]
    async fn set_title_if_default_is_noop_when_title_already_changed() {
        let (_dir, pool) = temp_pool().await;
        let chat = create_chat(&pool, "owner-1", "Новый чат", &ChatSettings::default())
            .await
            .expect("чат");
        update_chat(&pool, "owner-1", &chat.id, Some("Название клиента"), None, &ChatSettings::default())
            .await
            .expect("переименование");

        let updated = set_title_if_default(&pool, "owner-1", &chat.id, "Название от модели", "Новый чат")
            .await
            .expect("обновление названия");
        assert!(!updated);
        let loaded = load_chat(&pool, "owner-1", &chat.id, &ChatSettings::default())
            .await
            .expect("чат");
        assert_eq!(loaded.title, "Название клиента");
    }

    // --- Роль "system" в хранилище (design.md, решение 4) ---

    #[test]
    fn role_from_str_accepts_system() {
        assert!(matches!(role_from_str("system"), Ok(Role::System)));
    }

    #[test]
    fn role_to_str_round_trips_system() {
        assert_eq!(role_to_str(Role::System), "system");
        assert!(matches!(role_from_str(role_to_str(Role::System)), Ok(Role::System)));
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

        let first = load_messages(&pool, &chat.id, 0, 4, None, 8).await.expect("первая часть");
        assert_eq!(first.messages.len(), 4);
        assert_eq!(first.messages.iter().map(|m| m.seq).collect::<Vec<_>>(), vec![1, 2, 3, 4]);
        assert_eq!(first.next_after, Some(4));

        let second = load_messages(&pool, &chat.id, 4, 20, None, 8)
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

        let loaded = load_messages(&pool, &chat.id, 0, 10, None, 8).await.expect("чтение");
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

        let loaded = load_messages(&pool, &chat.id, 0, 10, None, 8).await.expect("чтение");
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

        let loaded = load_messages(&pool, &chat.id, 0, 10, None, 8).await.expect("чтение");
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
        let loaded = load_messages(&pool, &chat.id, 0, 10, None, 8).await.expect("чтение");
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

        let page = load_messages(&pool, &chat.id, 0, 10, None, 8).await.expect("чтение");
        let mut seqs: Vec<i64> = page.messages.iter().map(|m| m.seq).collect();
        seqs.sort_unstable();
        assert_eq!(seqs, vec![1, 2, 3, 4]);
    }

    // --- 3.2 Миграция 0003: перенос старой схемы messages в корневые ветки ---

    #[tokio::test]
    async fn migration_0003_moves_legacy_messages_into_one_active_root_branch() {
        let dir = tempfile::tempdir().expect("временный каталог");
        let path = dir.path().join("agentd.db");
        let path_str = path.to_str().expect("путь").to_string();

        // Схема до 0003: применяются только первые две миграции, а третья
        // не отмечается применённой — так `open_pool` ниже увидит её как
        // ожидающую, ровно как на боевой базе перед обновлением сервиса.
        let legacy_dir = dir.path().join("legacy-migrations");
        std::fs::create_dir_all(&legacy_dir).expect("каталог старых миграций");
        for name in ["0001_init.sql", "0002_chat_summaries.sql"] {
            std::fs::copy(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("migrations").join(name),
                legacy_dir.join(name),
            )
            .expect("копия старой миграции");
        }

        let options = SqliteConnectOptions::from_str(&format!("sqlite:{path_str}"))
            .expect("адрес базы")
            .create_if_missing(true)
            .foreign_keys(true);
        let legacy_pool = SqlitePoolOptions::new()
            .connect_with(options)
            .await
            .expect("пул старой схемы");
        sqlx::migrate::Migrator::new(legacy_dir)
            .await
            .expect("загрузка старых миграций")
            .run(&legacy_pool)
            .await
            .expect("применение старых миграций");

        // Данные по старой схеме: chats без active_branch, messages без
        // branch_id, обычный UNIQUE (chat_id, seq).
        sqlx::query(
            "INSERT INTO chats (id, owner, title, settings, created_at, updated_at) \
             VALUES ('chat-1', 'owner-1', 'Старый чат', '{}', 1000, 1000)",
        )
        .execute(&legacy_pool)
        .await
        .expect("вставка чата по старой схеме");
        for seq in 1..=3i64 {
            sqlx::query(
                "INSERT INTO messages (id, chat_id, seq, role, content, created_at) \
                 VALUES (?, 'chat-1', ?, 'user', ?, 1000)",
            )
            .bind(format!("msg-{seq}"))
            .bind(seq)
            .bind(format!("сообщение {seq}"))
            .execute(&legacy_pool)
            .await
            .expect("вставка сообщения по старой схеме");
        }
        legacy_pool.close().await;

        // `open_pool` применяет все миграции крейта, включая 0003.
        let pool = open_pool(&path_str, 5, 5000).await.expect("применение 0003");

        let branch_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM chat_branches WHERE chat_id = 'chat-1'",
        )
        .fetch_one(&pool)
        .await
        .expect("число веток чата");
        assert_eq!(branch_count, 1, "чат должен получить ровно одну корневую ветку");

        let active_branch: Option<String> =
            sqlx::query_scalar("SELECT active_branch FROM chats WHERE id = 'chat-1'")
                .fetch_one(&pool)
                .await
                .expect("активная ветка чата");
        let active_branch = active_branch.expect("активная ветка задана после миграции");

        let message_branches: Vec<String> = sqlx::query_scalar(
            "SELECT branch_id FROM messages WHERE chat_id = 'chat-1' ORDER BY seq",
        )
        .fetch_all(&pool)
        .await
        .expect("ветки сообщений");
        assert_eq!(message_branches.len(), 3, "все сообщения читаются без потерь");
        assert!(
            message_branches.iter().all(|b| *b == active_branch),
            "все сообщения перенесены в активную (корневую) ветку чата"
        );
    }

    // --- Миграция 0004 (памяти) на существующей БД с данными ---

    #[tokio::test]
    async fn migration_0004_backfills_active_task_id_and_keeps_old_chats_readable() {
        let dir = tempfile::tempdir().expect("временный каталог");
        let path = dir.path().join("agentd.db");
        let path_str = path.to_str().expect("путь").to_string();

        // Схема до 0004: применяются миграции 0001–0003, а 0004 остаётся
        // ожидающей — как на боевой базе перед обновлением сервиса.
        let legacy_dir = dir.path().join("legacy-migrations");
        std::fs::create_dir_all(&legacy_dir).expect("каталог старых миграций");
        for name in ["0001_init.sql", "0002_chat_summaries.sql", "0003_context_strategies.sql"] {
            std::fs::copy(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("migrations").join(name),
                legacy_dir.join(name),
            )
            .expect("копия старой миграции");
        }

        let options = SqliteConnectOptions::from_str(&format!("sqlite:{path_str}"))
            .expect("адрес базы")
            .create_if_missing(true)
            .foreign_keys(true);
        let legacy_pool = SqlitePoolOptions::new()
            .connect_with(options)
            .await
            .expect("пул старой схемы");
        sqlx::migrate::Migrator::new(legacy_dir)
            .await
            .expect("загрузка старых миграций")
            .run(&legacy_pool)
            .await
            .expect("применение старых миграций");

        sqlx::query(
            "INSERT INTO chats (id, owner, title, settings, created_at, updated_at) \
             VALUES ('chat-1', 'owner-1', 'Старый чат', '{}', 1000, 1000)",
        )
        .execute(&legacy_pool)
        .await
        .expect("вставка чата по старой схеме");
        sqlx::query(
            "INSERT INTO chat_branches (id, chat_id, parent_id, fork_seq, name, created_at) \
             VALUES ('branch-1', 'chat-1', NULL, NULL, 'root', 1000)",
        )
        .execute(&legacy_pool)
        .await
        .expect("вставка корневой ветки по старой схеме");
        let root_branch_id = "branch-1".to_string();
        sqlx::query("UPDATE chats SET active_branch = ? WHERE id = 'chat-1'")
            .bind(&root_branch_id)
            .execute(&legacy_pool)
            .await
            .expect("активная ветка чата");
        legacy_pool.close().await;

        // `open_pool` применяет все миграции крейта, включая 0004.
        let pool = open_pool(&path_str, 5, 5000).await.expect("применение 0004");

        let active_task_id: Option<String> =
            sqlx::query_scalar("SELECT active_task_id FROM chats WHERE id = 'chat-1'")
                .fetch_one(&pool)
                .await
                .expect("активная задача чата");
        assert_eq!(active_task_id.as_deref(), Some("chat-1"), "бэкофилл задаёт задачу равной id чата");

        let chat = load_chat(&pool, "owner-1", "chat-1", &ChatSettings::default())
            .await
            .expect("старый чат читается после миграции 0004");
        assert_eq!(chat.active_task_id, "chat-1");
        assert_eq!(chat.title, "Старый чат");

        let working = load_working_memory(&pool, "chat-1", "chat-1").await.expect("рабочая память");
        assert!(working.is_empty(), "у старого чата рабочей памяти ещё нет");
    }

    // --- 8.1 Миграция chat_summaries применяется при открытии пула ---

    #[tokio::test]
    async fn chat_summaries_table_exists_after_open_pool() {
        let (_dir, pool) = temp_pool().await;
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM chat_summaries")
            .fetch_one(&pool)
            .await
            .expect("таблица chat_summaries существует");
        assert_eq!(count, 0);
    }

    // --- 8.2 Хранение пересказа ---

    #[tokio::test]
    async fn load_summary_of_chat_without_summary_is_none() {
        let (_dir, pool) = temp_pool().await;
        let defaults = ChatSettings::default();
        let chat = create_chat(&pool, "owner-1", "Чат", &defaults).await.expect("чат");
        let summary = load_summary(&pool, &chat.id).await.expect("чтение");
        assert!(summary.is_none());
    }

    #[tokio::test]
    async fn save_summary_persists_and_overwrites() {
        let (_dir, pool) = temp_pool().await;
        let defaults = ChatSettings::default();
        let chat = create_chat(&pool, "owner-1", "Чат", &defaults).await.expect("чат");

        save_summary(&pool, &chat.id, "первый пересказ", 5)
            .await
            .expect("сохранение");
        let loaded = load_summary(&pool, &chat.id)
            .await
            .expect("чтение")
            .expect("пересказ сохранён");
        assert_eq!(loaded.summary, "первый пересказ");
        assert_eq!(loaded.through_seq, 5);

        save_summary(&pool, &chat.id, "второй пересказ", 15)
            .await
            .expect("перезапись");
        let loaded = load_summary(&pool, &chat.id)
            .await
            .expect("чтение")
            .expect("пересказ сохранён");
        assert_eq!(loaded.summary, "второй пересказ");
        assert_eq!(loaded.through_seq, 15);
    }

    #[tokio::test]
    async fn save_summary_does_not_move_through_seq_backward() {
        let (_dir, pool) = temp_pool().await;
        let defaults = ChatSettings::default();
        let chat = create_chat(&pool, "owner-1", "Чат", &defaults).await.expect("чат");

        save_summary(&pool, &chat.id, "новый пересказ", 20)
            .await
            .expect("сохранение");
        // Запоздавшая запись с меньшей границей не должна откатить прогресс
        // (design.md, риск «Параллельные запросы в один чат»).
        save_summary(&pool, &chat.id, "устаревший пересказ", 10)
            .await
            .expect("запись не должна падать");

        let loaded = load_summary(&pool, &chat.id)
            .await
            .expect("чтение")
            .expect("пересказ сохранён");
        assert_eq!(loaded.summary, "новый пересказ");
        assert_eq!(loaded.through_seq, 20);
    }

    // --- 8.3 Каскад и полнота load_messages ---

    #[tokio::test]
    async fn deleting_chat_cascades_summary() {
        let (_dir, pool) = temp_pool().await;
        let defaults = ChatSettings::default();
        let chat = create_chat(&pool, "owner-1", "Чат", &defaults).await.expect("чат");
        save_summary(&pool, &chat.id, "пересказ", 5).await.expect("сохранение");

        delete_chat(&pool, "owner-1", &chat.id).await.expect("удаление");

        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM chat_summaries WHERE chat_id = ?")
            .bind(&chat.id)
            .fetch_one(&pool)
            .await
            .expect("подсчёт пересказов");
        assert_eq!(count, 0, "foreign_keys=ON должен каскадно удалить пересказ");
    }

    // --- 6.1 Хранение фактов ---

    #[tokio::test]
    async fn facts_of_chat_without_facts_is_empty() {
        let (_dir, pool) = temp_pool().await;
        let defaults = ChatSettings::default();
        let chat = create_chat(&pool, "owner-1", "Чат", &defaults).await.expect("чат");
        let facts = load_facts(&pool, &chat.id).await.expect("чтение");
        assert!(facts.is_empty());
    }

    #[tokio::test]
    async fn set_fact_persists_and_overwrites_existing_key() {
        let (_dir, pool) = temp_pool().await;
        let defaults = ChatSettings::default();
        let chat = create_chat(&pool, "owner-1", "Чат", &defaults).await.expect("чат");

        set_fact(&pool, &chat.id, "budget", "100000", 3).await.expect("установка");
        let facts = load_facts(&pool, &chat.id).await.expect("чтение");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].value, "100000");

        set_fact(&pool, &chat.id, "budget", "200000", 7).await.expect("перезапись");
        let facts = load_facts(&pool, &chat.id).await.expect("чтение");
        assert_eq!(facts.len(), 1, "перезапись ключа не создаёт вторую запись");
        assert_eq!(facts[0].value, "200000");
        assert_eq!(facts[0].through_seq, 7);
    }

    #[tokio::test]
    async fn delete_fact_removes_key_and_missing_key_is_not_found() {
        let (_dir, pool) = temp_pool().await;
        let defaults = ChatSettings::default();
        let chat = create_chat(&pool, "owner-1", "Чат", &defaults).await.expect("чат");
        set_fact(&pool, &chat.id, "budget", "100000", 1).await.expect("установка");

        delete_fact(&pool, &chat.id, "budget").await.expect("удаление");
        assert!(load_facts(&pool, &chat.id).await.expect("чтение").is_empty());

        let err = delete_fact(&pool, &chat.id, "budget").await.expect_err("ключа уже нет");
        assert!(matches!(err, StoreError::NotFound));
    }

    #[tokio::test]
    async fn deleting_chat_cascades_facts() {
        let (_dir, pool) = temp_pool().await;
        let defaults = ChatSettings::default();
        let chat = create_chat(&pool, "owner-1", "Чат", &defaults).await.expect("чат");
        set_fact(&pool, &chat.id, "budget", "100000", 1).await.expect("установка");

        delete_chat(&pool, "owner-1", &chat.id).await.expect("удаление чата");

        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM chat_facts WHERE chat_id = ?")
            .bind(&chat.id)
            .fetch_one(&pool)
            .await
            .expect("подсчёт фактов");
        assert_eq!(count, 0, "foreign_keys=ON должен каскадно удалить факты");
    }

    // --- 7.1 Ветки: создание, список, активация ---

    #[tokio::test]
    async fn new_chat_has_one_active_root_branch() {
        let (_dir, pool) = temp_pool().await;
        let defaults = ChatSettings::default();
        let chat = create_chat(&pool, "owner-1", "Чат", &defaults).await.expect("чат");

        let branches = list_branches(&pool, "owner-1", &chat.id).await.expect("список веток");
        assert_eq!(branches.len(), 1);
        assert!(branches[0].active);
        assert!(branches[0].parent_id.is_none());
        assert_eq!(branches[0].id, chat.active_branch);
    }

    #[tokio::test]
    async fn branch_created_from_message_and_activated() {
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
        .expect("обмен");

        let branch = create_branch(&pool, "owner-1", &chat.id, 1, "альтернатива")
            .await
            .expect("создание ветки");
        assert_eq!(branch.fork_seq, Some(1));
        assert_eq!(branch.parent_id.as_deref(), Some(chat.active_branch.as_str()));

        activate_branch(&pool, "owner-1", &chat.id, &branch.id)
            .await
            .expect("активация");
        let branches = list_branches(&pool, "owner-1", &chat.id).await.expect("список веток");
        let active: Vec<_> = branches.iter().filter(|b| b.active).collect();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, branch.id);
    }

    #[tokio::test]
    async fn branch_from_nonexistent_message_is_not_found() {
        let (_dir, pool) = temp_pool().await;
        let defaults = ChatSettings::default();
        let chat = create_chat(&pool, "owner-1", "Чат", &defaults).await.expect("чат");

        let err = create_branch(&pool, "owner-1", &chat.id, 99, "ветка")
            .await
            .expect_err("сообщения с таким seq нет");
        assert!(matches!(err, StoreError::NotFound));
    }

    #[tokio::test]
    async fn two_branches_from_one_point_keep_shared_prefix_and_own_suffix() {
        let (_dir, pool) = temp_pool().await;
        let defaults = ChatSettings::default();
        let chat = create_chat(&pool, "owner-1", "Чат", &defaults).await.expect("чат");
        append_exchange(
            &pool,
            "owner-1",
            &chat.id,
            new_message(Role::User, "общий вопрос"),
            new_message(Role::Assistant, "общий ответ"),
        )
        .await
        .expect("общий обмен");
        let root = chat.active_branch.clone();

        let branch_a = create_branch(&pool, "owner-1", &chat.id, 2, "A").await.expect("ветка A");
        activate_branch(&pool, "owner-1", &chat.id, &branch_a.id).await.expect("активация A");
        append_messages(
            &pool,
            "owner-1",
            &chat.id,
            vec![new_message(Role::User, "только в A")],
        )
        .await
        .expect("сообщение в A");

        activate_branch(&pool, "owner-1", &chat.id, &root).await.expect("возврат к root");
        let branch_b = create_branch(&pool, "owner-1", &chat.id, 2, "B").await.expect("ветка B");
        activate_branch(&pool, "owner-1", &chat.id, &branch_b.id).await.expect("активация B");
        append_messages(
            &pool,
            "owner-1",
            &chat.id,
            vec![new_message(Role::User, "только в B")],
        )
        .await
        .expect("сообщение в B");

        let history_a = load_branch_history(&pool, &chat.id, &branch_a.id, 8).await.expect("история A");
        let history_b = load_branch_history(&pool, &chat.id, &branch_b.id, 8).await.expect("история B");

        assert!(history_a.iter().any(|m| m.content == "общий вопрос"));
        assert!(history_a.iter().any(|m| m.content == "только в A"));
        assert!(!history_a.iter().any(|m| m.content == "только в B"));

        assert!(history_b.iter().any(|m| m.content == "общий вопрос"));
        assert!(history_b.iter().any(|m| m.content == "только в B"));
        assert!(!history_b.iter().any(|m| m.content == "только в A"));
    }

    #[tokio::test]
    async fn nested_branch_sees_root_and_intermediate_chain() {
        let (_dir, pool) = temp_pool().await;
        let defaults = ChatSettings::default();
        let chat = create_chat(&pool, "owner-1", "Чат", &defaults).await.expect("чат");
        append_exchange(
            &pool,
            "owner-1",
            &chat.id,
            new_message(Role::User, "корневой вопрос"),
            new_message(Role::Assistant, "корневой ответ"),
        )
        .await
        .expect("корневой обмен");

        let middle = create_branch(&pool, "owner-1", &chat.id, 2, "middle").await.expect("промежуточная ветка");
        activate_branch(&pool, "owner-1", &chat.id, &middle.id).await.expect("активация middle");
        append_exchange(
            &pool,
            "owner-1",
            &chat.id,
            new_message(Role::User, "промежуточный вопрос"),
            new_message(Role::Assistant, "промежуточный ответ"),
        )
        .await
        .expect("промежуточный обмен");

        let leaf = create_branch(&pool, "owner-1", &chat.id, 2, "leaf").await.expect("вложенная ветка");
        activate_branch(&pool, "owner-1", &chat.id, &leaf.id).await.expect("активация leaf");
        append_messages(&pool, "owner-1", &chat.id, vec![new_message(Role::User, "листовое сообщение")])
            .await
            .expect("сообщение в leaf");

        let history = load_branch_history(&pool, &chat.id, &leaf.id, 8).await.expect("история leaf");
        let contents: Vec<&str> = history.iter().map(|m| m.content.as_str()).collect();
        assert_eq!(
            contents,
            vec!["корневой вопрос", "корневой ответ", "промежуточный вопрос", "промежуточный ответ", "листовое сообщение"],
        );
    }

    #[tokio::test]
    async fn load_messages_ignores_summary_and_returns_full_history() {
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
        .expect("обмен");
        save_summary(&pool, &chat.id, "пересказ", 1).await.expect("сохранение");

        let page = load_messages(&pool, &chat.id, 0, 10, None, 8).await.expect("чтение");
        assert_eq!(page.messages.len(), 2, "пересказ не должен скрывать сообщения");
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
