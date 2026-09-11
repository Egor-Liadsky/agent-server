//! Типы запроса и ответа контракта `/v1`.

use crate::store;
use agentcore::agent::{AgentReply, Message, MessageMeta, Role};
use agentcore::config::{
    ChatSettings, Provider, ReasoningMode, ResponseFormat, SamplingParams, ThinkingMode,
};
use agentcore::pipeline::PolicyLog;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
pub struct ChatRequest {
    /// Эквивалент истории из одного пользовательского сообщения.
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub messages: Option<Vec<MessageDto>>,
    #[serde(default)]
    pub settings: Option<ChatSettingsDto>,
    /// Чат, в который пишется обмен. Заданный вместе с `messages` — `400`
    /// (см. `POST /v1/chat` в specs/chat-api).
    #[serde(default)]
    pub chat_id: Option<String>,
    /// Произвольные данные клиента: принимаются и на вызов модели не влияют.
    #[serde(default)]
    #[allow(dead_code)]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDto {
    pub role: RoleDto,
    pub content: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RoleDto {
    User,
    Assistant,
}

impl From<RoleDto> for Role {
    fn from(role: RoleDto) -> Self {
        match role {
            RoleDto::User => Role::User,
            RoleDto::Assistant => Role::Assistant,
        }
    }
}

impl MessageDto {
    pub fn into_message(self) -> Message {
        match self.role {
            RoleDto::User => Message::user(self.content),
            RoleDto::Assistant => Message::assistant(self.content),
        }
    }
}

/// Различает «поле не задано» и «поле задано как null»: первое оставляет
/// значение чата прежним, второе снимает его. Без двойного `Option` эти два
/// случая в serde неразличимы (specs/chat-message-api, «Настройки чата
/// задаются целиком»).
fn double_option<'de, D, T>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::deserialize(deserializer).map(Some)
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ChatSettingsDto {
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub thinking: Option<String>,
    #[serde(default)]
    pub experts: Option<Vec<String>>,
    #[serde(default)]
    pub response_format: Option<ResponseFormatDto>,
    /// Включает и выключает кастомный формат ответа. Незаданное поле
    /// оставляет режим чата прежним.
    #[serde(default)]
    pub custom_response_mode: Option<bool>,
    #[serde(default, deserialize_with = "double_option")]
    pub temperature: Option<Option<f32>>,
    #[serde(default, deserialize_with = "double_option")]
    pub top_p: Option<Option<f32>>,
    #[serde(default, deserialize_with = "double_option")]
    pub top_k: Option<Option<u32>>,
    #[serde(default, deserialize_with = "double_option")]
    pub frequency_penalty: Option<Option<f32>>,
    #[serde(default, deserialize_with = "double_option")]
    pub presence_penalty: Option<Option<f32>>,
    /// Лимит контекстного окна чата. Незаданное поле оставляет сохранённый
    /// лимит чата прежним, явный `null` снимает его, число — задаёт
    /// (сохраняется при `PATCH`, наравне с параметрами сэмплирования).
    /// `POST /v1/chat` использует это же поле как разовое переопределение
    /// поверх сохранённого лимита, не сохраняя его (specs/context-limit,
    /// «Разовое переопределение лимита на один запрос»).
    #[serde(default, deserialize_with = "double_option")]
    pub max_context_tokens: Option<Option<u32>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ResponseFormatDto {
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub max_length: Option<u32>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub stop_instruction: Option<String>,
}

impl ResponseFormatDto {
    fn into_format(self) -> ResponseFormat {
        ResponseFormat {
            description: self.description,
            max_length: self.max_length,
            stop: self.stop,
            stop_instruction: self.stop_instruction,
        }
    }
}

/// Разбор провайдера из тела запроса.
pub fn parse_provider(value: &str) -> Option<Provider> {
    match value.trim().to_ascii_lowercase().as_str() {
        "cloud" => Some(Provider::Cloud),
        "ollama" => Some(Provider::Ollama),
        _ => None,
    }
}

impl ChatSettingsDto {
    /// Накладывает заданные поля поверх серверных значений по умолчанию.
    /// Незаданные поля остаются от конфигурации сервиса.
    pub fn apply_to(self, mut defaults: ChatSettings) -> Result<ChatSettings, String> {
        if let Some(provider) = &self.provider {
            defaults.provider = parse_provider(provider)
                .ok_or_else(|| format!("неизвестный провайдер: {provider}"))?;
        }
        if let Some(model) = self.model {
            defaults.model = Some(model);
        }
        if let Some(reasoning) = &self.reasoning {
            defaults.reasoning = ReasoningMode::parse(reasoning)
                .ok_or_else(|| format!("неизвестная стратегия рассуждения: {reasoning}"))?;
        }
        if let Some(thinking) = &self.thinking {
            defaults.thinking = ThinkingMode::parse(thinking)
                .ok_or_else(|| format!("неизвестный режим размышления: {thinking}"))?;
        }
        if let Some(experts) = self.experts {
            defaults.experts = experts;
        }
        if let Some(format) = self.response_format {
            defaults.response_format = format.into_format();
            // Формат ответа действует только в кастомном режиме, поэтому
            // заданный клиентом формат этот режим и включает. Явное поле
            // custom_response_mode ниже может его же и выключить.
            defaults.custom_response_mode = true;
        }
        if let Some(custom_response_mode) = self.custom_response_mode {
            defaults.custom_response_mode = custom_response_mode;
        }
        // Заданное значение подменяет прежнее, явный null снимает его, а
        // отсутствие поля оставляет как было.
        let sampling = SamplingParams {
            temperature: self.temperature.unwrap_or(defaults.sampling.temperature),
            top_p: self.top_p.unwrap_or(defaults.sampling.top_p),
            top_k: self.top_k.unwrap_or(defaults.sampling.top_k),
            frequency_penalty: self
                .frequency_penalty
                .unwrap_or(defaults.sampling.frequency_penalty),
            presence_penalty: self
                .presence_penalty
                .unwrap_or(defaults.sampling.presence_penalty),
        };
        defaults.sampling = sampling;
        defaults.max_context_tokens = self
            .max_context_tokens
            .unwrap_or(defaults.max_context_tokens);
        Ok(defaults)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub request_id: String,
    pub content: String,
    pub reasoning: Option<String>,
    /// Модель, которой ответили на самом деле.
    pub model: String,
    pub usage: UsageDto,
    pub timing: TimingDto,
    /// Результаты стадий конвейера. Присутствует всегда, даже когда пуст.
    pub policy: PolicyLog,
    /// Чат, в который записан обмен. `null` — запрос без `chat_id`, история
    /// нигде не сохранена (specs/chat-api, «Разовый вызов без чата»).
    #[serde(default)]
    pub chat_id: Option<String>,
    /// Номер сохранённого ответа модели в чате.
    #[serde(default)]
    pub seq: Option<i64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageDto {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
    pub reasoning_tokens: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimingDto {
    pub duration_ms: Option<u64>,
    pub sent_at: Option<i64>,
    pub received_at: Option<i64>,
}

impl From<&MessageMeta> for UsageDto {
    fn from(meta: &MessageMeta) -> Self {
        Self {
            prompt_tokens: meta.prompt_tokens,
            completion_tokens: meta.completion_tokens,
            total_tokens: meta.total_tokens,
            reasoning_tokens: meta.reasoning_tokens,
        }
    }
}

impl From<&MessageMeta> for TimingDto {
    fn from(meta: &MessageMeta) -> Self {
        Self {
            duration_ms: meta.duration_ms,
            sent_at: meta.sent_at,
            received_at: meta.received_at,
        }
    }
}

impl ChatResponse {
    pub fn new(request_id: String, model: String, reply: &AgentReply, policy: PolicyLog) -> Self {
        Self {
            request_id,
            content: reply.content.clone(),
            reasoning: reply.reasoning.clone(),
            model,
            usage: UsageDto::from(&reply.meta),
            timing: TimingDto::from(&reply.meta),
            policy,
            chat_id: None,
            seq: None,
        }
    }

    pub fn with_chat(mut self, chat_id: String, seq: i64) -> Self {
        self.chat_id = Some(chat_id);
        self.seq = Some(seq);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsResponse {
    pub models: Vec<String>,
}

// --- Управление чатами ---

/// Представление чата. `ChatSettings` сериализуется собственным `Serialize`
/// ядра — тем же форматом, каким принимается в `settings` запросов.
#[derive(Debug, Clone, Serialize)]
pub struct ChatDto {
    pub id: String,
    pub title: String,
    pub settings: ChatSettings,
    pub created_at: i64,
    pub updated_at: i64,
    pub message_count: i64,
}

impl From<store::Chat> for ChatDto {
    fn from(chat: store::Chat) -> Self {
        Self {
            id: chat.id,
            title: chat.title,
            settings: chat.settings,
            created_at: chat.created_at,
            updated_at: chat.updated_at,
            message_count: chat.message_count,
        }
    }
}

/// Сообщение чата в ответах на чтение. Поля телеметрии заполнены только у
/// ответов модели (specs/chat-storage, «Полнота сохраняемого сообщения»).
#[derive(Debug, Clone, Serialize)]
pub struct MessageView {
    pub seq: i64,
    pub role: RoleDto,
    pub content: String,
    pub created_at: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageDto>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing: Option<TimingDto>,
}

impl From<store::ChatMessage> for MessageView {
    fn from(message: store::ChatMessage) -> Self {
        let role = match message.role {
            Role::User => RoleDto::User,
            Role::Assistant => RoleDto::Assistant,
        };
        let (reasoning, model, usage, timing) = if matches!(message.role, Role::Assistant) {
            let usage = message.meta.as_ref().map(UsageDto::from);
            let timing = message.meta.as_ref().map(TimingDto::from);
            let model = message.meta.as_ref().and_then(|meta| meta.model.clone());
            (message.reasoning, model, usage, timing)
        } else {
            (None, None, None, None)
        };
        Self {
            seq: message.seq,
            role,
            content: message.content,
            created_at: message.created_at,
            reasoning,
            model,
            usage,
            timing,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct CreateChatRequest {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub settings: Option<ChatSettingsDto>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct UpdateChatRequest {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub settings: Option<ChatSettingsDto>,
}

/// Готовое сообщение для дозаписи в чат. Роль и текст обязательны,
/// остальное — телеметрия, которую клиент получил от локальной модели сам
/// (specs/chat-message-api, «Дозапись готовых сообщений в чат»).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewMessageDto {
    pub role: RoleDto,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageDto>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timing: Option<TimingDto>,
}

impl NewMessageDto {
    /// Телеметрия и рассуждение имеют смысл только у ответа модели: у
    /// реплики пользователя их нет, как и в `MessageView` на чтении.
    pub fn into_new_message(self) -> store::NewMessage {
        let role = Role::from(self.role);
        if matches!(role, Role::User) {
            return store::NewMessage {
                role,
                content: self.content,
                reasoning: None,
                meta: None,
            };
        }

        let usage = self.usage.unwrap_or_default();
        let timing = self.timing.unwrap_or_default();
        let meta = MessageMeta {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            reasoning_tokens: usage.reasoning_tokens,
            duration_ms: timing.duration_ms,
            sent_at: timing.sent_at,
            received_at: timing.received_at,
            model: self.model,
        };
        // Пустая телеметрия не пишется: иначе у реплики без метрик в
        // хранилище появился бы объект из одних отсутствующих полей.
        let has_telemetry = meta.prompt_tokens.is_some()
            || meta.completion_tokens.is_some()
            || meta.total_tokens.is_some()
            || meta.reasoning_tokens.is_some()
            || meta.duration_ms.is_some()
            || meta.sent_at.is_some()
            || meta.received_at.is_some()
            || meta.model.is_some();
        store::NewMessage {
            role,
            content: self.content,
            reasoning: self.reasoning,
            meta: has_telemetry.then_some(meta),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendMessagesRequest {
    pub messages: Vec<NewMessageDto>,
}

/// Ответ дозаписи: назначенные хранилищем номера сообщений в том же
/// порядке, в котором они переданы, и идентификатор запроса.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendMessagesResponse {
    pub request_id: String,
    pub chat_id: String,
    pub seqs: Vec<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ListChatsQuery {
    #[serde(default)]
    pub limit: Option<u32>,
    #[serde(default)]
    pub cursor: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GetChatQuery {
    #[serde(default)]
    pub limit: Option<u32>,
    #[serde(default)]
    pub after: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ListChatsResponse {
    pub chats: Vec<ChatDto>,
    pub next_cursor: Option<String>,
}

/// Чат вместе со страницей его сообщений: поля чата на верхнем уровне
/// объекта (specs/chat-api, «Чтение чата с сообщениями»).
#[derive(Debug, Clone, Serialize)]
pub struct ChatWithMessagesResponse {
    #[serde(flatten)]
    pub chat: ChatDto,
    pub messages: Vec<MessageView>,
    pub next_after: Option<i64>,
}
