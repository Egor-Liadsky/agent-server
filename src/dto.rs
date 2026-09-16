//! Типы запроса и ответа контракта `/v1`.

use crate::store;
use agentcore::agent::{AgentReply, Message, MessageMeta, Role};
use agentcore::config::{
    ChatSettings, ContextStrategy, Provider, ReasoningMode, ResponseFormat, SamplingParams, ThinkingMode,
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

/// `summary` и `summary_through_seq` — только для чтения, поэтому этот DTO
/// их не объявляет: `deny_unknown_fields` отклоняет попытку прислать их
/// (или любое другое неизвестное поле) как некорректный запрос
/// (specs/context-summary, «Наблюдаемость компактизации»).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
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
    /// Включает или выключает компактизацию истории для этого чата. Та же
    /// семантика присутствия поля, что и у `max_context_tokens`: поля нет —
    /// сохранённое значение остаётся, `null` — возврат к операторскому
    /// умолчанию (`AGENTD_SUMMARY_ENABLED`), значение — задаёт
    /// (specs/context-summary, «Настройки компактизации на уровне чата»).
    #[serde(default, deserialize_with = "double_option")]
    pub summary_enabled: Option<Option<bool>>,
    /// Потолок дословного хвоста компактизации для этого чата: не может
    /// превышать `AGENTD_SUMMARY_KEEP_MESSAGES` (проверяется в
    /// `merge_settings`).
    #[serde(default, deserialize_with = "double_option")]
    pub summary_keep_messages: Option<Option<u32>>,
    /// Шаг пересказа для этого чата: не может быть меньше
    /// `AGENTD_SUMMARY_STEP_MESSAGES` (проверяется в `merge_settings`).
    #[serde(default, deserialize_with = "double_option")]
    pub summary_step_messages: Option<Option<u32>>,
    /// Стратегия управления контекстом. Разбор и проверка допустимости —
    /// вне `apply_to`, отдельным шагом с собственными кодами ошибок
    /// (`context_strategy_invalid`, `context_strategy_not_allowed`), а не
    /// generic `invalid_request` (specs/context-strategies).
    #[serde(default, deserialize_with = "double_option")]
    pub context_strategy: Option<Option<String>>,
    #[serde(default, deserialize_with = "double_option")]
    pub context_window_messages: Option<Option<u32>>,
    /// Слоистая память этого чата, независимо от `context_strategy`. Та же
    /// семантика присутствия поля, что и у `summary_enabled`.
    #[serde(default, deserialize_with = "double_option")]
    pub memory_layers_enabled: Option<Option<bool>>,
    /// Автомаршрутизатор памяти этого чата. Имеет смысл только при
    /// включённой слоистой памяти. Та же семантика присутствия поля, что и
    /// у `summary_enabled`.
    #[serde(default, deserialize_with = "double_option")]
    pub memory_router_enabled: Option<Option<bool>>,
    #[serde(default, deserialize_with = "double_option")]
    pub memory_working_max_entries: Option<Option<u32>>,
    #[serde(default, deserialize_with = "double_option")]
    pub memory_long_term_max_entries: Option<Option<u32>>,
    /// Профиль этого чата: встроенный или собственный владельца. Та же
    /// семантика присутствия поля, что и у `summary_enabled` — поля нет,
    /// сохранённое значение остаётся, `null` снимает профиль, значение
    /// задаёт (specs/user-profiles, «Профиль выбирается настройкой чата
    /// поверх операторского умолчания»).
    #[serde(default, deserialize_with = "double_option")]
    pub profile_id: Option<Option<String>>,
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
        defaults.summary_enabled = self.summary_enabled.unwrap_or(defaults.summary_enabled);
        defaults.summary_keep_messages = self
            .summary_keep_messages
            .unwrap_or(defaults.summary_keep_messages);
        defaults.summary_step_messages = self
            .summary_step_messages
            .unwrap_or(defaults.summary_step_messages);
        defaults.memory_layers_enabled = self.memory_layers_enabled.unwrap_or(defaults.memory_layers_enabled);
        defaults.memory_router_enabled = self.memory_router_enabled.unwrap_or(defaults.memory_router_enabled);
        defaults.memory_working_max_entries =
            self.memory_working_max_entries.unwrap_or(defaults.memory_working_max_entries);
        defaults.memory_long_term_max_entries =
            self.memory_long_term_max_entries.unwrap_or(defaults.memory_long_term_max_entries);
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
    /// Сведения о компактизации истории этого запроса. `null` — запрос без
    /// `chat_id`: компактизации не подлежит (specs/context-summary,
    /// «Разовый вызов без чата не компактизуется»). Для запроса с `chat_id`
    /// присутствует всегда, включая некомпактизованные запросы — иначе
    /// настройкам не с чем сравниваться (specs/context-summary,
    /// «Наблюдаемость компактизации»).
    #[serde(default)]
    pub context: Option<ContextDto>,
}

/// Что действующая стратегия сделала со сборкой истории этого запроса.
/// Поля, не имеющие смысла для стратегии, опускаются, а не несут ноль/`false`
/// (specs/context-strategies, «Ответ сообщает, что сделала стратегия»).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextDto {
    pub strategy: ContextStrategy,
    /// Сколько сообщений отправлено провайдеру (`sliding_window`, `facts`,
    /// `branching`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sent_messages: Option<u32>,
    /// Сколько сохранённых сообщений отброшено без замены (`sliding_window`,
    /// `facts`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dropped_messages: Option<u32>,
    /// Сколько сохранённых сообщений заменено пересказом (`summary`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replaced_messages: Option<u32>,
    /// Строился ли новый пересказ на этом запросе (`summary`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary_built: Option<bool>,
    /// Сколько фактов подставлено в запрос (`facts`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub facts_applied: Option<u32>,
    /// Обновились ли факты после ответа (`facts`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub facts_updated: Option<bool>,
    /// Ветка, из которой собрана история (`branching`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch_id: Option<String>,
    /// Число записей долговременной памяти, подставленных в контекст
    /// (слоистая память включена, любая стратегия).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_long_term_entries: Option<u32>,
    /// Объём долговременной памяти в контексте, в символах (слоистая память
    /// включена).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_long_term_chars: Option<u32>,
    /// Число записей рабочей памяти, подставленных в контекст (слоистая
    /// память включена).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_working_entries: Option<u32>,
    /// Объём рабочей памяти в контексте, в символах (слоистая память
    /// включена).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_working_chars: Option<u32>,
    /// Число сообщений краткосрочной истории, фактически собранной
    /// действующей стратегией (слоистая память включена).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_short_term_messages: Option<u32>,
    /// Объём этой краткосрочной истории, в символах (слоистая память
    /// включена).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_short_term_chars: Option<u32>,
    /// Сколько операций `set` маршрутизатора памяти применено (новые ключи,
    /// счётчики относятся к маршрутизации ПРЕДЫДУЩЕГО сообщения; слоистая
    /// память включена).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_router_applied_set: Option<u32>,
    /// Сколько операций маршрутизатора обновили существующий ключ (слоистая
    /// память включена).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_router_applied_update: Option<u32>,
    /// Сколько операций `delete` маршрутизатора применено (слоистая память
    /// включена).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_router_applied_delete: Option<u32>,
    /// Сколько операций маршрутизатора отброшено валидацией (слоистая память
    /// включена).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_router_rejected: Option<u32>,
    /// Идентификатор применённого профиля (specs/user-profiles,
    /// «Наблюдаемость подстановки профиля»). Отсутствует без профиля.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile_id: Option<String>,
    /// Объём собранного раздела профиля в символах, после возможного
    /// усечения. Отсутствует без профиля.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile_chars: Option<u32>,
}

impl ContextDto {
    pub fn for_summary(replaced_messages: u32, summary_built: bool) -> Self {
        Self {
            strategy: ContextStrategy::Summary,
            sent_messages: None,
            dropped_messages: None,
            replaced_messages: Some(replaced_messages),
            summary_built: Some(summary_built),
            facts_applied: None,
            facts_updated: None,
            branch_id: None,
            memory_long_term_entries: None,
            memory_long_term_chars: None,
            memory_working_entries: None,
            memory_working_chars: None,
            memory_short_term_messages: None,
            memory_short_term_chars: None,
            memory_router_applied_set: None,
            memory_router_applied_update: None,
            memory_router_applied_delete: None,
            memory_router_rejected: None,
            profile_id: None,
            profile_chars: None,
        }
    }

    pub fn for_window(sent_messages: u32, dropped_messages: u32) -> Self {
        Self {
            strategy: ContextStrategy::SlidingWindow,
            sent_messages: Some(sent_messages),
            dropped_messages: Some(dropped_messages),
            replaced_messages: None,
            summary_built: None,
            facts_applied: None,
            facts_updated: None,
            branch_id: None,
            memory_long_term_entries: None,
            memory_long_term_chars: None,
            memory_working_entries: None,
            memory_working_chars: None,
            memory_short_term_messages: None,
            memory_short_term_chars: None,
            memory_router_applied_set: None,
            memory_router_applied_update: None,
            memory_router_applied_delete: None,
            memory_router_rejected: None,
            profile_id: None,
            profile_chars: None,
        }
    }

    pub fn for_facts(
        sent_messages: u32,
        dropped_messages: u32,
        facts_applied: u32,
        facts_updated: bool,
    ) -> Self {
        Self {
            strategy: ContextStrategy::Facts,
            sent_messages: Some(sent_messages),
            dropped_messages: Some(dropped_messages),
            replaced_messages: None,
            summary_built: None,
            facts_applied: Some(facts_applied),
            facts_updated: Some(facts_updated),
            branch_id: None,
            memory_long_term_entries: None,
            memory_long_term_chars: None,
            memory_working_entries: None,
            memory_working_chars: None,
            memory_short_term_messages: None,
            memory_short_term_chars: None,
            memory_router_applied_set: None,
            memory_router_applied_update: None,
            memory_router_applied_delete: None,
            memory_router_rejected: None,
            profile_id: None,
            profile_chars: None,
        }
    }

    pub fn for_branching(sent_messages: u32, branch_id: String) -> Self {
        Self {
            strategy: ContextStrategy::Branching,
            sent_messages: Some(sent_messages),
            dropped_messages: None,
            replaced_messages: None,
            summary_built: None,
            facts_applied: None,
            facts_updated: None,
            branch_id: Some(branch_id),
            memory_long_term_entries: None,
            memory_long_term_chars: None,
            memory_working_entries: None,
            memory_working_chars: None,
            memory_short_term_messages: None,
            memory_short_term_chars: None,
            memory_router_applied_set: None,
            memory_router_applied_update: None,
            memory_router_applied_delete: None,
            memory_router_rejected: None,
            profile_id: None,
            profile_chars: None,
        }
    }
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
            context: None,
        }
    }

    pub fn with_chat(mut self, chat_id: String, seq: i64) -> Self {
        self.chat_id = Some(chat_id);
        self.seq = Some(seq);
        self
    }

    pub fn with_context(mut self, context: ContextDto) -> Self {
        self.context = Some(context);
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
    /// Текст пересказа вытесненной части истории. Только для чтения: входные
    /// DTO это поле не объявляют (specs/context-summary, «Наблюдаемость
    /// компактизации»).
    #[serde(default)]
    pub summary: Option<String>,
    /// Номер последнего пересказанного сообщения.
    #[serde(default)]
    pub summary_through_seq: Option<i64>,
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
            summary: None,
            summary_through_seq: None,
        }
    }
}

impl ChatDto {
    /// Наполняет поля пересказа, только для чтения (`GET /v1/chats` и
    /// `GET /v1/chats/{id}`); отсутствие пересказа оставляет оба поля
    /// `null`.
    pub fn with_summary(mut self, summary: Option<store::ChatSummary>) -> Self {
        if let Some(summary) = summary {
            self.summary = Some(summary.summary);
            self.summary_through_seq = Some(summary.through_seq);
        }
        self
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
            // Системные сообщения хранилище никогда не пишет (design.md,
            // решение 4): ветка — только защита от несуществующего на
            // практике случая, `RoleDto` варианта `System` не имеет.
            Role::System => RoleDto::User,
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
    /// Явное указание ветки. `None` — история активной ветки чата
    /// (specs/chat-branching, «Чтение истории учитывает ветку»).
    #[serde(default)]
    pub branch: Option<String>,
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
    /// Ветка, чья история отдана (specs/chat-branching, «Чтение истории
    /// учитывает ветку»).
    pub branch_id: String,
}

// --- Факты (specs/context-facts) ---

#[derive(Debug, Clone, Serialize)]
pub struct FactDto {
    pub key: String,
    pub value: String,
    pub through_seq: i64,
    pub updated_at: i64,
}

impl From<store::Fact> for FactDto {
    fn from(fact: store::Fact) -> Self {
        Self {
            key: fact.key,
            value: fact.value,
            through_seq: fact.through_seq,
            updated_at: fact.updated_at,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct FactsResponse {
    pub facts: Vec<FactDto>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SetFactRequest {
    pub value: String,
}

// --- Ветки (specs/chat-branching) ---

#[derive(Debug, Clone, Serialize)]
pub struct BranchDto {
    pub id: String,
    pub name: String,
    pub parent_id: Option<String>,
    pub fork_seq: Option<i64>,
    pub message_count: i64,
    pub active: bool,
}

impl From<store::Branch> for BranchDto {
    fn from(branch: store::Branch) -> Self {
        Self {
            id: branch.id,
            name: branch.name,
            parent_id: branch.parent_id,
            fork_seq: branch.fork_seq,
            message_count: branch.message_count,
            active: branch.active,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BranchesResponse {
    pub branches: Vec<BranchDto>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CreateBranchRequest {
    pub from_seq: i64,
    #[serde(default)]
    pub name: Option<String>,
}

// --- Память (specs/memory-layers) ---

#[derive(Debug, Clone, Serialize)]
pub struct WorkingMemoryEntryDto {
    pub key: String,
    pub value: String,
    pub source: String,
    pub updated_at: i64,
}

impl From<store::WorkingMemoryEntry> for WorkingMemoryEntryDto {
    fn from(entry: store::WorkingMemoryEntry) -> Self {
        Self { key: entry.key, value: entry.value, source: entry.source, updated_at: entry.updated_at }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkingMemoryResponse {
    pub entries: Vec<WorkingMemoryEntryDto>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SetWorkingMemoryRequest {
    pub key: String,
    pub value: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeleteWorkingMemoryQuery {
    pub key: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct FinishTaskRequest {
    #[serde(default)]
    pub carry_forward_keys: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LongTermMemoryEntryDto {
    pub id: String,
    pub entry_type: String,
    pub key: Option<String>,
    pub value: String,
    pub source: String,
    pub source_chat_id: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

impl From<store::LongTermMemoryEntry> for LongTermMemoryEntryDto {
    fn from(entry: store::LongTermMemoryEntry) -> Self {
        Self {
            id: entry.id,
            entry_type: entry.entry_type,
            key: entry.key,
            value: entry.value,
            source: entry.source,
            source_chat_id: entry.source_chat_id,
            created_at: entry.created_at,
            updated_at: entry.updated_at,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct LongTermMemoryResponse {
    pub entries: Vec<LongTermMemoryEntryDto>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SetLongTermMemoryRequest {
    pub entry_type: String,
    #[serde(default)]
    pub key: Option<String>,
    pub value: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeleteLongTermMemoryQuery {
    pub id: String,
}

// --- Профили (specs/user-profiles) ---

#[derive(Debug, Clone, Serialize)]
pub struct ProfileDto {
    pub id: String,
    pub name: String,
    pub persona: String,
    pub style: String,
    pub format: String,
    pub constraints: Vec<String>,
    /// Встроенный профиль (`teacher`/`psychologist`/`reviewer`) только для
    /// чтения; собственный профиль владельца — `false`
    /// (specs/user-profiles, «Ручное управление профилями через HTTP»).
    pub built_in: bool,
}

impl From<crate::profile::Profile> for ProfileDto {
    fn from(profile: crate::profile::Profile) -> Self {
        Self {
            id: profile.id,
            name: profile.name,
            persona: profile.persona,
            style: profile.style,
            format: profile.format,
            constraints: profile.constraints,
            built_in: profile.built_in,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ProfilesResponse {
    pub profiles: Vec<ProfileDto>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreateProfileRequest {
    pub name: String,
    #[serde(default)]
    pub persona: String,
    #[serde(default)]
    pub style: String,
    #[serde(default)]
    pub format: String,
    #[serde(default)]
    pub constraints: Vec<String>,
}

/// Частичное изменение: незаданное поле сохраняет прежнее значение
/// (specs/user-profiles, «Частичное изменение профиля»).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateProfileRequest {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub persona: Option<String>,
    #[serde(default)]
    pub style: Option<String>,
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub constraints: Option<Vec<String>>,
}
