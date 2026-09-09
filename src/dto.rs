//! Типы запроса и ответа контракта `/v1`.

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
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
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
            // заданный клиентом формат этот режим и включает.
            defaults.custom_response_mode = true;
        }
        let sampling = SamplingParams {
            temperature: self.temperature.or(defaults.sampling.temperature),
            top_p: self.top_p.or(defaults.sampling.top_p),
            top_k: self.top_k.or(defaults.sampling.top_k),
            frequency_penalty: self.frequency_penalty.or(defaults.sampling.frequency_penalty),
            presence_penalty: self.presence_penalty.or(defaults.sampling.presence_penalty),
        };
        defaults.sampling = sampling;
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
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsResponse {
    pub models: Vec<String>,
}
