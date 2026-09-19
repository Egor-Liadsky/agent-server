//! Состояние задачи чата: конечный автомат этапов, шаг и ожидаемое
//! действие, пауза и бриф возобновления, раздел системного сообщения и
//! автоматический трекер (specs/task-state, design.md).

use crate::dto::ContextDto;
use crate::state::AppState;
use crate::store;
use agentcore::agent::{Agent, Message};
use agentcore::config::{ChatSettings, ReasoningMode, ThinkingMode};
use serde::Deserialize;

const TASK_SECTION_HEADING: &str = "Состояние задачи:";

/// Отличает вызов трекера состояния задачи от обычного диалогового вызова в
/// тестах, мокающих провайдера по содержимому тела запроса (по образцу
/// `memory::MEMORY_ROUTER_MARKER`).
#[cfg_attr(not(test), allow(dead_code))]
pub const TASK_TRACKER_MARKER: &str = "Верни JSON с предлагаемым переходом состояния задачи";

/// Этап задачи — закрытый набор (specs/task-state, «Состояние задачи — этап,
/// шаг и ожидаемое действие»).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStage {
    Planning,
    Clarification,
    Execution,
    Validation,
    Done,
}

impl TaskStage {
    pub fn parse(value: &str) -> Option<TaskStage> {
        match value {
            "planning" => Some(TaskStage::Planning),
            "clarification" => Some(TaskStage::Clarification),
            "execution" => Some(TaskStage::Execution),
            "validation" => Some(TaskStage::Validation),
            "done" => Some(TaskStage::Done),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            TaskStage::Planning => "planning",
            TaskStage::Clarification => "clarification",
            TaskStage::Execution => "execution",
            TaskStage::Validation => "validation",
            TaskStage::Done => "done",
        }
    }
}

/// Допустимые рёбра автомата (design.md, решение 1): пропуск этапа и
/// переход из `done` отклоняются, переход в тот же этап не входит в список
/// — он не ребро, а обновление шага/ожидаемого действия без смены этапа.
/// Прямое ребро `planning → execution` отсутствует: уточнение плана
/// обязательно проходит через `clarification` (design.md, решение 1).
const EDGES: [(TaskStage, TaskStage); 7] = [
    (TaskStage::Planning, TaskStage::Clarification),
    (TaskStage::Clarification, TaskStage::Execution),
    (TaskStage::Clarification, TaskStage::Planning),
    (TaskStage::Execution, TaskStage::Validation),
    (TaskStage::Validation, TaskStage::Done),
    (TaskStage::Validation, TaskStage::Execution),
    (TaskStage::Execution, TaskStage::Planning),
];

pub fn can_transition(from: TaskStage, to: TaskStage) -> bool {
    EDGES.iter().any(|(a, b)| *a == from && *b == to)
}

/// Причина отказа применения перехода или обновления полей —
/// записывается в журнал так же, как `memory::RejectReason`.
#[derive(Debug)]
pub enum TaskError {
    /// Имя этапа не входит в закрытый набор этапов (specs/task-state,
    /// «Неизвестный этап отличается от недопустимого перехода»).
    UnknownStage,
    /// Переход не входит в список допустимых рёбер (включая пропуск этапа
    /// и переход из `done`).
    InvalidEdge,
    /// Задача на паузе — переходы отклоняются до снятия с паузы.
    Paused,
    /// Текст шага длиннее операторского лимита.
    StepTooLong,
    /// Текст ожидаемого действия длиннее операторского лимита.
    ExpectedActionTooLong,
    /// Чат не найден или принадлежит другому владельцу.
    NotFound,
    Backend(store::StoreError),
}

impl From<store::StoreError> for TaskError {
    fn from(err: store::StoreError) -> Self {
        match err {
            store::StoreError::NotFound => TaskError::NotFound,
            other => TaskError::Backend(other),
        }
    }
}

impl PartialEq for TaskError {
    fn eq(&self, other: &Self) -> bool {
        self.as_log_str() == other.as_log_str()
    }
}

impl TaskError {
    pub fn as_log_str(&self) -> &'static str {
        match self {
            TaskError::UnknownStage => "unknown_stage",
            TaskError::InvalidEdge => "invalid_edge",
            TaskError::Paused => "paused",
            TaskError::StepTooLong => "step_too_long",
            TaskError::ExpectedActionTooLong => "expected_action_too_long",
            TaskError::NotFound => "not_found",
            TaskError::Backend(_) => "backend_error",
        }
    }
}

pub struct Limits {
    pub step_max_chars: usize,
    pub expected_action_max_chars: usize,
}

impl Limits {
    pub fn from_config(config: &crate::config::AgentdConfig) -> Self {
        Self {
            step_max_chars: config.task_step_max_chars as usize,
            expected_action_max_chars: config.task_expected_action_max_chars as usize,
        }
    }
}

fn check_text_limits(step: Option<&str>, expected_action: Option<&str>, limits: &Limits) -> Result<(), TaskError> {
    if let Some(step) = step {
        if step.chars().count() > limits.step_max_chars {
            return Err(TaskError::StepTooLong);
        }
    }
    if let Some(expected_action) = expected_action {
        if expected_action.chars().count() > limits.expected_action_max_chars {
            return Err(TaskError::ExpectedActionTooLong);
        }
    }
    Ok(())
}

/// Применяет переход этапа с проверкой автомата, паузы и лимитов длины
/// (specs/task-state, «Переходы ограничены конечным автоматом», «Переход
/// задаёт шаг и ожидаемое действие», «Пауза на любом этапе»). Отказ не
/// меняет состояние (design.md, решение 3 — частичного применения нет).
/// Переход в `done` не проходит через эту функцию: он необратим и
/// проводится отдельным путём через `store::finish_task` (specs/task-state,
/// «Переход в done завершает задачу»).
pub async fn apply_manual_transition(
    state: &AppState,
    owner: &str,
    chat_id: &str,
    to_stage: &str,
    step: Option<&str>,
    expected_action: Option<&str>,
) -> Result<store::TaskState, TaskError> {
    let current = store::load_task_state(&state.db, owner, chat_id).await?;
    let from = TaskStage::parse(&current.stage).ok_or(TaskError::UnknownStage)?;
    // Имя вне закрытого набора этапов — отдельная причина отказа, отличная
    // от недопустимого ребра (specs/task-state, «Неизвестный этап
    // отличается от недопустимого перехода»).
    let to = TaskStage::parse(to_stage).ok_or(TaskError::UnknownStage)?;
    if current.paused {
        return Err(TaskError::Paused);
    }
    if !can_transition(from, to) {
        return Err(TaskError::InvalidEdge);
    }
    let limits = Limits::from_config(&state.config);
    check_text_limits(step, expected_action, &limits)?;

    let updated =
        store::write_task_transition(&state.db, &current.id, from.as_str(), to.as_str(), step, expected_action, "manual", "")
            .await?;
    log_transition(state, chat_id, from.as_str(), to.as_str(), "manual", step, expected_action, "");
    Ok(updated)
}

/// Проверяет, что задачу можно завершить переходом в `done`: ребро
/// `validation → done` и отсутствие паузы (specs/task-state, «Переход в
/// done завершает задачу»). Завершение идёт отдельным путём через
/// `store::finish_task`, но правило автомата остаётся здесь, а не в
/// обработчике HTTP (design.md, решение 3).
pub async fn check_can_finish(state: &AppState, owner: &str, chat_id: &str) -> Result<(), TaskError> {
    let current = store::load_task_state(&state.db, owner, chat_id).await?;
    if current.paused {
        return Err(TaskError::Paused);
    }
    let from = TaskStage::parse(&current.stage).ok_or(TaskError::UnknownStage)?;
    if !can_transition(from, TaskStage::Done) {
        return Err(TaskError::InvalidEdge);
    }
    Ok(())
}

/// Пишет запись в журнал сервиса о применённом переходе: тексты шага,
/// ожидаемого действия и причины — только при `AGENTD_LOG_CONTENT=true`,
/// иначе только этапы, источник и длины (specs/task-state, «Тексты
/// состояния задачи маскируются в журнале»), тем же приёмом, что
/// `memory::log_applied`.
fn log_transition(
    state: &AppState,
    chat_id: &str,
    from_stage: &str,
    to_stage: &str,
    source: &str,
    step: Option<&str>,
    expected_action: Option<&str>,
    reason: &str,
) {
    if state.config.log_content {
        tracing::info!(
            chat_id = %chat_id,
            from_stage,
            to_stage,
            source,
            step,
            expected_action,
            reason,
            "переход состояния задачи применён"
        );
    } else {
        tracing::info!(
            chat_id = %chat_id,
            from_stage,
            to_stage,
            source,
            step_chars = step.map(|s| s.chars().count()),
            expected_action_chars = expected_action.map(|s| s.chars().count()),
            reason_chars = reason.chars().count(),
            "переход состояния задачи применён"
        );
    }
}

/// Пишет запись об обновлении шага и ожидаемого действия без смены этапа
/// по предложению трекера. Маскирование текстов — то же, что у
/// `log_transition` (specs/task-state, «Тексты состояния задачи
/// маскируются в журнале»).
fn log_fields_update(
    state: &AppState,
    chat_id: &str,
    stage: &str,
    step: Option<&str>,
    expected_action: Option<&str>,
) {
    if state.config.log_content {
        tracing::info!(
            chat_id = %chat_id,
            stage,
            source = "auto",
            step,
            expected_action,
            "шаг и ожидаемое действие обновлены без смены этапа"
        );
    } else {
        tracing::info!(
            chat_id = %chat_id,
            stage,
            source = "auto",
            step_chars = step.map(|s| s.chars().count()),
            expected_action_chars = expected_action.map(|s| s.chars().count()),
            "шаг и ожидаемое действие обновлены без смены этапа"
        );
    }
}

/// Меняет только шаг и/или ожидаемое действие, без смены этапа
/// (specs/task-state, «Обновление шага без смены этапа»). Задача на паузе
/// неизменяема, кроме снятия с паузы (design.md, решение 6).
pub async fn update_fields(
    state: &AppState,
    owner: &str,
    chat_id: &str,
    step: Option<&str>,
    expected_action: Option<&str>,
) -> Result<store::TaskState, TaskError> {
    let current = store::load_task_state(&state.db, owner, chat_id).await?;
    if current.paused {
        return Err(TaskError::Paused);
    }
    let limits = Limits::from_config(&state.config);
    check_text_limits(step, expected_action, &limits)?;
    Ok(store::update_task_fields(&state.db, owner, chat_id, step, expected_action).await?)
}

/// Бриф возобновления, собранный детерминированно из этапа, шага, ожидаемого
/// действия и записей рабочей памяти задачи, без обращения к модели
/// (specs/task-state, «Бриф возобновления собирается при паузе без вызова
/// модели»). Обрезается по операторскому лимиту: сначала по числу записей
/// памяти, затем по длине значения — этап, шаг и ожидаемое действие
/// остаются целиком (design.md, решение 7).
pub fn build_resume_brief(
    stage: &str,
    step: &str,
    expected_action: &str,
    working_memory: &[store::WorkingMemoryEntry],
    max_chars: usize,
) -> String {
    let mut lines = vec![
        format!("Этап на паузе: {stage}"),
        format!("Шаг: {step}"),
        format!("Ожидаемое действие: {expected_action}"),
    ];
    if !working_memory.is_empty() {
        lines.push("Рабочая память задачи:".to_string());
        for entry in working_memory {
            lines.push(format!("- {}: {}", entry.key, entry.value));
        }
    }
    let mut brief = lines.join("\n");
    while brief.chars().count() > max_chars && lines.len() > 3 {
        lines.pop();
        brief = lines.join("\n");
    }
    if brief.chars().count() > max_chars {
        brief = brief.chars().take(max_chars).collect();
    }
    brief
}

/// Ставит задачу на паузу, сохраняя бриф возобновления, собранный на месте
/// (specs/task-state, «Пауза сохраняет состояние», «Пауза не обращается к
/// модели»). Идемпотентно: повторная пауза перезаписывает бриф новым шагом
/// (specs/task-state, «Повторная пауза обновляет бриф»).
pub async fn pause(state: &AppState, owner: &str, chat_id: &str) -> Result<store::TaskState, TaskError> {
    let current = store::load_task_state(&state.db, owner, chat_id).await?;
    let working_memory = store::load_working_memory(&state.db, chat_id, &current.id).await.unwrap_or_default();
    let max_chars = state.config.task_resume_brief_max_chars as usize;
    let brief =
        build_resume_brief(&current.stage, &current.step, &current.expected_action, &working_memory, max_chars);
    let updated = store::set_task_paused(&state.db, &current.id, true, Some(&brief)).await?;
    if state.config.log_content {
        tracing::info!(chat_id = %chat_id, stage = %current.stage, resume_brief = %brief, "задача поставлена на паузу");
    } else {
        tracing::info!(chat_id = %chat_id, stage = %current.stage, resume_brief_chars = brief.chars().count(), "задача поставлена на паузу");
    }
    Ok(updated)
}

/// Снимает задачу с паузы, сохраняя прежний этап, шаг и ожидаемое действие
/// (specs/task-state, «Снятие с паузы возвращает тот же этап»). Идемпотентно.
pub async fn resume(state: &AppState, owner: &str, chat_id: &str) -> Result<store::TaskState, TaskError> {
    let current = store::load_task_state(&state.db, owner, chat_id).await?;
    Ok(store::set_task_paused(&state.db, &current.id, false, None).await?)
}

// --- Раздел системного сообщения (specs/task-state, «Состояние задачи
// подставляется в системное сообщение») ---

/// Раздел состояния задачи, готовый к вставке в список `sections`
/// `system_message::build` наравне с разделами слоистой памяти
/// (design.md, «Опорные факты»). `None`, только если задача уже завершена
/// (`done`) — свежесозданная задача имеет пустые шаг и ожидаемое действие,
/// но раздел всё равно присутствует, пока состояние задачи включено.
pub fn system_message_section(task: &store::TaskState) -> String {
    let mut lines = vec![
        TASK_SECTION_HEADING.to_string(),
        format!("Этап: {}", task.stage),
        format!("Шаг: {}", if task.step.is_empty() { "(не задан)" } else { &task.step }),
        format!(
            "Ожидаемое действие: {}",
            if task.expected_action.is_empty() { "(не задано)" } else { &task.expected_action }
        ),
    ];
    if task.paused {
        lines.push("Задача на паузе.".to_string());
        if !task.resume_brief.is_empty() {
            lines.push(format!("Бриф возобновления:\n{}", task.resume_brief));
        }
    }
    lines.join("\n")
}

// --- Операторские умолчания и переопределения на чат (design.md, решение 8) ---

pub fn effective_enabled(state: &AppState, settings: &ChatSettings) -> bool {
    settings.task_state_enabled.unwrap_or(state.config.task_state_enabled)
}

pub fn effective_auto_enabled(state: &AppState, settings: &ChatSettings) -> bool {
    settings.task_state_auto_enabled.unwrap_or(state.config.task_state_auto_enabled)
}

/// Подмешивает состояние задачи в уже собранный `ContextDto`, на момент
/// сборки запроса (specs/task-state, «Наблюдаемость состояния задачи в
/// ответе»).
pub fn merge_into_context(context: &mut ContextDto, task: &store::TaskState) {
    context.task_stage = Some(task.stage.clone());
    context.task_step = Some(task.step.clone());
    context.task_expected_action = Some(task.expected_action.clone());
    context.task_paused = Some(task.paused);
}

/// Подмешивает счётчики последнего завершившегося прогона трекера — по
/// ПРЕДЫДУЩЕМУ сообщению, трекер фоновый (design.md, решение 5).
pub fn merge_tracker_counters(context: &mut ContextDto, outcome: TrackOutcome) {
    context.task_tracker_applied = Some(outcome.applied);
    context.task_tracker_updated = Some(outcome.updated);
    context.task_tracker_rejected = Some(outcome.rejected);
}

// --- Автоматический трекер (design.md, решение 5, по образцу
// `memory::route_after_exchange`) ---

#[derive(Debug, Deserialize)]
struct RawTrackerProposal {
    stage: String,
    #[serde(default)]
    step: Option<String>,
    #[serde(default)]
    expected_action: Option<String>,
    #[serde(default)]
    reason: Option<String>,
}

/// Разбирает ответ трекера как один объект перехода — тот же приём поиска
/// первой `{`/последней `}`, что у `facts::parse_operations`.
fn parse_proposal(content: &str) -> Option<RawTrackerProposal> {
    let start = content.find('{')?;
    let end = content.rfind('}')?;
    if end < start {
        return None;
    }
    serde_json::from_str(&content[start..=end]).ok()
}

/// Промпт трекера подаёт обе реплики последнего обмена отдельными блоками
/// (design.md, решение 1): критерий перехода в `execution` — подтверждение
/// пользователя, и оно живёт только в его сообщении, поэтому по одному
/// ответу модели этот переход непроверяем.
fn tracker_prompt(task: &store::TaskState, user_message: &str, assistant_reply: &str) -> String {
    format!(
        "Текущее состояние задачи: этап {}, шаг «{}», ожидаемое действие «{}».\n\n\
         Последнее сообщение пользователя:\n{}\n\n\
         Последний ответ модели:\n{}\n\n\
         {TASK_TRACKER_MARKER} на основе этого обмена, если он сдвигает задачу вперёд, назад или \
         не меняет её. Переход в clarification предлагай, когда план собран и последний ответ \
         модели задаёт пользователю уточняющие вопросы по этому плану. Переход из clarification \
         в execution предлагай тогда, когда последнее сообщение пользователя подтверждает \
         готовность приступить к выполнению — отвечает на заданные вопросы или прямо требует \
         начать работу. Прямое требование начать («начинай выполнение», «пиши код») достаточно \
         само по себе: неотвеченные вопросы и новые вопросы в ответе модели его не отменяют — \
         решать, хватает ли данных, пользователь уже решил. Если этап не \
         меняется, всё равно верни текущий этап с обновлёнными шагом и ожидаемым действием по \
         этому обмену. Ответь только JSON-объектом вида \
         {{\"stage\":\"planning\"|\"clarification\"|\"execution\"|\"validation\"|\"done\",\"step\":\"...\",\
         \"expected_action\":\"...\",\"reason\":\"...\"}} — этапом, который, по-твоему, сейчас \
         действителен, текущим шагом, ожидаемым действием и краткой причиной.",
        task.stage, task.step, task.expected_action, user_message, assistant_reply
    )
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TrackOutcome {
    pub applied: u32,
    /// Предложения с ТЕКУЩИМ этапом, применённые как обновление шага и
    /// ожидаемого действия: это не ребро автомата и не отказ
    /// (specs/task-state, «Ответ различает обновление полей и переход»).
    pub updated: u32,
    pub rejected: u32,
}

/// Запускает трекер после отправки ответа пользователю: один
/// дополнительный вызов модели, разбор предложения, проверка тем же
/// автоматом и теми же лимитами, что и явный переход (specs/task-state,
/// «Автоматический трекер предлагает переход»). Не запускается для
/// приостановленной задачи (specs/task-state, «Трекер не запускается на
/// паузе»). Предложение `done` отбрасывается всегда — завершение задачи
/// остаётся ручным действием (specs/task-state, «Трекер не завершает
/// задачу»). Отказ вызова или разбора логируется и не меняет состояние
/// (specs/task-state, «Отказ трекера не ломает ответ»).
pub async fn track_after_exchange(
    state: &AppState,
    chat: &store::Chat,
    settings: &ChatSettings,
    owner: &str,
    user_message: &str,
    assistant_reply: &str,
) -> TrackOutcome {
    let current = match store::load_task_state(&state.db, owner, &chat.id).await {
        Ok(task) => task,
        Err(err) => {
            tracing::warn!(chat_id = %chat.id, error = %err, "не удалось прочитать состояние задачи для трекера");
            return TrackOutcome::default();
        }
    };
    if current.paused {
        return TrackOutcome::default();
    }

    let prompt = tracker_prompt(&current, user_message, assistant_reply);
    let mut tracker_settings = settings.clone();
    if let Some(model) = &state.config.task_state_model {
        tracker_settings.model = Some(model.clone());
    }
    tracker_settings.reasoning = ReasoningMode::Default;
    tracker_settings.thinking = ThinkingMode::Disabled;
    tracker_settings.custom_response_mode = false;

    let reply = match state.agent.ask(&[Message::user(prompt)], &tracker_settings).await {
        Ok(reply) => reply,
        Err(err) => {
            tracing::warn!(chat_id = %chat.id, error = %err, "не удалось запросить предложение трекера состояния задачи: вызов модели не удался");
            return TrackOutcome::default();
        }
    };
    let Some(proposal) = parse_proposal(&reply.content) else {
        tracing::warn!(chat_id = %chat.id, "не удалось разобрать ответ трекера состояния задачи как JSON-объект");
        return TrackOutcome::default();
    };

    let mut outcome = TrackOutcome::default();
    let limits = Limits::from_config(&state.config);
    let log_reject = |reason: &str| {
        tracing::info!(
            chat_id = %chat.id,
            reject_reason = reason,
            accepted = false,
            "предложение трекера состояния задачи отброшено"
        );
    };

    let Some(to) = TaskStage::parse(&proposal.stage) else {
        log_reject("unknown_stage");
        outcome.rejected += 1;
        return outcome;
    };
    if to == TaskStage::Done {
        // Предложение done отбрасывается всегда, даже если ребро в автомате
        // формально допустимо (specs/task-state, «Трекер не завершает
        // задачу»): завершение задачи необратимо чистит рабочую память.
        log_reject("done_is_never_applied_by_tracker");
        outcome.rejected += 1;
        return outcome;
    }
    let Some(from) = TaskStage::parse(&current.stage) else {
        log_reject("unknown_stage");
        outcome.rejected += 1;
        return outcome;
    };
    // Предложение с текущим этапом — не ребро автомата, а обновление шага и
    // ожидаемого действия (specs/task-state, «Предложение текущего этапа
    // обновляет шаг и ожидаемое действие»). Без этой ветки шаг замерзает на
    // значениях, записанных при входе в этап, и системное сообщение начинает
    // работать против автомата (design.md, решение 2).
    if to == from {
        if check_text_limits(proposal.step.as_deref(), proposal.expected_action.as_deref(), &limits).is_err() {
            log_reject("text_too_long");
            outcome.rejected += 1;
            return outcome;
        }
        if proposal.step.is_none() && proposal.expected_action.is_none() {
            return outcome;
        }
        // Запись в журнал переходов не делается: журнал остаётся журналом
        // переходов (add-task-state-machine, решение 3).
        match store::update_task_fields(
            &state.db,
            owner,
            &chat.id,
            proposal.step.as_deref(),
            proposal.expected_action.as_deref(),
        )
        .await
        {
            Ok(_) => {
                log_fields_update(
                    state,
                    &chat.id,
                    from.as_str(),
                    proposal.step.as_deref(),
                    proposal.expected_action.as_deref(),
                );
                outcome.updated += 1;
            }
            Err(err) => {
                tracing::warn!(chat_id = %chat.id, error = %err, "не удалось обновить шаг и ожидаемое действие по предложению трекера");
                outcome.rejected += 1;
            }
        }
        return outcome;
    }
    if !can_transition(from, to) {
        log_reject("invalid_edge");
        outcome.rejected += 1;
        return outcome;
    }
    if check_text_limits(proposal.step.as_deref(), proposal.expected_action.as_deref(), &limits).is_err() {
        log_reject("text_too_long");
        outcome.rejected += 1;
        return outcome;
    }

    let reason = proposal.reason.unwrap_or_default();
    match store::write_task_transition(
        &state.db,
        &current.id,
        from.as_str(),
        to.as_str(),
        proposal.step.as_deref(),
        proposal.expected_action.as_deref(),
        "auto",
        &reason,
    )
    .await
    {
        Ok(_) => {
            log_transition(
                state,
                &chat.id,
                from.as_str(),
                to.as_str(),
                "auto",
                proposal.step.as_deref(),
                proposal.expected_action.as_deref(),
                &reason,
            );
            outcome.applied += 1;
        }
        Err(err) => {
            tracing::warn!(chat_id = %chat.id, error = %err, "не удалось применить предложение трекера состояния задачи");
            outcome.rejected += 1;
        }
    }
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- 4.1 Автомат: допустимые рёбра и отказы ---

    #[test]
    fn all_designed_edges_are_allowed() {
        assert!(can_transition(TaskStage::Planning, TaskStage::Clarification));
        assert!(can_transition(TaskStage::Clarification, TaskStage::Execution));
        assert!(can_transition(TaskStage::Clarification, TaskStage::Planning));
        assert!(can_transition(TaskStage::Execution, TaskStage::Validation));
        assert!(can_transition(TaskStage::Validation, TaskStage::Done));
        assert!(can_transition(TaskStage::Validation, TaskStage::Execution));
        assert!(can_transition(TaskStage::Execution, TaskStage::Planning));
    }

    #[test]
    fn direct_planning_to_execution_is_rejected() {
        assert!(!can_transition(TaskStage::Planning, TaskStage::Execution));
    }

    #[test]
    fn skipping_a_stage_is_rejected() {
        assert!(!can_transition(TaskStage::Planning, TaskStage::Validation));
        assert!(!can_transition(TaskStage::Planning, TaskStage::Done));
    }

    #[test]
    fn transition_from_done_is_rejected() {
        assert!(!can_transition(TaskStage::Done, TaskStage::Planning));
        assert!(!can_transition(TaskStage::Done, TaskStage::Execution));
    }

    #[test]
    fn same_stage_is_not_an_edge() {
        assert!(!can_transition(TaskStage::Execution, TaskStage::Execution));
    }

    #[test]
    fn unknown_stage_name_does_not_parse() {
        assert_eq!(TaskStage::parse("mystery"), None);
    }

    // --- 4.2 Лимиты длины ---

    #[test]
    fn step_over_limit_is_rejected() {
        let limits = Limits { step_max_chars: 5, expected_action_max_chars: 100 };
        assert_eq!(check_text_limits(Some(&"a".repeat(6)), None, &limits), Err(TaskError::StepTooLong));
    }

    #[test]
    fn expected_action_over_limit_is_rejected() {
        let limits = Limits { step_max_chars: 100, expected_action_max_chars: 5 };
        assert_eq!(
            check_text_limits(None, Some(&"a".repeat(6)), &limits),
            Err(TaskError::ExpectedActionTooLong)
        );
    }

    #[test]
    fn text_within_limit_is_accepted() {
        let limits = Limits { step_max_chars: 5, expected_action_max_chars: 5 };
        assert!(check_text_limits(Some("ok"), Some("ok"), &limits).is_ok());
    }

    // --- 4.4 Бриф возобновления ---

    #[test]
    fn resume_brief_contains_stage_step_expected_action_and_memory_keys() {
        let memory = vec![store::WorkingMemoryEntry {
            key: "target".to_string(),
            value: "iOS 17+".to_string(),
            source: "manual".to_string(),
            updated_at: 0,
        }];
        let brief = build_resume_brief("execution", "правит парсер", "ждёт ревью", &memory, 10_000);
        assert!(brief.contains("execution"));
        assert!(brief.contains("правит парсер"));
        assert!(brief.contains("ждёт ревью"));
        assert!(brief.contains("target: iOS 17+"));
    }

    #[test]
    fn resume_brief_truncates_memory_entries_before_dropping_stage_fields() {
        let memory: Vec<_> = (0..50)
            .map(|i| store::WorkingMemoryEntry {
                key: format!("k{i}"),
                value: "значение подлиннее чем один символ".to_string(),
                source: "manual".to_string(),
                updated_at: 0,
            })
            .collect();
        let brief = build_resume_brief("execution", "шаг", "действие", &memory, 60);
        assert!(brief.contains("execution"));
        assert!(brief.contains("шаг"));
        assert!(brief.contains("действие"));
        assert!(brief.chars().count() <= 60);
    }

    // --- Раздел системного сообщения ---

    #[test]
    fn section_reports_stage_step_and_expected_action() {
        let task = store::TaskState {
            id: "t1".to_string(),
            chat_id: "c1".to_string(),
            stage: "execution".to_string(),
            step: "правит парсер".to_string(),
            expected_action: "ждёт ответа модели".to_string(),
            paused: false,
            resume_brief: String::new(),
            created_at: 0,
            updated_at: 0,
        };
        let section = system_message_section(&task);
        assert!(section.starts_with(TASK_SECTION_HEADING));
        assert!(section.contains("execution"));
        assert!(section.contains("правит парсер"));
        assert!(section.contains("ждёт ответа модели"));
        assert!(!section.contains("паузе"));
    }

    #[test]
    fn section_marks_pause_and_carries_resume_brief() {
        let task = store::TaskState {
            id: "t1".to_string(),
            chat_id: "c1".to_string(),
            stage: "execution".to_string(),
            step: "шаг".to_string(),
            expected_action: "действие".to_string(),
            paused: true,
            resume_brief: "бриф возобновления".to_string(),
            created_at: 0,
            updated_at: 0,
        };
        let section = system_message_section(&task);
        assert!(section.contains("паузе"));
        assert!(section.contains("бриф возобновления"));
    }

    // --- Трекер: разбор ---

    #[test]
    fn parses_valid_proposal() {
        let content = r#"вот ответ: {"stage":"execution","step":"шаг","expected_action":"действие","reason":"причина"}"#;
        let proposal = parse_proposal(content).expect("разбор");
        assert_eq!(proposal.stage, "execution");
        assert_eq!(proposal.step.as_deref(), Some("шаг"));
    }

    #[test]
    fn unparseable_proposal_is_none() {
        assert!(parse_proposal("не JSON вовсе").is_none());
    }

    // --- Операторские умолчания и переопределения ---

    #[tokio::test]
    async fn chat_setting_overrides_operator_default_task_state_enabled() {
        let state = AppState::for_tests().await;
        assert_eq!(effective_enabled(&state, &ChatSettings::default()), state.config.task_state_enabled);
        let settings = ChatSettings { task_state_enabled: Some(true), ..ChatSettings::default() };
        assert!(effective_enabled(&state, &settings));
    }

    #[tokio::test]
    async fn chat_setting_overrides_operator_default_task_state_auto_enabled() {
        let state = AppState::for_tests().await;
        let settings = ChatSettings { task_state_auto_enabled: Some(false), ..ChatSettings::default() };
        assert!(!effective_auto_enabled(&state, &settings));
    }

    // --- 6.4 Тексты переходов маскируются в журнале по умолчанию ---

    fn captured(f: impl FnOnce()) -> String {
        let capture = crate::telemetry::capture::Capture::default();
        let subscriber = tracing_subscriber::fmt().json().with_writer(capture.clone()).finish();
        tracing::subscriber::with_default(subscriber, f);
        capture.text()
    }

    #[test]
    fn transition_step_is_absent_from_log_by_default() {
        let runtime = tokio::runtime::Runtime::new().expect("runtime");
        let mut state = runtime.block_on(AppState::for_tests());
        std::sync::Arc::make_mut(&mut state.config).log_content = false;
        let chat = runtime
            .block_on(store::create_chat(&state.db, "owner-1", "Чат", &ChatSettings::default()))
            .expect("чат");
        runtime.block_on(store::load_task_state(&state.db, "owner-1", &chat.id)).expect("состояние задачи");

        let output = captured(|| {
            let _ = runtime.block_on(apply_manual_transition(&state, "owner-1", &chat.id, "clarification", Some("секретный шаг"), None));
        });
        assert!(output.contains("\"to_stage\":\"clarification\""), "этап должен остаться в журнале: {output}");
        assert!(!output.contains("секретный шаг"), "текст шага не должен попасть в журнал: {output}");
    }

    #[test]
    fn transition_step_is_logged_when_content_logging_enabled() {
        let runtime = tokio::runtime::Runtime::new().expect("runtime");
        let mut state = runtime.block_on(AppState::for_tests());
        std::sync::Arc::make_mut(&mut state.config).log_content = true;
        let chat = runtime
            .block_on(store::create_chat(&state.db, "owner-1", "Чат", &ChatSettings::default()))
            .expect("чат");
        runtime.block_on(store::load_task_state(&state.db, "owner-1", &chat.id)).expect("состояние задачи");

        let output = captured(|| {
            let _ = runtime.block_on(apply_manual_transition(&state, "owner-1", &chat.id, "clarification", Some("видимый шаг"), None));
        });
        assert!(output.contains("видимый шаг"), "текст шага должен попасть в журнал при AGENTD_LOG_CONTENT=true: {output}");
    }
}
