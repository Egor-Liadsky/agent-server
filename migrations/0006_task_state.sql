-- Явное состояние активной задачи чата: этап, шаг, ожидаемое действие,
-- пауза и бриф возобновления (design.md, решение 1); журнал переходов —
-- отдельная таблица, привязанная к задаче, а не к чату, чтобы история
-- прежней задачи не исчезала при смене active_task_id (design.md, решение 1).
-- Строка задачи создаётся лениво при первом чтении или изменении
-- (design.md, решение 2) — миграция существующие чаты не трогает.

CREATE TABLE chat_tasks (
    id            TEXT    PRIMARY KEY,
    chat_id       TEXT    NOT NULL REFERENCES chats (id) ON DELETE CASCADE,
    stage         TEXT    NOT NULL DEFAULT 'planning',
    step          TEXT    NOT NULL DEFAULT '',
    expected_action TEXT  NOT NULL DEFAULT '',
    paused        INTEGER NOT NULL DEFAULT 0,
    resume_brief  TEXT    NOT NULL DEFAULT '',
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL
);
CREATE INDEX chat_tasks_chat_id ON chat_tasks (chat_id);

CREATE TABLE chat_task_transitions (
    id         TEXT    PRIMARY KEY,
    task_id    TEXT    NOT NULL REFERENCES chat_tasks (id) ON DELETE CASCADE,
    from_stage TEXT    NOT NULL,
    to_stage   TEXT    NOT NULL,
    source     TEXT    NOT NULL,   -- 'manual' | 'auto'
    reason     TEXT    NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL
);
CREATE INDEX chat_task_transitions_task_id ON chat_task_transitions (task_id);
