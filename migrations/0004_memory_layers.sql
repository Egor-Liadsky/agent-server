-- Три слоя памяти: рабочая (задача чата) и долговременная (владелец)
-- (design.md, решения 1 и 2). Краткосрочный слой — существующая таблица
-- `messages`, схему не трогаем.

ALTER TABLE chats ADD COLUMN active_task_id TEXT;
UPDATE chats SET active_task_id = id WHERE active_task_id IS NULL;

CREATE TABLE chat_working_memory (
    id          TEXT    PRIMARY KEY,
    chat_id     TEXT    NOT NULL REFERENCES chats (id) ON DELETE CASCADE,
    task_id     TEXT    NOT NULL,
    key         TEXT    NOT NULL,
    value       TEXT    NOT NULL,
    source      TEXT    NOT NULL,   -- 'router' | 'manual'
    updated_at  INTEGER NOT NULL,
    UNIQUE (chat_id, task_id, key)
);
CREATE INDEX chat_working_memory_chat_task ON chat_working_memory (chat_id, task_id);

CREATE TABLE owner_long_term_memory (
    id             TEXT    PRIMARY KEY,
    owner          TEXT    NOT NULL,
    entry_type     TEXT    NOT NULL,   -- 'profile' | 'decision' | 'knowledge'
    key            TEXT,               -- NULL — запись без явного ключа
    value          TEXT    NOT NULL,
    source         TEXT    NOT NULL,   -- 'router' | 'manual'
    source_chat_id TEXT    REFERENCES chats (id) ON DELETE SET NULL,
    created_at     INTEGER NOT NULL,
    updated_at     INTEGER NOT NULL
);
CREATE INDEX owner_long_term_memory_owner ON owner_long_term_memory (owner, entry_type);
CREATE UNIQUE INDEX owner_long_term_memory_owner_key
    ON owner_long_term_memory (owner, entry_type, key)
    WHERE key IS NOT NULL;
