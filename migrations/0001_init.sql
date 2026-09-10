-- Схема чатов и сообщений. Описание развилок — в design.md (решение 3).

CREATE TABLE chats (
    id          TEXT    PRIMARY KEY,
    owner       TEXT    NOT NULL,
    title       TEXT    NOT NULL,
    settings    TEXT    NOT NULL,   -- JSON ChatSettings
    created_at  INTEGER NOT NULL,   -- секунды Unix
    updated_at  INTEGER NOT NULL
);
CREATE INDEX chats_owner_updated ON chats (owner, updated_at DESC, id DESC);

CREATE TABLE messages (
    id          TEXT    PRIMARY KEY,
    chat_id     TEXT    NOT NULL REFERENCES chats (id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    role        TEXT    NOT NULL,   -- 'user' | 'assistant'
    content     TEXT    NOT NULL,
    reasoning   TEXT,
    meta        TEXT,               -- JSON MessageMeta
    created_at  INTEGER NOT NULL,
    UNIQUE (chat_id, seq)
);
