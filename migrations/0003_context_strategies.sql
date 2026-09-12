-- Ветвление диалога и устойчивые факты (design.md, решения 3 и 5).
-- BREAKING: заменяет UNIQUE (chat_id, seq) у messages на
-- UNIQUE (branch_id, seq) — SQLite не умеет менять ограничения существующей
-- таблицы, поэтому messages пересоздаётся и наполняется заново в этой же
-- транзакции.

CREATE TABLE chat_branches (
    id           TEXT    PRIMARY KEY,
    chat_id      TEXT    NOT NULL REFERENCES chats (id) ON DELETE CASCADE,
    parent_id    TEXT    REFERENCES chat_branches (id) ON DELETE CASCADE,
    fork_seq     INTEGER,              -- NULL у корневой ветки
    name         TEXT    NOT NULL,
    created_at   INTEGER NOT NULL
);
CREATE INDEX chat_branches_chat ON chat_branches (chat_id);

CREATE TABLE chat_facts (
    chat_id     TEXT    NOT NULL REFERENCES chats (id) ON DELETE CASCADE,
    key         TEXT    NOT NULL,
    value       TEXT    NOT NULL,
    through_seq INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL,
    PRIMARY KEY (chat_id, key)
);

-- Каждому существующему чату — корневая ветка, чтобы сообщения было куда
-- перенести (design.md, «Миграция существующих данных»).
ALTER TABLE chats ADD COLUMN active_branch TEXT;

INSERT INTO chat_branches (id, chat_id, parent_id, fork_seq, name, created_at)
SELECT lower(hex(randomblob(16))), id, NULL, NULL, 'root', created_at FROM chats;

UPDATE chats SET active_branch = (
    SELECT b.id FROM chat_branches b WHERE b.chat_id = chats.id
);

CREATE TABLE messages_new (
    id          TEXT    PRIMARY KEY,
    chat_id     TEXT    NOT NULL REFERENCES chats (id) ON DELETE CASCADE,
    branch_id   TEXT    NOT NULL REFERENCES chat_branches (id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    role        TEXT    NOT NULL,
    content     TEXT    NOT NULL,
    reasoning   TEXT,
    meta        TEXT,
    created_at  INTEGER NOT NULL,
    UNIQUE (branch_id, seq)
);

INSERT INTO messages_new (id, chat_id, branch_id, seq, role, content, reasoning, meta, created_at)
SELECT m.id, m.chat_id, c.active_branch, m.seq, m.role, m.content, m.reasoning, m.meta, m.created_at
FROM messages m JOIN chats c ON c.id = m.chat_id;

DROP TABLE messages;
ALTER TABLE messages_new RENAME TO messages;

CREATE INDEX messages_chat_seq ON messages (chat_id, seq);
CREATE INDEX messages_branch_seq ON messages (branch_id, seq);
