-- Ручной откат миграции 0003_context_strategies.sql — на случай возврата
-- к образу сервиса до появления веток и фактов. sqlx не поддерживает down-
-- миграции в этом проекте, поэтому скрипт не запускается автоматически:
-- выполнить его вручную перед разворачиванием старого образа.
--
-- ЦЕНА ОТКАТА: переносятся только сообщения АКТИВНОЙ ветки каждого чата —
-- сообщения прочих веток теряются безвозвратно, старая схема их не знает.
-- Перед откатом убедитесь, что нужные ветки уже неактуальны или
-- экспортированы вручную.

CREATE TABLE messages_old (
    id          TEXT    PRIMARY KEY,
    chat_id     TEXT    NOT NULL REFERENCES chats (id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    role        TEXT    NOT NULL,
    content     TEXT    NOT NULL,
    reasoning   TEXT,
    meta        TEXT,
    created_at  INTEGER NOT NULL,
    UNIQUE (chat_id, seq)
);

INSERT INTO messages_old (id, chat_id, seq, role, content, reasoning, meta, created_at)
SELECT m.id, m.chat_id, m.seq, m.role, m.content, m.reasoning, m.meta, m.created_at
FROM messages m
JOIN chats c ON c.id = m.chat_id
WHERE m.branch_id = c.active_branch;

DROP TABLE messages;
ALTER TABLE messages_old RENAME TO messages;

CREATE INDEX messages_chat_seq ON messages (chat_id, seq);

ALTER TABLE chats DROP COLUMN active_branch;
DROP TABLE chat_branches;
DROP TABLE chat_facts;
