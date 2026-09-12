-- Пересказ вытесненной части истории чата. Отношение один к одному с
-- chats, но отдельная таблица: производное состояние отделено от
-- пользовательских настроек в chats.settings (design.md, решение 4).

CREATE TABLE chat_summaries (
    chat_id     TEXT    PRIMARY KEY REFERENCES chats (id) ON DELETE CASCADE,
    summary     TEXT    NOT NULL,
    through_seq INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);
