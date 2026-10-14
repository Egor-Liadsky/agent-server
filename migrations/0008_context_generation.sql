-- Граница контекста сохраняет архив сообщений, но отделяет новые ходы.
ALTER TABLE chats ADD COLUMN context_generation INTEGER NOT NULL DEFAULT 0;
ALTER TABLE messages ADD COLUMN context_generation INTEGER NOT NULL DEFAULT 0;
