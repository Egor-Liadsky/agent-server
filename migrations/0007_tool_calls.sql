-- Вызовы инструментов в истории чата (tool calling). Колонки nullable:
-- строки, записанные до миграции, читаются как сообщения без вызовов.
-- tool_calls   — JSON Vec<ToolCall> у ответа модели, запросившего инструменты;
-- tool_call_id — у сообщения роли tool: на какой вызов оно отвечает;
-- tool_name    — у сообщения роли tool: имя инструмента (нужно Ollama).
ALTER TABLE messages ADD COLUMN tool_calls   TEXT;
ALTER TABLE messages ADD COLUMN tool_call_id TEXT;
ALTER TABLE messages ADD COLUMN tool_name    TEXT;
