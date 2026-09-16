-- Собственные профили владельца: явно заданный набор предпочтений
-- (роль, стиль, формат, ограничения), подставляемый в системное сообщение
-- каждого запроса чата (design.md, решение 1). Встроенные профили
-- (`teacher`, `psychologist`, `reviewer`) в этой таблице не хранятся —
-- они константы кода (design.md, решение 3).

CREATE TABLE owner_profiles (
    id          TEXT    PRIMARY KEY,
    owner       TEXT    NOT NULL,
    name        TEXT    NOT NULL,
    persona     TEXT    NOT NULL DEFAULT '',
    style       TEXT    NOT NULL DEFAULT '',
    format      TEXT    NOT NULL DEFAULT '',
    constraints TEXT    NOT NULL DEFAULT '[]',  -- JSON-массив строк
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL,
    UNIQUE (owner, id)
);
CREATE INDEX owner_profiles_owner ON owner_profiles (owner);
