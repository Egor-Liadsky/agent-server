## 1. Автомат состояния задачи

- [x] 1.1 Добавить вариант `TaskStage::Clarification` в `src/task.rs`
      (`parse`, `as_str`) и убедиться, что `cargo build` проходит без
      предупреждений о неисчерпывающем `match`
- [x] 1.2 Обновить `EDGES`: убрать `planning → execution`, добавить
      `planning → clarification`, `clarification → execution`,
      `clarification → planning`; проверить тестом
      `can_transition(Planning, Execution) == false`
- [x] 1.3 Добавить unit-тесты в `src/task.rs` на новые рёбра
      (`planning→clarification`, `clarification→execution`,
      `clarification→planning`) и на отказ прямого
      `planning→execution`, по образцу существующих тестов
      `can_transition_*`

## 2. Автотрекер

- [x] 2.1 Обновить `tracker_prompt` в `src/task.rs`: добавить
      `clarification` в перечисление допустимых значений `stage` в
      JSON-схеме ответа и указать критерий перехода
      `clarification → execution` — только явное подтверждение
      пользователя в последнем ответе модели (design.md, решение 3)
- [x] 2.2 Добавить тест в `src/tests.rs`, мокающий трекер
      (`wiremock`) так, чтобы проверить: предложение
      `clarification` из `planning` применяется, предложение
      `execution` из `clarification` без признаков подтверждения
      пользователя в промпте всё равно уходит на модель как есть (сама
      классификация — ответственность модели, тест проверяет только
      применение/отклонение по автомату и лимитам)

## 3. Проверка

- [x] 3.1 Прогнать `cargo test` в `agent-sever` и убедиться, что все
      тесты (включая новые из 1.3 и 2.2) проходят
- [x] 3.2 Обновить README сервиса, если в нём перечислены этапы
      состояния задачи, добавив `clarification` в список
