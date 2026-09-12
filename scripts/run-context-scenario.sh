#!/bin/bash
# Прогоняет tests/data/context-scenario.json на одной стратегии контекста
# против запущенного agentd и печатает по каждому сообщению ответ, блок
# context и usage — данные для отчёта docs/context-strategies-comparison.md.
#
# Использование: AGENTD_URL=http://127.0.0.1:8090 ./run-context-scenario.sh <strategy>
# strategy — summary | sliding_window | facts (ветвление — отдельный
# run-context-scenario-branching.sh, сценарий требует двух веток).
set -euo pipefail
STRATEGY="$1"
BASE="${AGENTD_URL:-http://127.0.0.1:8090}"
SCENARIO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/tests/data/context-scenario.json"

CHAT=$(curl -s -X POST "$BASE/v1/chats" -H 'content-type: application/json' \
  -d "$(jq -n --arg s "$STRATEGY" '{title: ("scenario-" + $s), settings: {
    context_strategy: $s,
    custom_response_mode: true,
    response_format: {description: "Отвечай кратко и по делу, не больше 6-8 коротких абзацев или пунктов, без лишних преамбул.", max_length: 500}
  }}')")
CHAT_ID=$(echo "$CHAT" | jq -r '.id')
echo "chat_id=$CHAT_ID strategy=$STRATEGY"

TOTAL_TOKENS=0
N=$(jq '.messages | length' "$SCENARIO")
for i in $(seq 0 $((N-1))); do
  MSG=$(jq -r ".messages[$i]" "$SCENARIO")
  RESP=$(curl -s -X POST "$BASE/v1/chat" -H 'content-type: application/json' \
    -d "$(jq -n --arg cid "$CHAT_ID" --arg p "$MSG" '{chat_id:$cid,prompt:$p}')")
  TOKENS=$(echo "$RESP" | jq -r '.usage.total_tokens // 0')
  TOTAL_TOKENS=$((TOTAL_TOKENS + TOKENS))
  echo "--- сообщение $((i+1))/$N (tokens=$TOKENS) ---"
  echo "$RESP" | jq -c '{content, context, usage, error}'
done
echo "TOTAL_TOKENS_MAIN=$TOTAL_TOKENS"
echo "CHAT_ID_FINAL=$CHAT_ID"
