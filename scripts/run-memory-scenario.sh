#!/bin/bash
# Прогоняет tests/data/memory-scenario.json на одной стратегии контекста
# (memory_layers | sliding_window) против запущенного agentd: сначала все
# сообщения сценария в чате A (внутричатный контрольный вопрос — последнее
# сообщение сценария), затем создаёт новый чат B того же владельца и задаёт
# межчатный контрольный вопрос — данные для
# openspec/changes/add-memory-layers/comparison.md.
#
# Использование: AGENTD_URL=http://127.0.0.1:8090 ./run-memory-scenario.sh <strategy>
set -euo pipefail
STRATEGY="$1"
BASE="${AGENTD_URL:-http://127.0.0.1:8090}"
SCENARIO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/tests/data/memory-scenario.json"

CHAT_A=$(curl -s -X POST "$BASE/v1/chats" -H 'content-type: application/json' \
  -d "$(jq -n --arg s "$STRATEGY" '{title: ("memory-scenario-A-" + $s), settings: {
    context_strategy: $s, custom_response_mode: true, response_format: {
      description: "Отвечай кратко и по делу, не больше 6-8 коротких абзацев или пунктов, без лишних преамбул.",
      max_length: 500
    }
  }}')")
CHAT_A_ID=$(echo "$CHAT_A" | jq -r '.id')
echo "chat_a_id=$CHAT_A_ID strategy=$STRATEGY"

TOTAL_TOKENS=0
N=$(jq '.messages | length' "$SCENARIO")
for i in $(seq 0 $((N-1))); do
  MSG=$(jq -r ".messages[$i]" "$SCENARIO")
  RESP=$(curl -s -X POST "$BASE/v1/chat" -H 'content-type: application/json' \
    -d "$(jq -n --arg cid "$CHAT_A_ID" --arg p "$MSG" '{chat_id:$cid,prompt:$p}')")
  TOKENS=$(echo "$RESP" | jq -r '.usage.total_tokens // 0')
  TOTAL_TOKENS=$((TOTAL_TOKENS + TOKENS))
  echo "--- сообщение $((i+1))/$N (tokens=$TOKENS) ---"
  echo "$RESP" | jq -c '{content, context, usage, error}'
  if [ "$i" -eq 5 ]; then
    sleep 2 # дать фоновому маршрутизатору отработать сообщение перед сменой темы
  fi
done
echo "TOTAL_TOKENS_MAIN=$TOTAL_TOKENS"
echo "CHAT_A_ID_FINAL=$CHAT_A_ID"

sleep 2 # дать фоновому маршрутизатору отработать последнее сообщение чата A

CROSS_Q=$(jq -r '.cross_chat_control_question' "$SCENARIO")
CHAT_B=$(curl -s -X POST "$BASE/v1/chats" -H 'content-type: application/json' \
  -d "$(jq -n --arg s "$STRATEGY" '{title: ("memory-scenario-B-" + $s), settings: {
    context_strategy: $s, custom_response_mode: true, response_format: {
      description: "Отвечай кратко и по делу, не больше 6-8 коротких абзацев или пунктов, без лишних преамбул.",
      max_length: 500
    }
  }}')")
CHAT_B_ID=$(echo "$CHAT_B" | jq -r '.id')
echo "chat_b_id=$CHAT_B_ID"
RESP=$(curl -s -X POST "$BASE/v1/chat" -H 'content-type: application/json' \
  -d "$(jq -n --arg cid "$CHAT_B_ID" --arg p "$CROSS_Q" '{chat_id:$cid,prompt:$p}')")
echo "--- межчатный контрольный вопрос ---"
echo "$RESP" | jq -c '{content, context, usage, error}'
