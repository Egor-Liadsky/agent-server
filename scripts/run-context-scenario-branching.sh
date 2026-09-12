#!/bin/bash
# Прогоняет tests/data/context-scenario.json на стратегии `branching`:
# первая половина сообщений — общий ствол (checkpoint), затем от него
# создаются и последовательно проходятся две ветки.
#
# Использование: AGENTD_URL=http://127.0.0.1:8090 ./run-context-scenario-branching.sh
set -euo pipefail
BASE="${AGENTD_URL:-http://127.0.0.1:8090}"
SCENARIO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/tests/data/context-scenario.json"

CHAT=$(curl -s -X POST "$BASE/v1/chats" -H 'content-type: application/json' \
  -d '{"title":"scenario-branching","settings":{"context_strategy":"branching","custom_response_mode":true,"response_format":{"description":"Отвечай кратко и по делу, не больше 6-8 коротких абзацев или пунктов, без лишних преамбул.","max_length":500}}}')
CHAT_ID=$(echo "$CHAT" | jq -r '.id')
echo "chat_id=$CHAT_ID"

TOTAL=0
N=$(jq '.messages | length' "$SCENARIO")
MID=$((N/2))
for i in $(seq 0 $((MID-1))); do
  MSG=$(jq -r ".messages[$i]" "$SCENARIO")
  RESP=$(curl -s -X POST "$BASE/v1/chat" -H 'content-type: application/json' \
    -d "$(jq -n --arg cid "$CHAT_ID" --arg p "$MSG" '{chat_id:$cid,prompt:$p}')")
  T=$(echo "$RESP" | jq -r '.usage.total_tokens // 0'); TOTAL=$((TOTAL+T))
  echo "--- root $((i+1))/$MID (tokens=$T) ---"; echo "$RESP" | jq -c '{content,context}'
done

ROOT_BRANCH=$(curl -s "$BASE/v1/chats/$CHAT_ID" | jq -r '.branch_id')
CHECKPOINT_SEQ=$(curl -s "$BASE/v1/chats/$CHAT_ID" | jq -r '.message_count')
echo "checkpoint_seq=$CHECKPOINT_SEQ root_branch=$ROOT_BRANCH"

BRANCH_A=$(curl -s -X POST "$BASE/v1/chats/$CHAT_ID/branches" -H 'content-type: application/json' \
  -d "$(jq -n --argjson s "$CHECKPOINT_SEQ" '{from_seq:$s, name:"вариант: Jira"}')")
BRANCH_A_ID=$(echo "$BRANCH_A" | jq -r '.id')
curl -s -X POST "$BASE/v1/chats/$CHAT_ID/branches/$BRANCH_A_ID/activate" > /dev/null
echo "branch_a=$BRANCH_A_ID"

for i in $(seq $MID $((N-1))); do
  MSG=$(jq -r ".messages[$i]" "$SCENARIO")
  MSG="[считаем, что трекер — Jira] $MSG"
  RESP=$(curl -s -X POST "$BASE/v1/chat" -H 'content-type: application/json' \
    -d "$(jq -n --arg cid "$CHAT_ID" --arg p "$MSG" '{chat_id:$cid,prompt:$p}')")
  T=$(echo "$RESP" | jq -r '.usage.total_tokens // 0'); TOTAL=$((TOTAL+T))
  echo "--- A $((i+1))/$N (tokens=$T) ---"; echo "$RESP" | jq -c '{content,context}'
done

curl -s -X POST "$BASE/v1/chats/$CHAT_ID/branches/$ROOT_BRANCH/activate" > /dev/null

BRANCH_B=$(curl -s -X POST "$BASE/v1/chats/$CHAT_ID/branches" -H 'content-type: application/json' \
  -d "$(jq -n --argjson s "$CHECKPOINT_SEQ" '{from_seq:$s, name:"вариант: собственный трекер"}')")
BRANCH_B_ID=$(echo "$BRANCH_B" | jq -r '.id')
curl -s -X POST "$BASE/v1/chats/$CHAT_ID/branches/$BRANCH_B_ID/activate" > /dev/null
echo "branch_b=$BRANCH_B_ID"

for i in $(seq $MID $((N-1))); do
  MSG=$(jq -r ".messages[$i]" "$SCENARIO")
  MSG="[считаем, что используем собственный трекер задач] $MSG"
  RESP=$(curl -s -X POST "$BASE/v1/chat" -H 'content-type: application/json' \
    -d "$(jq -n --arg cid "$CHAT_ID" --arg p "$MSG" '{chat_id:$cid,prompt:$p}')")
  T=$(echo "$RESP" | jq -r '.usage.total_tokens // 0'); TOTAL=$((TOTAL+T))
  echo "--- B $((i+1))/$N (tokens=$T) ---"; echo "$RESP" | jq -c '{content,context}'
done

echo "TOTAL_TOKENS_MAIN=$TOTAL"
echo "CHAT_ID_FINAL=$CHAT_ID BRANCH_A=$BRANCH_A_ID BRANCH_B=$BRANCH_B_ID"
