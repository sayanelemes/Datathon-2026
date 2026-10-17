#!/usr/bin/env bash
# Запуск всего сервиса одной командой (RunPod / Linux):
#   ./start.sh [профиль]            по умолчанию l4
# Профиль — profiles/<имя>.env: l4 | l4_30b | 3050 | 3050_2b | 4050. Переменная, заданная вручную, важнее профиля:
#   THRESH=0.6 ./start.sh l4
# Ключ API: переменная API_KEY или файл /workspace/2mp_api_key (его создаёт runpod/setup.sh).
# Сторож: если упадёт vLLM или веб-сервис — перезапускается сам. Журналы — в $LOGS.
# После скачивания весов интернет не нужен (HF_HUB_OFFLINE=1); туннель cloudflared — по желанию (TUNNEL=1).
set -e
cd "$(dirname "$0")"
PROFILE="${1:-${PROFILE:-l4}}"
PF="profiles/$PROFILE.env"
[ -f "$PF" ] || { echo "Нет профиля $PF. Есть: $(ls profiles | sed 's/\.env//' | tr '\n' ' ')"; exit 1; }

# --- профиль: KEY=VALUE, строки с # пропускаются, уже заданные переменные не трогаем ---
while IFS='=' read -r k v; do
  k="${k//[[:space:]]/}"; v="${v%$'\r'}"
  [[ -z "$k" || "$k" == \#* ]] && continue
  [ -z "${!k+x}" ] && export "$k=$v"
done < "$PF"
export PROFILE

WS=/workspace; [ -d "$WS" ] || WS="$PWD"
[ -z "$API_KEY" ] && [ -f "$WS/2mp_api_key" ] && API_KEY="$(cat "$WS/2mp_api_key")"
export API_KEY="${API_KEY:?Укажите ключ: API_KEY=... ./start.sh $PROFILE (или запустите runpod/setup.sh)}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$WS/.cache/huggingface}" VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$WS/.cache/vllm}"
export DATA_DIR="${DATA_DIR:-$WS/2mp_jobs}" UPLOAD_DIR="${UPLOAD_DIR:-$WS/2mp_uploads}" RESULTS_DIR="${RESULTS_DIR:-$WS/2mp_results}"
export LEGAL_INDEX_DIR="${LEGAL_INDEX_DIR:-$DATA_DIR/_legal_index}"
[ "$WS" = /workspace ] && export MODELS_DIR="${MODELS_DIR:-$WS/models/detector}"   # иначе — ./models (как у fetch_models.py)
LOGS="${LOGS:-$WS/logs}"; mkdir -p "$LOGS" "$DATA_DIR"
PY="${PY:-python3}"; [ -x .venv/bin/python ] && PY=.venv/bin/python
echo "Профиль: $PROFILE  (модель $VLM_MODEL, бэкенд $BACKEND, видео одновременно: ${WORKERS:-1})"

up() { curl -sf -m 5 "$1" >/dev/null 2>&1; }

start_vllm() {
  [ -f "$MODEL_DIR/config.json" ] || { echo "[!] Нет весов в $MODEL_DIR — запустите runpod/setup.sh $PROFILE"; exit 1; }
  echo "$(date +%T) запускаю vLLM ($MODEL_DIR) — первый старт 2–5 минут, журнал: $LOGS/vllm.log"
  nohup vllm serve "$MODEL_DIR" --served-model-name qwen --host 127.0.0.1 --port 8001 \
    --max-model-len "${MAX_LEN:-16384}" --max-num-seqs "${MAX_NUM_SEQS:-32}" \
    --limit-mm-per-prompt "{\"image\":${VLM_MAX_IMAGES:-16}}" --gpu-memory-utilization "${GPU_UTIL:-0.80}" \
    --enable-prefix-caching >> "$LOGS/vllm.log" 2>&1 &
  echo $! > "$LOGS/vllm.pid"
  for i in $(seq 1 120); do                 # до 10 минут
    up localhost:8001/health && { echo "$(date +%T) vLLM готов"; return 0; }
    kill -0 "$(cat "$LOGS/vllm.pid")" 2>/dev/null || { echo "[!] vLLM упал при запуске:"; tail -30 "$LOGS/vllm.log"; exit 1; }
    sleep 5
  done
  echo "[!] vLLM не поднялся за 10 минут"; tail -30 "$LOGS/vllm.log"; exit 1
}

start_ollama() {
  if ! up localhost:11434/api/tags; then
    OLLAMA_NUM_PARALLEL="$PARALLEL" OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH:-0}" OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV:-f16}" \
      OLLAMA_KEEP_ALIVE=-1 nohup ollama serve >> "$LOGS/ollama.log" 2>&1 &
    until up localhost:11434/api/tags; do sleep 2; done
  fi
}

model_up() { if [ "$BACKEND" = "vllm" ]; then up localhost:8001/health; else up localhost:11434/api/tags; fi; }
start_model() { if [ "$BACKEND" = "vllm" ]; then start_vllm; else start_ollama; fi; }

model_up || start_model

# --- прогрев: первый запрос с картинкой компилирует графы, жюри не должно его ждать ---
echo "$(date +%T) прогрев модели..."
$PY -I tools/selftest.py --quick >/dev/null 2>&1 || echo "[i] прогрев не прошёл — подробности: $PY tools/selftest.py"

if [ "${TUNNEL:-0}" = "1" ]; then
  CF="$(command -v cloudflared || echo "$WS/bin/cloudflared")"
  if [ -x "$CF" ]; then
    nohup "$CF" tunnel --no-autoupdate --url http://localhost:8000 > "$LOGS/tunnel.log" 2>&1 &
    echo "Туннель: ссылка появится в $LOGS/tunnel.log (grep trycloudflare)"
  fi
fi

if [ -n "$RUNPOD_POD_ID" ]; then
  echo "Публичная ссылка RunPod: https://${RUNPOD_POD_ID}-8000.proxy.runpod.net"
fi
echo "Сервис: http://0.0.0.0:8000   Ключ API: $API_KEY"

# --- сторож: модель упала — поднять заново ---
(
  fails=0
  while sleep 20; do
    if model_up; then fails=0; continue; fi
    fails=$((fails + 1))
    [ "$fails" -lt 3 ] && continue                     # 3 провала подряд (~1 мин) — не дёргаем из-за одной заминки
    echo "$(date +%T) модель не отвечает минуту — перезапуск" >> "$LOGS/watchdog.log"
    if [ -f "$LOGS/vllm.pid" ]; then kill "$(cat "$LOGS/vllm.pid")" 2>/dev/null || true; sleep 10; fi
    ( start_model ) >> "$LOGS/watchdog.log" 2>&1 || true
    fails=0
  done
) &
WD=$!
trap 'kill $WD 2>/dev/null' EXIT

# --- веб-сервис; упал — перезапуск через 3 с ---
while true; do
  $PY -m uvicorn app:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 75 2>&1 | tee -a "$LOGS/app.log" || true
  echo "$(date +%T) веб-сервис остановился — перезапуск через 3 с" | tee -a "$LOGS/app.log"
  sleep 3
done
