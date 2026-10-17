#!/usr/bin/env bash
# Полная остановка сервиса 2MP: наш start.sh (со сторожем), веб-сервис, vLLM и его процесс EngineCore.
#   bash stop.sh
# ВАЖНО: не искать процессы по имени «start.sh» — так называется и главный процесс контейнера RunPod (/start.sh),
# его остановка перезапускает весь под. Наш start.sh останавливаем только по PID из /tmp/2mp_start.pid.
if [ -f /tmp/2mp_start.pid ]; then
  P=$(cat /tmp/2mp_start.pid)
  if [ "$P" != "1" ] && [ "$(cat /proc/$P/cmdline 2>/dev/null | tr '\0' ' ' | grep -c 'bash start.sh')" = "1" ]; then
    kill -- "-$P" 2>/dev/null || kill "$P" 2>/dev/null       # вся группа: start.sh, сторож, цикл веб-сервиса
  fi
  rm -f /tmp/2mp_start.pid
fi
pkill -f "uvicorn app:app" 2>/dev/null
pkill -f "vllm serve" 2>/dev/null
pkill -f "VLLM::EngineCore" 2>/dev/null           # дочерний процесс vLLM держит видеопамять, «vllm serve» его не задевает
for i in $(seq 1 15); do
  pgrep -f "vllm serve|VLLM::EngineCore" >/dev/null || break
  sleep 1
done
pkill -9 -f "vllm serve|VLLM::EngineCore" 2>/dev/null
command -v nvidia-smi >/dev/null && echo "видеопамять: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo "остановлено"
