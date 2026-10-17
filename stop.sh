#!/usr/bin/env bash
# Полная остановка сервиса: start.sh (со сторожем), веб-сервис, vLLM и его процесс EngineCore.
#   bash stop.sh
pkill -f "start.sh" 2>/dev/null
pkill -f "uvicorn app:app" 2>/dev/null
pkill -f "vllm serve" 2>/dev/null
pkill -f -i "enginecore" 2>/dev/null          # дочерний процесс vLLM держит видеопамять, «vllm serve» его не задевает
for i in $(seq 1 15); do
  pgrep -f -i "vllm serve|enginecore" >/dev/null || break
  sleep 1
done
pkill -9 -f -i "vllm serve|enginecore" 2>/dev/null
command -v nvidia-smi >/dev/null && echo "видеопамять: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo "остановлено"
