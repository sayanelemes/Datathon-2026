#!/usr/bin/env bash
# Start Command для шаблона RunPod:  bash /workspace/cashguard/runpod/boot.sh
# При каждом старте пода: доставить библиотеки (быстро, из кэша), поднять сервис в фоне, держать контейнер живым.
cd "$(dirname "$0")/.."
PROFILE="${PROFILE:-l4}"
mkdir -p /workspace/logs
{
  bash runpod/setup.sh "$PROFILE" && bash start.sh "$PROFILE"
} >> /workspace/logs/boot.log 2>&1 &
sleep infinity
