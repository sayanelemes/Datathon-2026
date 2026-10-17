#!/usr/bin/env bash
# Удаление всех данных проверок после хакатона (ТЗ: тестовые видео нельзя хранить и передавать третьим лицам).
#   bash tools/wipe_data.sh            — видео, кадры, результаты, журналы
#   bash tools/wipe_data.sh --all      — плюс ключ API и веса моделей
set -e
WS=/workspace; [ -d "$WS" ] || WS="$(cd "$(dirname "$0")/.." && pwd)"
echo "Удаляю: результаты, загруженные видео, видео с разметкой, журналы"
rm -rf "$WS/2mp_jobs" "$WS/2mp_uploads" "$WS/2mp_results" "$WS/logs"; (cd "$(dirname "$0")/.." && rm -rf jobs uploads results)
if [ "$1" = "--all" ]; then
  echo "Удаляю ключ API и веса: $WS/2mp_api_key, $WS/models"
  rm -rf "$WS/2mp_api_key" "$WS/models"
fi
echo "Готово. Не забудьте удалить сетевой диск в RunPod (Storage → Delete), если он больше не нужен."
