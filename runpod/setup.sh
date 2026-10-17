#!/usr/bin/env bash
# Подготовка пода RunPod (шаблон vllm-latest, 1x L4). Повторный запуск безопасен и быстрый:
# всё тяжёлое (веса, кэши, ключ) лежит на сетевом диске /workspace и переживает Terminate.
#   bash runpod/setup.sh [профиль]      по умолчанию l4
set -e
cd "$(dirname "$0")/.."
PROFILE="${1:-l4}"
WS=/workspace; [ -d "$WS" ] || { echo "[!] Нет /workspace — подключите сетевой диск к поду"; exit 1; }
export HF_HOME=$WS/.cache/huggingface PIP_CACHE_DIR=$WS/.cache/pip
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$WS/models" "$WS/logs" "$WS/bin" "$WS/.cache/rtmlib"
set -a; . "profiles/$PROFILE.env"; set +a
PY=python3
export PIP_BREAK_SYSTEM_PACKAGES=1 PIP_DISABLE_PIP_VERSION_CHECK=1
pipi() {                                     # pip, а если его нет у системного Python — uv
  if $PY -m pip --version >/dev/null 2>&1; then $PY -m pip install -q "$@"
  else uv pip install --system -q "$@"; fi
}
ok() { echo "  ✓ $*"; }

echo "== 1/5 Проверка образа"
$PY -c "import vllm, torch; print('  vLLM', vllm.__version__, '| torch', torch.__version__, '| CUDA', torch.version.cuda)" \
  || { echo "[!] В образе нет vLLM — выберите шаблон vllm-latest"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/  GPU: /'
command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl >/dev/null)

echo "== 2/5 Библиотеки сервиса (в окружение vLLM, не ломая его версии)"
NP=$($PY -c 'import numpy; print(numpy.__version__)')
# rtmlib без зависимостей: иначе он поставит opencv-python и CPU-onnxruntime поверх версий vLLM
pipi --no-deps rtmlib
pipi "numpy==$NP" onnxruntime-gpu python-multipart pyyaml requests starlette uvicorn tqdm pypdf imageio-ffmpeg
$PY -c "import fastapi, pydantic" 2>/dev/null || pipi "numpy==$NP" fastapi "pydantic>=2"
$PY -c "import cv2" 2>/dev/null || pipi "numpy==$NP" opencv-python-headless
for i in 1 2 3 4 5; do                       # докачать то, что rtmlib ещё попросит при импорте
  miss=$($PY -c "import rtmlib" 2>&1 | sed -n "s/.*No module named '\([^'.]*\).*/\1/p" | head -1)
  [ -z "$miss" ] && break
  echo "  доставляю $miss"; pipi "numpy==$NP" "$miss"
done
$PY -c "import vllm, cv2, rtmlib, onnxruntime, fastapi, pydantic, uvicorn, multipart, yaml; print('  импорт ок, onnxruntime', onnxruntime.__version__, '| fastapi', fastapi.__version__)"

echo "== 3/5 Веса Qwen ($HF_REPO → $MODEL_DIR)"
if [ -f "$MODEL_DIR/config.json" ] && ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1; then
  ok "уже скачаны ($(du -sh "$MODEL_DIR" | cut -f1))"
else
  $PY - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download("$HF_REPO", local_dir="$MODEL_DIR", max_workers=8)
PYEOF
  ok "скачано ($(du -sh "$MODEL_DIR" | cut -f1))"
fi

echo "== 4/5 Следопыт: RTMPose и YOLOX COCO (веса — на /workspace)"
mkdir -p ~/.cache
[ -L ~/.cache/rtmlib ] || { rm -rf ~/.cache/rtmlib; ln -s "$WS/.cache/rtmlib" ~/.cache/rtmlib; }
MODELS_DIR=$WS/models/detector $PY tools/fetch_models.py --models yolox_s,yolox_tiny --pose \
  || echo "  [i] часть весов Следопыта не скачалась — Qwen-анализ работает и без них"

echo "== 5/5 Ключ API и cloudflared"
if [ ! -s "$WS/2mp_api_key" ]; then
  $PY -c "import secrets; print('2mp-' + secrets.token_urlsafe(24))" > "$WS/2mp_api_key"
fi
chmod 600 "$WS/2mp_api_key"
ok "ключ: $(cat "$WS/2mp_api_key")   (файл $WS/2mp_api_key, доступ только владельцу)"
# шрифт сайта — на диск: страница не обращается к Google Fonts (работа без интернета, внутренний контур)
if [ ! -s static/fonts/onest.css ]; then
  mkdir -p static/fonts
  UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
  if curl -fsSL -A "$UA" "https://fonts.googleapis.com/css2?family=Onest:wght@400;500;600;700;800&display=swap" -o /tmp/onest.css; then
    i=0; cp /tmp/onest.css static/fonts/onest.css
    for u in $(grep -o 'https://fonts.gstatic.com[^)]*' /tmp/onest.css | sort -u); do
      i=$((i+1)); curl -fsSL "$u" -o "static/fonts/onest_$i.woff2" && sed -i "s#$u#/static/fonts/onest_$i.woff2#g" static/fonts/onest.css
    done
    ok "шрифт Onest сохранён локально ($i файлов)"
  else
    echo "  [i] шрифт не скачался — сайт покажет системный шрифт"
  fi
fi
if [ ! -x "$WS/bin/cloudflared" ]; then
  curl -fsSL -o "$WS/bin/cloudflared" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    && chmod +x "$WS/bin/cloudflared" && ok "cloudflared скачан" || echo "  [i] cloudflared не скачался — не страшно, есть ссылка RunPod"
fi

echo
echo "Готово. Запуск:  bash start.sh $PROFILE      (или runpod/boot.sh как Start Command шаблона)"
