@echo off
chcp 65001 >nul
rem ===== Установка на ноутбук (Windows). Запускать один раз из папки cashguard\laptop =====
cd /d "%~dp0.."

where python >nul 2>nul || (echo [!] Установите Python 3.11 с python.org и отметьте "Add to PATH" & pause & exit /b 1)
where ollama >nul 2>nul || (echo [!] Установите Ollama с ollama.com & pause & exit /b 1)

echo === Библиотеки Python ===
python -m pip install --upgrade pip
python -m pip install -r requirements.txt || (echo [!] Ошибка установки библиотек & pause & exit /b 1)

echo === Модель Qwen3-VL-4B (instruct, ~3.3 ГБ) ===
ollama pull qwen3-vl:4b-instruct || (echo [!] Не удалось скачать модель & pause & exit /b 1)

echo === Следопыт: RTMPose и YOLOX (веса скачиваются один раз) ===
python tools\fetch_models.py --models yolox_tiny,yolox_s --pose

echo === Cloudflare Tunnel (публичная ссылка) ===
where cloudflared >nul 2>nul || winget install --id Cloudflare.cloudflared -e --accept-source-agreements --accept-package-agreements

echo.
echo Готово. Дальше запускайте laptop\start.bat
pause
