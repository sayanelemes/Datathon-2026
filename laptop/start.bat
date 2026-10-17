@echo off
chcp 65001 >nul
rem ===== Запуск сервиса на ноутбуке (Windows) =====
rem   start.bat            — профиль 4050 (ноутбук-сервер)
rem   start.bat 3050       — тестовый ноутбук (RTMPose на CPU)
rem   start.bat 3050_2b    — запасной: Qwen3-VL-2B, если длинное видео не укладывается в 300 с
rem Настройки лежат в profiles\<имя>.env. Переменная, заданная вручную (set THRESH=0.6), важнее профиля.
setlocal
cd /d "%~dp0.."

set "PROFILE=%~1"
if "%PROFILE%"=="" set "PROFILE=4050"
if not exist "profiles\%PROFILE%.env" (
  echo [!] Нет профиля profiles\%PROFILE%.env. Есть:
  dir /b profiles
  pause
  exit /b 1
)

rem --- ключ для жюри: поменяйте на свой длинный ---
if "%API_KEY%"=="" set "API_KEY=change-me-long-secret-key"

rem --- профиль: KEY=VALUE, строки с # пропускаются, уже заданные переменные не трогаем ---
for /f "usebackq eol=# tokens=1,* delims==" %%a in ("profiles\%PROFILE%.env") do if not defined %%a set "%%a=%%b"
if "%OLLAMA_CTX%"=="" set "OLLAMA_CTX=8192"
echo Профиль: %PROFILE%   модель %VLM_MODEL%   шаг %STRIDE_SEC% с   RTMPose: %POSE_DEVICE%

if /i not "%BACKEND%"=="ollama" (
  echo [!] Профиль %PROFILE% рассчитан на сервер с vLLM — запускайте start.sh на Linux.
  pause
  exit /b 1
)

rem --- модель есть? ---
ollama list | findstr /i /c:"%VLM_MODEL%" >nul || (
  echo Скачиваю %VLM_MODEL% ...
  ollama pull %VLM_MODEL% || (echo [!] Не удалось скачать модель & pause & exit /b 1)
)

rem --- Ollama: PARALLEL запросов параллельно ---
curl -s http://127.0.0.1:11434/api/tags >nul 2>nul
if errorlevel 1 (
  echo Запускаю Ollama: %PARALLEL% параллельных запроса, контекст %OLLAMA_CTX%...
  start "ollama" cmd /c "set OLLAMA_NUM_PARALLEL=%PARALLEL%&& set OLLAMA_CONTEXT_LENGTH=%OLLAMA_CTX%&& set OLLAMA_KEEP_ALIVE=-1&& ollama serve"
  timeout /t 5 >nul
) else (
  echo [i] Ollama уже запущен. Для ускорения закройте Ollama в трее и перезапустите этот файл —
  echo     тогда он запустится с %PARALLEL% параллельными запросами.
)

rem --- прогрев модели (первый запрос медленный) ---
echo Прогрев модели...
curl -s http://127.0.0.1:11434/api/generate -d "{\"model\":\"%VLM_MODEL%\",\"prompt\":\"ok\",\"stream\":false}" >nul

rem --- публичная ссылка в отдельном окне ---
start "tunnel" cmd /k "cloudflared tunnel --url http://localhost:8000"
echo.
echo Публичная ссылка появится в окне "tunnel" (https://....trycloudflare.com)
echo Локально: http://localhost:8000   Ключ API: %API_KEY%
echo.
python -m uvicorn app:app --host 0.0.0.0 --port 8000
pause
