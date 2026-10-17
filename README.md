# 2MP — контроль кассовой дисциплины по видео

Сервис принимает запись с камеры над кассой и находит признаки возможных нарушений: тип, таймкод, кадры-доказательства, описание и пункт нормы НБРК. Все модели работают внутри сервиса, без облачных AI-API.

## Как устроен анализ

```
POST /api/v1/analyze (.mov, X-API-Key) → uploads/{job_id}.mov → очередь (память процесса или Redis) → воркер
  1. Зоны       Qwen3-VL находит рабочее место кассира, стойку и сторону клиента (или зоны из configs/camera_1.yaml)
  2. Просмотр   Qwen3-VL смотрит всё видео окнами крупным планом (новые типы нарушений ловит именно он)
  2б. Следопыт  параллельно, 5 кадров/с: YOLOX (люди — rtmlib; телефоны и предметы — COCO) → RTMPose → ByteTrack
                → роли CASHIER / CLIENT / UNKNOWN → Rule Engine (configs/rules.yaml):
                transfer_of_item, phone_use_during_service, cashier_left_workspace,
                unauthorized_person_in_restricted_zone, client_in_restricted_zone, + правила по рукам
  3. Перепроверка  каждый кандидат — 5–12 кадров вокруг события в Qwen, строгий JSON (Pydantic):
                violation, type, object_type (banknote|card|phone|paper|unknown), confidence, описание, кадры
  4. Решение    подтверждено Qwen / очень уверенный просмотр / детерминированное правило, которое Qwen не отклонил;
                Qwen не ответил — нарушение по правилу с пометкой verification=unverified
  5. Норма      Legal-RAG по legal_base/: top-3 фрагмента → Qwen выбирает НОМЕР фрагмента; документ, пункт, URL
                и цитата берутся из текста документа (цитата Qwen принимается, только если дословно есть в тексте)
  6. Ответ      GET /api/v1/jobs/{job_id}; затем в фоне — results/{job_id}_annotated.mp4 (зоны, рамки, скелет,
                телефоны, линия передачи, бейдж нарушения, таймлайн)
```

## Основной вариант: RunPod, 1× L4 24 ГБ (шаблон vllm-latest)

Под: 1× L4 · 24 ГБ VRAM · 55 ГБ RAM · 6 vCPU, сетевой диск на `/workspace`, открытый HTTP-порт **8000**.
Модель — `Qwen/Qwen3-VL-8B-Instruct-FP8` (vLLM), RTMPose — на той же GPU. Профиль `l4`.

**Первый раз (≈15 мин, в основном скачивание ~10 ГБ весов):**
```bash
cd /workspace && unzip -o cashguard.zip && cd cashguard
bash runpod/setup.sh l4     # библиотеки, веса Qwen и RTMPose, ключ API, cloudflared
bash start.sh l4            # vLLM + сервис + сторож; в конце печатает ссылку и ключ
```
Ссылка для жюри: `https://<POD_ID>-8000.proxy.runpod.net` (печатается при старте), ключ — в `/workspace/2mp_api_key`.
Запасная ссылка: `TUNNEL=1 bash start.sh l4` → адрес `trycloudflare.com` в `/workspace/logs/tunnel.log`.

**Проверка после запуска** (во втором терминале):
```bash
cd /workspace/cashguard && set -a && . profiles/l4.env && set +a
python3 tools/selftest.py --video /workspace/dataset/<самое_длинное>.mov
```
Покажет: отвечает ли модель валидным JSON, скорость окна, где считает RTMPose (`cuda`/`cpu`),
полный анализ с таймкодами и время (должно быть < 300 с). `GET /health` — профиль, очередь, `pose`.

**Перед отправкой жюри — проверка API по ТЗ через публичную ссылку** (с любого компьютера):
```bash
python3 tools/check_api.py --url https://<POD_ID>-8000.proxy.runpod.net --key <ключ> a.mov b.mov c.mov
```
Ключ (401), job_id, статусы, время < 300 с от загрузки, схема строго по разделу 10, verdict ⇔ violations,
1–3 кадра JPEG base64, таймкоды в пределах видео. Все видео отправляются одновременно, как может сделать жюри.

**Доказательство локальности** (для питчинга):
```bash
python3 tools/selftest.py --offline --video <видео>.mov
```
Полный анализ при запрете внешней сети в процессе анализа (соединения и DNS не на localhost блокируются и
записываются, прокси снимаются) + наблюдение за исходящими соединениями vLLM и сервиса; vLLM слушает только
127.0.0.1, `HF_HUB_OFFLINE=1`. Итог «анализ работает полностью локально» = 0 попыток и 0 внешних соединений.

**Чтобы сервис поднимался сам после рестарта пода:** в шаблоне Start Command поставьте
`bash /workspace/cashguard/runpod/boot.sh` (вместо `sleep infinity`). Журналы — `/workspace/logs/`.

Важно:
- Ссылка RunPod привязана к поду: после **Terminate** будет новый POD_ID — новую ссылку придётся переслать жюри.
  На время проверки под не останавливать. Данные на `/workspace` (веса, ключ, кэши) после Terminate сохраняются.
- Сторож перезапускает vLLM, если модель минуту не отвечает, и веб-сервис — если он упал;
  незаконченные задачи после перезапуска продолжаются сами (не зависают в `processing`).
- Два видео обрабатываются одновременно (`WORKERS=2`); лимит 5 минут считается от загрузки, поэтому
  видео, долго ждавшее в очереди, просматривается с более широким шагом (без «дыр» — не шире окна).
- Эксперимент с 30B: `bash runpod/setup.sh l4_30b && bash start.sh l4_30b` (~17 ГБ весов), сравнить через `tools/eval_api.py`.

## Запасной вариант: ноутбук (Windows, RTX 4050 / 3050, 6 ГБ)

1. Установите **Python 3.11** (python.org, отметьте «Add to PATH») и **Ollama** (ollama.com).
2. Распакуйте папку `cashguard`, откройте `laptop` и запустите **`install.bat`** (один раз, ~10 мин):
   библиотеки, модель `qwen3-vl:4b-instruct` (3,3 ГБ), веса RTMPose, cloudflared.
3. Откройте `laptop\start.bat` в Блокноте и поменяйте `API_KEY` на свой длинный ключ.
4. Запустите **`start.bat`** (профиль `4050`). Откроется окно `tunnel` — в нём публичная ссылка `https://…trycloudflare.com`.
   Сайт: эта ссылка; API: `…/api/v1/analyze` с заголовком `X-API-Key`.
   На ноутбуке с **RTX 3050** запускайте **`start_3050.bat`** (то же, что `start.bat 3050`).
5. Для скорости закройте Ollama в трее перед запуском — `start.bat` поднимет его с нужным числом параллельных запросов.

Пока идёт проверка жюри: ноутбук на зарядке, сон и гибернация отключены, интернет по кабелю, крышку не закрывать.
Ссылка `trycloudflare.com` меняется при перезапуске — для постоянной ссылки нужен именованный туннель (свой домен в Cloudflare).

### Профили (апгрейд железа = другое имя профиля)

Все настройки лежат в `profiles/<имя>.env`; запуск — `start.bat <имя>` (Windows) или `./start.sh <имя>` (Linux).

| Профиль | Железо | Модель | Запуск |
|---|---|---|---|
| `3050` | RTX 3050 6 ГБ, RTMPose на CPU | qwen3-vl:4b-instruct (Ollama) | `start_3050.bat` |
| `3050_2b` | запасной, если не укладываемся в 300 с | qwen3-vl:2b-instruct (Ollama) | `start.bat 3050_2b` |
| `4050` | RTX 4050 6 ГБ (ноутбук-сервер) | qwen3-vl:4b-instruct (Ollama) | `start.bat` |
| `l4` | L4 / A5000 / 4090, 24 ГБ | Qwen3-VL-8B-Instruct-FP8 (vLLM) | `./start.sh l4` |
| `l4_30b` | эксперимент на 24 ГБ | Qwen3-VL-30B-A3B AWQ (vLLM) | `./start.sh l4_30b` |
| `3050_offline` | RTX 3050, локальные тесты (не для жюри) | qwen3-vl:4b-instruct (Q4_K_M), 1–3 кадра, 640 px, Следопыт до Qwen | `start.bat 3050_offline` |
| `compose` | docker compose на L4 | Qwen3-VL-8B FP8, vLLM 0.60 / 8192 / 2 seqs / 8 картинок | `docker compose up -d` |

Переменная, заданная вручную, важнее профиля: `set THRESH=0.6` перед `start.bat` или `THRESH=0.6 ./start.sh l4`.
Активный профиль виден в `GET /health` (поле `profile`).
Обязательно проверьте время на самом длинном видео датасета: должно быть < 300 с.
Если дольше — поднимите `STRIDE_SEC` до 4–5 или перейдите на профиль `3050_2b`.

## Вариант с Docker: `docker compose up -d`

Для машины с Docker и NVIDIA Container Toolkit (на поде RunPod Docker не запускается — там `start.sh`).

```bash
cp env.example .env           # впишите API_KEY (env.example = .env.example, видимая копия)
docker compose up -d          # api, worker, vllm (Qwen3-VL-8B-Instruct-FP8), redis
docker compose logs -f worker # первый старт: скачивание весов Qwen (~10 ГБ) и YOLOX/RTMPose
curl localhost:8000/health
```

| Сервис | Что делает |
|---|---|
| `api` | FastAPI: API жюри, веб-интерфейс, `/results/{job_id}_annotated.mp4`; порт `${PORT:-8000}` |
| `worker` | кадры, YOLOX, RTMPose, ByteTrack, правила, Qwen, Legal-RAG, видео с разметкой (GPU) |
| `vllm` | `vllm/vllm-openai:v0.11.0`: `--gpu-memory-utilization 0.60 --max-model-len 8192 --max-num-seqs 2 --limit-mm-per-prompt {"image":8}` |
| `redis` | очередь задач и статусы |

Тома: `hf_cache` (веса Hugging Face), `data` (загрузки, результаты, статусы), `models` и `rtmlib_cache` (веса Следопыта).
`configs/` и `legal_base/` подключены с хоста — правьте зоны, пороги и нормы без пересборки образа (перезапуск `worker`).
Настройки compose — `profiles/compose.env` (до 8 картинок на запрос из-за `limit-mm-per-prompt`).
Почему FP8: при `gpu-memory-utilization 0.60` на L4 vLLM получает ~14 ГБ, а BF16-веса 8B — ~17 ГБ.

## API (для жюри)

```bash
curl -X POST https://<адрес>/api/v1/analyze -H "X-API-Key: <ключ>" -F "file=@video.mov"
# {"job_id": "3f2a9c1e", "status": "processing"}
curl https://<адрес>/api/v1/jobs/3f2a9c1e -H "X-API-Key: <ключ>"
# {"job_id","status": processing|done|failed, "verdict": violation|no_violation, "violations": [...]}
```

Без верного ключа — `401`. По умолчанию ответ `done` содержит строго поля ТЗ: `job_id, status, verdict, violations[type, timecode_sec, description, evidence_frames[timestamp_sec, image_base64]]` (это проверяет `tools/check_api.py`).
`failed` — `{"job_id", "status": "failed", "error"}`.

Расширенный формат — `API_EXTENDED=1`: у нарушения ещё `confidence`, `legal_reference`
(`source, document, article, title, quote, url, relevance, confidence`), `legal_confidence` (`high|medium|low`),
`object_type`, `verification` (`confirmed|rule|unverified|scan`), у задачи — `annotated_video_url`.
Веб-интерфейс получает все поля всегда (`/web/jobs/{id}`). Документация OpenAPI — `/docs`.

## Конфигурация

| Файл | Что внутри |
|---|---|
| `configs/camera_1.yaml` | зоны `cashier_zone`, `client_zone`, `transfer_zone`, `restricted_zone` (нормированные 0..1); `zones_mode: auto` — Qwen ищет зоны на каждом видео, заданные вручную важнее |
| `configs/rules.yaml` | детектор (классы COCO, пороги), ByteTrack, роли, все правила (длительности, расстояния, `requires_vlm`, `maps_to`), решение, визуализация |
| `configs/violations.yaml` | каталог типов нарушений и норм (бывший `rules.yaml`) |
| `legal_base/` | нормативные документы (TXT/PDF) + `sources.yaml` (название, URL, источник) |
| `profiles/*.env` | профили железа |

Пути к файлам — переменные `CAMERA_CONFIG`, `RULES_CONFIG`, `VIOLATIONS_PATH`, `LEGAL_BASE_DIR`; полный список — `.env.example`.

## Нормативная база (Legal-RAG)

В `legal_base/` лежит выдержка из Правил НБРК № 231 (пункты 5, 12–16, 19–25, 31–35, 74–80), скопированная с adilet.zan.kz 17.10.2026.
Добавить документ: положить `.txt`/`.pdf` в `legal_base/` и описать в `legal_base/sources.yaml` — индекс пересоберётся сам
(кэш в `LEGAL_INDEX_DIR`). Эмбеддер по умолчанию — встроенный hash-TF-IDF (работает офлайн, без скачивания);
нейросетевой — `LEGAL_EMBED_MODEL=<путь к sentence-transformers>`; FAISS используется, если установлен.

Правила выбора нормы: пункт из каталога `configs/violations.yaml` (проверен людьми) главнее; Qwen согласен или поиск
сам ставит его первым → `high`; Qwen выбрал другой → остаётся пункт каталога, `medium`; пункта в каталоге нет —
берётся выбор Qwen только при уверенности `medium/high`; иначе `legal_reference = null`, `legal_confidence = low`.

## Работа с датасетом

1. **Черновая разметка типов** (Qwen описывает каждую метку организаторов):
   ```bash
   python tools/autolabel.py --videos /workspace/dataset --labels labels_raw.csv --out labels_draft.csv
   ```
   Просмотрите `labels_draft.csv`, исправьте `type_id`, сохраните как `labels.csv`.
   Формат: `video,t_sec,type_id,note`; пустой `t_sec` — видео без нарушений.
   Новые типы добавьте в `configs/violations.yaml` (с нормой!).
2. **Отложите 20% видео** для честной проверки промптов и порогов: их имена — в `val.txt`.
3. **Проверка качества через API**, как у жюри:
   ```bash
   python tools/eval_api.py --key <ключ> --videos /workspace/dataset --labels labels.csv --only val.txt
   ```
   Цель: 0 пропусков, минимум ложных срабатываний, все видео < 300 с.

## Настройки (переменные окружения при запуске)

| Переменная | По умолч. | Что делает |
|---|---|---|
| `WIN_SEC` / `STRIDE_SEC` | 4 / 2 | окно и шаг просмотра; больше шаг — быстрее, но выше риск пропуска |
| `SCAN_FRAMES` | 8 | кадров в окне просмотра |
| `VERIFY_FRAMES` / `VOTES` | 12 / 2 | плотность и число перепроверок подозрительного момента |
| `CAND_T` | 0.35 | порог «подозрительно» — ниже = меньше пропусков |
| `THRESH` | 0.5 | порог подтверждения — выше = меньше ложных срабатываний |
| `STRONG` | 0.8 | очень уверенный просмотр засчитывается даже без подтверждения |
| `USE_ZONE` | 1 | искать рабочее место кассира и смотреть крупный план |
| `USE_POSE` / `POSE_FPS` | 1 / 5 | Следопыт (RTMPose) и его частота кадров |
| `TIME_BUDGET` | 240 | лимит на видео от момента загрузки, с (ТЗ — 300) |
| `WORKERS` | 1 | сколько видео анализируются одновременно |
| `MOTION_T` | 0 | >0 — окна без движения не отправляются в Qwen (ускорение) |
| `VLM_URL` / `VLM_MODEL` | vLLM / qwen | адрес и имя модели (задаются профилем) |
| `VLM_MOCK` | — | `1` — запуск без модели (только интерфейс и демо) |
| `VLLM_URL` | — | базовый адрес vLLM (`http://vllm:8000/v1`); `VLM_URL` — полный адрес chat/completions |
| `VLM_RETRIES` / `VLM_TIMEOUT` | 2 / 150 | повторы при сбое сети, 5xx и невалидном JSON; таймаут запроса, с |
| `VLM_MAX_IMAGES` | 16 | ≤ `--limit-mm-per-prompt` у vLLM; ограничивает `SCAN_FRAMES` и `VERIFY_FRAMES` |
| `OBJ_DET_MODEL` / `USE_OBJ_DET` | yolox_s.onnx / 1 | YOLOX COCO для телефонов и предметов (файл в `MODELS_DIR`) |
| `POSE_SEQUENTIAL` | 0 | `1` — Следопыт до Qwen, а не одновременно (слабая GPU) |
| `VIS_ENABLED` | 1 | видео с разметкой после ответа |
| `FAIL_ON_VLM_DOWN` | 1 | `0` — при недоступном Qwen нарушения по правилам (`unverified`) вместо `failed` |
| `API_EXTENDED` | 0 | расширенный ответ API (см. выше) |
| `REDIS_URL` | — | очередь Redis (иначе — в памяти процесса) |

Значения по умолчанию в таблице — это код; профили их переопределяют.
Пример: `THRESH=0.6 STRIDE_SEC=3 API_KEY=... ./start.sh l4`

## Без GPU (посмотреть интерфейс)

```bash
pip install -r requirements.txt
VLM_MOCK=1 API_KEY=test uvicorn app:app --port 8000
```
Откройте `http://localhost:8000` → «Открыть пример проверки».

## Тесты

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests
```
Правила (синтетические сцены), разбор JSON Qwen и решение по голосам, Legal-RAG (в т.ч. «каждый тип каталога находит
свой пункт в базе»), ByteTrack и постобработка YOLOX, API (ключ, статусы, строгая и расширенная схемы, ошибки),
очередь Redis и возобновление после сбоя (нужен `redis-server`, иначе пропускается). Модель и веса не нужны.

## Структура

```
app.py           FastAPI: API по ТЗ, интерфейс, /results; воркеры-потоки, если очередь в памяти
worker.py        воркер: очередь → анализ → результат → видео с разметкой (отдельный процесс в Docker)
store.py         задачи и очередь: в памяти процесса или Redis; возобновление прерванных задач
pipeline.py      анализ: зоны → просмотр + Следопыт → кандидаты → перепроверка → решение → норма
pose.py          Следопыт: YOLOX + RTMPose + ByteTrack → роли → Rule Engine
perception/      detector.py (YOLOX COCO, onnxruntime), bytetrack.py (ByteTrack)
rule_engine/     роли и правила, по модулю на правило
zones.py         зоны из YAML или от Qwen
legal_rag/       нормативная база: загрузка, индекс, выбор нормы
visualization.py аннотированное видео
schemas.py       Pydantic: ответы Qwen и API
settings.py      пути и конфиги (окружение → YAML)
configs/         camera_1.yaml, rules.yaml, violations.yaml
legal_base/      нормативные документы + sources.yaml
profiles/        профили железа (*.env)
static/          интерфейс + демо
tools/           fetch_models, autolabel, eval_api, selftest (+ --offline), check_api, netguard
tests/           pytest
runpod/          setup.sh (подготовка пода), boot.sh (Start Command)
Dockerfile, docker-compose.yml, .env.example
docs/            материалы для сдачи
```
