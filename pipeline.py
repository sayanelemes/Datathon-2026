"""Анализ видео кассира.

Этапы:
  1. Зона кассы   — Qwen один раз находит рабочее место кассира, дальше смотрим крупный план.
  2. Скан         — всё видео окнами (WIN_SEC, шаг STRIDE_SEC), Qwen оценивает каждое окно.
  2б. Следопыт    — параллельно со сканом: YOLOX + RTMPose + ByteTrack, зоны, роли, Rule Engine (pose.py).
  3. Кандидаты    — подозрительные моменты из скана и от правил.
  5. Проверка     — каждый кандидат перепроверяется плотнее (больше кадров, подсказка), VOTES раз.
  6. Решение      — пропуск хуже ложной тревоги: подтверждено проверкой ИЛИ очень уверенный скан
                    ИЛИ детерминированное правило, которое Qwen не отклонил.
  7. Норма        — Legal-RAG: фрагмент из legal_base/ (Qwen выбирает только номер фрагмента).
Все модели работают локально (vLLM на localhost). Внешние AI-сервисы не используются.
"""
import base64, json, logging, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests
import yaml

import legal_rag
import settings
import zones as zones_mod
from schemas import validate_verdict

log = logging.getLogger("pipeline")
E = os.getenv
VLM_URL = settings.vlm_url()
VLM_MODEL = E("VLM_MODEL", "qwen")
VLM_MOCK = E("VLM_MOCK") == "1"
BACKEND = E("BACKEND", "vllm")              # vllm (OpenAI API) | ollama (родной API: контекст и JSON задаются в запросе)
OLLAMA_URL = E("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_CTX = int(E("OLLAMA_CTX", 8192))     # контекст на запрос; по умолчанию у Ollama 4096 — промпт бы обрезался
OLLAMA_JSON = E("OLLAMA_JSON", "1") == "1"  # format=json: модель обязана вернуть валидный JSON
VLM_TIMEOUT = float(E("VLM_TIMEOUT", 150))
VLM_RETRIES = int(E("VLM_RETRIES", 2))      # повторы при сбое сети, 5xx и невалидном JSON
VLM_MAX_IMAGES = int(E("VLM_MAX_IMAGES", 16))  # = --limit-mm-per-prompt у vLLM
WIN_SEC = float(E("WIN_SEC", 4))            # длина окна скана
STRIDE_SEC = float(E("STRIDE_SEC", 2))      # шаг окна скана
SCAN_FRAMES = min(int(E("SCAN_FRAMES", 8)), VLM_MAX_IMAGES)      # кадров в окне скана
VERIFY_SEC = float(E("VERIFY_SEC", 5))      # длина окна проверки
VERIFY_FRAMES = min(int(E("VERIFY_FRAMES", 12)), VLM_MAX_IMAGES)  # кадров в окне проверки
VOTES = int(E("VOTES", 2))                  # сколько раз перепроверять кандидата
MAX_SIDE = int(E("MAX_SIDE", 640))          # размер кадра для VLM
CACHE_SIDE = int(E("CACHE_SIDE", 960))      # размер кадров в памяти (и кадров-доказательств)
VERIFY_SIDE = int(E("VERIFY_SIDE", MAX_SIDE))  # размер кадра при перепроверке: крупнее — видна купюра в руке
PARALLEL = int(E("PARALLEL", 16))
CAND_T = float(E("CAND_T", 0.35))           # порог кандидата по скану
THRESH = float(E("THRESH", 0.5))            # порог подтверждения проверкой
STRONG = float(E("STRONG", 0.8))            # «сильный» скан: засчитываем даже без подтверждения
USE_ZONE = E("USE_ZONE", "1") == "1"
USE_POSE = E("USE_POSE", "1") == "1"
MOTION_T = float(E("MOTION_T", 0))          # >0: окна без движения не отправляются в Qwen (ускорение на слабой GPU)
TIME_BUDGET = float(E("TIME_BUDGET", 240))  # лимит ТЗ — 300 с
SKIP_GRAB = int(E("SKIP_GRAB", 150))        # до стольких кадров пропускаем grab(), дальше — перемотка
POSE_SEQUENTIAL = E("POSE_SEQUENTIAL", "0") == "1"  # 1 — Следопыт до Qwen, не одновременно (слабая GPU, офлайн-режим)
FAIL_ON_VLM_DOWN = E("FAIL_ON_VLM_DOWN", "1") == "1"  # Qwen недоступен целиком: 1 — failed; 0 — только правила (unverified)
USE_LEGAL_RAG = E("USE_LEGAL_RAG", "1") == "1"

RULES_PATH = settings.VIOLATIONS_PATH
with open(RULES_PATH, encoding="utf-8") as _f:
    RULES = {r["id"]: r for r in yaml.safe_load(_f)["rules"]}
_DEC = settings.rules_config().get("decision") or {}
UNVERIFIED_MIN = float(E("UNVERIFIED_MIN", _DEC.get("unverified_min", 0.5)))
VETO_T = float(E("VETO_T", _DEC.get("veto_confidence", 0.25)))

try:
    import pose  # Следопыт (rtmlib); если не установлен — выключен
except Exception as e:  # noqa
    pose = None
    log.warning("Следопыт выключен: %r", e)



# ======================= видео =======================
class Video:
    def __init__(self, path):
        self.path = path
        cap = cv2.VideoCapture(path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        self.dur = self.n / self.fps if self.fps else 0
        self._cache = {}

    def clamp(self, t):
        return round(min(max(t, 0.0), max(self.dur - 1.0 / self.fps, 0)), 2)

    def iter(self, times):
        """Кадры по одному, без хранения в памяти (уменьшены до CACHE_SIDE)."""
        cap = cv2.VideoCapture(self.path)
        pos, ok, fr, small = -1, False, None, None    # индекс последнего прочитанного кадра
        for t in sorted({self.clamp(t) for t in times}):
            idx = int(round(t * self.fps))
            if idx <= pos:                         # тот же кадр уже прочитан
                if ok and fr is not None:
                    yield t, small
                continue
            gap = idx - pos - 1
            if 0 <= gap <= SKIP_GRAB:              # близко — пропускаем кадры без декодирования в BGR
                for _ in range(gap):
                    cap.grab()
            else:                                  # далеко — перемотка
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            pos = idx
            if ok:
                small = resize(fr, CACHE_SIDE)
                yield t, small
        cap.release()

    def frames(self, times):
        """{t: кадр BGR}; читает файл последовательно и кэширует (уменьшенные кадры)."""
        times = [self.clamp(t) for t in times]
        need = set(times) - set(self._cache)
        if need:
            for t, fr in self.iter(need):
                self._cache[t] = fr
        return {t: self._cache[t] for t in times if t in self._cache}


def resize(fr, max_side=MAX_SIDE):
    h, w = fr.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(fr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else fr


def jpg_b64(fr, q=85):
    ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf).decode()


def crop(fr, zone):
    if not zone:
        return fr
    h, w = fr.shape[:2]
    x1, y1, x2, y2 = zone
    return fr[int(y1 * h / 1000):int(y2 * h / 1000), int(x1 * w / 1000):int(x2 * w / 1000)]


def box_to_full(b, zone):
    """Рамка 0..1000 внутри зоны -> 0..1000 всего кадра."""
    if not isinstance(b, list) or len(b) != 4:
        return None
    try:
        b = [min(max(float(x), 0), 1000) for x in b]
    except (TypeError, ValueError):
        return None
    if b[2] <= b[0] or b[3] <= b[1]:
        return None
    if zone:
        zx1, zy1, zx2, zy2 = zone
        zw, zh = zx2 - zx1, zy2 - zy1
        b = [zx1 + b[0] * zw / 1000, zy1 + b[1] * zh / 1000, zx1 + b[2] * zw / 1000, zy1 + b[3] * zh / 1000]
    return [round(x) for x in b]


# ======================= VLM =======================
def rules_text():
    return "\n".join(f'- {r["id"]}: {r["visual"]}' for r in RULES.values())


BASE = (
    "Ты — контролёр кассовой дисциплины банка. Камера стоит сверху над рабочим местом кассира.\n"
    "Роли: кассир сидит за столом с компьютером и/или счётчиком банкнот; клиент — по другую сторону стола; "
    "инкассатор — приносит или забирает сумки с ценностями.\n"
    "Оценивай ТОЛЬКО то, что действительно видно на кадрах. Не додумывай то, чего не видно "
    "(документы, проводки, суммы). Обычная работа кассира — НЕ нарушение.\n"
    "Камера далеко и сверху: купюры выглядят как небольшие светлые или цветные прямоугольники бумаги, "
    "часто сложенные пачкой. То, что клиент передаёт кассиру, а кассир пересчитывает, — как правило, наличные. "
    "Если не можешь различить, купюра это или карта, так и напиши («купюра или карта») и оценивай действия рук.\n"
    "Сокрытие денег видно по рукам, даже если сам предмет не различим: после получения денег кассир отделяет часть "
    "и опускает руку ниже уровня стола, к коленям, в карман, рукав, под клавиатуру или в ящик вне обзора клиента, "
    "пока другая рука работает со счётчиком или стопкой.\n"
    "confidence — вероятность того, что НАРУШЕНИЕ ЕСТЬ: 0 — точно нет, 1 — точно есть "
    "(это НЕ уверенность в ответе: если нарушения нет, confidence близко к 0).\n"
    "Возможные нарушения:\n{rules}\n"
)

SCAN_PROMPT = BASE + (
    "\nТебе дают кадры одного фрагмента видео по порядку, у каждого подписано время.\n"
    "Сначала опиши, что делают руки кассира и где деньги/ценности, затем оцени.\n"
    "Ответ строго одним JSON:\n"
    '{{"observation": "1-2 предложения по-русски", "violation": true или false, '
    '"type_id": "<id из списка или NONE>", "frame_index": <номер самого показательного кадра>, '
    '"confidence": <вероятность нарушения 0..1>}}'
)

VERIFY_PROMPT = BASE + (
    "\nЭто ПЕРЕПРОВЕРКА подозрительного момента. Кадры идут плотно, по порядку, с подписанным временем.\n"
    "Подсказка предварительного анализа (может быть ошибочной): {hint}\n"
    "Шаг 1: про себя опиши по кадрам, что делает кассир руками и что происходит с деньгами, сумками, телефоном.\n"
    "Шаг 2: реши, есть ли нарушение из списка. Если действие обычное или его не видно — violation=false.\n"
    "Определи предмет, если применимо: banknote (купюры), card (карта), phone (телефон), paper (документ), unknown.\n"
    "Не выдумывай факты. Если данных недостаточно — object_type=unknown и низкий confidence.\n"
    "Ответ строго одним JSON без markdown:\n"
    '{{"observation": "1-2 предложения по-русски: что именно видно и почему это может быть нарушением", '
    '"violation": true или false, "type_id": "<id или NONE>", '
    '"object_type": "banknote" | "card" | "phone" | "paper" | "unknown", '
    '"frame_index": <номер кадра с самым явным моментом>, "evidence_frames": [<номера кадров, подтверждающих событие>], '
    '"confidence": <вероятность нарушения 0..1>, "bbox": [x1,y1,x2,y2] — рамка вокруг главного объекта на этом кадре '
    'в координатах 0..1000, или null}}'
)

ZONE_PROMPT = (
    "На кадре — вид сверху на кассу банка. Найди:\n"
    "bbox — всё рабочее место кассира: стол, кассира, его руки, счётчик банкнот, место передачи денег клиенту, "
    "сейф и ящики рядом;\n"
    "cashier — область за стойкой, где сидит/стоит кассир (служебная сторона);\n"
    "counter — стойка/край стола, через который передают деньги и документы;\n"
    "client — область перед стойкой, где стоит клиент.\n"
    "Если какой-то области не видно — null. Координаты 0..1000 всего кадра.\n"
    'Ответ строго JSON: {"bbox": [x1,y1,x2,y2], "cashier": [x1,y1,x2,y2], "counter": [x1,y1,x2,y2], "client": [x1,y1,x2,y2] или null}'
)


def parse_json(text):
    """Первый JSON-объект в ответе модели (терпит ```json, текст до/после, «умные» кавычки)."""
    text = re.sub(r"```(json)?", "", text or "").replace("“", '"').replace("”", '"')
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def norm_answer(a):
    """Ответ модели → проверенный Pydantic-схемой dict (violation — bool, confidence — 0..1, object_type …)
    или None, если ответ невалиден."""
    return validate_verdict(a, set(RULES))


def _post(url, body):
    last = None
    for attempt in range(VLM_RETRIES + 1):         # повтор: Ollama бывает занята загрузкой модели, vLLM — перегружен
        try:
            r = requests.post(url, json=body, timeout=VLM_TIMEOUT)
            if r.status_code >= 500 and attempt < VLM_RETRIES:
                last = RuntimeError(f"{r.status_code}: {r.text[:200]}")
                time.sleep(1 + attempt)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout) as e:
            last = e
            if attempt < VLM_RETRIES:
                time.sleep(1 + attempt)
    raise last


def chat(system, content, max_tokens=400):
    """content — список частей в формате OpenAI (text / image_url). Невалидный JSON — повтор (VLM_RETRIES)."""
    out = None
    for attempt in range(VLM_RETRIES + 1):
        out = _chat_once(system, content, max_tokens)
        if out is not None:
            return out
        log.info("модель вернула не JSON — повтор %d/%d", attempt + 1, VLM_RETRIES)
    return out


def chat_text(prompt, max_tokens=300):
    """Текстовый запрос к Qwen (выбор нормы в Legal-RAG)."""
    if VLM_MOCK:
        return None
    return chat(None, [{"type": "text", "text": prompt}], max_tokens=max_tokens)


def _chat_once(system, content, max_tokens):
    if BACKEND == "ollama":
        texts, images = [], []
        for c in content:
            if c["type"] == "text":
                texts.append(c["text"])
            else:
                images.append(c["image_url"]["url"].split(",", 1)[1])
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": "\n".join(texts), "images": images}]
        body = {"model": VLM_MODEL, "messages": msgs, "stream": False, "keep_alive": -1,
                "options": {"temperature": 0.0, "num_ctx": OLLAMA_CTX, "num_predict": max_tokens}}
        if OLLAMA_JSON:
            body["format"] = "json"
        return parse_json(_post(OLLAMA_URL.rstrip("/") + "/api/chat", body)["message"]["content"])
    body = {"model": VLM_MODEL, "temperature": 0.0, "max_tokens": max_tokens,
            "messages": ([{"role": "system", "content": system}] if system else []) +
                        [{"role": "user", "content": content}]}
    return parse_json(_post(VLM_URL, body)["choices"][0]["message"]["content"])


def frames_content(frames_t, zone, tail, side=None):
    content = []
    if BACKEND == "ollama":                        # родной API Ollama: картинки идут по порядку, подписи — одним текстом
        content.append({"type": "text", "text": "Кадры по порядку: " +
                        ", ".join(f"кадр {i} — t={t:.1f}с" for i, (t, _) in enumerate(frames_t)) + "."})
    for i, (t, fr) in enumerate(frames_t):
        if BACKEND != "ollama":
            content.append({"type": "text", "text": f"Кадр {i}, t={t:.1f}с"})
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + jpg_b64(resize(crop(fr, zone), side or MAX_SIDE))}})
    content.append({"type": "text", "text": tail})
    return content


def _mock(frames_t):
    return {"observation": "Тестовый режим: модель не подключена.", "violation": False, "type_id": "NONE",
            "frame_index": len(frames_t) // 2, "confidence": 0.0, "bbox": None}


def ask_scan(frames_t, zone):
    if VLM_MOCK:
        return _mock(frames_t)
    return norm_answer(chat(SCAN_PROMPT.format(rules=rules_text()),
                            frames_content(frames_t, zone, "Есть ли в этом фрагменте нарушение? Ответь JSON.")))


def ask_verify(frames_t, zone, hint_text):
    if VLM_MOCK:
        return None            # тестовый режим: «ответа нет» — правила дают unverified, а не отклоняются
    return norm_answer(chat(VERIFY_PROMPT.format(rules=rules_text(), hint=hint_text),
                            frames_content(frames_t, zone, "Перепроверь фрагмент. Ответь JSON.", side=VERIFY_SIDE),
                            max_tokens=600))


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _box(x):
    try:
        b = [float(v) for v in x][:4]
    except (TypeError, ValueError):
        return None
    return b if len(b) == 4 and 0 <= b[0] < b[2] <= 1000 and 0 <= b[1] < b[3] <= 1000 else None


def locate(video):
    """Qwen по двум кадрам: (рамка крупного плана 0..1000 | None, {cashier, counter, client} — для зон)."""
    if VLM_MOCK or not USE_ZONE or video.dur <= 0:
        return None, {}
    boxes, parts = [], {"cashier": [], "counter": [], "client": []}
    for t, fr in video.frames([video.dur * 0.3, video.dur * 0.6]).items():
        try:
            a = chat(None, [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + jpg_b64(resize(fr))}},
                            {"type": "text", "text": ZONE_PROMPT}], max_tokens=200) or {}
            b = _box(a.get("bbox") or [])
            if b:
                boxes.append(b)
            for k in parts:
                pb = _box(a.get(k) or [])
                if pb:
                    parts[k].append(pb)
        except Exception as e:  # noqa
            log.warning("zone error: %r", e)
    found = {k: _union(v) for k, v in parts.items() if v}
    if not boxes:
        return None, found
    x1, y1, x2, y2 = _union(boxes)
    mx, my = (x2 - x1) * 0.2, (y2 - y1) * 0.2          # запас 20%, чтобы не отрезать руки
    z = [max(0, x1 - mx), max(0, y1 - my), min(1000, x2 + mx), min(1000, y2 + my)]
    if (z[2] - z[0]) * (z[3] - z[1]) > 0.85e6 or (z[2] - z[0]) < 150 or (z[3] - z[1]) < 150:
        return None, found                             # почти весь кадр или подозрительно мало
    return [round(x) for x in z], found


def find_zone(video):
    """Рамка рабочего места кассира (0..1000) по двум кадрам; None — смотрим весь кадр."""
    return locate(video)[0]


# ======================= анализ =======================
def analyze(path, progress=lambda p, msg: None, t0=None):
    """t0 — момент загрузки видео: время в очереди тоже входит в лимит жюри (5 минут)."""
    t0 = t0 or time.time()
    left = lambda: TIME_BUDGET - (time.time() - t0)  # noqa: E731
    v = Video(path)
    if v.dur <= 0:
        raise ValueError("Не удалось прочитать видео. Проверьте, что файл — видеозапись .mov или .mp4.")

    progress(3, "Поиск рабочего места кассира")
    zone, qboxes = locate(v)
    zs = zones_mod.resolve(settings.camera_config(), qboxes)
    log.info("зоны: %s %s", zs.source, sorted(zs.polys))

    # ---- 0. Следопыт (YOLOX + RTMPose + ByteTrack + правила) — параллельно со сканом ----
    pose_res = {}
    pose_thread = None
    if USE_POSE and pose is not None and pose.available():
        def _pose():
            try:
                pose_res.update(pose.analyze(v, zone, zs))
            except Exception:  # noqa
                log.exception("Следопыт упал")
        pose_thread = threading.Thread(target=_pose, daemon=True)
        pose_thread.start()
        if POSE_SEQUENTIAL:                            # слабая GPU: не держим YOLOX/RTMPose и Qwen одновременно
            progress(5, "Анализ людей и предметов")
            pose_thread.join(timeout=max(left() - 90, 1))

    # ---- 1. скан всего видео ----
    # если видео долго стояло в очереди — шаг шире (но не больше окна, чтобы не было «дыр»)
    stride = STRIDE_SEC
    if left() < TIME_BUDGET * 0.6:
        stride = min(max(STRIDE_SEC * TIME_BUDGET * 0.6 / max(left(), 30), STRIDE_SEC), WIN_SEC)
        log.info("мало времени (%.0f с): шаг просмотра %.1f с", left(), stride)
    starts, s = [], 0.0
    while s < max(v.dur - 0.5, 0.01):
        starts.append(round(s, 2))
        s += stride
    step = WIN_SEC / SCAN_FRAMES
    windows = [sorted({v.clamp(st + i * step) for i in range(SCAN_FRAMES)}) for st in starts]
    progress(8, "Чтение кадров")
    v.frames([t for w in windows for t in w])

    scan = [None] * len(windows)
    still = [MOTION_T > 0 and motion(v, w, zone) < MOTION_T for w in windows]
    if any(still):
        log.info("motion skip: %d из %d окон без движения", sum(still), len(windows))
    ex = ThreadPoolExecutor(PARALLEL)
    try:
        sent = [sorted(v.frames(w).items()) for w in windows]          # кадры, которые реально уйдут в модель
        windows = [[t for t, _ in fr] or w for fr, w in zip(sent, windows)]
        futs = [None if st else ex.submit(ask_scan, fr, zone) for fr, st in zip(sent, still)]
        for i, f in enumerate(futs):
            if f is None:
                scan[i] = {"violation": False, "confidence": 0.0, "observation": "без движения"}
                continue
            try:
                scan[i] = f.result(timeout=max(left() - 40, 1))   # оставляем время на проверку
            except Exception as e:  # noqa
                log.warning("scan error: %r", e)
            progress(10 + int(50 * (i + 1) / len(windows)), f"Просмотр записи: фрагмент {i + 1} из {len(windows)}")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    vlm_down = bool(windows) and not all(still) and all(x is None for x, st in zip(scan, still) if not st)
    if vlm_down and FAIL_ON_VLM_DOWN:
        raise RuntimeError("Модель анализа недоступна. Проверьте, что Qwen запущен (vLLM или Ollama), и повторите.")
    if vlm_down:
        log.warning("Qwen недоступен — решение только по правилам (verification=unverified)")

    curve = []
    for w, a in zip(windows, scan):
        a = a or {}
        c = float(a.get("confidence") or 0)
        sv = c if a.get("violation") else min(c, 1 - c) * 0.3
        curve.append({"t": w[len(w) // 2], "s": round(sv, 3), "a": a, "w": w})

    if pose_thread is not None:
        progress(61, "Анализ движений рук")
        pose_thread.join(timeout=max(left() - 60, 1))

    # ---- 3. кандидаты ----
    cands = list(pose_res.get("candidates", []))
    for c in curve:
        if c["s"] >= CAND_T:
            a = c["a"]
            fi = min(max(int(a.get("frame_index") or 0), 0), len(c["w"]) - 1)
            cands.append({"t": c["w"][fi], "s": c["s"], "type_id": a.get("type_id"),
                          "obs": a.get("observation", ""), "src": "scan"})
    # детерминированные правила не «съедаются» соседними кандидатами — у них своя логика решения
    det = nms([c for c in cands if c.get("requires_vlm") is False], gap=VERIFY_SEC * 0.6)
    cands = nms([c for c in cands if c.get("requires_vlm") is not False], gap=VERIFY_SEC * 0.6) + det

    # ---- 4. перепроверка + голосование ----
    events = []
    if cands and left() > 20 and not vlm_down:
        progress(70, f"Перепроверка подозрительных моментов: {len(cands)}")
        jobs = []
        for c in cands:
            for k in range(VOTES):
                off = k * (VERIFY_SEC / VERIFY_FRAMES) / VOTES      # для голосования — сдвинутые наборы кадров
                st = c["t"] - VERIFY_SEC / 2 + off
                jobs.append((c, sorted({v.clamp(st + i * VERIFY_SEC / VERIFY_FRAMES) for i in range(VERIFY_FRAMES)})))
        v.frames([t for _, ts in jobs for t in ts])
        ex = ThreadPoolExecutor(PARALLEL)
        try:
            futs = []
            for c, ts in jobs:
                fr = sorted(v.frames(ts).items())
                futs.append((c, [t for t, _ in fr] or ts, ex.submit(ask_verify, fr, zone, hint(c))))
            for c, ts, f in futs:
                try:
                    a = f.result(timeout=max(left(), 1))
                except Exception as e:  # noqa
                    log.warning("verify error: %r", e)
                    a = None
                c.setdefault("votes", []).append((a, ts))
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
        for c in cands:
            events += decide(c, zone)
    else:  # нет времени на проверку или Qwen недоступен — решаем по скану и правилам
        for c in cands:
            if c["src"] == "scan" and c["s"] >= THRESH:
                events.append({"type_id": norm_type(c["type_id"]), "t": c["t"], "conf": c["s"],
                               "observation": c["obs"], "bbox": None, "verification": "scan"})
            elif c["src"] == "rule" and (c.get("requires_vlm") is False or c["s"] >= UNVERIFIED_MIN):
                events.append(_rule_event(c, "unverified"))

    progress(92, "Подготовка кадров-доказательств и норм")
    events = merge(events)
    use_qwen_legal = not vlm_down and left() > 15
    violations = [build_violation(v, e, legal_chat=chat_text if use_qwen_legal else None) for e in events]

    th = v.frames([v.dur * (i + 0.5) / 16 for i in range(16)])
    thumbs = [{"t": t, "b64": jpg_b64(resize(fr, 220), 70)} for t, fr in sorted(th.items())]
    scores = [{"t": c["t"], "s": c["s"]} for c in curve]
    return {"verdict": "violation" if violations else "no_violation", "violations": violations,
            "duration": round(v.dur, 2), "thumbs": thumbs, "scores": scores, "model": VLM_MODEL,
            "zone": zone, "zones": zs.to_json(), "tracks": pose_res.get("tracks", []),
            "rule_events": pose_res.get("rule_events", []), "roles": pose_res.get("roles", {}),
            "elapsed_sec": round(time.time() - t0, 1), "windows": len(windows), "candidates": len(cands),
            "_vis": {"scene": pose_res.get("scene"), "zones": zs}}


def motion(v, w, zone):
    """Средняя разница соседних кадров окна (0..255) в зоне кассы — есть ли движение."""
    fr = [cv2.cvtColor(cv2.resize(crop(f, zone), (64, 64)), cv2.COLOR_BGR2GRAY).astype(np.float32)
          for _, f in sorted(v.frames(w).items())]
    if len(fr) < 2:
        return 255.0
    return float(max(np.abs(a - b).mean() for a, b in zip(fr, fr[1:])))


def norm_type(tid):
    return tid if tid in RULES else "OTHER"


def hint(c):
    parts = []
    if c.get("type_id") in RULES:
        parts.append(f'возможно «{RULES[c["type_id"]]["name"]}» около t={c["t"]:.1f}с')
    for o in [c.get("obs")] + c.get("extra", []):
        if o:
            parts.append(f'предварительно: {o}')
    return "; ".join(parts) or "подозрительный момент"


def _rule_event(c, verification, a=None, ts=None, zone=None):
    """Событие из кандидата правила (с подтверждением Qwen или без него)."""
    e = {"type_id": norm_type(c["type_id"]), "t": c["t"], "conf": c["s"], "observation": c.get("obs", ""),
         "bbox": None, "verification": verification, "object_type": "unknown", "rule_event": c.get("rule_event")}
    if a:
        fi = min(max(int(a.get("frame_index") or 0), 0), len(ts) - 1)
        same = norm_type(a.get("type_id")) == e["type_id"]
        e["conf"] = max(c["s"], float(a.get("confidence") or 0))
        if c.get("requires_vlm") is not False:          # Qwen уточнил тип и момент (кроме детерминированных правил)
            e.update(type_id=norm_type(a.get("type_id")), t=ts[fi])
            same = True
        if same:                                         # описание и предмет — только если Qwen говорит о том же нарушении
            e.update(object_type=a.get("object_type", "unknown"), observation=a.get("observation") or e["observation"],
                     bbox=box_to_full(a.get("bbox"), zone))
    return e


def decide(c, zone):
    """Голосование по перепроверкам. Пропуск хуже ложной тревоги."""
    answered = [(a, ts) for a, ts in c.get("votes", []) if a]
    yes = [(a, ts) for a, ts in answered if a.get("violation") and float(a.get("confidence") or 0) >= THRESH]
    if c.get("src") == "rule" and c.get("requires_vlm") is False:
        # детерминированное правило: Qwen может только отклонить (уверенное «нет» во всех ответах)
        if answered and not yes and all(not a.get("violation") and float(a.get("confidence") or 0) < VETO_T
                                         for a, _ in answered):
            log.info("Qwen отклонил правило %s в %.1f с", c.get("rule"), c["t"])
            return []
        best = max(yes, key=lambda x: float(x[0].get("confidence") or 0)) if yes else (None, None)
        return [_rule_event(c, "confirmed" if yes else ("rule" if answered else "unverified"), best[0], best[1], zone)]
    if yes and len(yes) * 2 >= len(answered):          # большинство или ничья — за нарушение
        a, ts = max(yes, key=lambda x: float(x[0].get("confidence") or 0))
        fi = min(max(int(a.get("frame_index") or 0), 0), len(ts) - 1)
        conf = sum(float(x[0].get("confidence") or 0) for x in yes) / len(yes)
        return [{"type_id": norm_type(a.get("type_id")), "t": ts[fi], "conf": conf,
                 "observation": a.get("observation", ""), "bbox": box_to_full(a.get("bbox"), zone),
                 "verification": "confirmed", "object_type": a.get("object_type", "unknown"),
                 "rule_event": c.get("rule_event")}]
    # Qwen не ответил валидным JSON ни разу — правило всё равно может дать нарушение (unverified)
    if not answered and c.get("src") == "rule" and c["s"] >= UNVERIFIED_MIN:
        return [_rule_event(c, "unverified")]
    # перепроверка не подтвердила, но скан очень уверен — оставляем (приоритет полноты)
    if c["src"] == "scan" and c["s"] >= STRONG:
        return [{"type_id": norm_type(c["type_id"]), "t": c["t"], "conf": c["s"] * 0.8,
                 "observation": c["obs"], "bbox": None, "verification": "scan"}]
    return []


def nms(items, gap):
    """Самый сильный кандидат в окрестности gap секунд; подсказки слабых соседей сохраняются."""
    out = []
    for c in sorted(items, key=lambda x: -x["s"]):
        near = [o for o in out if abs(c["t"] - o["t"]) <= gap]
        if near:
            if c.get("obs"):
                near[0].setdefault("extra", []).append(c["obs"])
        else:
            out.append(dict(c))
    return sorted(out, key=lambda x: x["t"])


def merge(events, gap=4.0):
    """Склеивает события одного типа, близкие по времени."""
    events = sorted(events, key=lambda e: (e["type_id"], e["t"]))
    groups = []
    for e in events:
        if groups and groups[-1][0]["type_id"] == e["type_id"] and e["t"] - groups[-1][-1]["t"] <= gap:
            groups[-1].append(e)
        else:
            groups.append([e])
    return sorted([max(g, key=lambda x: x["conf"]) for g in groups], key=lambda e: e["t"])


def build_violation(v, e, legal_chat=None):
    r = RULES[e["type_id"]]
    ts = sorted({v.clamp(e["t"] + d) for d in (-1.5, 0, 1.5)})
    fr = v.frames(ts)
    if not fr:                                     # ТЗ: 1–3 кадра — ищем ближайший читаемый кадр
        for d in (0.3, -0.3, 1.0, -1.0, 2.0, -2.0):
            fr = v.frames([v.clamp(e["t"] + d)])
            if fr:
                ts = list(fr)
                break
    obs = (e.get("observation") or "").strip()
    if obs and not obs.endswith("."):
        obs += "."
    legal_ref, legal_conf = None, "low"
    if USE_LEGAL_RAG:
        legal_ref, legal_conf = legal_rag.lookup(r["id"], r, obs, chat=legal_chat)
    if legal_ref:
        norm = f'{legal_ref["article"]} {legal_ref["document"]}: «{legal_ref["quote"].rstrip(" .;")}»'
    else:
        norm = f'{r["point"]} — {r["norm"]}'
    desc = f'{obs} Признак возможного нарушения: {r["name"].lower()}. Норма: {norm}.'.strip()
    return {
        "type": r["name"],
        "timecode_sec": round(e["t"], 1),
        "description": desc,
        "evidence_frames": [{"timestamp_sec": t, "image_base64": jpg_b64(fr[t])} for t in ts if t in fr],
        # дополнительные поля для интерфейса и расширенного API
        "rule": {"id": r["id"], "document": r["doc"], "point": r["point"], "text": r["norm"]},
        "confidence": round(e["conf"], 2),
        "legal_reference": legal_ref,
        "legal_confidence": legal_conf,
        "object_type": e.get("object_type") or "unknown",
        "verification": e.get("verification", "confirmed"),
        "rule_event": e.get("rule_event"),
        "bbox": e.get("bbox"),
        "key_frame_sec": v.clamp(e["t"]),
    }
