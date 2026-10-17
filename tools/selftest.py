"""Самопроверка сервиса на сервере: модель, JSON, скорость, RTMPose, целое видео.

  python3 tools/selftest.py                 # все проверки на демо-видео
  python3 tools/selftest.py --video X.mov   # плюс полный анализ этого видео с замером времени
  python3 tools/selftest.py --quick         # один запрос к модели (прогрев; так вызывает start.sh)
  python3 tools/selftest.py --offline [--video X.mov]   # доказательство локальности для жюри

Настройки берутся из окружения, как у сервиса (start.sh экспортирует профиль).
Чтобы запустить вручную с профилем:  set -a; . profiles/l4.env; set +a; python3 tools/selftest.py
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline  # noqa: E402
from tools import netguard  # noqa: E402

DEMO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "demo", "demo.mp4")
OK, BAD = "  ✓", "  ✗"
fails = []


def check(name, fn):
    t = time.time()
    try:
        msg = fn()
        print(f"{OK} {name} ({time.time() - t:.1f} с){': ' + msg if msg else ''}")
        return True
    except Exception as e:  # noqa
        print(f"{BAD} {name} ({time.time() - t:.1f} с): {e}")
        fails.append(name)
        return False


def frames(video, n, t0=None, span=4.0):
    v = pipeline.Video(video)
    t0 = v.dur * 0.4 if t0 is None else t0
    return v, sorted(v.frames([t0 + i * span / n for i in range(n)]).items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="полный анализ этого видео")
    ap.add_argument("--quick", action="store_true", help="только один запрос к модели")
    ap.add_argument("--offline", action="store_true",
                    help="доказательство локальности: полный анализ с запретом внешней сети и наблюдением за соединениями")
    a = ap.parse_args()
    src = a.video or DEMO

    print(f"Бэкенд {pipeline.BACKEND}, модель {pipeline.VLM_MODEL}, профиль {os.getenv('PROFILE', '-')}")
    if pipeline.VLM_MOCK:
        print("[i] VLM_MOCK=1 — модель не вызывается")

    if a.offline:
        sys.exit(offline(src))

    if a.quick:
        _, fr = frames(src, 2)
        try:
            r = pipeline.ask_scan(fr, None)
        except Exception as e:  # noqa
            print("модель не отвечает:", e)
            r = None
        sys.exit(0 if r else 1)

    print("\n1. Видео")
    check("чтение кадров " + os.path.basename(src),
          lambda: "%.1f с, %.0f fps, кадров %d" % (lambda v: (v.dur, v.fps, v.n))(pipeline.Video(src)))

    print("\n2. Модель")

    def one_scan():
        _, fr = frames(src, pipeline.SCAN_FRAMES)
        r = pipeline.ask_scan(fr, None)
        if not r:
            raise RuntimeError("ответ не разобран как JSON")
        return f'violation={r["violation"]} conf={r["confidence"]:.2f} type={r["type_id"]} — «{r["observation"][:90]}»'
    check(f"скан: {pipeline.SCAN_FRAMES} кадров → JSON", one_scan)

    def one_verify():
        _, fr = frames(src, pipeline.VERIFY_FRAMES, span=pipeline.VERIFY_SEC)
        r = pipeline.ask_verify(fr, None, "подозрительный момент")
        if not r:
            raise RuntimeError("ответ не разобран как JSON")
        return f'violation={r["violation"]} conf={r["confidence"]:.2f} object={r.get("object_type")} bbox={r["bbox"]}'
    check(f"перепроверка: {pipeline.VERIFY_FRAMES} кадров → JSON", one_verify)

    def detector():
        import settings
        from perception import detector as objdet
        d = objdet.get(settings.rules_config().get("detector") or {})
        if d is None:
            raise RuntimeError("нет ONNX YOLOX COCO — python3 tools/fetch_models.py")
        _, fr = frames(src, 1)
        t = time.time()
        n = len(d(fr[0][1]))
        return f"считает на {d.provider}, {1000 * (time.time() - t):.0f} мс/кадр, объектов на кадре: {n}"
    check("YOLOX COCO: телефоны и предметы", detector)

    def legal():
        import legal_rag
        idx = legal_rag.get_index()
        if not idx.chunks:
            raise RuntimeError("legal_base/ пуста — нормы будут из каталога без цитат")
        top = idx.search("кассир покинул рабочее место до окончания обслуживания клиента", 1)[0][0]
        return f"{len(idx.chunks)} фрагментов, поиск {idx.backend}; «кассир ушёл» → {top.article}"
    check("Legal-RAG: нормативная база", legal)

    def zone():
        z = pipeline.find_zone(pipeline.Video(src))
        return f"зона кассы {z}" if z else "зона не найдена — будет весь кадр (это допустимо)"
    check("поиск рабочего места кассира", zone)

    def throughput():
        n = max(pipeline.PARALLEL, 2)
        v = pipeline.Video(src)
        jobs = [sorted(v.frames([(v.dur * k / n) + i * 0.5 for i in range(pipeline.SCAN_FRAMES)]).items()) for k in range(n)]
        t = time.time()
        with ThreadPoolExecutor(pipeline.PARALLEL) as ex:
            res = list(ex.map(lambda f: pipeline.ask_scan(f, None), jobs))
        dt = time.time() - t
        bad = sum(r is None for r in res)
        if bad * 4 >= n:
            raise RuntimeError(f"{bad} из {n} ответов не разобраны — модель отвечает не JSON, смотрите журнал")
        per = dt / n
        win = 145 / pipeline.STRIDE_SEC
        return (f"не разобрано {bad} из {n}; " if bad else "") + f"{n} окон параллельно за {dt:.1f} с → {per:.2f} с на окно; видео 145 с ≈ {win:.0f} окон ≈ {win * per:.0f} с на скан"
    check(f"скорость: {pipeline.PARALLEL} запросов параллельно", throughput)

    print("\n3. Следопыт (RTMPose)")
    if pipeline.pose is None or not pipeline.pose.available() or not pipeline.USE_POSE:
        print(f"{BAD} rtmlib не установлен или USE_POSE=0 — Следопыт выключен")
        fails.append("pose")
    else:
        def pose_load():
            pipeline.pose._load()
            return f"считает на {pipeline.pose.PROVIDER}"
        if check("загрузка RTMPose", pose_load):
            def pose_speed():
                _, fr = frames(src, 10, span=3)
                t = time.time()
                people = [len(pipeline.pose.detect(f)) for _, f in fr]
                per = (time.time() - t) / len(fr)
                return (f"{per * 1000:.0f} мс на кадр, людей в кадре: {people}; "
                        f"видео 145 с при POSE_FPS={pipeline.pose.POSE_FPS:g} ≈ {145 * pipeline.pose.POSE_FPS * per:.0f} с")
            check("скорость RTMPose", pose_speed)

    if a.video:
        print("\n4. Полный анализ (как у жюри)")

        def full():
            t = time.time()
            last = [None]

            def prog(p, msg):
                if msg != last[0]:
                    last[0] = msg
                    print(f"     {time.time() - t:5.1f} с  {p:3d}%  {msg}")
            r = pipeline.analyze(a.video, prog)
            dt = time.time() - t
            for v in r["violations"]:
                ref = v.get("legal_reference") or {}
                print(f"     → {v['timecode_sec']:6.1f} с  {v['type']}  (уверенность {v['confidence']}, {v.get('verification')}, "
                      f"кадров {len(v['evidence_frames'])}, норма {ref.get('article', '—')} [{v.get('legal_confidence')}])")
            if dt > 300:
                raise RuntimeError(f"{dt:.0f} с — дольше лимита 5 минут! Увеличьте STRIDE_SEC или уменьшите VOTES")
            return (f"вердикт {r['verdict']}, нарушений {len(r['violations'])}, окон {r.get('windows')}, "
                    f"кандидатов {r.get('candidates')}, время {dt:.0f} с из 300")
        check("анализ " + os.path.basename(a.video), full)

    print("\nИтог:", "всё в порядке" if not fails else "есть проблемы: " + ", ".join(fails))
    sys.exit(1 if fails else 0)


def offline(src):
    """Полный анализ видео при запрете внешней сети. Код возврата 0 — анализ прошёл, внешних соединений нет."""
    print("\nПроверка локальности: анализ без доступа к внешней сети\n")
    netguard.install()
    probe = []
    import requests
    for host in ("huggingface.co", "api.openai.com", "8.8.8.8"):      # убеждаемся, что запрет действует
        try:
            requests.get(f"https://{host}/", timeout=3)
            probe.append(f"{host}: ДОСТУПЕН")
            fails.append(f"{host} доступен при запрете")
        except Exception:  # noqa
            probe.append(f"{host}: заблокирован")
    print(f"{OK} запрет внешней сети в процессе анализа включён ({'; '.join(probe)})")
    if netguard.PROXIES:
        print(f"  [i] сняты переменные прокси {', '.join(netguard.PROXIES)} — через прокси трафик ушёл бы мимо проверки")
    netguard.BLOCKED.clear()

    vport = int(__import__("re").search(r":(\d+)/", pipeline.VLM_URL + "/").group(1)) if pipeline.BACKEND == "vllm" else 11434
    vl = netguard.listen_addr(vport)
    if vl:
        mark = OK if all(x in ("127.0.0.1", "::1") for x in vl) else BAD
        print(f"{mark} vLLM слушает только {', '.join(vl)}:{vport} — снаружи модель недоступна")
        if mark == BAD:
            fails.append("vLLM слушает внешний адрес")
    mp = netguard.port_pid(vport)
    off = netguard.pid_env(mp, "HF_HUB_OFFLINE") if mp and pipeline.BACKEND == "vllm" else None
    if off is not None:
        print(f"{OK if off == '1' else BAD} у процесса модели (PID {mp}: {netguard.cmdline(mp)[:40]}…) HF_HUB_OFFLINE={off}")
        if off != "1":
            fails.append("HF_HUB_OFFLINE у vLLM")
    print(f"{OK if os.getenv('HF_HUB_OFFLINE') == '1' else BAD} у сервиса HF_HUB_OFFLINE={os.getenv('HF_HUB_OFFLINE', '(не задана)')}")

    sm = netguard.Sampler()
    sm.start()
    t = time.time()
    try:
        r = pipeline.analyze(src)
        res = f"вердикт {r['verdict']}, нарушений {len(r['violations'])}, Следопыт: {'да' if r.get('tracks') else 'нет'}"
        good = True
    except Exception as e:  # noqa
        res, good = f"анализ упал: {e}", False
    dt = time.time() - t
    sm.stop()
    print(f"{OK if good else BAD} полный анализ {os.path.basename(src)} без сети: {res} ({dt:.0f} с)")
    if not good:
        fails.append("анализ без сети")

    if netguard.BLOCKED:
        print(f"{BAD} попытки выйти в сеть из кода анализа: {len(netguard.BLOCKED)}")
        for addr, where in sorted(set(netguard.BLOCKED))[:10]:
            print(f"     {addr}  ← {where}")
        fails.append("попытки выхода в сеть")
    else:
        print(f"{OK} попыток выйти в сеть из кода анализа: 0")
    if sm.found:
        print(f"{BAD} исходящие соединения vLLM/сервиса на внешние адреса: {len(sm.found)}")
        for addr, who in sm.found.items():
            print(f"     {addr}  ← {who}")
        fails.append("внешние соединения")
    else:
        print(f"{OK} исходящих соединений vLLM/сервиса на внешние адреса: 0 (проверок соединений: {sm.samples})")

    print("\nИтог:", "анализ работает полностью локально" if not fails else "есть проблемы: " + ", ".join(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    main()
