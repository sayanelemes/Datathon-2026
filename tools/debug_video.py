"""Что модель «видит» в каждом окне видео — для разбора пропусков и ложных срабатываний.

  set -a; . profiles/l4.env; set +a
  python3 tools/debug_video.py static/demo/demo.mp4 --at 13.7        # + перепроверка вокруг 13.7 с

Печатает по каждому окну скана: время, violation, confidence, type_id, наблюдение модели.
С --at — ещё перепроверку вокруг момента: без подсказки и с подсказкой (как в пайплайне).
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline as p  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--at", type=float, help="момент для перепроверки, с")
    ap.add_argument("--zone", action="store_true", help="искать зону кассы (как в пайплайне)")
    ap.add_argument("--near", type=float, default=0, help="показывать только окна в пределах ±N с от --at")
    a = ap.parse_args()

    v = p.Video(a.video)
    zone = p.find_zone(v) if a.zone else None
    print(f"видео {v.dur:.1f} с, {v.fps:.0f} fps; зона {zone}; кадр для модели ≤ {p.MAX_SIDE}px; окно {p.WIN_SEC} с / шаг {p.STRIDE_SEC} с / {p.SCAN_FRAMES} кадров\n")
    starts, s = [], 0.0
    while s < max(v.dur - 0.5, 0.01):
        starts.append(round(s, 2))
        s += p.STRIDE_SEC
    if a.near and a.at is not None:
        starts = [st for st in starts if st + p.WIN_SEC >= a.at - a.near and st <= a.at + a.near]
    step = p.WIN_SEC / p.SCAN_FRAMES
    wins = [sorted(v.frames([v.clamp(st + i * step) for i in range(p.SCAN_FRAMES)]).items()) for st in starts]
    with ThreadPoolExecutor(p.PARALLEL) as ex:
        res = list(ex.map(lambda fr: p.ask_scan(fr, zone), wins))
    print("СКАН (окно: violation / confidence / тип — наблюдение)")
    for fr, r in zip(wins, res):
        t0, t1 = fr[0][0], fr[-1][0]
        if not r:
            print(f"  {t0:5.1f}–{t1:5.1f} с: ответ не разобран")
            continue
        mark = "!!" if r["violation"] else "  "
        print(f"{mark}{t0:5.1f}–{t1:5.1f} с: {str(r['violation']):5} {r['confidence']:.2f} {r['type_id']:<28} {r['observation'][:150]}")

    if a.at is not None:
        st = a.at - p.VERIFY_SEC / 2
        fr = sorted(v.frames([v.clamp(st + i * p.VERIFY_SEC / p.VERIFY_FRAMES) for i in range(p.VERIFY_FRAMES)]).items())
        print(f"\nПЕРЕПРОВЕРКА вокруг {a.at} с ({fr[0][0]:.1f}–{fr[-1][0]:.1f} с, {len(fr)} кадров)")
        for name, hint in (("без подсказки", "подозрительный момент"),
                           ("с подсказкой", "возможно кассир отделяет купюру и прячет её ниже стола или в руке")):
            r = p.ask_verify(fr, zone, hint)
            if r:
                print(f"  {name}: violation={r['violation']} conf={r['confidence']:.2f} type={r['type_id']} кадр={r['frame_index']}\n    «{r['observation']}»")
            else:
                print(f"  {name}: ответ не разобран")


if __name__ == "__main__":
    main()
