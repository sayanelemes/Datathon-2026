"""Скачивает веса Следопыта один раз (дальше всё работает без интернета):
  YOLOX COCO (Megvii, Apache 2.0) — телефоны и предметы → MODELS_DIR;
  RTMPose / YOLOX людей (rtmlib) — в кэш rtmlib (~/.cache/rtmlib), если указан --pose.

  python3 tools/fetch_models.py [--models yolox_s,yolox_tiny] [--pose]
"""
import argparse
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import settings  # noqa: E402

URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{name}.onnx"
SIZE_MIN = {"yolox_nano": 3_000_000, "yolox_tiny": 15_000_000, "yolox_s": 30_000_000, "yolox_m": 90_000_000}


def fetch(name: str) -> str:
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    dst = os.path.join(settings.MODELS_DIR, f"{name}.onnx")
    if os.path.exists(dst) and os.path.getsize(dst) >= SIZE_MIN.get(name, 1):
        print(f"  ✓ {name}: уже есть ({os.path.getsize(dst) / 1e6:.0f} МБ)")
        return dst
    tmp = dst + ".part"
    print(f"  ↓ {name} ← {URL.format(name=name)}")
    urllib.request.urlretrieve(URL.format(name=name), tmp)
    if os.path.getsize(tmp) < SIZE_MIN.get(name, 1):
        os.remove(tmp)
        raise RuntimeError(f"{name}: файл подозрительно маленький — скачивание прервано?")
    os.replace(tmp, dst)
    print(f"  ✓ {name}: {os.path.getsize(dst) / 1e6:.0f} МБ")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.getenv("FETCH_MODELS", "yolox_s,yolox_tiny"))
    ap.add_argument("--pose", action="store_true", help="также прогреть веса RTMPose (rtmlib)")
    a = ap.parse_args()
    ok = True
    for name in [m.strip() for m in a.models.split(",") if m.strip()]:
        try:
            fetch(name)
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  [!] {name}: {e} — без него телефоны и предметы не детектируются, остальное работает")
    if a.pose:
        try:
            import pose
            pose._load()
            print(f"  ✓ RTMPose: считает на {pose.PROVIDER}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  [!] RTMPose: {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
