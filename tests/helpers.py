"""Синтетические сцены для правил: люди как набор ключевых точек, без видео и нейросетей."""
import numpy as np

from rule_engine import CASHIER, CLIENT, Frame, Obj, Person, Scene
from zones import Zones

W, H, FPS = 1000, 1000, 5.0

# кассир сверху, клиент снизу, стойка посередине (нормированные координаты)
ZONES = Zones({
    "cashier_zone": [(0.2, 0.0), (0.8, 0.0), (0.8, 0.45), (0.2, 0.45)],
    "transfer_zone": [(0.2, 0.4), (0.8, 0.4), (0.8, 0.6), (0.2, 0.6)],
    "client_zone": [(0.2, 0.55), (0.8, 0.55), (0.8, 1.0), (0.2, 1.0)],
    "restricted_zone": [(0.2, 0.0), (0.8, 0.0), (0.8, 0.4), (0.2, 0.4)],
}, "test")


def person(pid, cx, cy, role, s=80.0, lw=None, rw=None, nose=None):
    kp = np.zeros((17, 2))
    sc = np.full(17, 0.9)
    kp[0] = nose if nose is not None else (cx, cy - s)
    kp[5], kp[6] = (cx - s / 2, cy - s / 2), (cx + s / 2, cy - s / 2)
    kp[7], kp[8] = (cx - s / 2, cy), (cx + s / 2, cy)
    kp[9] = lw if lw is not None else (cx - s / 2, cy + s / 2)
    kp[10] = rw if rw is not None else (cx + s / 2, cy + s / 2)
    kp[11], kp[12] = (cx - s / 3, cy + s / 2), (cx + s / 3, cy + s / 2)
    kp[13:] = (cx, cy + s)
    return Person(pid, [cx - s, cy - 1.5 * s, cx + s, cy + 1.5 * s], kp, sc, 0.9, role)


def scene(n_sec, make, zones=ZONES):
    """make(t) -> (persons, objects) для каждого кадра."""
    frames = []
    for i in range(int(n_sec * FPS) + 1):
        t = round(i / FPS, 2)
        ps, os_ = make(t)
        frames.append(Frame(t, ps, os_))
    return Scene(frames, zones, W, H, FPS)


def cashier(t=0, **kw):
    return person(1, 500, 200, CASHIER, **kw)


def client(t=0, pid=2, cx=500, cy=750, **kw):
    return person(pid, cx, cy, CLIENT, **kw)


def phone(x, y, oid=10):
    return Obj("phone", [x - 15, y - 25, x + 15, y + 25], 0.8, oid)


def small(x, y, oid=20):
    return Obj("small_object", [x - 20, y - 20, x + 20, y + 20], 0.7, oid)
