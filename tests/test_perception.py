"""ByteTrack и постобработка YOLOX — на синтетических данных (без весов)."""
import numpy as np

from perception.bytetrack import ByteTracker, iou_matrix, track_sequence
from perception.detector import decode, nms, postprocess, preprocess


def test_iou():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 1 / 3, 0.0], atol=1e-6)


def test_bytetrack_keeps_ids_through_weak_and_missing_frames():
    seq = []
    for k in range(30):
        a = {"box": [10 + 5 * k, 50, 60 + 5 * k, 150], "score": 0.9 if k != 10 else 0.3}
        b = {"box": [400 - 5 * k, 60, 450 - 5 * k, 160], "score": 0.9}
        seq.append((k / 5, [a, b] if k != 15 else [b]))
    out = track_sequence(seq, {}, 5)
    ids = {p["id"] for _, ps in out for p in ps}
    assert ids == {1, 2}


def test_lost_track_not_inherited_by_other_person():
    trk = ByteTracker({"track_buffer_sec": 5}, 5)
    trk.update(0.0, np.array([[0, 0, 100, 200]]), np.array([0.9]))
    trk.update(0.2, np.zeros((0, 4)), np.zeros(0))                      # кассир ушёл
    res = trk.update(0.4, np.array([[60, 0, 160, 200]]), np.array([0.9]))  # другой человек рядом (IoU 0.25)
    assert res[0][1] != 1


def test_yolox_postprocess_synthetic():
    size = 64
    n = sum((size // s) ** 2 for s in (8, 16, 32))
    raw = np.zeros((n, 85), np.float32)
    raw[:, 4] = -10
    # одна «уверенная» ячейка: класс 67 (телефон) в ячейке (2,3) страйда 8
    i = 3 * 8 + 2
    raw[i, :4] = [0.5, 0.5, np.log(2.0), np.log(3.0)]
    raw[i, 4] = 0.9
    raw[i, 5 + 67] = 0.95
    dets = postprocess(raw, 1.0, size, {67: "phone", 0: "person"}, 0.3, 0.45)
    assert len(dets) == 1 and dets[0].cls == "phone"
    cx = (dets[0].box[0] + dets[0].box[2]) / 2
    assert abs(cx - 20) < 1e-3                                           # (2 + 0.5) * 8


def test_preprocess_and_nms():
    x, r = preprocess(np.zeros((100, 200, 3), np.uint8), 64)
    assert x.shape == (1, 3, 64, 64) and abs(r - 0.32) < 1e-6
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    assert sorted(nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5)) == [0, 2]
    assert decode(np.zeros((sum((64 // s) ** 2 for s in (8, 16, 32)), 85)), 64).shape[1] == 85
