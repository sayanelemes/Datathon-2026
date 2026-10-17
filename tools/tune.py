"""Подбор порогов решения по журналу (cands_debug) — без повторных вызовов модели.

  python3 tools/tune.py --labels /root/dataset/labels.csv --report /root/dataset/eval_report.csv \
                        --jobs /workspace/2mp_jobs --val /root/dataset/val.txt

Нужен прогон tools/eval_api.py на версии сервиса, которая пишет cands_debug в результат задачи.
Подбирает на обучающей части (всё, кроме val.txt), отдельно показывает отложенную часть.
Печатает строку настроек для профиля (profiles/l4.env).
"""
import argparse
import csv
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.common import read_labels  # noqa: E402

INACTION = ("NO_RECOUNT_IN_FRONT", "PAYOUT_NO_RECOUNT", "BAG_ACCEPTED_NO_CHECK")
DEFAULT = dict(thresh=0.5, strong=0.8, vote="tie", det="veto", inaction=0.0, veto=0.25)


def decide(c, P):
    votes = [v for v in c.get("votes", []) if v]
    tthr = {k: P["inaction"] for k in INACTION} if P["inaction"] else {}
    yes = [v for v in votes if v["v"] and v["c"] >= max(P["thresh"], tthr.get(v["type"], 0.0))]
    if c.get("det"):
        if P["det"] == "off":
            return None
        if P["det"] == "veto":
            if votes and not yes and all(not v["v"] and v["c"] < P["veto"] for v in votes):
                return None
            return (c["type"], c["t"])
    need = {"all": len(votes), "majority": len(votes) // 2 + 1}.get(P["vote"], (len(votes) + 1) // 2)
    if yes and len(yes) >= max(need, 1):
        b = max(yes, key=lambda v: v["c"])
        return (b["type"], b["t"])
    if not votes and c.get("src") == "rule" and c.get("s", 0) >= 0.5:
        return (c["type"], c["t"])
    if c.get("src") == "scan" and c.get("s", 0) >= P["strong"]:
        return (c["type"], c["t"])
    return None


def events_of(job, P):
    if "cands_debug" not in job:              # старая версия сервиса — берём итог как есть
        return [(v.get("rule", {}).get("id", "?"), v["timecode_sec"]) for v in job.get("violations", [])]
    evs = []
    for c in job["cands_debug"]:
        e = decide(c, P)
        if e and e[0] not in P.get("disabled", ()):
            evs.append(e)
    return evs


def score(data, P, names):
    tp = fp = fn = tn = 0
    errs, fp_types, tp_types = [], {}, {}
    for n in names:
        truth, job = data[n]
        evs = events_of(job, P)
        want, got = bool(truth), bool(evs)
        tp += want and got
        fn += want and not got
        fp += (not want) and got
        tn += (not want) and not got
        for typ, _ in evs:
            (tp_types if want else fp_types)[typ] = (tp_types if want else fp_types).get(typ, 0) + 1
        if want and got:
            errs.append(min(abs(t - m["t"]) for _, t in evs for m in truth))
    N = len(names) or 1
    return {"acc": (tp + tn) / N, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "tc": sum(errs) / len(errs) if errs else None, "fp_types": fp_types, "tp_types": tp_types}


def fmt(P):
    return (f"THRESH={P['thresh']} STRONG={P['strong']} VOTE_RULE={P['vote']} DET_MODE={P['det']}"
            + (f" inaction≥{P['inaction']}" if P["inaction"] else ""))


def env_line(P):
    parts = [f"THRESH={P['thresh']}", f"STRONG={P['strong']}", f"VOTE_RULE={P['vote']}", f"DET_MODE={P['det']}"]
    if P["inaction"]:
        parts.append("TYPE_THRESH=" + ",".join(f"{k}={P['inaction']}" for k in INACTION))
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--jobs", default="/workspace/2mp_jobs")
    ap.add_argument("--val", default=None)
    a = ap.parse_args()

    labels = read_labels(a.labels)
    data = {}
    with open(a.report, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = os.path.join(a.jobs, r["job_id"], "result.json")
            if r["video"] in labels and os.path.exists(p):
                data[r["video"]] = (labels[r["video"]], json.load(open(p, encoding="utf-8")))
    if not data:
        sys.exit("Нет результатов задач: проверьте --report и --jobs")
    have_dbg = sum("cands_debug" in j for _, j in data.values())
    val = set()
    if a.val and os.path.exists(a.val):
        val = {os.path.splitext(x.strip())[0] for x in open(a.val, encoding="utf-8") if x.strip()}
    train = sorted(n for n in data if n not in val)
    held = sorted(n for n in data if n in val)
    print(f"видео: {len(data)} (журнал решений есть у {have_dbg}); подбор на {len(train)}, проверка на {len(held)}\n")

    base = score(data, DEFAULT, train)
    grid = []
    for th, st, vote, det, ina in itertools.product([0.5, 0.6, 0.7, 0.8, 0.9], [0.8, 0.9, 0.95, 1.01],
                                                   ["tie", "majority", "all"], ["veto", "confirm", "off"],
                                                   [0.0, 0.8, 0.9, 1.01]):
        P = dict(DEFAULT, thresh=th, strong=st, vote=vote, det=det, inaction=ina)
        grid.append((score(data, P, train), P))
    # лучшие: точность, затем меньше пропусков, затем ближе к настройкам по умолчанию
    grid.sort(key=lambda x: (-x[0]["acc"], x[0]["fn"], sum(x[1][k] != DEFAULT[k] for k in DEFAULT)))

    def line(tag, s, P, names):
        h = score(data, P, held) if held else None
        tc = f"{s['tc']:.1f} с" if s["tc"] is not None else "—"
        hv = f" | отложенные: {h['acc']:.0%} (FN {h['fn']}, FP {h['fp']})" if h else ""
        print(f"{tag:10s} {s['acc']:.0%}  FN {s['fn']:2d}  FP {s['fp']:2d}  таймкод {tc:7s}{hv}   {fmt(P)}")

    line("сейчас", base, DEFAULT, train)
    print()
    for s, P in grid[:12]:
        line("", s, P, train)
    best_s, best = grid[0]
    no_miss = [g for g in grid if g[0]["fn"] == 0]
    print("\nЛожные срабатывания по типам (лучшие настройки, обучающая часть):",
          ", ".join(f"{k}: {v}" for k, v in sorted(best_s["fp_types"].items(), key=lambda x: -x[1])) or "нет")
    print("Найденные нарушения по типам на видео с нарушением:",
          ", ".join(f"{k}: {v}" for k, v in sorted(best_s["tp_types"].items(), key=lambda x: -x[1])) or "нет")
    if no_miss:
        s0, P0 = no_miss[0]
        print(f"\nЛучшее БЕЗ пропусков: {s0['acc']:.0%}, FP {s0['fp']}   {fmt(P0)}")
    print("\nСтрока для profiles/l4.env (лучшие по точности):\n" + env_line(best))


if __name__ == "__main__":
    main()
