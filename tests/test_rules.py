"""Rule Engine: каждое правило срабатывает на своём сценарии и молчит на обычной работе."""
import settings
from helpers import CASHIER, CLIENT, ZONES, cashier, client, person, phone, scene, small

import rule_engine
from rule_engine import UNKNOWN, cashier_left, hand_rules, phone_use, restricted_zone, transfer_of_item
from rule_engine.base import segments
from zones import Zones

CFG = settings.rules_config()["rules"]
REQUIRED = {"violation_type", "start_timecode_sec", "end_timecode_sec", "involved_track_ids",
            "evidence_frame_timestamps", "preliminary_confidence", "raw_evidence"}


def test_segments_merge_short_gaps():
    times = [i * 0.2 for i in range(20)]
    flags = [True] * 5 + [False] + [True] * 5 + [False] * 9
    assert segments(flags, times, 1.0, 0.2, max_gap=0.2) == [(0.0, 2.0)]
    assert segments(flags, times, 1.5, 0.2, max_gap=0.0) == []


# ---------------- transfer_of_item ----------------
def _transfer(t, with_obj=True, active=lambda t: 2 <= t <= 3.4):
    if active(t):
        c = cashier(rw=(500, 470))
        k = client(lw=(520, 520))
        objs = [small(505, 470 + (t - 2) * 30)] if with_obj else []
        return [c, k], objs
    return [cashier(), client()], []


def test_transfer_with_object_moving_to_client():
    res = transfer_of_item.run(scene(6, _transfer), CFG["transfer_of_item"])
    assert len(res) == 1
    r = res[0]
    assert REQUIRED <= set(r.to_dict())
    assert r.violation_type == "transfer_of_item" and r.requires_vlm
    assert 1.8 <= r.start_timecode_sec <= 2.2 and 3.2 <= r.end_timecode_sec <= 3.6
    assert r.raw_evidence["object_progress"] > 0.15
    assert r.preliminary_confidence == CFG["transfer_of_item"]["confidence_with_object"]
    assert set(r.involved_track_ids) == {1, 2}


def test_transfer_without_object_is_weaker_and_require_object_blocks_it():
    res = transfer_of_item.run(scene(6, lambda t: _transfer(t, with_obj=False)), CFG["transfer_of_item"])
    assert len(res) == 1 and res[0].preliminary_confidence == CFG["transfer_of_item"]["confidence"]
    strict = dict(CFG["transfer_of_item"], require_object=True)
    assert transfer_of_item.run(scene(6, lambda t: _transfer(t, with_obj=False)), strict) == []


def test_transfer_too_short_or_outside_zone():
    assert transfer_of_item.run(scene(6, lambda t: _transfer(t, active=lambda t: 2 <= t <= 2.2)), CFG["transfer_of_item"]) == []

    def far(t):  # руки рядом, но у кассира над столом (не в зоне передачи)
        return [cashier(rw=(500, 250)), client(cy=750, lw=(520, 280))], []
    assert transfer_of_item.run(scene(4, far), CFG["transfer_of_item"]) == []


# ---------------- phone_use_during_service ----------------
def _phone(t, start=3, end=6, with_client=True):
    ps = [cashier()] + ([client()] if with_client else [])
    return ps, ([phone(500, 125)] if start <= t <= end else [])


def test_phone_near_face_with_client():
    res = phone_use.run(scene(10, _phone), CFG["phone_use_during_service"])
    assert len(res) == 1
    r = res[0]
    assert r.violation_type == "phone_use_during_service" and r.maps_to == "PHONE_AT_CASH_DESK"
    assert r.raw_evidence["client_present"] is True
    assert r.preliminary_confidence == CFG["phone_use_during_service"]["confidence"]


def test_phone_short_or_far_is_ignored():
    assert phone_use.run(scene(10, lambda t: _phone(t, 3, 4)), CFG["phone_use_during_service"]) == []

    def far(t):
        return [cashier(), client()], [phone(900, 900)]
    assert phone_use.run(scene(10, far), CFG["phone_use_during_service"]) == []


def test_phone_without_client_kept_but_weaker():
    res = phone_use.run(scene(10, lambda t: _phone(t, with_client=False)), CFG["phone_use_during_service"])
    assert len(res) == 1 and res[0].preliminary_confidence < CFG["phone_use_during_service"]["confidence"]


# ---------------- cashier_left_workspace ----------------
def _away(a, b):
    return lambda t: ([client()] + ([] if a <= t <= b else [cashier()]), [])


def test_cashier_left_long_is_deterministic():
    res = cashier_left.run(scene(25, _away(5, 17)), CFG["cashier_left_workspace"])
    assert len(res) == 1
    r = res[0]
    assert not r.requires_vlm and r.raw_evidence["absence_sec"] >= 10
    assert r.maps_to == "CASHIER_LEFT_DURING_SERVICE"


def test_cashier_left_short_needs_vlm_and_tiny_ignored():
    res = cashier_left.run(scene(25, _away(5, 9)), CFG["cashier_left_workspace"])
    assert len(res) == 1 and res[0].requires_vlm
    assert cashier_left.run(scene(25, _away(5, 6)), CFG["cashier_left_workspace"]) == []


def test_cashier_left_without_client_is_not_deterministic():
    res = cashier_left.run(scene(25, lambda t: ([] if 5 <= t <= 17 else [cashier()], [])), CFG["cashier_left_workspace"])
    assert len(res) == 1 and res[0].requires_vlm


# ---------------- restricted zone ----------------
def test_unauthorized_person_in_restricted_zone():
    def mk(t):
        ps = [cashier(), client()]
        if 2 <= t <= 6:
            ps.append(person(3, 300, 150, UNKNOWN))
        return ps, []
    res = restricted_zone.run_unauthorized(scene(10, mk), CFG["unauthorized_person_in_restricted_zone"])
    assert len(res) == 1 and res[0].involved_track_ids == [3] and not res[0].requires_vlm


def test_unauthorized_short_visit_and_cashier_ignored():
    def mk(t):
        ps = [cashier(), client()]
        if 2 <= t <= 3.5:
            ps.append(person(3, 300, 150, UNKNOWN))
        return ps, []
    assert restricted_zone.run_unauthorized(scene(10, mk), CFG["unauthorized_person_in_restricted_zone"]) == []


def test_client_crosses_counter():
    def mk(t):
        cy = 150 if 3 <= t <= 6 else 750
        return [cashier(), client(cy=cy, cx=300)], []
    res = restricted_zone.run_client(scene(10, mk), CFG["client_in_restricted_zone"])
    assert len(res) == 1 and res[0].violation_type == "client_in_restricted_zone"
    assert res[0].raw_evidence["crossed_from"] == "client_zone"


def test_rules_without_zones_do_not_crash():
    sc = scene(10, lambda t: ([cashier(), client()], [phone(500, 125)]), zones=Zones())
    out = rule_engine.run(sc, settings.rules_config())
    assert all(r.violation_type != "unauthorized_person_in_restricted_zone" for r in out)


# ---------------- legacy hand rules ----------------
def test_hand_below_desk():
    def mk(t):
        if 2 <= t <= 4:
            return [cashier(lw=(460, 400)), client()], []        # левая рука ниже бёдер, правая у стола
        return [cashier(), client()], []
    res = hand_rules.run_below(scene(8, mk), CFG["hand_below_desk"])
    assert len(res) == 1 and res[0].maps_to == "CASH_CONCEALMENT" and res[0].requires_vlm


# ---------------- roles ----------------
def test_roles_cashier_client_and_intruder():
    def mk(t):
        ps = [client()]
        if t < 8 or t > 14:
            ps.append(person(1 if t < 8 else 4, 500, 200, UNKNOWN))   # кассир вернулся с новым id на своё место
        if 9 <= t <= 13:
            ps.append(person(3, 700, 120, UNKNOWN))                    # посторонний в другом месте за стойкой
        return ps, []
    sc = scene(20, mk)
    roles = rule_engine.assign_roles(sc, settings.rules_config())
    assert roles[1] == CASHIER and roles[4] == CASHIER
    assert roles[2] == CLIENT
    assert roles[3] != CASHIER


def test_engine_output_is_candidate_ready():
    sc = scene(10, lambda t: _phone(t))
    for f in sc.frames:
        for p in f.persons:
            p.role = CASHIER if p.id == 1 else CLIENT
    out = rule_engine.run(sc, settings.rules_config())
    assert out
    c = out[0].to_candidate()
    assert {"t", "s", "type_id", "src", "obs", "requires_vlm", "rule_event"} <= set(c)
    assert c["src"] == "rule"
    assert ZONES.has("transfer_zone")
