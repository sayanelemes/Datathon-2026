"""Разбор ответов Qwen: JSON в тексте, форматы чисел, синонимы ключей, невалидные ответы."""
import pipeline
from schemas import LegalChoice, QwenVerdict, validate_verdict


def test_parse_json_variants():
    assert pipeline.parse_json('```json\n{"violation": true}\n```') == {"violation": True}
    assert pipeline.parse_json('Ответ: {"a": 1} и ещё текст') == {"a": 1}
    assert pipeline.parse_json('“{“violation”: false}”'.replace("“{", "{").replace("}”", "}")) == {"violation": False}
    assert pipeline.parse_json("нет json") is None
    assert pipeline.parse_json('{"a": [1, 2}') is None


def test_verdict_normalization():
    d = validate_verdict({"violation": "да", "type": "cash_concealment", "object_type": "Купюры", "confidence": "85%",
                          "description": "кассир прячет купюру", "frame_index": "3", "evidence_frames": [1, "2", {"frame_index": 4}]},
                         {"CASH_CONCEALMENT"})
    assert d["violation"] is True and d["type_id"] == "CASH_CONCEALMENT"
    assert d["object_type"] == "banknote" and d["confidence"] == 0.85
    assert d["observation"] == "кассир прячет купюру" and d["frame_index"] == 3
    assert d["evidence_frames"] == [1, 2, 4]


def test_confidence_formats():
    for raw, exp in ((0.7, 0.7), (70, 0.7), ("0,7", 0.7), ("high", 0.85), ("низкая", 0.15), (2.5, 0.025), (-1, 0.0)):
        assert abs(QwenVerdict.model_validate({"violation": True, "confidence": raw}).confidence - exp) < 1e-9


def test_unknown_object_and_type():
    d = validate_verdict({"violation": False, "type_id": "SOMETHING", "object_type": "кошка", "confidence": 0.1}, {"OTHER"})
    assert d["object_type"] == "unknown" and d["type_id"] == "NONE"


def test_invalid_answers():
    assert validate_verdict(None) is None
    assert validate_verdict("строка") is None
    assert validate_verdict({"observation": "что-то"}) is None            # нет ни violation, ни confidence
    assert validate_verdict({"confidence": 0.9})["violation"] is True      # violation выводится из confidence


def test_bbox_validation():
    assert validate_verdict({"violation": True, "bbox": [1, 2, 3]})["bbox"] is None
    assert validate_verdict({"violation": True, "bbox": ["1", 2, 3, 4]})["bbox"] == [1.0, 2.0, 3.0, 4.0]


def test_legal_choice():
    c = LegalChoice.model_validate({"choice": "2", "confidence": "HIGH"})
    assert c.choice == 2 and c.confidence == "high"
    assert LegalChoice.model_validate({"choice": None}).choice is None
    assert LegalChoice.model_validate({"choice": 1, "confidence": 0.5}).confidence == "medium"


def test_chat_retries_invalid_json(monkeypatch):
    calls = []

    def fake(system, content, max_tokens):
        calls.append(1)
        return None if len(calls) < 2 else {"violation": False}
    monkeypatch.setattr(pipeline, "_chat_once", fake)
    assert pipeline.chat(None, [], 10) == {"violation": False}
    assert len(calls) == 2


def test_decide_rule_paths(monkeypatch):
    ts = [1.0, 1.5, 2.0]
    det = {"t": 1.5, "s": 0.7, "type_id": "UNAUTHORIZED_PERSON", "src": "rule", "requires_vlm": False, "obs": "x"}
    # Qwen уверенно «нет» — детерминированное правило отклонено
    no = {"violation": False, "confidence": 0.05, "type_id": "NONE", "frame_index": 0, "object_type": "unknown"}
    assert pipeline.decide(dict(det, votes=[(no, ts)]), None) == []
    # Qwen не ответил — нарушение остаётся, unverified
    e = pipeline.decide(dict(det, votes=[(None, ts)]), None)
    assert e and e[0]["verification"] == "unverified" and e[0]["object_type"] == "unknown"
    # правило, требующее Qwen: подтверждено
    vlm = {"t": 1.5, "s": 0.6, "type_id": "PHONE_AT_CASH_DESK", "src": "rule", "requires_vlm": True, "obs": "x"}
    yes = {"violation": True, "confidence": 0.8, "type_id": "PHONE_AT_CASH_DESK", "frame_index": 2, "object_type": "phone"}
    e = pipeline.decide(dict(vlm, votes=[(yes, ts)]), None)
    assert e[0]["verification"] == "confirmed" and e[0]["object_type"] == "phone" and e[0]["t"] == 2.0
    # правило, требующее Qwen: Qwen сказал «нет» — нарушения нет
    assert pipeline.decide(dict(vlm, votes=[(no, ts)]), None) == []
    # Qwen недоступен — правило с достаточной уверенностью даёт unverified
    e = pipeline.decide(dict(vlm, votes=[(None, ts)]), None)
    assert e and e[0]["verification"] == "unverified"
