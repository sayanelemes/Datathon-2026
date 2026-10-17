"""Legal-RAG: разбиение на пункты, поиск top-3, цитаты только из текста, null без уверенности."""
import os

import pytest

import legal_rag
import settings
from legal_rag import index as lindex
from legal_rag.loader import load_chunks, split_articles

DOC = """Правила тестовые
Глава 1. Общие положения
1. Настоящие правила определяют порядок кассовых операций.
2. Наличные деньги принимаются и пересчитываются кассовым работником в присутствии клиента.
3. Кассовому работнику не допускается покидать рабочее место до окончания обслуживания клиента.
Глава 2. Хранение
4. Вход в кладовую разрешается лицам, ответственным за сохранность наличных денег.
5. Кассовый работник:
1) подготавливает сумму;
2) пересчитывает подготовленные к выдаче наличные деньги в присутствии клиента;
3) выдает документ.
"""


@pytest.fixture()
def base(tmp_path, monkeypatch):
    d = tmp_path / "legal"
    d.mkdir()
    (d / "doc.txt").write_text(DOC, encoding="utf-8")
    (d / "sources.yaml").write_text('doc.txt:\n  document: "Тестовые правила"\n  url: "https://example.org/doc"\n  source: "test"\n',
                                    encoding="utf-8")
    monkeypatch.setattr(settings, "LEGAL_BASE_DIR", str(d))
    monkeypatch.setattr(settings, "LEGAL_INDEX_DIR", str(tmp_path / "idx"))
    legal_rag.reset()
    yield d
    legal_rag.reset()


def test_split_articles():
    arts = split_articles(DOC)
    labels = [a[0] for a in arts]
    assert labels[:1] == ["преамбула"] and "п. 3" in labels and "п. 5" in labels
    p5 = next(a for a in arts if a[0] == "п. 5")
    assert "2) пересчитывает" in p5[3] and p5[2].startswith("Глава 2")


def test_index_top3_and_cache(base):
    idx = legal_rag.get_index()
    assert len(idx.chunks) == 6
    top = idx.search("кассир ушёл с рабочего места, не закончив обслуживание клиента", 3)
    assert len(top) == 3 and top[0][0].article == "п. 3"
    legal_rag.reset()
    files = os.listdir(settings.LEGAL_INDEX_DIR)
    assert any(f.startswith("index_") for f in files)
    assert legal_rag.get_index().chunks[0].document == "Тестовые правила"      # загрузка из кэша


def test_lookup_catalog_without_qwen(base):
    rule = {"name": "Кассир покинул рабочее место", "visual": "кассир уходит", "norm": "не покидать рабочее место", "point": "п. 3"}
    ref, conf = legal_rag.lookup("LEFT", rule, "кассир ушёл")
    assert conf == "high" and ref["article"] == "п. 3"
    assert ref["url"] == "https://example.org/doc" and ref["source"] == "test"
    assert ref["quote"] in DOC


def test_subpoint_quote(base):
    rule = {"name": "Выдача без пересчёта", "visual": "", "norm": "пересчёт при клиенте", "point": "п. 5, пп. 2"}
    ref, _ = legal_rag.lookup("PAYOUT", rule, "")
    assert ref["quote"].startswith("2) пересчитывает")


def test_qwen_cannot_invent_quote_or_article(base):
    rule = {"name": "Нечто новое", "visual": "", "norm": "", "point": ""}
    ref, conf = legal_rag.lookup("NEW", rule, "посторонний вошёл в кладовую",
                                 chat=lambda p: {"choice": 1, "confidence": "high", "quote": "статья 999 выдуманная цитата"})
    assert ref is not None and "999" not in ref["quote"] and ref["quote"].strip() in DOC.replace("\n", " ")
    legal_rag.reset()
    ref, conf = legal_rag.lookup("NEW2", rule, "x", chat=lambda p: {"choice": 7, "confidence": "high"})   # номер вне списка
    assert ref is None and conf == "low"


def test_no_confidence_gives_null(base):
    rule = {"name": "Нечто", "visual": "", "norm": "", "point": ""}
    assert legal_rag.lookup("X", rule, "погода", chat=lambda p: {"choice": None, "confidence": "low"}) == (None, "low")
    assert legal_rag.lookup("Y", rule, "погода", chat=None) == (None, "low")


def test_empty_base(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "LEGAL_BASE_DIR", str(tmp_path / "none"))
    monkeypatch.setattr(settings, "LEGAL_INDEX_DIR", str(tmp_path / "idx"))
    legal_rag.reset()
    assert legal_rag.lookup("A", {"point": "п. 1"}, "") == (None, "low")
    legal_rag.reset()


def test_real_base_maps_every_catalog_type():
    """Поставляемая база НБРК № 231: для каждого типа из каталога находится пункт каталога."""
    import yaml
    with open(settings.VIOLATIONS_PATH, encoding="utf-8") as f:
        cat = yaml.safe_load(f)["rules"]
    real = os.path.join(settings.BASE_DIR, "legal_base")
    idx = lindex.build_or_load(real, None)
    for r in cat:
        art = legal_rag.catalog_article(r["point"])
        found = legal_rag.candidates(idx, f'{r["name"]}. {r["visual"]}. {r["norm"]}', art)
        assert any(c.article_num == art for c, _ in found), r["id"]
