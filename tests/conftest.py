"""Общие настройки тестов: всё во временных папках, модель не нужна (VLM_MOCK=1)."""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_TMP = tempfile.mkdtemp(prefix="cg-tests-")
for k, sub in (("DATA_DIR", "jobs"), ("UPLOAD_DIR", "uploads"), ("RESULTS_DIR", "results"),
               ("LEGAL_INDEX_DIR", "legal_index"), ("MODELS_DIR", "models")):
    os.environ.setdefault(k, os.path.join(_TMP, sub))
os.environ.setdefault("VLM_MOCK", "1")
os.environ.setdefault("API_KEY", "test-key-0123456789")
os.environ.setdefault("VALIDATE_VIDEO", "0")       # в тестах «видео» — байты-заглушки
os.environ.setdefault("USE_POSE", "0")
os.environ.setdefault("VIS_ENABLED", "0")
os.environ.pop("REDIS_URL", None)
