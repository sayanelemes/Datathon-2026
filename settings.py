"""Пути и конфигурация. Всё читается из окружения или YAML в configs/ — в коде не хардкодим пути, URL и пороги.

Порядок: переменная окружения → YAML → значение по умолчанию.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
E = os.getenv


def _path(env: str, default: str) -> str:
    p = E(env) or default
    return p if os.path.isabs(p) else os.path.join(BASE_DIR, p)


CONFIG_DIR = _path("CONFIG_DIR", "configs")
CAMERA_CONFIG = _path("CAMERA_CONFIG", os.path.join(CONFIG_DIR, "camera_1.yaml"))
RULES_CONFIG = _path("RULES_CONFIG", os.path.join(CONFIG_DIR, "rules.yaml"))
VIOLATIONS_PATH = _path("VIOLATIONS_PATH", os.path.join(CONFIG_DIR, "violations.yaml"))

DATA_DIR = _path("DATA_DIR", "jobs")                 # статусы и результаты задач (result.json)
UPLOAD_DIR = _path("UPLOAD_DIR", "uploads")          # загруженные видео: uploads/{job_id}.mov
RESULTS_DIR = _path("RESULTS_DIR", "results")        # results/{job_id}_annotated.mp4
MODELS_DIR = _path("MODELS_DIR", "models")           # ONNX детектора объектов (YOLOX COCO)
LEGAL_BASE_DIR = _path("LEGAL_BASE_DIR", "legal_base")
LEGAL_INDEX_DIR = _path("LEGAL_INDEX_DIR", os.path.join(DATA_DIR, "_legal_index"))

LOG_LEVEL = E("LOG_LEVEL", "INFO")


def setup_logging() -> None:
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")


@lru_cache(maxsize=None)
def load_yaml(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        logging.getLogger("settings").warning("нет конфига %s — используются значения по умолчанию", path)
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def rules_config() -> dict[str, Any]:
    return load_yaml(RULES_CONFIG)


def camera_config() -> dict[str, Any]:
    return load_yaml(CAMERA_CONFIG)


def env_bool(name: str, default: bool) -> bool:
    v = E(name)
    return default if v is None or v == "" else v.strip().lower() in ("1", "true", "yes", "on")


def vlm_url() -> str:
    """VLM_URL (полный адрес chat/completions) или VLLM_URL (базовый адрес vLLM, как в мастер-промпте)."""
    url = E("VLM_URL") or E("VLLM_URL") or "http://127.0.0.1:8001/v1/chat/completions"
    url = url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url + "/v1/chat/completions"
