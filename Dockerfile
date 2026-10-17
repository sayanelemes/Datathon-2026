# Образ сервиса (api и worker): FastAPI + Следопыт (YOLOX/RTMPose на onnxruntime-gpu) + Legal-RAG.
# Qwen3-VL работает в отдельном контейнере vllm (см. docker-compose.yml).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-pip ffmpeg curl libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# rtmlib без зависимостей: иначе он поставит opencv-python и CPU-onnxruntime поверх нужных версий
RUN grep -v '^rtmlib' requirements.txt > /tmp/req.txt \
 && pip3 install -r /tmp/req.txt \
 && pip3 install --no-deps rtmlib

COPY . .
RUN mkdir -p /data/jobs /data/uploads /data/results /models

ENV DATA_DIR=/data/jobs UPLOAD_DIR=/data/uploads RESULTS_DIR=/data/results MODELS_DIR=/models \
    LEGAL_INDEX_DIR=/data/jobs/_legal_index
EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "75"]
