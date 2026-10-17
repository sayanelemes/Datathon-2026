#!/usr/bin/env bash
# Ноутбук под Linux/WSL: тот же start.sh, профиль 4050 и туннель по умолчанию.
#   ./laptop/start.sh          — RTX 4050
#   ./laptop/start.sh 3050     — RTX 3050
# Один раз: pip install -r requirements.txt && ollama pull qwen3-vl:4b-instruct
export API_KEY="${API_KEY:-change-me-long-secret-key}" TUNNEL="${TUNNEL:-1}"
exec "$(dirname "$0")/../start.sh" "${1:-4050}"
