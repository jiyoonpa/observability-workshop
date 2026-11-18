#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# venv: 이 폴더 안에 한 번만 만들고, requirements가 바뀔 때만 재설치
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
if [ requirements.txt -nt venv/.installed ]; then
  pip install -q -r requirements.txt && touch venv/.installed
fi

# 환경변수 (setup_vectordb.py도 읽도록 쉘에 올림)
set -a; source .env; set +a

# Postgres 컨테이너, AWS 세션 확인
docker start healthcare-postgres >/dev/null 2>&1 || true
aws sts get-caller-identity >/dev/null 2>&1 || { echo "❌ aws login 이 필요합니다"; exit 1; }

# 최초 1회: 컬렉션이 없으면 적재
if ! python -c "import sys; from config import DOMAIN; from helpers.pgvector_utils import collection_exists; sys.exit(0 if collection_exists(DOMAIN, 'local') else 1)"; then
  echo "▶ 데이터 적재 실행"
  python helpers/setup_vectordb.py local
fi

exec streamlit run app.py
