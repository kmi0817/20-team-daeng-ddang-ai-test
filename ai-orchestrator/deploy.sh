#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/app/ai"
COMPOSE_FILE="${APP_DIR}/docker-compose.yml"
ENV_FILE="${APP_DIR}/.env"

# GitHub Actions에서 넘겨주는 값
IMAGE="${IMAGE:-}"
RELEASE_ID="${RELEASE_ID:-}"
ENV_FILE_B64="${ENV_FILE_B64:-}"

if [ -z "${IMAGE}" ] || [ -z "${RELEASE_ID}" ] || [ -z "${ENV_FILE_B64}" ]; then
  echo "❌ IMAGE, RELEASE_ID, ENV_FILE_B64은 필수 값입니다."
  exit 1
fi

echo "🌐 AI Orchestrator deploy (docker compose)"
echo "🏷️  RELEASE_ID=${RELEASE_ID}"

cd "${APP_DIR}"
test -f "${COMPOSE_FILE}"

DOCKER_IMAGE="${IMAGE}:${RELEASE_ID}"

# rollback용 현재 이미지 기록
CURRENT_IMAGE=""
if docker inspect ai-orchestrator >/dev/null 2>&1; then
  CURRENT_IMAGE="$(docker inspect -f '{{.Config.Image}}' ai-orchestrator 2>/dev/null || true)"
fi
echo "🧩 CURRENT_IMAGE=${CURRENT_IMAGE:-none}"

# .env 생성 및 교체
umask 077
echo "${ENV_FILE_B64}" | base64 -d > "${ENV_FILE}"
echo "DOCKER_IMAGE=${FULL_IMAGE}" >> "${ENV_FILE}"
echo "🔐 ${ENV_FILE} 작성 (mode 600)"

# Docker Hub 로그인
if [ -n "${DOCKERHUB_USERNAME:-}" ] && [ -n "${DOCKERHUB_TOKEN:-}" ]; then
  echo "🔑 docker login to Docker Hub..."
  echo "${DOCKERHUB_TOKEN}" | docker login -u "${DOCKERHUB_USERNAME}" --password-stdin
else
  echo "ℹ️ DOCKERHUB_USERNAME/TOKEN not provided. assume already logged in."
fi

# 배포
echo "📦 docker compose pull..."
docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" pull

echo "🚀 docker compose up -d..."
docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" up -d

echo "📋 docker compose ps"
docker compose -f "${COMPOSE_FILE}" ps

# 헬스 체크
echo "🏥 헬스 체크 시작..."

ok=0
for i in {1..10}; do
  sleep 5
  if wget -qO- http://localhost:8000/health >/dev/null 2>&1; then
    ok=1
    echo "✅ 헬스 체크 성공!"
    break
  fi
  echo "⏳ 시작 대기 중.... ($i/10)"
done

if [ "$ok" -ne 1 ]; then
  echo "🚨 헬스 체크 실패. 배포를 중단합니다."

  if [ -n "${CURRENT_IMAGE}" ]; then
    echo "🔙 직전 버전(${CURRENT_IMAGE})으로 롤백합니다."

    sed -i.bak "s|^IMAGE=.*$|IMAGE=${CURRENT_IMAGE}|g" "${ENV_FILE}"
    docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" pull || true
    docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" up -d
    echo "⚠️ 롤백 완료. 제대로 롤백됐는지 확인해주세요."
  else
    echo "❌ 롤백할 직전 버전을 찾을 수 없습니다."
  fi
  exit 1
fi

echo "🧹 prune old images"
docker image prune -f >/dev/null 2>&1 || true

echo "🎉 deploy success"