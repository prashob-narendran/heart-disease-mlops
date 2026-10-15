#!/usr/bin/env bash
# Deploy the heart-disease API to the current Kubernetes context and verify it.
#
#   ./scripts/k8s_deploy.sh                 # deploy :latest from GHCR
#   ./scripts/k8s_deploy.sh sha-c6c4a6c     # deploy a specific image tag
#   ./scripts/k8s_deploy.sh local           # build heart-api:local and deploy it
#   URL=http://localhost:8080 ./scripts/k8s_deploy.sh
#
# Steps: [build] -> apply manifests (kustomize) -> optionally pin the tag ->
# wait for the rollout -> show pods/service/HPA -> smoke-test the service.
set -euo pipefail

NS=heart-disease
TAG="${1:-}"
URL="${URL:-http://localhost:8080}"
IMAGE=ghcr.io/prashob-narendran/heart-disease-api
cd "$(dirname "$0")/.."

echo "== Context: $(kubectl config current-context)"
kubectl get nodes

if [ "$TAG" = "local" ]; then
  echo "== Building heart-api:local"
  docker build -t heart-api:local .
  echo "== Applying manifests (kubectl apply -k k8s/local/)"
  kubectl apply -k k8s/local/
  # same tag as before -> force new pods so they run the fresh build
  kubectl -n "$NS" rollout restart deployment/heart-api
else
  echo "== Applying manifests (kubectl apply -k k8s/)"
  kubectl apply -k k8s/
fi

if [ -n "$TAG" ] && [ "$TAG" != "local" ]; then
  echo "== Pinning image to $IMAGE:$TAG"
  kubectl -n "$NS" set image deployment/heart-api api="$IMAGE:$TAG"
fi

echo "== Waiting for rollout"
kubectl -n "$NS" rollout status deployment/heart-api --timeout=180s

echo "== Resources"
kubectl -n "$NS" get deploy,pods,svc,hpa,pdb -o wide

echo "== Smoke test against $URL"
python scripts/smoke_test_api.py --url "$URL" --wait 60
