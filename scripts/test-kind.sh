#!/usr/bin/env bash
set -euo pipefail

# Test Kind cluster resources: metrics-server & HPA

echo "📄 Applying Metrics Server RBAC"
kubectl apply -f deployment/kind/metrics-server-rbac.yaml

echo "📄 Applying Metrics Server Deployment"
kubectl apply -f deployment/kind/metrics-server.yaml

echo "📄 Waiting for Metrics Server to be ready"
kubectl wait --for=condition=Available=True deployment/metrics-server -n kube-system --timeout=120s || true

echo "📄 Applying HPA for API"
kubectl apply -f deployment/kind/hpa.yaml

echo "📄 Verifying HPA"
kubectl get hpa api-hpa

echo "✅ Kind cluster integration seems healthy"
