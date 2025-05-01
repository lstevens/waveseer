#!/usr/bin/env bash
set -euo pipefail

# Helm chart integration tests
echo "📄 Running Helm Lint"
helm lint charts/waveseer

echo "📄 Rendering templates"
helm template charts/waveseer > /dev/null

echo "✅ Helm chart lint and template succeeded"
