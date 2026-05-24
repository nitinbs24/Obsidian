#!/usr/bin/env bash
# =============================================================================
# hybrid-plugin-setup.sh
# Configures OpenSearch Neural Search plugin for hybrid BM25 + kNN retrieval.
# Owner: Search Database Administrator (Priyadarshini Sarja)
#
# Usage:
#   chmod +x hybrid-plugin-setup.sh
#   ./hybrid-plugin-setup.sh [OPENSEARCH_HOST] [INDEX_NAME]
#
# Defaults:
#   OPENSEARCH_HOST = http://localhost:9200
#   INDEX_NAME      = obsidian-files
# =============================================================================

set -euo pipefail

OS_HOST="${1:-http://localhost:9200}"
INDEX_NAME="${2:-obsidian-files}"
MAPPING_FILE="$(dirname "$0")/index-mapping.json"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[~]${NC} $*"; }
die()  { echo -e "${RED}[!]${NC} $*" >&2; exit 1; }

# ── 0. Wait for OpenSearch to be ready ────────────────────────────────────────
log "Waiting for OpenSearch at ${OS_HOST} ..."
for i in $(seq 1 30); do
  STATUS=$(curl -sf "${OS_HOST}/_cluster/health" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "unreachable")
  if [[ "$STATUS" == "green" || "$STATUS" == "yellow" ]]; then
    log "Cluster is ${STATUS}"; break
  fi
  warn "Attempt ${i}/30 — status: ${STATUS}. Retrying in 5 s ..."
  sleep 5
  [[ $i -eq 30 ]] && die "OpenSearch did not become ready in time."
done

# ── 1. Enable ML Commons (needed for neural search) ───────────────────────────
log "Configuring ML Commons cluster settings ..."
curl -sf -X PUT "${OS_HOST}/_cluster/settings" \
  -H "Content-Type: application/json" -d '{
    "persistent": {
      "plugins.ml_commons.only_run_on_ml_node":               false,
      "plugins.ml_commons.allow_registering_model_via_url":   true,
      "plugins.ml_commons.memory_feature_enabled":            true,
      "plugins.ml_commons.max_ml_task_per_node":              10
    }
  }' | python3 -m json.tool
echo ""

# ── 2. Configure Neural Search hybrid pipeline ────────────────────────────────
log "Creating hybrid search pipeline: obsidian-hybrid-pipeline ..."
curl -sf -X PUT "${OS_HOST}/_search/pipeline/obsidian-hybrid-pipeline" \
  -H "Content-Type: application/json" -d '{
    "description": "Obsidian hybrid BM25 + kNN normalization pipeline",
    "phase_results_processors": [
      {
        "normalization-processor": {
          "normalization": { "technique": "min_max" },
          "combination": {
            "technique": "arithmetic_mean",
            "parameters": { "weights": [0.4, 0.6] }
          }
        }
      }
    ]
  }' | python3 -m json.tool
echo ""

# ── 3. Create index (skip if already exists) ──────────────────────────────────
EXISTS=$(curl -sf -o /dev/null -w "%{http_code}" "${OS_HOST}/${INDEX_NAME}")

if [[ "$EXISTS" == "200" ]]; then
  warn "Index '${INDEX_NAME}' already exists — skipping creation."
  warn "To recreate: DELETE /${INDEX_NAME} first, then re-run this script."
else
  log "Creating index '${INDEX_NAME}' with mapping from ${MAPPING_FILE} ..."
  [[ -f "$MAPPING_FILE" ]] || die "Mapping file not found: ${MAPPING_FILE}"

  curl -sf -X PUT "${OS_HOST}/${INDEX_NAME}" \
    -H "Content-Type: application/json" \
    --data-binary "@${MAPPING_FILE}" | python3 -m json.tool
  echo ""
  log "Index '${INDEX_NAME}' created successfully."
fi

# ── 4. Attach pipeline as default for the index ───────────────────────────────
log "Attaching hybrid pipeline as default search pipeline on index ..."
curl -sf -X PUT "${OS_HOST}/${INDEX_NAME}/_settings" \
  -H "Content-Type: application/json" -d '{
    "index.search.default_pipeline": "obsidian-hybrid-pipeline"
  }' | python3 -m json.tool
echo ""

# ── 5. Verify ─────────────────────────────────────────────────────────────────
log "Verifying index mapping ..."
curl -sf "${OS_HOST}/${INDEX_NAME}/_mapping" | python3 -m json.tool | grep -E '"type"|"dimension"' | head -20
echo ""

log "Verifying pipeline ..."
curl -sf "${OS_HOST}/_search/pipeline/obsidian-hybrid-pipeline" | python3 -m json.tool
echo ""

log "✅  OpenSearch hybrid search setup complete."
log "    Index    : ${OS_HOST}/${INDEX_NAME}"
log "    Pipeline : obsidian-hybrid-pipeline"
log "    Dashboards: http://localhost:5601"
