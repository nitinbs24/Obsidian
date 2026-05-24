# Obsidian — Search Database & Cache Layer

A hybrid search engine that indexes uploaded files into OpenSearch and caches
results in Redis — built for the HPE Campus Connect Program.

---

## What This Role Does

1. Files uploaded to MinIO are ingested by the ingestion worker, which calls
   **my OpenSearch client** to upsert text chunks and vector embeddings into the search index
2. When a user searches, the Go API Gateway checks **my Redis cache** first —
   if it's a hit, results are returned in under 1 ms without touching OpenSearch
3. On a cache miss, **my hybrid search query** runs BM25 + vector search in parallel
   on OpenSearch, returns ranked results, and stores them in Redis for next time

---

## Tech Stack

| Tool | What it does | Why we chose it |
|------|-------------|-----------------|
| OpenSearch | Stores file chunks, metadata, and 384-dim vector embeddings | Free, open source, supports hybrid BM25 + kNN search |
| OpenSearch Neural Search Plugin | Combines BM25 and vector scores into one ranked result | Built into OpenSearch, no extra service needed |
| OpenSearch Dashboards | Web UI to inspect the search index | Free, built into OpenSearch |
| Redis | In-memory cache for search results with TTL | Sub-millisecond reads, LRU eviction, zero persistence overhead |
| Python (opensearch_client) | Upserts chunks and embeddings into OpenSearch | Used by Nithin & Praneeth's ingestion worker |
| Python (redis_cache) | Stores and retrieves cached search results | Plugs into the Go API Gateway cache check |

---

## Role Structure

```
Obsidian/
├── infrastructure/
│   ├── opensearch/
│   │   ├── docker-compose.yml       # Runs OpenSearch + Dashboards
│   │   ├── index-mapping.json       # Schema: vector field, metadata, BM25 fields
│   │   └── hybrid-plugin-setup.sh  # One-shot: creates index + hybrid pipeline
│   └── redis/
│       ├── docker-compose.yml       # Runs Redis container
│       └── redis.conf               # 256MB cap, LRU eviction, no persistence
├── workers/
│   └── ingestion/
│       └── opensearch_client.py     # Called by ingestion worker to upsert chunks
├── backend/
│   ├── search/
│   │   └── opensearch_query_builder.py  # Hybrid BM25+kNN query (source of truth for Go)
│   └── cache/
│       └── redis_cache.py           # Step 4: stores search results after OpenSearch returns
├── tests/
│   └── test_opensearch.py           # Unit + integration tests for all components
├── requirements.txt
└── .env.example
```

---

## Prerequisites

```bash
# 1. Docker Desktop
# Download: https://www.docker.com/products/docker-desktop/
docker --version

# 2. Python 3.10+
# Download: https://www.python.org/downloads/
# On Windows installer: tick "Add python.exe to PATH"
python --version

# 3. On Windows — run ONCE in Administrator PowerShell (or OpenSearch crashes)
wsl -d docker-desktop sysctl -w vm.max_map_count=262144

# To make it permanent, add to %USERPROFILE%\.wslconfig:
# [wsl2]
# kernelCommandLine = sysctl.vm.max_map_count=262144
```

---

## Setup (First Time Only)

### 1. Clone the repo

```bash
git clone https://github.com/your-org/Obsidian
cd Obsidian
```

### 2. Create your .env file

```bash
cp .env.example .env
# Defaults work for local. Update HOST values if running on a server.
```

### 3. Install Python dependencies

```bash
# Linux/Mac:
python3 -m venv .venv
source .venv/bin/activate

# Windows:
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

### 4. Start Redis

```bash
cd infrastructure/redis
docker compose up -d
```

### 5. Start OpenSearch

```bash
cd infrastructure/opensearch
docker compose up -d
# Takes 30-60 seconds — wait until healthy
docker logs -f opensearch-node1
# Ready when you see: Cluster health status changed from [RED] to [GREEN]
```

### 6. Create index and hybrid pipeline (run once only)

```bash
cd infrastructure/opensearch
chmod +x hybrid-plugin-setup.sh       # Linux/Mac only
./hybrid-plugin-setup.sh

# You should see:
# [+] Cluster is green/yellow
# [+] Creating hybrid search pipeline: obsidian-hybrid-pipeline
# [+] Creating index 'obsidian-files'
# [+] ✅ OpenSearch hybrid search setup complete
```

---

## Verify Everything is Running

```bash
# OpenSearch cluster health
curl http://localhost:9200/_cluster/health

# Index exists with correct mapping
curl http://localhost:9200/obsidian-files/_mapping

# Hybrid pipeline attached
curl http://localhost:9200/_search/pipeline/obsidian-hybrid-pipeline

# Redis is alive
docker exec obsidian-redis redis-cli ping
# Should reply: PONG
```

---

## Accessing the Services

| Service | URL |
|---------|-----|
| OpenSearch API | `http://localhost:9200` |
| OpenSearch Dashboards | `http://localhost:5601` |
| Redis | `localhost:6379` |

---

## Running Tests

```bash
# Unit tests — no containers needed, runs instantly
pytest tests/test_opensearch.py -v -m unit

# Integration tests — needs OpenSearch + Redis running
pytest tests/test_opensearch.py -v -m integration

# All tests
pytest tests/test_opensearch.py -v
```

---

## How Other Roles Use My Code

### Nithin & Praneeth (Ingestion Worker)

Import and call the OpenSearch client to upsert chunks after embedding:

```python
from workers.ingestion.opensearch_client import get_client, ChunkDocument

client = get_client()
client.bulk_upsert([
    ChunkDocument(
        object_key  = "bucket/reports/fire_safety.pdf",
        bucket      = "hpe-objects",
        filename    = "fire_safety.pdf",
        extension   = "pdf",
        mime_type   = "application/pdf",
        download_url= "http://minio:9000/bucket/reports/fire_safety.pdf",
        owner       = "alice",
        size_bytes  = 204800,
        uploaded_at = "2024-11-01T10:00:00Z",
        chunk_index = 0,
        chunk_total = 3,
        chunk_text  = "Fire safety protocols require regular drills.",
        embedding   = [0.1, 0.2, ...],   # 384 floats from all-MiniLM-L6-v2
        tags        = ["safety", "compliance"],
    )
])
```

### Prarthana (Go API Gateway)

The hybrid query JSON structure is defined in `opensearch_query_builder.py` —
replicate it exactly in `backend/search/opensearch.go`.

The Redis cache key format she must match in `backend/cache/redis.go`:

```
obsidian:search:result:<SHA-256 of canonical JSON>
```

Canonical JSON example:

```json
{"bucket":"","date_from":"","date_to":"","extension":"pdf","from":0,"owner":"","q":"fire safety","size":10,"tags":[]}
```

SHA-256 that, take the first 32 hex characters, prepend `obsidian:search:result:`.

---

## Environment Variables

Copy `.env.example` to `.env`. Never push `.env` to GitHub.

```
# OpenSearch
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_INDEX=obsidian-files

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# TTL (seconds)
REDIS_TTL_DEFAULT=300
REDIS_TTL_FILTERED=600
REDIS_TTL_POPULAR=1800
REDIS_POPULAR_THRESHOLD=10

# Search tuning
SEARCH_KNN_K=50
SEARCH_BM25_BOOST=0.4
SEARCH_KNN_BOOST=0.6
SEARCH_DEFAULT_SIZE=10
```

---

## Redis TTL Strategy

| Query type | TTL | Reason |
|------------|-----|--------|
| General query | 5 min | Results may change as files are uploaded |
| Filtered query (extension / bucket / owner) | 10 min | More specific → more stable |
| Popular query (10+ hits) | 30 min | Auto-promoted to save the most compute |

---

## Troubleshooting

**OpenSearch not starting?**
```bash
docker logs opensearch-node1 --tail 30
# Most common cause on Windows: vm.max_map_count too low
# Fix: wsl -d docker-desktop sysctl -w vm.max_map_count=262144
```

**Index already exists on setup script?**
```bash
# Safe to ignore — your index is already there
# To recreate from scratch:
curl -X DELETE http://localhost:9200/obsidian-files
./hybrid-plugin-setup.sh
```

**Redis connection refused?**
```bash
docker ps | grep redis
# If not running:
cd infrastructure/redis && docker compose up -d
```

**Python module not found?**
```bash
# Make sure virtualenv is activated
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Linux/Mac
pip install -r requirements.txt
```

**Embeddings dimension mismatch on upsert?**
```bash
# Embedding must be exactly 384 floats (all-MiniLM-L6-v2 output)
# Check model_spec.json in workers/model-server/
```

---

## Stopping Services

```bash
cd infrastructure/opensearch && docker compose down
cd infrastructure/redis      && docker compose down
```
