# Obsidian — Object Storage Search Engine

Each top-level directory maps to an ownership role; files are placed where they logically belong and may evolve as the project matures.

---

## Repository Structure

```
Obsidian/
├── infrastructure/
├── workers/
├── backend/
├── frontend/
├── shared/
└── docs/
```

---

## `infrastructure/` — Infrastructure & Messaging

Everything needed to stand up the backing services locally (MinIO, Kafka, Redis, OpenSearch). A root-level `docker-compose.yml` orchestrates the full stack; each sub-directory contains service-specific configuration.

| Path | Description |
|------|-------------|
| `minio/config.env` | Environment variables for the MinIO instance (access keys, region, port). |
| `minio/setup.sh` | Bucket creation and event-notification configuration so MinIO publishes upload events to Kafka. |
| `kafka/docker-compose.yml` | Compose file for the Kafka broker and Zookeeper. |
| `kafka/topics.sh` | Script that creates the required topics and configures partition counts. |
| `kafka/dead-letter/dlq-consumer.py` | A lightweight consumer that drains the dead-letter queue and logs failed messages for inspection. |
| `redis/redis.conf` | Redis configuration (max memory, eviction policy, persistence settings). |
| `opensearch/docker-compose.yml` | Compose file for the OpenSearch cluster. |
| `opensearch/index-mapping.json` | Index mapping that defines the vector field, keyword fields, and metadata schema. |
| `opensearch/hybrid-plugin-setup.sh` | Script that enables and configures the Neural Search plugin for hybrid retrieval. |
| `docker-compose.yml` | **Root compose file** — spins up the entire backing stack (MinIO + Kafka + Redis + OpenSearch) in one command. |

---

## `workers/` — Python Workers

Three independently deployable Python services that handle embedding, ingestion, and search.

### `workers/model-server/` — Embedding Model Server

A single, centralised FastAPI service that exposes embedding endpoints. Both the ingestion worker and the search worker call into this server, so embedding models are loaded exactly once.

| Path | Description |
|------|-------------|
| `main.py` | Application entrypoint; starts the FastAPI server. |
| `text_embedder.py` | Wraps SentenceTransformers (`all-MiniLM-L6-v2`) for text embedding. |
| `image_embedder.py` | Wraps CLIP / `nomic-embed-vision` for image embedding. |
| `server.py` | FastAPI app defining `/embed/text` and `/embed/image` routes. |
| `model_spec.json` | Single source of truth for model names, versions, and dimensions. |
| `requirements.txt` | Python dependencies (includes ML libraries). |
| `load-balancer/nginx.conf` | Nginx configuration for routing across model-server replicas at scale. |

### `workers/ingestion/` — Ingestion Worker

Consumes Kafka events triggered by file uploads, extracts text, chunks it, obtains embeddings from the model server, and upserts everything into OpenSearch.

| Path | Description |
|------|-------------|
| `main.py` | Kafka consumer entrypoint; listens for new-object events. |
| `tika_extractor.py` | Uses Apache Tika to extract text from PDF, DOCX, PPTX, and TXT files. |
| `image_handler.py` | Preprocessing pipeline for JPEG/PNG files before they are sent for embedding. |
| `chunker.py` | LangChain-based text chunking with configurable overlap. |
| `model_client.py` | HTTP client that calls the model server's `/embed/*` endpoints. |
| `opensearch_client.py` | Upserts vectors and metadata into OpenSearch. |
| `requirements.txt` | Python dependencies — no ML/embedding libs; the model server handles those. |

### `workers/search/` — Search Worker

A gRPC service that receives a raw query string, parses intent, vectorises the query via the model server, and returns it ready for hybrid retrieval.

| Path | Description |
|------|-------------|
| `main.py` | gRPC server entrypoint. |
| `nlp_parser.py` | Parses intent, filters, and keywords from a raw query string. |
| `model_client.py` | HTTP client calling the model server (same interface as the ingestion worker). |
| `grpc/search.proto` | Protobuf definition — symlinked from `shared/proto/`. |
| `grpc/search_pb2.py` | Auto-generated Python bindings from the proto file. |
| `grpc/search_pb2_grpc.py` | Auto-generated gRPC service stubs. |
| `requirements.txt` | Python dependencies — no ML/embedding libs. |

---

## `backend/` — Go API Server

The central orchestration layer. Accepts HTTP requests from the frontend, checks the Redis cache, fans out to OpenSearch and the search worker, and assembles the response.

| Path | Description |
|------|-------------|
| `main.go` | Application entrypoint. |
| `go.mod` | Go module definition and dependency list. |
| `api/routes.go` | HTTP route definitions (`/search`, `/health`, etc.). |
| `cache/redis.go` | Query normalisation, hashing, and TTL-based caching logic. |
| `search/opensearch.go` | Builds and executes hybrid BM25 + vector queries via the Neural Search plugin. |
| `grpc/client.go` | gRPC client that calls the search worker for query vectorisation. |
| `config/config.go` | Centralised configuration loading (env vars, defaults). |

---

## `frontend/` — Next.js UI

A Next.js application providing the search interface. Communicates exclusively with the Go backend.

| Path | Description |
|------|-------------|
| `app/page.tsx` | Search landing page. |
| `app/results/page.tsx` | Results page displaying ranked matches. |
| `app/layout.tsx` | Root layout (shared header, metadata, providers). |
| `components/SearchBar.tsx` | Debounced search input that calls the Go API. |
| `components/ResultCard.tsx` | Displays a single result — filename, text snippet, download link. |
| `components/FilterSidebar.tsx` | Faceted filters that map to OpenSearch filter parameters. |
| `components/UploadStatus.tsx` | Polls the Go API to show real-time ingestion progress. |
| `lib/api.ts` | Centralised HTTP client for all calls to the Go backend. |
| `package.json` | Node dependencies and scripts. |
| `next.config.ts` | Next.js configuration. |

---

## `shared/` — Cross-Team Contracts

Artefacts that are referenced by multiple services. Changes here require consensus across the team.

| Path | Description |
|------|-------------|
| `proto/search.proto` | The single source of truth for the gRPC contract between the Go backend and the search worker. |
| `model_spec.json` | Symlink to `workers/model-server/model_spec.json` — keeps model metadata in sync across services. |

---

## `docs/` — Documentation

| Path | Description |
|------|-------------|
| `architecture.md` | High-level architecture diagram and data-flow explanation. |
| `setup.md` | One-command local stack setup guide. |
| `roles.md` | Ownership map — which person/role owns which files and directories. |

---
# 2. Follow the setup guide for remaining steps
cat docs/setup.md
```
