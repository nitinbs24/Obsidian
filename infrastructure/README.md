# Role 1 — Infrastructure & Messaging Engineer

## Overview
This folder contains all configuration and setup scripts for the MinIO object storage and Apache Kafka messaging pipeline.

## What This Sets Up
- **MinIO** — S3-compatible object storage running on Kubernetes (minikube)
- **Apache Kafka** — 3-broker message queue running via Strimzi operator
- **Event Pipeline** — Every file uploaded to MinIO automatically fires an `s3:ObjectCreated` event to the `file-upload-events` Kafka topic

## Prerequisites
- Windows with WSL2 (Ubuntu 24.04)
- Docker installed inside WSL2
- minikube, kubectl, helm, aws cli, mc installed

## Quick Start

### 1. Start minikube
```bash
minikube start --driver=docker
```

### 2. Deploy Kafka (Strimzi)
```bash
kubectl create namespace kafka
kubectl create -f 'https://strimzi.io/install/latest?namespace=kafka' -n kafka
kubectl apply -f kafka/kafka-nodepool.yaml -n kafka
kubectl apply -f kafka/kafka-cluster.yaml -n kafka
kubectl apply -f kafka/kafka-topic.yaml -n kafka
```

### 3. Deploy MinIO
```bash
helm repo add minio https://charts.min.io
helm repo update
helm install minio minio/minio \
  --set rootUser=minioadmin,rootPassword=minioadmin123 \
  --set mode=standalone \
  --set persistence.size=1Gi \
  --set resources.requests.memory=512Mi
```

### 4. Configure MinIO → Kafka notifications
```bash
bash minio/setup.sh
```

### 5. Every time you restart WSL
```bash
bash startup.sh
```

## Folder Structure
infrastructure/
minio/
config.env          # MinIO environment variables
setup.sh            # Bucket creation + event notification config
kafka/
kafka-nodepool.yaml # 3-broker KafkaNodePool (KRaft mode)
kafka-cluster.yaml  # Kafka cluster definition
kafka-topic.yaml    # file-upload-events topic (3 partitions, 3 replicas)
topics.sh           # Topic creation script
dead-letter/
dlq-consumer.py   # Dead letter queue consumer
startup.sh            # Startup script after WSL restart
README.md             # This file
## Kafka Details for Other Roles

| Property | Value |
|---|---|
| Bootstrap Server | `my-cluster-kafka-bootstrap.kafka.svc.cluster.local:9092` |
| Topic | `file-upload-events` |
| DLQ Topic | `file-upload-events-dlq` |
| Partitions | 3 |
| Replication Factor | 3 |

## MinIO Details

| Property | Value |
|---|---|
| S3 API Endpoint | `http://localhost:9000` |
| Console URL | `http://localhost:9001` |
| Bucket | `uploads` |
| Access Key | `minioadmin` |
| Secret Key | `minioadmin123` |

## Test the Pipeline
```bash
# Upload a file
aws s3 cp test.txt s3://uploads/ --endpoint-url http://localhost:9000

# Check Kafka for the event
kubectl exec -it my-cluster-dual-role-0 -n kafka -- \
  bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic file-upload-events \
  --from-beginning \
  --timeout-ms 5000
```
