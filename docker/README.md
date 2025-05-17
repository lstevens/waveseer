# Waveseer Docker Deployment Guide

This guide provides instructions for deploying Waveseer using Docker and Docker Compose. The Docker configuration supports both development and production environments.

## Quick Start

```bash
# Start the API server
docker-compose up api

# Start the development environment with hot reload
docker-compose --profile dev up api-dev

# Start the annotation interface
docker-compose --profile annotation up annotate
```

## Available Services

Waveseer Docker setup includes several services:

| Service | Description | Port | URL |
|---------|-------------|------|-----|
| `api` | Production API server | 9000 | http://localhost:9000 |
| `api-dev` | Development API server with hot reload | 9000 | http://localhost:9000 |
| `chart-service` | Chart rendering service | 8010 | http://localhost:8010 |
| `ui` | Web UI for visualization | 8050 | http://localhost:8050 |
| `seer` | Pattern detection background worker | - | - |
| `ingest` | Data ingestion service | 8000 | http://localhost:8000 |
| `annotate` | Pattern annotation interface | 8501 | http://localhost:8501 |

## Docker Profiles

The docker-compose configuration uses profiles to manage different deployment scenarios:

- Default (no profile): Core services only (`api`, `chart-service`, `ui`, `seer`, `ingest`)
- `dev`: Development environment with hot reload (`api-dev`)
- `annotation`: Includes annotation interface (`annotate`)

To use profiles:

```bash
# Start services with a specific profile
docker-compose --profile dev up

# Start multiple profiles
docker-compose --profile dev --profile annotation up
```

## Volume Management

The Docker setup includes persistent volumes:

- `models`: Stores trained and exported models
- `data`: Stores datasets and annotations

Data will persist between container restarts. The volumes are mapped to the following host directories:

- `./models:/app/models`: Model storage
- `./data:/app/data`: Data storage
- `./config.yml:/app/config.yml`: Configuration file

## Development Setup

For development, use the `dev` profile which provides:

- Hot reload for code changes
- Volume mounting for live editing
- Debug-level logging

```bash
# Start development environment
docker-compose --profile dev up api-dev

# For full development environment with annotation
docker-compose --profile dev --profile annotation up
```

## Production Deployment

For production deployment:

```bash
# Build production images
docker-compose build

# Start the production stack
docker-compose up -d

# Check logs
docker-compose logs -f
```

### Production Optimization

The production Dockerfile includes:

- Multi-stage build for smaller image size
- Non-root user for improved security
- Proper signal handling with tini
- Health checks for reliability

## Environment Variables

You can customize the deployment with environment variables:

- `PYTHONPATH`: Path for Python modules
- `WAVESEER_MODELS_DIR`: Directory for model storage
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)
- `API_URL`: URL for the API service

Set environment variables in the docker-compose.yml file or use a .env file.

## Building Custom Images

```bash
# Build all images
docker-compose build

# Build specific service
docker-compose build api

# Build with no cache
docker-compose build --no-cache
```

## Troubleshooting

### API Connection Issues

If the annotation interface cannot connect to the API:

1. Check that the API service is running: `docker-compose ps`
2. Verify the API URL in the annotation service environment variables
3. Check the API logs: `docker-compose logs api`

### Volume Permission Issues

If you encounter permission issues with volumes:

```bash
# Fix permissions (run from host)
sudo chown -R 1000:1000 ./models ./data
```

### Container Health

Check container health:

```bash
docker-compose ps
```

The API and chart service include health checks to ensure they're functioning correctly.
