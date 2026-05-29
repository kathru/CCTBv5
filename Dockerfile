# CCTBv5 — Production Dockerfile
# Multi-stage build: keeps final image lean

# ── Stage 1: builder ──────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into /install
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Version injected at build time via --build-arg (computed from git by docker-compose)
ARG GIT_MINOR=0
ARG GIT_PATCH=0

WORKDIR /app

# Install pg_isready (libpq-client) para o entrypoint wait-for-postgres.
# Garante que a app nunca sobe antes do postgres aceitar conexões,
# independentemente de como o container foi iniciado (docker compose up / docker start).
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Bake version into static files readable at runtime
RUN echo -n "$GIT_MINOR" > /app/_git_minor && echo -n "$GIT_PATCH" > /app/_git_patch

# Copy application code
COPY src/                  ./src/
COPY infra/                ./infra/
COPY main.py               .
COPY _version.txt          .
COPY docker-entrypoint.sh  .

# Non-root user for security
RUN useradd -m -u 1000 cctb && \
    mkdir -p /app/logs && \
    chmod +x /app/docker-entrypoint.sh && \
    chown -R cctb:cctb /app
USER cctb

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health', timeout=8)" || exit 1

# Entrypoint aguarda postgres+redis antes de iniciar a aplicação.
# Funciona com docker compose up E com docker start direto.
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "main.py"]
