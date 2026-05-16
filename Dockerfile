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

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Bake version into static files readable at runtime
RUN echo -n "$GIT_MINOR" > /app/_git_minor && echo -n "$GIT_PATCH" > /app/_git_patch

# Copy application code
COPY src/     ./src/
COPY infra/   ./infra/
COPY main.py  .

# Non-root user for security
RUN useradd -m -u 1000 cctb && \
    mkdir -p /app/logs && \
    chown -R cctb:cctb /app
USER cctb

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

CMD ["python", "main.py"]
