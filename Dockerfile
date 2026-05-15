# CCTBv5 — Production Dockerfile
# Multi-stage build: keeps final image lean

# ── Stage 1: builder ──────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies (git needed for version computation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into /install
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# Compute version from git and write to static file
COPY .git .git
RUN git tag | wc -l   | tr -d '[:space:]' > /tmp/git_minor && \
    git rev-list --count HEAD | tr -d '[:space:]' > /tmp/git_patch && \
    echo "Generated version: 5.$(cat /tmp/git_minor).$(cat /tmp/git_patch)"

# ── Stage 2: runtime ──────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy version info computed at build time
COPY --from=builder /tmp/git_minor  /app/_git_minor
COPY --from=builder /tmp/git_patch  /app/_git_patch

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
