#!/bin/sh
# CCTBv5 — Entrypoint com wait-for-postgres robusto.
#
# Garante que a aplicação só sobe APÓS o postgres aceitar conexões,
# independentemente de como o container foi iniciado:
#   - docker compose up  (respeita depends_on)
#   - docker start       (ignora depends_on — este script compensa)
#   - restart automático pelo Docker daemon após falha
#
# Estratégia: pg_isready com retry exponencial (máx 60s).
# Não usa ferramentas externas (wait-for-it, netcat, etc.) — só sh e pg_isready.
# pg_isready está disponível via libpq-client instalado no stage de runtime.

set -e

# ── Extrai host/port da DATABASE_URL ─────────────────────────────────────────
# Formato esperado: postgresql+asyncpg://user:pass@host:port/db
# Extrai apenas host e port para pg_isready
DB_URL="${DATABASE_URL:-}"
PG_HOST="postgres"   # default: nome do serviço docker-compose
PG_PORT="5432"

if [ -n "$DB_URL" ]; then
    # Remove prefixo de driver (postgresql+asyncpg:// → postgresql://)
    _clean=$(echo "$DB_URL" | sed 's|+[^/]*||')
    # Extrai host:port da URL
    _hostport=$(echo "$_clean" | sed 's|.*@||' | sed 's|/.*||')
    _host=$(echo "$_hostport" | cut -d: -f1)
    _port=$(echo "$_hostport" | cut -d: -f2)
    [ -n "$_host" ] && PG_HOST="$_host"
    [ -n "$_port" ] && [ "$_port" != "$_host" ] && PG_PORT="$_port"
fi

# ── Wait-for-PostgreSQL ───────────────────────────────────────────────────────
MAX_WAIT=120    # segundos máximos de espera
INTERVAL=3      # intervalo entre tentativas
elapsed=0

echo "[entrypoint] Aguardando PostgreSQL em ${PG_HOST}:${PG_PORT}..."

until pg_isready -h "$PG_HOST" -p "$PG_PORT" -q 2>/dev/null; do
    if [ "$elapsed" -ge "$MAX_WAIT" ]; then
        echo "[entrypoint] ERRO: PostgreSQL não ficou disponível em ${MAX_WAIT}s — abortando."
        exit 1
    fi
    echo "[entrypoint] PostgreSQL indisponível — tentando novamente em ${INTERVAL}s... (${elapsed}s/${MAX_WAIT}s)"
    sleep "$INTERVAL"
    elapsed=$((elapsed + INTERVAL))
done

echo "[entrypoint] PostgreSQL disponível em ${PG_HOST}:${PG_PORT} (${elapsed}s)"

# ── Wait-for-Redis (usa redis-cli ping para check de liveness real) ──────────
# Usa python urlparse para extração segura do host (suporta senhas na URL)
REDIS_URL="${REDIS_URL:-redis://redis:6379/0}"
_redis_parse=$(python -c "
from urllib.parse import urlparse
u = urlparse('${REDIS_URL}')
h = u.hostname or 'redis'
p = u.port or 6379
print(h)
print(p)
" 2>/dev/null)
REDIS_HOST=$(echo "$_redis_parse" | head -1)
REDIS_PORT=$(echo "$_redis_parse" | tail -1)
REDIS_HOST="${REDIS_HOST:-redis}"
REDIS_PORT="${REDIS_PORT:-6379}"

redis_elapsed=0
redis_max=30

echo "[entrypoint] Aguardando Redis em ${REDIS_HOST}:${REDIS_PORT}..."
# Prefere redis-cli (liveness real via PING/PONG), fallback para socket TCP
if command -v redis-cli >/dev/null 2>&1; then
    until redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping 2>/dev/null | grep -q PONG; do
        if [ "$redis_elapsed" -ge "$redis_max" ]; then
            echo "[entrypoint] AVISO: Redis não respondeu em ${redis_max}s — continuando mesmo assim."
            break
        fi
        sleep 2
        redis_elapsed=$((redis_elapsed + 2))
    done
else
    # Fallback: socket TCP (sem redis-cli instalado)
    until python -c "
import socket, sys
try:
    s = socket.create_connection(('${REDIS_HOST}', ${REDIS_PORT}), timeout=2)
    s.close(); sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; do
        if [ "$redis_elapsed" -ge "$redis_max" ]; then
            echo "[entrypoint] AVISO: Redis não respondeu em ${redis_max}s — continuando mesmo assim."
            break
        fi
        sleep 2
        redis_elapsed=$((redis_elapsed + 2))
    done
fi

echo "[entrypoint] Dependências OK — iniciando CCTBv5..."

# ── Executa o comando principal ───────────────────────────────────────────────
exec "$@"
