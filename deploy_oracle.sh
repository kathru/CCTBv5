#!/bin/bash
# deploy_oracle.sh — Executado no Oracle via SSH pelo deploy.ps1
# NÃO usa set -e: erros são tratados explicitamente para que o loop
# de health-check não seja abortado quando o container ainda está subindo.

STEP() { echo ""; echo ">> $1"; }
OK()   { echo "   OK $1"; }
FAIL() { echo "   ERRO $1" >&2; exit 1; }

# ── 1. Atualiza repositório ───────────────────────────────────────────────────
STEP "Atualizando repositorio..."
cd ~/CCTBv5 || FAIL "cd ~/CCTBv5 falhou"
git pull origin main || FAIL "git pull falhou"

# ── 2. Copia arquivos para o container ───────────────────────────────────────
STEP "Copiando arquivos para o container..."
docker cp src/ cctb_app:/app/src/         || FAIL "docker cp src/ falhou"
docker cp _version.txt cctb_app:/app/_version.txt || FAIL "docker cp _version.txt falhou"

# ── 3. Reinicia o container ───────────────────────────────────────────────────
STEP "Reiniciando container..."
docker restart cctb_app || FAIL "docker restart falhou"

# ── 4. Aguarda health ok (max 180s) ──────────────────────────────────────────
STEP "Aguardando subida (max 180s)..."
WAITED=0
while [ "$WAITED" -lt 180 ]; do
    sleep 5
    WAITED=$((WAITED + 5))

    # || STATUS="" garante que falha de curl/python3 não abortа o loop
    STATUS=$(curl -s --max-time 3 http://localhost:8001/health 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" \
        2>/dev/null) || STATUS=""

    echo "   aguardando... (${WAITED}s) status=${STATUS:-sem resposta}"

    if [ "$STATUS" = "ok" ]; then
        VERSION=$(curl -s --max-time 3 http://localhost:8001/health 2>/dev/null \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('version',''))" \
            2>/dev/null) || VERSION="?"
        OK "Deploy Oracle concluido com sucesso -- v${VERSION}"
        exit 0
    fi
done

FAIL "container nao respondeu em 180s — verifique: docker logs cctb_app"
