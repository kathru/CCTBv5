#!/bin/bash
# deploy_oracle.sh -- Executado no Oracle via SSH pelo deploy.ps1
# NAO usa set -e: erros sao tratados explicitamente para que o loop
# de health-check nao seja abortado quando o container ainda esta subindo.
#
# BUG CORRIGIDO: "docker cp src/" criava /app/src/src/ (diretorio aninhado).
# Correto: "docker cp src/." copia os CONTEUDOS de src/ para /app/src/.

STEP() { echo ""; echo ">> $1"; }
OK()   { echo "   OK $1"; }
FAIL() { echo "   ERRO $1" >&2; exit 1; }

# -- 1. Atualiza repositorio --------------------------------------------------
STEP "Atualizando repositorio..."
cd ~/CCTBv5 || FAIL "cd ~/CCTBv5 falhou"
git pull origin main || FAIL "git pull falhou"

# -- 2. Copia arquivos para o container ---------------------------------------
STEP "Copiando arquivos para o container..."

# src/. = copia CONTEUDOS de src/ para /app/src/ (sem criar src/src/ aninhado)
docker cp src/. cctb_app:/app/src/                     || FAIL "docker cp src/. falhou"
docker cp _version.txt cctb_app:/app/_version.txt      || FAIL "docker cp _version.txt falhou"

# Remove o diretorio aninhado /app/src/src/ criado por deploys anteriores com bug
docker exec cctb_app bash -c "rm -rf /app/src/src" 2>/dev/null || true
OK "Arquivos copiados (src/. -> /app/src/)"

# -- 3. Reinicia o container --------------------------------------------------
STEP "Reiniciando container..."
docker restart cctb_app || FAIL "docker restart falhou"

# -- 4. Aguarda health ok (max 180s) ------------------------------------------
STEP "Aguardando subida (max 180s)..."
WAITED=0
while [ "$WAITED" -lt 180 ]; do
    sleep 5
    WAITED=$((WAITED + 5))

    # || STATUS="" garante que falha de curl/python3 nao aborta o loop
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

FAIL "container nao respondeu em 180s -- verifique: docker logs cctb_app"
