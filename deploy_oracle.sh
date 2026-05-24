#!/bin/bash
set -e

echo ">> Atualizando repositorio..."
cd ~/CCTBv5
git pull origin main

echo ">> Copiando arquivos para o container..."
docker cp src/ cctb_app:/app/src/

echo ">> Reiniciando container..."
docker restart cctb_app

echo ">> Aguardando subida (max 90s)..."
WAITED=0
while [ $WAITED -lt 90 ]; do
  sleep 5
  WAITED=$((WAITED + 5))
  STATUS=$(curl -s http://localhost:8001/health 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null)
  echo "   aguardando... (${WAITED}s) status=${STATUS:-sem resposta}"
  if [ "$STATUS" = "ok" ]; then
    VERSION=$(curl -s http://localhost:8001/health 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('version',''))" 2>/dev/null)
    echo "OK Deploy Oracle concluido com sucesso -- v${VERSION}"
    exit 0
  fi
done
echo "ERRO container nao respondeu em 90s"
exit 1
