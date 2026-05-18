# deploy.ps1 — Sincroniza local e Oracle com um único comando
#
# Uso:
#   .\deploy.ps1                  # commit automático + deploy em ambos
#   .\deploy.ps1 -msg "fix: ..."  # commit com mensagem customizada
#   .\deploy.ps1 -OracleOnly      # só Oracle (sem rebuild local)
#   .\deploy.ps1 -LocalOnly       # só local (sem SSH)
#
# Pré-requisito: SSH key configurada para oracle (ssh ubuntu@137.131.220.216)

param(
    [string]$msg       = "",
    [switch]$OracleOnly,
    [switch]$LocalOnly
)

$ErrorActionPreference = "Stop"

$ORACLE_HOST = "ubuntu@137.131.220.216"
$ORACLE_KEY  = "D:\oracle server\ssh-key-2026-05-03.key"
$PROJECT_DIR = $PSScriptRoot

function Write-Step($text) {
    Write-Host ""
    Write-Host "  >> $text" -ForegroundColor Cyan
}

function Write-Ok($text) {
    Write-Host "  OK $text" -ForegroundColor Green
}

function Write-Fail($text) {
    Write-Host "  ERRO $text" -ForegroundColor Red
}

Set-Location $PROJECT_DIR

# ── 0. Git pull — sincroniza com remoto antes de qualquer coisa ───────────────
Write-Step "Sincronizando com GitHub (git pull)..."
git pull --rebase
if ($LASTEXITCODE -ne 0) { Write-Fail "git pull falhou - resolva conflitos manualmente"; exit 1 }
Write-Ok "Repositório atualizado"

# ── 1. Git status ──────────────────────────────────────────────────────────────
Write-Step "Verificando git status..."
$status = git status --porcelain
if ($status) {
    if ($msg -eq "") {
        $date = Get-Date -Format "yyyy-MM-dd HH:mm"
        $msg  = "chore: deploy $date"
    }
    Write-Step "Commitando alterações: '$msg'"
    git add -A
    $coauthor = "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    git commit -m "$msg`n`n$coauthor"
    if ($LASTEXITCODE -ne 0) { Write-Fail "git commit falhou"; exit 1 }
    Write-Ok "Commit feito"
} else {
    Write-Host "  (nenhuma alteração para commitar)" -ForegroundColor Gray
}

# ── 2. Git push ────────────────────────────────────────────────────────────────
Write-Step "Pushing para GitHub..."
git push
if ($LASTEXITCODE -ne 0) { Write-Fail "git push falhou"; exit 1 }
Write-Ok "Push feito"

# ── 3. Calcular versão ─────────────────────────────────────────────────────────
# Y = número de tags fase/* (fases estruturais da memória.md)
# Z = commits desde a última tag fase/* (reseta a cada nova fase)
$GIT_MINOR = (git tag | Where-Object { $_ -like 'fase/*' } | Measure-Object -Line).Lines
$LAST_TAG  = git tag | Where-Object { $_ -like 'fase/*' } | Select-Object -Last 1
if ($LAST_TAG) {
    $GIT_PATCH = git rev-list --count "$LAST_TAG..HEAD"
} else {
    $GIT_PATCH = git rev-list --count HEAD
}
$VERSION   = "5.$GIT_MINOR.$GIT_PATCH"
Write-Host ""
Write-Host "  Versão: v$VERSION" -ForegroundColor Yellow

# ── 4. Deploy LOCAL ────────────────────────────────────────────────────────────
if (-not $OracleOnly) {
    Write-Step "Rebuilding container LOCAL (localhost:8001)..."
    $env:GIT_MINOR = $GIT_MINOR
    $env:GIT_PATCH = $GIT_PATCH
    docker compose up -d --build cctb
    if ($LASTEXITCODE -ne 0) { Write-Fail "Build local falhou"; exit 1 }
    Write-Ok "Local atualizado → http://localhost:8001"
}

# ── 5. Deploy ORACLE ───────────────────────────────────────────────────────────
if (-not $LocalOnly) {
    Write-Step "Deploying no Oracle Cloud (137.131.220.216:8001)..."

    # Usa single-quoted heredoc (sem expansão PS) + stdin para evitar CRLF
    $remote_script = @'
set -e
cd ~/CCTBv5
git pull
export GIT_MINOR=$(git tag -l 'fase/*' | wc -l | tr -d ' ')
LAST_TAG=$(git tag -l 'fase/*' | tail -1)
if [ -n "$LAST_TAG" ]; then
  export GIT_PATCH=$(git rev-list --count "$LAST_TAG..HEAD" | tr -d ' ')
else
  export GIT_PATCH=$(git rev-list --count HEAD | tr -d ' ')
fi
echo "Versao: 5.$GIT_MINOR.$GIT_PATCH"
sudo -E docker compose up -d --build cctb
echo "Oracle OK"
'@
    # Remove CRLF do Windows antes de enviar
    $remote_script = $remote_script -replace "`r`n", "`n"
    $remote_script | ssh -i $ORACLE_KEY -o StrictHostKeyChecking=no $ORACLE_HOST "bash -s"
    if ($LASTEXITCODE -ne 0) { Write-Fail "Deploy Oracle falhou"; exit 1 }
    Write-Ok "Oracle atualizado → http://137.131.220.216:8001"
}

# ── 6. Health check ────────────────────────────────────────────────────────────
Write-Step "Verificando health (aguardando containers iniciarem)..."
Start-Sleep -Seconds 20

if (-not $OracleOnly) {
    try {
        $local_health = Invoke-RestMethod "http://localhost:8001/health" -TimeoutSec 10
        $local_ok = $local_health.status -eq "ok"
        if ($local_ok) { Write-Ok "Local: OK (v$($local_health.version))" }
        else { Write-Fail "Local: DEGRADED" }
    } catch {
        Write-Host "  AVISO Local: sem resposta ainda (aguarde ~15s)" -ForegroundColor Yellow
    }
}

if (-not $LocalOnly) {
    try {
        $oracle_health = Invoke-RestMethod "http://137.131.220.216:8001/health" -TimeoutSec 15
        $oracle_ok = $oracle_health.status -eq "ok"
        if ($oracle_ok) { Write-Ok "Oracle: OK (v$($oracle_health.version))" }
        else { Write-Fail "Oracle: DEGRADED" }
    } catch {
        Write-Host "  AVISO Oracle: sem resposta ainda (aguarde ~15s)" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "  >> Deploy v$VERSION concluido!" -ForegroundColor Green
Write-Host ""
