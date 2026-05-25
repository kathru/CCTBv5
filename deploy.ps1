# deploy.ps1 -- Commit, push e deploy em Oracle + local com um unico comando
#
# Uso:
#   .\deploy.ps1                  # commit automatico + deploy em ambos
#   .\deploy.ps1 -msg "fix: ..."  # commit com mensagem customizada
#   .\deploy.ps1 -OracleOnly      # so Oracle (sem rebuild local)
#   .\deploy.ps1 -LocalOnly       # so local (sem SSH)
#
# Pre-requisito: SSH key em "D:\oracle server\ssh-key-2026-05-03.key"

param(
    [string]$msg        = "",
    [switch]$OracleOnly,
    [switch]$LocalOnly
)

$ErrorActionPreference = "Stop"

$ORACLE_HOST = "ubuntu@137.131.220.216"
$ORACLE_KEY  = "D:\oracle server\ssh-key-2026-05-03.key"
$COAUTHOR    = "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
$PROJECT_DIR = $PSScriptRoot

function Write-Step($text) { Write-Host ""; Write-Host "  >> $text" -ForegroundColor Cyan }
function Write-Ok($text)   { Write-Host "  OK $text"    -ForegroundColor Green }
function Write-Warn($text) { Write-Host "  AVISO $text" -ForegroundColor Yellow }
function Write-Fail($text) { Write-Host "  ERRO $text"  -ForegroundColor Red }

Set-Location $PROJECT_DIR

# -- 1. Commit das alteracoes de codigo (se houver) ---------------------------
Write-Step "Verificando git status..."
$dirty = git status --porcelain | Where-Object { $_ -notmatch '^\?\? ' -and $_ -notmatch '_version\.txt' }
if ($dirty) {
    if ($msg -eq "") {
        $date = Get-Date -Format "yyyy-MM-dd HH:mm"
        $msg  = "chore: deploy $date"
    }
    Write-Step "Commitando alteracoes: '$msg'"
    git add -A
    git commit -m "$msg`n`n$COAUTHOR"
    if ($LASTEXITCODE -ne 0) { Write-Fail "git commit falhou"; exit 1 }
    Write-Ok "Commit feito"
} else {
    Write-Host "  (nenhuma alteracao de codigo para commitar)" -ForegroundColor Gray
}

# -- 2. Calcular versao -------------------------------------------------------
# REGRA DE VERSIONAMENTO: vX.Y.Z
#   X = 5      versao geral do projeto (fixo)
#   Y = fases  total de fases implementadas -- lido de _phase.txt
#              (atualizar _phase.txt manualmente ao concluir cada nova fase)
#   Z = commits total de commits no branch (git rev-list --count HEAD)
$PHASE_FILE = "$PROJECT_DIR\_phase.txt"
if (Test-Path $PHASE_FILE) {
    $GIT_MINOR = [int](Get-Content $PHASE_FILE -Raw).Trim()
} else {
    Write-Warn "_phase.txt nao encontrado -- usando 0"
    $GIT_MINOR = 0
}
$GIT_PATCH = [int](git rev-list --count HEAD)
$VERSION   = "5.$GIT_MINOR.$GIT_PATCH"
Write-Host ""
Write-Host "  Versao calculada: v$VERSION" -ForegroundColor Yellow

# -- 3. Gravar _version.txt e commitar (antes do push) -----------------------
# Critico: _version.txt deve chegar ao Oracle via git pull,
# e tambem e copiado ao container em deploy_oracle.sh.
Write-Step "Atualizando _version.txt -> v$VERSION..."
Set-Content -Path "$PROJECT_DIR\_version.txt" -Value $VERSION -NoNewline
$vDirty = git status --porcelain "_version.txt"
if ($vDirty) {
    git add "_version.txt"
    git commit --no-verify -m "chore: bump version -> v$VERSION`n`n$COAUTHOR"
    if ($LASTEXITCODE -ne 0) { Write-Fail "git commit _version.txt falhou"; exit 1 }
    Write-Ok "_version.txt commitado (v$VERSION)"
} else {
    Write-Host "  (_version.txt ja estava em v$VERSION)" -ForegroundColor Gray
}

# -- 4. Sync com GitHub (pull + push) ----------------------------------------
Write-Step "Sincronizando com GitHub..."
git pull --rebase
if ($LASTEXITCODE -ne 0) { Write-Fail "git pull falhou - resolva conflitos manualmente"; exit 1 }
git push
if ($LASTEXITCODE -ne 0) { Write-Fail "git push falhou"; exit 1 }
Write-Ok "GitHub atualizado (v$VERSION)"

# -- 5. Deploy LOCAL (monitor SSH tunnel -> Oracle) ---------------------------
if (-not $OracleOnly) {
    Write-Step "Iniciando monitor local (SSH tunnel -> Oracle)..."
    $env:GIT_MINOR = $GIT_MINOR
    $env:GIT_PATCH = $GIT_PATCH
    pwsh -NoProfile -File "$PROJECT_DIR\monitor.ps1"
    if ($LASTEXITCODE -ne 0) { Write-Fail "Monitor local falhou"; exit 1 }
    Write-Ok "Local -> http://localhost:8001 (dados via Oracle)"
}

# -- 6. Deploy ORACLE ---------------------------------------------------------
if (-not $LocalOnly) {
    Write-Step "Deploying no Oracle Cloud (137.131.220.216:8001)..."
    ssh -i $ORACLE_KEY -o StrictHostKeyChecking=no $ORACLE_HOST "bash ~/CCTBv5/deploy_oracle.sh"
    if ($LASTEXITCODE -ne 0) { Write-Fail "Deploy Oracle falhou"; exit 1 }
    Write-Ok "Oracle atualizado -> http://137.131.220.216:8001"
}

# -- 7. Health check final (informativo, nao falha o deploy) ------------------
Write-Step "Verificando health..."
Start-Sleep -Seconds 3

if (-not $OracleOnly) {
    try {
        $h = Invoke-RestMethod "http://localhost:8001/health" -TimeoutSec 8
        if ($h.status -eq "ok") { Write-Ok "Local:  OK (v$($h.version))" }
        else                     { Write-Warn "Local: status=$($h.status)" }
    } catch {
        Write-Warn "Local: sem resposta (tunnel SSH pode estar iniciando)"
    }
}

if (-not $LocalOnly) {
    try {
        $h = Invoke-RestMethod "http://137.131.220.216:8001/health" -TimeoutSec 10
        if ($h.status -eq "ok") { Write-Ok "Oracle: OK (v$($h.version))" }
        else                     { Write-Warn "Oracle: status=$($h.status)" }
    } catch {
        Write-Warn "Oracle: sem resposta no health check final"
    }
}

Write-Host ""
Write-Host "  >> Deploy v$VERSION concluido!" -ForegroundColor Green
Write-Host ""
