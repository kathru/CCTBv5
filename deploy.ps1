# deploy.ps1 -- Commit, push e deploy em Oracle + local
#
# Uso:
#   .\deploy.ps1                  # commit automatico + deploy em ambos
#   .\deploy.ps1 -msg "fix: ..."  # commit com mensagem customizada
#   .\deploy.ps1 -OracleOnly      # so Oracle (sem rebuild local)
#   .\deploy.ps1 -LocalOnly       # so local (sem SSH)
#
# VERSIONAMENTO: vX.Y.Z
#   X = 5            (fixo: versao geral do projeto)
#   Y = fase atual   (lido de phases.json -> current)
#   Z = total commits (git rev-list --count HEAD)
#
# REGRA DE DEPLOY:
#   Antes de executar, sempre apresentar fase + commits ao usuario para confirmacao.
#   O Claude atualiza phases.json ao concluir cada nova fase.

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

# -- 1. Calcular versao a partir de phases.json + git -------------------------
$PHASES_FILE = "$PROJECT_DIR\phases.json"
if (-not (Test-Path $PHASES_FILE)) {
    Write-Fail "phases.json nao encontrado em $PROJECT_DIR"
    exit 1
}
$phases    = Get-Content $PHASES_FILE | ConvertFrom-Json
$GIT_MINOR = [int]$phases.current
$GIT_PATCH = [int](git rev-list --count HEAD)
$VERSION   = "5.$GIT_MINOR.$GIT_PATCH"
$PHASE_NAME= ($phases.phases | Where-Object { $_.n -eq $GIT_MINOR } | Select-Object -First 1).name

# -- Resumo do deploy (apresentar ao usuario antes de prosseguir) -------------
Write-Host ""
Write-Host "  ============================================" -ForegroundColor DarkCyan
Write-Host "   RESUMO DO DEPLOY" -ForegroundColor Yellow
Write-Host "  ============================================" -ForegroundColor DarkCyan
Write-Host "   Fase    : $GIT_MINOR -- $PHASE_NAME" -ForegroundColor White
Write-Host "   Commits : $GIT_PATCH" -ForegroundColor White
Write-Host "   Versao  : v$VERSION" -ForegroundColor Cyan
Write-Host "  ============================================" -ForegroundColor DarkCyan
Write-Host ""

# -- 2. Commit das alteracoes de codigo (se houver) ---------------------------
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

    # Recalcular patch apos novo commit
    $GIT_PATCH = [int](git rev-list --count HEAD)
    $VERSION   = "5.$GIT_MINOR.$GIT_PATCH"
} else {
    Write-Host "  (nenhuma alteracao de codigo para commitar)" -ForegroundColor Gray
}

# -- 3. Gravar _version.txt e commitar (antes do push) -----------------------
Write-Step "Atualizando _version.txt -> v$VERSION..."
Set-Content -Path "$PROJECT_DIR\_version.txt" -Value $VERSION -NoNewline
$vDirty = git status --porcelain "_version.txt"
if ($vDirty) {
    git add "_version.txt"
    git commit --no-verify -m "chore: bump version -> v$VERSION`n`n$COAUTHOR"
    if ($LASTEXITCODE -ne 0) { Write-Fail "git commit _version.txt falhou"; exit 1 }
    # Recalcular novamente apos commit do version
    $GIT_PATCH = [int](git rev-list --count HEAD)
    $VERSION   = "5.$GIT_MINOR.$GIT_PATCH"
    Write-Ok "_version.txt commitado -> v$VERSION"
} else {
    Write-Host "  (_version.txt ja estava em v$VERSION)" -ForegroundColor Gray
}

# -- 4. Sync com GitHub (pull + push) -----------------------------------------
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

# -- 7. Health check final (informativo) -------------------------------------
Write-Step "Verificando health..."
Start-Sleep -Seconds 3

if (-not $OracleOnly) {
    try {
        $h = Invoke-RestMethod "http://localhost:8001/health" -TimeoutSec 8
        if ($h.status -eq "ok") { Write-Ok "Local:  OK (v$($h.version))" }
        else                     { Write-Warn "Local: status=$($h.status)" }
    } catch { Write-Warn "Local: sem resposta (tunnel SSH pode estar iniciando)" }
}

if (-not $LocalOnly) {
    try {
        $h = Invoke-RestMethod "http://137.131.220.216:8001/health" -TimeoutSec 10
        if ($h.status -eq "ok") { Write-Ok "Oracle: OK (v$($h.version))" }
        else                     { Write-Warn "Oracle: status=$($h.status)" }
    } catch { Write-Warn "Oracle: sem resposta no health check final" }
}

Write-Host ""
Write-Host "  >> Deploy v$VERSION concluido! (Fase $GIT_MINOR -- $PHASE_NAME)" -ForegroundColor Green
Write-Host ""
