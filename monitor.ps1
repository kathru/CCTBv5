# monitor.ps1 — Inicia o dashboard local em modo monitor apontando para o Oracle
#
# O que faz:
#   1. Abre SSH tunnels para PostgreSQL e Redis do Oracle
#   2. Inicia o container local com docker-compose.monitor.yml
#   3. Dashboard em localhost:8001 mostra dados reais do Oracle (read-only)
#
# Pré-requisito: SSH key configurada para oracle

$ErrorActionPreference = "Stop"

$ORACLE_HOST = "ubuntu@137.131.220.216"
$ORACLE_KEY  = "D:\oracle server\ssh-key-2026-05-03.key"
$TUNNEL_PG   = 15432   # porta local → Oracle PostgreSQL 5432
$TUNNEL_REDIS= 16379   # porta local → Oracle Redis 6379

function Write-Step($text) { Write-Host "" ; Write-Host "  >> $text" -ForegroundColor Cyan }
function Write-Ok($text)   { Write-Host "  OK $text" -ForegroundColor Green }

Set-Location $PSScriptRoot

# ── 1. Verifica se tunnels já estão ativos ──────────────────────────────────
Write-Step "Verificando SSH tunnels..."

$pgOk    = $false
$redisOk = $false

try {
    $tcp = New-Object System.Net.Sockets.TcpClient
    $tcp.Connect("127.0.0.1", $TUNNEL_PG)
    $tcp.Close()
    $pgOk = $true
    Write-Host "  Tunnel PostgreSQL ($TUNNEL_PG) já ativo" -ForegroundColor Gray
} catch { }

try {
    $tcp = New-Object System.Net.Sockets.TcpClient
    $tcp.Connect("127.0.0.1", $TUNNEL_REDIS)
    $tcp.Close()
    $redisOk = $true
    Write-Host "  Tunnel Redis ($TUNNEL_REDIS) já ativo" -ForegroundColor Gray
} catch { }

# ── 2. Abre tunnels se necessário ───────────────────────────────────────────
if (-not $pgOk -or -not $redisOk) {
    Write-Step "Abrindo SSH tunnels para Oracle..."
    $sshArgs = @(
        "-N",
        "-L", "${TUNNEL_PG}:localhost:5432",
        "-L", "${TUNNEL_REDIS}:localhost:6379",
        "-i", $ORACLE_KEY,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=30",
        $ORACLE_HOST
    )
    Start-Process ssh -ArgumentList $sshArgs -WindowStyle Hidden
    Start-Sleep -Seconds 3

    # Verifica se tunnel abriu
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient
        $tcp.Connect("127.0.0.1", $TUNNEL_PG)
        $tcp.Close()
        Write-Ok "Tunnel PostgreSQL ativo em localhost:$TUNNEL_PG"
    } catch {
        Write-Host "  ERRO: Tunnel PostgreSQL nao abriu. Verifique SSH key e firewall." -ForegroundColor Red
        exit 1
    }
}

# ── 3. Calcula versão ────────────────────────────────────────────────────────
$GIT_MINOR = (git tag | Where-Object { $_ -like 'fase/*' } | Measure-Object -Line).Lines
$LAST_TAG  = git tag | Where-Object { $_ -like 'fase/*' } | Select-Object -Last 1
if ($LAST_TAG) {
    $GIT_PATCH = git rev-list --count "$LAST_TAG..HEAD"
} else {
    $GIT_PATCH = git rev-list --count HEAD
}
$env:GIT_MINOR = $GIT_MINOR
$env:GIT_PATCH = $GIT_PATCH

Write-Host ""
Write-Host "  Modo: MONITOR ONLY → Oracle" -ForegroundColor Yellow
Write-Host "  PostgreSQL: localhost:$TUNNEL_PG → Oracle:5432" -ForegroundColor Gray
Write-Host "  Redis:      localhost:$TUNNEL_REDIS → Oracle:6379" -ForegroundColor Gray

# ── 4. Sobe container em modo monitor ───────────────────────────────────────
Write-Step "Iniciando dashboard monitor..."
docker compose -f docker-compose.yml -f docker-compose.monitor.yml up -d --build cctb
if ($LASTEXITCODE -ne 0) { Write-Host "  ERRO: build falhou" -ForegroundColor Red; exit 1 }
Write-Ok "Dashboard monitor → http://localhost:8001"

# ── 5. Health check ─────────────────────────────────────────────────────────
Write-Step "Aguardando startup..."
Start-Sleep -Seconds 8
try {
    $h = Invoke-RestMethod "http://localhost:8001/health" -TimeoutSec 10
    Write-Ok "Health OK (v$($h.version)) — dados do Oracle em tempo real"
} catch {
    Write-Host "  AVISO: sem resposta ainda — aguarde ~10s" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "  >> Monitor ativo! Para parar: docker compose stop cctb" -ForegroundColor Green
Write-Host ""
