# monitor.ps1 — Dashboard local consumindo dados do Oracle via SSH tunnel
#
# Uso: .\monitor.ps1          (inicia tudo)
#       .\monitor.ps1 -Stop   (para tudo)

param([switch]$Stop)

$ErrorActionPreference = "Stop"

$ORACLE_HOST  = "ubuntu@137.131.220.216"
$ORACLE_KEY   = "D:\oracle server\ssh-key-2026-05-03.key"
$SSH_PG       = 25432   # SSH tunnel PostgreSQL (127.0.0.1)
$SSH_REDIS    = 26379   # SSH tunnel Redis      (127.0.0.1)
$PROXY_PG     = 15432   # Porta Docker (0.0.0.0)
$PROXY_REDIS  = 16379

$PID_FILE     = "$PSScriptRoot\.monitor_pids"

function Write-Step($t) { Write-Host ""; Write-Host "  >> $t" -ForegroundColor Cyan }
function Write-Ok($t)   { Write-Host "  OK $t" -ForegroundColor Green }

function Stop-Monitor {
    Write-Step "Parando monitor..."
    if (Test-Path $PID_FILE) {
        Get-Content $PID_FILE | ForEach-Object {
            try { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue } catch {}
        }
        Remove-Item $PID_FILE -Force
    }
    docker compose stop cctb 2>$null
    Write-Ok "Monitor parado."
    exit 0
}

if ($Stop) { Stop-Monitor }

Set-Location $PSScriptRoot

# Para qualquer monitor anterior
if (Test-Path $PID_FILE) { Stop-Monitor }
$pids = @()

# ── 1. SSH tunnel (processo independente) ──────────────────────────────────
Write-Step "Abrindo SSH tunnels para Oracle..."
$sshProc = Start-Process "ssh" -ArgumentList @(
    "-N",
    "-L", "127.0.0.1:${SSH_PG}:localhost:5432",
    "-L", "127.0.0.1:${SSH_REDIS}:localhost:6379",
    "-i", "`"$ORACLE_KEY`"",
    "-o", "StrictHostKeyChecking=no",
    "-o", "ServerAliveInterval=30",
    $ORACLE_HOST
) -PassThru -WindowStyle Hidden
$pids += $sshProc.Id
Start-Sleep -Seconds 6

# Verifica tunnel
$ok = $false
try { $t=New-Object Net.Sockets.TcpClient; $t.Connect("127.0.0.1",$SSH_PG); $ok=$t.Connected; $t.Close() } catch {}
if (-not $ok) { Write-Host "  ERRO: Tunnel SSH nao abriu." -ForegroundColor Red; exit 1 }
Write-Ok "Tunnels SSH ativos (PG:$SSH_PG / Redis:$SSH_REDIS)"

# ── 2. Python proxy 0.0.0.0 (processo independente) ────────────────────────
Write-Step "Iniciando proxy TCP para Docker..."
$proxyScript = "$PSScriptRoot\.monitor_proxy.py"
@"
import socket,threading,time
def pipe(a,b):
    try:
        while True:
            d=a.recv(4096)
            if not d:break
            b.sendall(d)
    except:pass
    finally:
        [x.close() for x in [a,b] if x]
def proxy(lp,rp):
    s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    s.bind(('0.0.0.0',lp));s.listen(20)
    while True:
        c,_=s.accept();u=socket.socket();u.connect(('127.0.0.1',rp))
        [threading.Thread(target=pipe,args=x,daemon=True).start() for x in [(c,u),(u,c)]]
for lp,rp in [($PROXY_PG,$SSH_PG),($PROXY_REDIS,$SSH_REDIS)]:
    threading.Thread(target=proxy,args=(lp,rp),daemon=True).start()
while True:time.sleep(60)
"@ | Set-Content $proxyScript

$python = (Get-Command python -ErrorAction SilentlyContinue)?.Source
if (-not $python) { $python = "$PSScriptRoot\.venv\Scripts\python.exe" }
$proxyProc = Start-Process $python -ArgumentList $proxyScript -PassThru -WindowStyle Hidden
$pids += $proxyProc.Id
Start-Sleep -Seconds 2

# Verifica proxy
$ok2 = $false
try { $t=New-Object Net.Sockets.TcpClient; $t.Connect("127.0.0.1",$PROXY_PG); $ok2=$t.Connected; $t.Close() } catch {}
if (-not $ok2) { Write-Host "  ERRO: Proxy TCP nao abriu." -ForegroundColor Red; exit 1 }
Write-Ok "Proxy TCP ativo (0.0.0.0:$PROXY_PG / $PROXY_REDIS)"

# Salva PIDs para poder parar depois
$pids | Set-Content $PID_FILE

# ── 3. Calcula versão ────────────────────────────────────────────────────────
$GIT_MINOR = (git tag | Where-Object { $_ -like 'fase/*' } | Measure-Object -Line).Lines
$LAST_TAG  = git tag | Where-Object { $_ -like 'fase/*' } | Select-Object -Last 1
$GIT_PATCH = if ($LAST_TAG) { git rev-list --count "$LAST_TAG..HEAD" } else { git rev-list --count HEAD }
$env:GIT_MINOR = $GIT_MINOR
$env:GIT_PATCH = $GIT_PATCH

Write-Host ""
Write-Host "  Modo: MONITOR ONLY → Oracle" -ForegroundColor Yellow
Write-Host "  PostgreSQL: host.docker.internal:$PROXY_PG → Oracle:5432" -ForegroundColor Gray
Write-Host "  Redis:      host.docker.internal:$PROXY_REDIS → Oracle:6379" -ForegroundColor Gray

# ── 4. Container Docker ─────────────────────────────────────────────────────
Write-Step "Iniciando dashboard monitor..."
docker compose -f docker-compose.yml -f docker-compose.monitor.yml up -d --build cctb
if ($LASTEXITCODE -ne 0) { Write-Host "  ERRO: build falhou" -ForegroundColor Red; exit 1 }

# ── 5. Health check ─────────────────────────────────────────────────────────
Write-Step "Aguardando startup..."
Start-Sleep -Seconds 12
try {
    $h = Invoke-RestMethod "http://localhost:8001/health" -TimeoutSec 15
    Write-Ok "Health OK (v$($h.version)) — dados Oracle em tempo real"
} catch {
    Write-Host "  AVISO: aguardando inicialização (~15s)..." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "  >> Monitor ativo → http://localhost:8001" -ForegroundColor Green
Write-Host "  >> Para parar:  .\monitor.ps1 -Stop" -ForegroundColor Gray
Write-Host ""
