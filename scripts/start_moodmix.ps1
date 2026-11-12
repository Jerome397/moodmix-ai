$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    Write-Host 'Python environment missing. Follow the Windows quick start in README.md.' -ForegroundColor Red
    exit 1
}
Set-Location -LiteralPath $projectRoot
$logDirectory = Join-Path $projectRoot 'tmp'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null

function Test-ServiceReady([string] $Url) {
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
        return $response.StatusCode -eq 200
    } catch { return $false }
}

$services = @(
    @{ Name = 'backend'; Health = 'http://127.0.0.1:8000/health'; Arguments = '-m uvicorn backend.main:app --host 127.0.0.1 --port 8000' },
    @{ Name = 'frontend'; Health = 'http://127.0.0.1:8501/_stcore/health'; Arguments = '-m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false' },
    @{ Name = 'studio'; Health = 'http://127.0.0.1:8502/_stcore/health'; Arguments = '-m streamlit run app/agent_app.py --server.address 127.0.0.1 --server.port 8502 --server.headless true --browser.gatherUsageStats false' }
)
foreach ($service in $services) {
    if (-not (Test-ServiceReady $service.Health)) {
        Write-Host "Starting $($service.Name)..."
        $logId = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
        Start-Process -FilePath $projectPython -ArgumentList $service.Arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDirectory "$($service.Name)-$logId.stdout.log") -RedirectStandardError (Join-Path $logDirectory "$($service.Name)-$logId.stderr.log")
    }
}
$deadline = (Get-Date).AddSeconds(60)
do {
    $pending = @($services | Where-Object { -not (Test-ServiceReady $_.Health) })
    if ($pending.Count -eq 0) { break }
    Start-Sleep -Milliseconds 750
} while ((Get-Date) -lt $deadline)
if ($pending.Count -gt 0) {
    Write-Host "Could not start: $($pending.Name -join ', '). Check the latest logs in $logDirectory." -ForegroundColor Red
    exit 1
}
Start-Process 'http://127.0.0.1:8501/'
Start-Process 'http://127.0.0.1:8502/'
Write-Host 'MoodMix and Studio are ready. You can close this window.' -ForegroundColor Green
