param([string]$Python = "python")
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Create .venv with a working Python 3.11 interpreter' }
}
$projectPython = Join-Path $projectRoot '.venv/Scripts/python.exe'
& $projectPython -c "import sys; assert sys.version_info[:2] == (3, 11), 'Use Python 3.11'"
if ($LASTEXITCODE -ne 0) { throw 'Incompatible environment; preserve it and choose a separate project directory' }
$dependencyFile = if (Test-Path requirements-lock-windows.txt) { 'requirements-lock-windows.txt' } else { 'requirements.txt' }
& $projectPython -m pip install --cache-dir work/pip-cache -r $dependencyFile
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $projectPython scripts/setup_upstream.py
if ($LASTEXITCODE -ne 0) { throw 'Upstream verification failed' }
& $projectPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency compatibility failed' }
Write-Output 'Use .venv/Scripts/python.exe explicitly; activation is unnecessary.'
