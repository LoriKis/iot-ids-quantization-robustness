# Optional bootstrap when no working Python 3.11 exists. Project-local only.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$archive = Join-Path $projectRoot 'work/python-3.11.9.zip'
$destination = Join-Path $projectRoot 'work/python-3.11.9'
$portablePython = Join-Path $destination 'tools/python.exe'
New-Item -ItemType Directory -Force work | Out-Null
if (-not (Test-Path -LiteralPath $portablePython)) {
    if (Test-Path -LiteralPath $destination) { throw 'Partial portable Python exists; preserve it and inspect manually.' }
    if (-not (Test-Path -LiteralPath $archive)) {
        Invoke-WebRequest 'https://www.nuget.org/api/v2/package/python/3.11.9' -OutFile $archive
    }
    $expectedHash = '9283876D58C017E0E846F95B490DA3BCA0FC0A6EE1134B2870677CFB7EEC3C67'
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expectedHash) {
        throw 'Portable Python archive hash mismatch'
    }
    Expand-Archive -LiteralPath $archive -DestinationPath $destination
}
& $portablePython --version
if ($LASTEXITCODE -ne 0) { throw 'Portable Python could not start' }
& "$PSScriptRoot/setup.ps1" -Python $portablePython
