$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    $python = Join-Path $projectRoot '../lab01/.venv/Scripts/python.exe'
}
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Create project/.venv and install project/requirements.txt first.'
}
Push-Location $projectRoot
try {
    & $python -u -m src.app
} finally {
    Pop-Location
}
