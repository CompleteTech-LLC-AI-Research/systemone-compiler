# Install locally and verify the no-key workflow. Never runs paid inference.
param([switch]$Full)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (-not (Test-Path ".venv")) {
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
}
$Python = Join-Path (Get-Location) ".venv/Scripts/python.exe"
if (-not (Test-Path $Python)) { throw "No Windows venv interpreter at $Python" }
$Extras = ".[dev]"
if ($Full) { $Extras = ".[all,dev]" }
& $Python -m pip install -e $Extras
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
& $Python -m typewright doctor
if ($LASTEXITCODE -ne 0) { throw "Core doctor checks failed." }
if ($Full) {
    & $Python -m typewright doctor --check-optional
    if ($LASTEXITCODE -ne 0) { throw "Optional dependency contracts failed." }
}
& $Python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Tests failed." }
$Run = "runs/setup-demo-" + (Get-Date -Format "yyyyMMdd-HHmmss") + "-" + $PID
& $Python -m typewright demo --out $Run
if ($LASTEXITCODE -ne 0) { throw "Synthetic demo failed." }
Write-Output "Local setup and synthetic demo complete. No live inference was run."
