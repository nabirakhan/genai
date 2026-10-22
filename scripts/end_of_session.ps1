# End-of-session routine for the shared lab PC: commit results, push, remove credentials.
# Usage: scripts\end_of_session.ps1 [-NoPush]
param([switch]$NoPush)
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

Write-Host "== committing tracked outputs =="
git add splits results web/public/results LAB_RUNBOOK.md 2>$null
$staged = git diff --cached --name-only
if ($staged) {
    $n = (Get-Content results/classifier_runs.csv -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
    git commit -m "Lab session $(Get-Date -Format 'yyyy-MM-dd HH:mm'): results ($n rows)"
    if (-not $NoPush) { git push }
} else {
    Write-Host "nothing to commit"
}
git status --short

Write-Host "== removing credentials =="
$kj = Join-Path $HOME ".kaggle\kaggle.json"
if (Test-Path $kj) { Remove-Item $kj -Force; Write-Host "removed $kj" }
if (Get-Command gh -ErrorAction SilentlyContinue) { gh auth logout --hostname github.com 2>$null }
if (Test-Path Env:HF_TOKEN) { Remove-Item Env:HF_TOKEN }
if (Test-Path (Join-Path $HOME ".cache\huggingface\token")) { Remove-Item (Join-Path $HOME ".cache\huggingface\token") -Force }
Write-Host "== backup raw zips to F: (if present) =="
$raw = Join-Path $env:GENAI_DATA "raw\aptos\aptos2019-blindness-detection.zip"
if ((Test-Path $raw) -and (Test-Path "F:\")) {
    New-Item -ItemType Directory -Force "F:\genai-backup" | Out-Null
    if (-not (Test-Path "F:\genai-backup\aptos2019-blindness-detection.zip")) { Copy-Item $raw "F:\genai-backup\" }
}
Write-Host "done. Safe to leave."
