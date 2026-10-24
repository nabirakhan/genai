# End-of-session routine for the shared lab PC: commit results, push, remove credentials.
# Usage: scripts\end_of_session.ps1 [-NoPush]
param([switch]$NoPush)
$ErrorActionPreference = "Continue"
$ok = $true
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

Write-Host "== committing tracked outputs =="
git add splits results web/public/results LAB_RUNBOOK.md src/classify/config.json 2>$null
if ($LASTEXITCODE -ne 0) { $ok = $false; Write-Host "!! git add failed (stale .git\index.lock?): run git add again to see why" -ForegroundColor Red }
$staged = git diff --cached --name-only
if ($staged) {
    $n = (Get-Content results/classifier_runs.csv -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
    git commit -m "Lab session $(Get-Date -Format 'yyyy-MM-dd HH:mm'): results ($n rows)"
    if ($LASTEXITCODE -ne 0) { $ok = $false; Write-Host "!! git commit failed (user.name/user.email set?)" -ForegroundColor Red }
} else {
    Write-Host "nothing to commit"
}
if (-not $NoPush) {
    git pull --rebase --autostash
    git push
    if ($LASTEXITCODE -ne 0) { $ok = $false; Write-Host "!! git push failed: results exist only on this PC" -ForegroundColor Red }
    if (git status --porcelain=v1 --branch | Select-String "ahead") { $ok = $false; Write-Host "!! commits not on origin" -ForegroundColor Red }
}
if (git status --porcelain -- splits results web/public/results LAB_RUNBOOK.md src/classify/config.json) { $ok = $false; Write-Host "!! outputs not committed" -ForegroundColor Red }
git status --short

Write-Host "== removing credentials =="
# Kaggle CLI 2.x: legacy kaggle.json, API-token file, or the `kaggle auth login` OAuth cache
foreach ($f in "kaggle.json", "access_token", "access_token.txt", "credentials.json") {
    $p = Join-Path $HOME ".kaggle\$f"
    if (Test-Path $p) { Remove-Item $p -Force; Write-Host "removed $p" }
}
Remove-Item Env:KAGGLE_API_TOKEN, Env:KAGGLE_KEY, Env:KAGGLE_USERNAME -ErrorAction SilentlyContinue
# git push authenticates through Git Credential Manager (Windows Credential Manager), not gh. Feed git a
# BOM-free file: a PowerShell 5.1 pipe prepends a BOM and git refuses it ("missing protocol field").
$desc = Join-Path $env:TEMP "github-erase.txt"
[System.IO.File]::WriteAllText($desc, "protocol=https`nhost=github.com`n`n")
cmd /c "git credential reject < `"$desc`""
Remove-Item $desc
cmdkey /delete:git:https://github.com 2>$null | Out-Null
if (Get-Command gh -ErrorAction SilentlyContinue) { gh auth logout --hostname github.com 2>$null }
if (Test-Path Env:HF_TOKEN) { Remove-Item Env:HF_TOKEN }
foreach ($f in "token", "stored_tokens") { Remove-Item (Join-Path $HOME ".cache\huggingface\$f") -Force -ErrorAction SilentlyContinue }
Write-Host "== backup raw zips to F: (if present) =="
$data = if ($env:GENAI_DATA) { $env:GENAI_DATA } else { "C:\genai-data" }  # a fresh shell has no $env:GENAI_DATA
$raw = Join-Path $data "raw\aptos\aptos2019-blindness-detection.zip"
if ((Test-Path $raw) -and (Test-Path "F:\")) {
    New-Item -ItemType Directory -Force "F:\genai-backup" | Out-Null
    if (-not (Test-Path "F:\genai-backup\aptos2019-blindness-detection.zip")) { Copy-Item $raw "F:\genai-backup\" }
}
if ($ok) { Write-Host "done. Safe to leave." } else { Write-Host "NOT safe to leave: see the errors above" -ForegroundColor Red; exit 1 }
