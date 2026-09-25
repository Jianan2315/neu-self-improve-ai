param(
    [string]$RuntimeRoot,
    [string]$Model = 'qwen3:8b',
    [int]$Rounds = 2,
    [string]$SeedHistory = (Join-Path $PSScriptRoot 'evidence/baseline/history.json'),
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'runtime.ps1')
$taskRuntime = Resolve-ExperimentRuntime -ExplicitRoot $RuntimeRoot
if (!(Test-Path -LiteralPath $SeedHistory -PathType Leaf)) { throw 'Seed history is missing.' }
if ($CheckOnly) { Write-Output "Runtime paths and seed history verified: $taskRuntime"; return }
$taskPython = Join-Path $taskRuntime 'venv\Scripts\python.exe'
$taskRunDir = Join-Path $PSScriptRoot ('designer-runs\' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
& $taskPython -X utf8 -u (Join-Path $PSScriptRoot 'designer_loop.py') --runtime-root $taskRuntime --model $Model --rounds $Rounds --seed-history $SeedHistory --output $taskRunDir
if ($LASTEXITCODE -ne 0) { throw "Designer loop stopped; inspect $taskRunDir\status.json" }
Write-Output "Designer loop completed. Results: $taskRunDir"
