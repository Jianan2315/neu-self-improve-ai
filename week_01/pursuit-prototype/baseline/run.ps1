param(
    [string]$RuntimeRoot,
    [string]$Spec = (Join-Path $PSScriptRoot 'scenario.json'),
    [int]$Rounds = 1,
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $taskProjectRoot 'runtime.ps1')
$taskRuntime = Resolve-ExperimentRuntime -ExplicitRoot $RuntimeRoot
if ($CheckOnly) { Write-Output "Runtime paths verified: $taskRuntime"; return }
$taskPython = Join-Path $taskRuntime 'venv\Scripts\python.exe'
$taskRunDir = Join-Path $taskProjectRoot ('runs\' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
& $taskPython -X utf8 -u (Join-Path $taskProjectRoot 'pursuit.py') --runtime-root $taskRuntime --spec $Spec --output $taskRunDir --rounds $Rounds
if ($LASTEXITCODE -ne 0) { throw "Scenario validation failed. Inspect $taskRunDir" }
Write-Output "Results saved to: $taskRunDir"
