function Resolve-ExperimentRuntime {
    param([string]$ExplicitRoot)
    $taskRoot = $ExplicitRoot
    if ([string]::IsNullOrWhiteSpace($taskRoot)) { $taskRoot = $env:OPENRA_RUNTIME_ROOT }
    if ([string]::IsNullOrWhiteSpace($taskRoot)) {
        $taskPointer = Join-Path $PSScriptRoot '.runtime-path'
        if (Test-Path -LiteralPath $taskPointer) { $taskRoot = (Get-Content -Raw -LiteralPath $taskPointer).Trim() }
    }
    if ([string]::IsNullOrWhiteSpace($taskRoot)) { $taskRoot = Join-Path $PSScriptRoot '.runtime' }
    if (!(Test-Path -LiteralPath $taskRoot -PathType Container)) { throw 'Runtime not found. Follow SETUP.md or supply -RuntimeRoot.' }
    $taskRoot = (Resolve-Path -LiteralPath $taskRoot).Path
    foreach ($taskPart in @('venv/Scripts/python.exe','source/openra_env','source/OpenRA/bin/OpenRA.Server.dll','source/OpenRA/mods/ra/maps/singles.oramap')) {
        if (!(Test-Path -LiteralPath (Join-Path $taskRoot $taskPart))) { throw "Runtime is missing $taskPart. See SETUP.md." }
    }
    if (!(Test-Path -LiteralPath (Join-Path $taskRoot 'dotnet/dotnet.exe')) -and !(Get-Command dotnet -ErrorAction SilentlyContinue)) {
        throw '.NET 8 runtime not found: install it or provide runtime/dotnet/dotnet.exe.'
    }
    return $taskRoot
}
