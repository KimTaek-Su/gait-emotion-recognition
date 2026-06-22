param(
    [switch]$Strict
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$renderYaml = Join-Path $repoRoot "render.yaml"
$dockerfile = Join-Path $repoRoot "Dockerfile"
$requirements = Join-Path $repoRoot "requirements.txt"
$modelPath = Join-Path $repoRoot "models\deployment\gait_emotion_api_model.joblib"
$deployScope = @(
    ".gitignore",
    "README.md",
    "Dockerfile",
    "render.yaml",
    "requirements.txt",
    "src/main.py",
    "frontend/index.html",
    "frontend/app.js",
    "models/deployment/gait_emotion_api_model.joblib",
    "scripts/start_local_web.cmd",
    "scripts/start_local_web.ps1",
    "scripts/start_public_tunnel.cmd",
    "scripts/start_public_tunnel.ps1",
    "scripts/render_preflight.cmd",
    "scripts/render_preflight.ps1"
)

$checks = New-Object System.Collections.Generic.List[pscustomobject]

function Add-Check {
    param(
        [string]$Name,
        [bool]$Passed,
        [string]$Detail,
        [bool]$Required = $true
    )

    $checks.Add([pscustomobject]@{
        Name = $Name
        Passed = $Passed
        Detail = $Detail
        Required = $Required
    }) | Out-Null
}

Add-Check -Name "render.yaml exists" -Passed (Test-Path $renderYaml) -Detail $renderYaml
Add-Check -Name "Dockerfile exists" -Passed (Test-Path $dockerfile) -Detail $dockerfile
Add-Check -Name "requirements.txt exists" -Passed (Test-Path $requirements) -Detail $requirements

if (Test-Path $modelPath) {
    $modelFile = Get-Item $modelPath
    Add-Check -Name "deployment model exists" -Passed ($modelFile.Length -gt 0) -Detail ("{0} bytes" -f $modelFile.Length)
} else {
    Add-Check -Name "deployment model exists" -Passed $false -Detail $modelPath
}

$remoteUrl = ""
try {
    Push-Location $repoRoot
    $remoteUrl = (git remote get-url origin 2>$null)
    $statusLines = @(git status --short 2>$null)
    $deployStatusLines = @(git status --short -- $deployScope 2>$null)
    $deployUnstaged = @(git diff --name-only -- $deployScope 2>$null)
    $deployStaged = @(git diff --cached --name-only -- $deployScope 2>$null)
    Pop-Location
} catch {
    try { Pop-Location } catch {}
}

Add-Check -Name "git remote origin configured" -Passed (-not [string]::IsNullOrWhiteSpace($remoteUrl)) -Detail $remoteUrl

if ($null -eq $statusLines) {
    Add-Check -Name "git working tree readable" -Passed $false -Detail "git status failed"
} else {
    $workspaceClean = $statusLines.Count -eq 0
    $workspaceDetail = if ($workspaceClean) { "clean" } else { ($statusLines | Select-Object -First 10) -join "; " }
    Add-Check -Name "git working tree clean" -Passed $workspaceClean -Detail $workspaceDetail -Required $false
}

if ($null -eq $deployStatusLines) {
    Add-Check -Name "deploy scope git status readable" -Passed $false -Detail "git status for deploy scope failed"
} else {
    $deployClean = $deployStatusLines.Count -eq 0
    $deployDetail = if ($deployClean) { "clean" } else { ($deployStatusLines | Select-Object -First 15) -join "; " }
    Add-Check -Name "deploy scope tracked changes isolated" -Passed ($deployUnstaged.Count -eq 0) -Detail $deployDetail

    $deployReadyDetail = if ($deployStaged.Count -gt 0) {
        "staged: " + ($deployStaged -join ", ")
    } elseif ($deployClean) {
        "clean"
    } else {
        "changes exist but are not staged"
    }
    Add-Check -Name "deploy scope staged or committed" -Passed ($deployClean -or $deployStaged.Count -gt 0) -Detail $deployReadyDetail
}

$failed = $checks | Where-Object { -not $_.Passed -and $_.Required }

$checks | Select-Object Name, Passed, Detail, Required | Format-Table -AutoSize | Out-String | Write-Host

if ($failed.Count -eq 0) {
    Write-Host "Render preflight passed for the deployment scope. Next step: push the staged deployment commit and create a Render Blueprint deployment." -ForegroundColor Green
    exit 0
}

if ($Strict) {
    throw "Render preflight failed."
}

Write-Host "Render preflight found issues. Review the failed checks above before deploying." -ForegroundColor Yellow