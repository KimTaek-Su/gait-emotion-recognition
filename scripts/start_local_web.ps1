param(
    [int]$Port = 8000,
    [string]$BindHost = "127.0.0.1",
    [switch]$Reload,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvDir = Join-Path $repoRoot ".venv312"
$pythonExe = Join-Path $venvDir "Scripts\python.exe"
$requirementsPath = Join-Path $repoRoot "requirements.txt"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter()]
        [string[]]$ArgumentList = @(),

        [Parameter(Mandatory = $true)]
        [string]$Step
    )

    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

function Ensure-Venv {
    if (Test-Path $pythonExe) {
        return
    }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        Write-Host "Creating Python 3.12 virtual environment at $venvDir"
        Invoke-Checked -FilePath $pyLauncher.Source -ArgumentList @('-3.12', '-m', 'venv', $venvDir) -Step 'Creating .venv312'
        return
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        Write-Host "Creating virtual environment at $venvDir with python from PATH"
        Invoke-Checked -FilePath $pythonCommand.Source -ArgumentList @('-m', 'venv', $venvDir) -Step 'Creating .venv312'
        return
    }

    throw "Python was not found on PATH. Install Python 3.12 or the py launcher first."
}

function Test-DependenciesInstalled {
    $checkCode = @'
import importlib.util

modules = ["fastapi", "uvicorn", "joblib", "numpy", "sklearn"]
missing = [name for name in modules if importlib.util.find_spec(name) is None]
raise SystemExit(0 if not missing else 1)
'@

    & $pythonExe -c $checkCode
    return $LASTEXITCODE -eq 0
}

Ensure-Venv

if (-not $SkipInstall -and -not (Test-DependenciesInstalled)) {
    Write-Host "Installing Python dependencies from requirements.txt"
    Invoke-Checked -FilePath $pythonExe -ArgumentList @('-m', 'pip', 'install', '--upgrade', 'pip') -Step 'Upgrading pip'
    Invoke-Checked -FilePath $pythonExe -ArgumentList @('-m', 'pip', 'install', '-r', $requirementsPath) -Step 'Installing requirements'
}

Push-Location $repoRoot
try {
    $uvicornArgs = @('-m', 'uvicorn', 'src.main:app', '--host', $BindHost, '--port', "$Port")
    if ($Reload) {
        $uvicornArgs += '--reload'
    }

    Write-Host "Starting FastAPI at http://${BindHost}:${Port}/"
    Invoke-Checked -FilePath $pythonExe -ArgumentList $uvicornArgs -Step 'Starting uvicorn'
}
finally {
    Pop-Location
}