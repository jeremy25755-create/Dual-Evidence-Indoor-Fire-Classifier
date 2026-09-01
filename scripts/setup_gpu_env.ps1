[CmdletBinding()]
param(
    [string]$PythonExecutable = "",
    [ValidateSet("cu128", "cu130")]
    [string]$CudaWheel = "cu128"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

$venvWorks = $false
if (Test-Path -LiteralPath $venvPython) {
    $venvProbe = & $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}' if sys.prefix != sys.base_prefix else '')" 2>$null
    $venvWorks = $LASTEXITCODE -eq 0 -and $venvProbe -eq "3.12"
}

if (-not $venvWorks) {
    if (Test-Path -LiteralPath $venvPath) {
        $expectedPath = [System.IO.Path]::GetFullPath(
            (Join-Path $projectRoot ".venv")
        ).TrimEnd("\")
        $actualPath = [System.IO.Path]::GetFullPath($venvPath).TrimEnd("\")
        if ($actualPath -ne $expectedPath -or (Split-Path -Leaf $actualPath) -ne ".venv") {
            throw "Refusing to remove unexpected environment path: $actualPath"
        }
        Remove-Item -LiteralPath $actualPath -Recurse -Force
    }

    if ($PythonExecutable) {
        if (-not (Test-Path -LiteralPath $PythonExecutable)) {
            throw "Python executable was not found: $PythonExecutable"
        }
        & $PythonExecutable -m venv $venvPath
    }
    else {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        $pyCommand = Get-Command py -ErrorAction SilentlyContinue
        if ($pythonCommand) {
            & $pythonCommand.Source -m venv $venvPath
        }
        elseif ($pyCommand) {
            & $pyCommand.Source -3.12 -m venv $venvPath
        }
        else {
            throw "Python 3.12 is required. Pass its path with -PythonExecutable."
        }
    }
}

& $venvPython -m pip install --upgrade pip "setuptools<82" wheel
$expectedCuda = if ($CudaWheel -eq "cu130") { "13.0" } else { "12.8" }
$installedCuda = & $venvPython -c "import importlib.util; print('' if importlib.util.find_spec('torch') is None else (__import__('torch').version.cuda or ''))"
if ([string]::IsNullOrWhiteSpace($installedCuda) -or $installedCuda.Trim() -ne $expectedCuda) {
    & $venvPython -m pip uninstall -y torch torchvision torchaudio
    & $venvPython -m pip install torch --index-url "https://download.pytorch.org/whl/$CudaWheel"
}
else {
    Write-Output "CUDA PyTorch is already installed; skipping the wheel download."
}
& $venvPython -m pip install -r (Join-Path $projectRoot "requirements.txt")
& $venvPython -m pip install -r (Join-Path $projectRoot "requirements-dev.txt")
& $venvPython (Join-Path $PSScriptRoot "check_gpu_environment.py")

Write-Output ""
Write-Output "TEFN GPU environment is ready."
Write-Output "Allow activation: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass"
Write-Output "Activate: & '$venvPath\Scripts\Activate.ps1'"
