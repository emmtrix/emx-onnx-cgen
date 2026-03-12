param(
    [string]$PythonExe = "python.exe",
    [string]$OutputZip = "emx-onnx-cgen-windows-amd64.zip",
    [string]$RequirementsFile = "requirements-ci.txt",
    [switch]$DisableOnnxruntimeFallback
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Description,
        [Parameter(Mandatory = $true)]
        [string[]]$Command
    )

    Write-Host "==> $Description"
    $exe = $Command[0]
    $args = @()
    if ($Command.Length -gt 1) {
        $args += $Command[1..($Command.Length - 1)]
    }
    & $exe @args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $($Command -join ' ')"
    }
}

function Invoke-OptionalCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Description,
        [Parameter(Mandatory = $true)]
        [string[]]$Command
    )

    Write-Host "==> $Description"
    $exe = $Command[0]
    $args = @()
    if ($Command.Length -gt 1) {
        $args += $Command[1..($Command.Length - 1)]
    }
    & $exe @args
    return $LASTEXITCODE
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $repoRoot

if (-not (Test-Path $RequirementsFile)) {
    throw "Requirements file not found: $RequirementsFile"
}

if (-not (Get-Command $PythonExe -ErrorAction SilentlyContinue)) {
    throw "Python executable not found in PATH: $PythonExe"
}

$pythonCommand = @($PythonExe)
$pythonVersion = (& $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
if ($pythonVersion -ne "3.11") {
    $fallbackPython311 = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"
    if (Test-Path $fallbackPython311) {
        $fallbackVersion = (& $fallbackPython311 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
        if ($fallbackVersion -eq "3.11") {
            Write-Warning "Configured Python is $pythonVersion. Falling back to $fallbackPython311 to match windows-release workflow."
            $pythonCommand = @($fallbackPython311)
        }
        else {
            throw "Python 3.11 is required for local Windows release builds. Current Python is $pythonVersion and fallback at '$fallbackPython311' is $fallbackVersion. Pass a Python 3.11 interpreter via -PythonExe."
        }
    }
    else {
        throw "Python 3.11 is required for local Windows release builds. Current Python is $pythonVersion. Pass a Python 3.11 interpreter via -PythonExe."
    }
}

function New-PythonCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )

    return @($pythonCommand + $Args)
}

$requirementsContent = Get-Content $RequirementsFile
$onnxruntimeLine = $requirementsContent | Where-Object { $_ -match '^\s*onnxruntime\s*==\s*([0-9][0-9\.]*)\s*$' } | Select-Object -First 1
$requirementsWithoutOrt = $requirementsContent | Where-Object { $_ -notmatch '^\s*onnxruntime\s*([<>=!~].*)?$' }
$tempDir = [System.IO.Path]::GetTempPath()
$tempRequirements = Join-Path $tempDir "emx-onnx-cgen-requirements-no-ort.txt"
Set-Content -Path $tempRequirements -Value $requirementsWithoutOrt -Encoding Ascii

try {
    # Keep these install/build steps aligned with .github/workflows/windows-release.yml.
    Invoke-CheckedCommand -Description "Upgrade pip" -Command (New-PythonCommand @("-m", "pip", "install", "--upgrade", "pip"))
    Invoke-CheckedCommand -Description "Install CI requirements (without onnxruntime)" -Command (New-PythonCommand @("-m", "pip", "install", "-q", "-r", $tempRequirements))

    if ($onnxruntimeLine) {
        $pinnedOnnxruntime = ($onnxruntimeLine -replace '^\s*onnxruntime\s*==\s*', '').Trim()
        $ortExitCode = Invoke-OptionalCommand -Description "Install pinned onnxruntime==$pinnedOnnxruntime" -Command (New-PythonCommand @("-m", "pip", "install", "onnxruntime==$pinnedOnnxruntime"))

        if ($ortExitCode -ne 0) {
            if ($DisableOnnxruntimeFallback) {
                throw "Pinned onnxruntime==$pinnedOnnxruntime is unavailable and fallback is disabled."
            }
            Write-Warning "Pinned onnxruntime==$pinnedOnnxruntime is unavailable. Falling back to 'onnxruntime>=$pinnedOnnxruntime,<2'."
            Invoke-CheckedCommand -Description "Install fallback onnxruntime>=$pinnedOnnxruntime,<2" -Command (New-PythonCommand @("-m", "pip", "install", "onnxruntime>=$pinnedOnnxruntime,<2"))
        }
    }
    else {
        Invoke-CheckedCommand -Description "Install onnxruntime" -Command (New-PythonCommand @("-m", "pip", "install", "onnxruntime"))
    }

    Invoke-CheckedCommand -Description "Install PyInstaller" -Command (New-PythonCommand @("-m", "pip", "install", "pyinstaller"))
    Invoke-CheckedCommand -Description "Build PyInstaller bundle" -Command (New-PythonCommand @("tools/pyinstaller_build.py"))

    if (Test-Path $OutputZip) {
        Remove-Item $OutputZip -Force
    }

    Compress-Archive -Path dist/emx-onnx-cgen/* -DestinationPath $OutputZip
    Write-Host "Windows release bundle created: $repoRoot/$OutputZip"
}
finally {
    if (Test-Path $tempRequirements) {
        Remove-Item $tempRequirements -Force
    }
}
