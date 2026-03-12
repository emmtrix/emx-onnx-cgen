param(
    [string]$PythonExe = "python",
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
    & $Command[0] $Command[1..($Command.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE: $($Command -join ' ')"
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
    & $Command[0] $Command[1..($Command.Length - 1)]
    return $LASTEXITCODE
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $repoRoot

if (-not (Test-Path $RequirementsFile)) {
    throw "Requirements file not found: $RequirementsFile"
}

$requirementsContent = Get-Content $RequirementsFile
$onnxruntimeLine = $requirementsContent | Where-Object { $_ -match '^\s*onnxruntime\s*==\s*([0-9][0-9\.]*)\s*$' } | Select-Object -First 1
$requirementsWithoutOrt = $requirementsContent | Where-Object { $_ -notmatch '^\s*onnxruntime\s*([<>=!~].*)?$' }
$tempRequirements = Join-Path $env:TEMP "emx-onnx-cgen-requirements-no-ort.txt"
Set-Content -Path $tempRequirements -Value $requirementsWithoutOrt

try {
    # Keep these install/build steps aligned with .github/workflows/windows-release.yml.
    Invoke-CheckedCommand -Description "Upgrade pip" -Command @($PythonExe, "-m", "pip", "install", "--upgrade", "pip")
    Invoke-CheckedCommand -Description "Install CI requirements (without onnxruntime)" -Command @($PythonExe, "-m", "pip", "install", "-q", "-r", $tempRequirements)

    if ($onnxruntimeLine) {
        $pinnedOnnxruntime = ($onnxruntimeLine -replace '^\s*onnxruntime\s*==\s*', '').Trim()
        $ortExitCode = Invoke-OptionalCommand -Description "Install pinned onnxruntime==$pinnedOnnxruntime" -Command @($PythonExe, "-m", "pip", "install", "onnxruntime==$pinnedOnnxruntime")

        if ($ortExitCode -ne 0) {
            if ($DisableOnnxruntimeFallback) {
                throw "Pinned onnxruntime==$pinnedOnnxruntime is unavailable and fallback is disabled."
            }
            Write-Warning "Pinned onnxruntime==$pinnedOnnxruntime is unavailable. Falling back to 'onnxruntime>=$pinnedOnnxruntime,<2'."
            Invoke-CheckedCommand -Description "Install fallback onnxruntime>=$pinnedOnnxruntime,<2" -Command @($PythonExe, "-m", "pip", "install", "onnxruntime>=$pinnedOnnxruntime,<2")
        }
    }
    else {
        Invoke-CheckedCommand -Description "Install onnxruntime" -Command @($PythonExe, "-m", "pip", "install", "onnxruntime")
    }

    Invoke-CheckedCommand -Description "Install PyInstaller" -Command @($PythonExe, "-m", "pip", "install", "pyinstaller")
    Invoke-CheckedCommand -Description "Build PyInstaller bundle" -Command @($PythonExe, "tools/pyinstaller_build.py")

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
