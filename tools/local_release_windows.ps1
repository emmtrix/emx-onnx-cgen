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

# Like windows-release.yml: build with the pinned python-build-standalone CPython
# (tools/release_python.json); $PythonExe only runs the download script.
$releasePythonDir = Join-Path $repoRoot "build/release-python"
$releaseVenv = Join-Path $repoRoot "build/release-venv"
$releasePython = (& $PythonExe tools/fetch_release_python.py --platform windows-amd64 --dest $releasePythonDir)
if ($LASTEXITCODE -ne 0) {
    throw "Downloading the release Python failed with exit code ${LASTEXITCODE}."
}
if (Test-Path $releaseVenv) {
    Remove-Item $releaseVenv -Recurse -Force
}
Invoke-CheckedCommand -Description "Create release virtual environment" -Command @($releasePython.Trim(), "-m", "venv", $releaseVenv)
$pythonCommand = @(Join-Path $releaseVenv "Scripts/python.exe")

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

    $outputSbom = $OutputZip -replace '\.zip$', '.cdx.json'
    $version = (& $pythonCommand[0] -m setuptools_scm).Trim()
    Invoke-CheckedCommand -Description "Generate SBOM" -Command (New-PythonCommand @(
        "tools/generate_sbom.py", "bundle",
        "--workpath", "build/emx-onnx-cgen",
        "--dist", "dist/emx-onnx-cgen",
        "--version", $version,
        "--platform", "windows-amd64",
        "--artifact", $OutputZip,
        "--output", $outputSbom,
        "--python-build", "tools/release_python.json"
    ))
}
finally {
    if (Test-Path $tempRequirements) {
        Remove-Item $tempRequirements -Force
    }
}
