param(
    [string]$PythonExe = "python",
    [string]$OutputZip = "emx-onnx-cgen-windows-amd64.zip"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $repoRoot

# Keep these install/build steps aligned with .github/workflows/windows-release.yml.
& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -q -r requirements-ci.txt
& $PythonExe -m pip install pyinstaller

& $PythonExe tools/pyinstaller_build.py

if (Test-Path $OutputZip) {
    Remove-Item $OutputZip -Force
}

Compress-Archive -Path dist/emx-onnx-cgen/* -DestinationPath $OutputZip
Write-Host "Windows release bundle created: $repoRoot/$OutputZip"
