$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $PSScriptRoot
$modelsDir = Join-Path $projectDir "data\pretrained_models"
$modelDir = Join-Path $modelsDir "vosk-model-small-vn-0.4"
$archive = Join-Path $modelsDir "vosk-model-small-vn-0.4.download.zip"
$modelUrl = "https://alphacephei.com/vosk/models/vosk-model-small-vn-0.4.zip"

if (Test-Path -LiteralPath $modelDir) {
    Write-Output "Vosk model already exists at $modelDir"
    exit 0
}

New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null
try {
    curl.exe -L --fail --output $archive $modelUrl
    if ($LASTEXITCODE -ne 0) {
        throw "Model download failed with exit code $LASTEXITCODE"
    }
    Expand-Archive -LiteralPath $archive -DestinationPath $modelsDir -Force
    Write-Output "Vosk model installed at $modelDir"
} finally {
    if (Test-Path -LiteralPath $archive) {
        Remove-Item -LiteralPath $archive -Force
    }
}
