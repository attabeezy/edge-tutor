param(
    [string]$OutputDir = "reports/android-mnn",
    [string]$Adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $Adb)) {
    throw "adb not found at $Adb"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$out = Join-Path $repoRoot $OutputDir
New-Item -ItemType Directory -Force -Path $out | Out-Null

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logFile = Join-Path $out "android-mnn-$stamp.log"

& $Adb logcat -c
Write-Host "Capturing EdgeTutor MNN logs to $logFile"
Write-Host "Run the android-mnn scenario on device, then press Ctrl+C to stop capture."
& $Adb logcat "EdgeTutorPerf:D" "MnnEngine:D" "EdgeTutorJNI:D" "AndroidRuntime:E" "*:S" | Tee-Object -FilePath $logFile
