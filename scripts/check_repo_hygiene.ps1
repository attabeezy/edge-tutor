param(
    [int]$MaxPathLength = 240,
    [int]$MaxTrackedFileSizeMB = 50
)

$ErrorActionPreference = "Stop"
$repoRoot = (Get-Location).Path
$maxBytes = $MaxTrackedFileSizeMB * 1MB

$forbiddenPatterns = @(
    '^android/app/build/',
    '^android/build/',
    '^android/.gradle/',
    '^android/.idea/',
    '^android-ltk/app/build/',
    '^android-ltk/build/',
    '^android-ltk/.gradle/',
    '^android-ltk/.idea/'
)

function Get-GitLines([string]$args) {
    $output = & git --no-pager $args 2>$null
    if (-not $output) { return @() }
    return @($output | Where-Object { $_ -and $_.Trim() -ne "" })
}

$tracked = Get-GitLines "ls-files"
$untracked = Get-GitLines "ls-files --others --exclude-standard"
$candidates = @($tracked + $untracked | Select-Object -Unique)

$violations = [System.Collections.Generic.List[string]]::new()

foreach ($path in $candidates) {
    foreach ($pattern in $forbiddenPatterns) {
        if ($path -match $pattern) {
            $violations.Add("Forbidden generated path: $path")
            break
        }
    }

    $fullPath = Join-Path $repoRoot $path
    if ((Test-Path -LiteralPath $fullPath) -and ($fullPath.Length -gt $MaxPathLength)) {
        $violations.Add("Long path ($($fullPath.Length) chars): $path")
    }
}

foreach ($path in $tracked) {
    $fullPath = Join-Path $repoRoot $path
    if (Test-Path -LiteralPath $fullPath) {
        $size = (Get-Item -LiteralPath $fullPath).Length
        if ($size -gt $maxBytes) {
            $sizeMB = [Math]::Round(($size / 1MB), 1)
            $violations.Add("Oversized tracked file (${sizeMB}MB): $path")
        }
    }
}

if ($violations.Count -gt 0) {
    Write-Host "Repo hygiene check failed:" -ForegroundColor Red
    $violations | Select-Object -Unique | ForEach-Object { Write-Host " - $_" -ForegroundColor Red }
    exit 1
}

Write-Host "Repo hygiene check passed." -ForegroundColor Green
exit 0
