<#
.SYNOPSIS
    Measure a whole library in one pass, with a log that survives the night.

.DESCRIPTION
    `mtx scan` handles a library in one run now: separation on the card
    overlaps the measuring on the cores, and `--prune-stems` drops each
    track's stems once its measurement is written, so the cache never grows
    past the handful of tracks in flight. This is only a launcher -- it exists
    for three things PowerShell 5.1 gets wrong on its own.

    First, `mtx` writes progress to stderr, and a native program's stderr
    arrives here as ErrorRecord objects; with $ErrorActionPreference = "Stop"
    the first progress line becomes a terminating NativeCommandError and a
    healthy scan looks like an instant crash. Second, Tee-Object writes
    UTF-16, so a log written that way comes back full of NUL bytes and grep
    finds nothing in it. Third, a long run wants a timestamped log at all.

    It is resumable. `mtx scan` skips tracks that already have a receipt, so
    re-running after a crash, a reboot or a Ctrl-C picks up where it stopped
    and recomputes nothing.

.PARAMETER LibraryRoot
    The library to measure.

.PARAMETER Jobs
    Worker processes. Do not raise this above the physical core count:
    measured throughput is flat from there to twice that, and only per-file
    service time goes up. The pool narrows itself below this on long or
    high-rate tracks, which is a memory bound and not a core one. See
    PERFORMANCE.md.

.PARAMETER KeepStems
    Keep every separation instead of dropping it once its track is measured.
    Four uncompressed wavs a track, about 165 MB: a library wants more disk
    for its stems than for itself, so this is for an album, not a library.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\scan_library.ps1 `
        -LibraryRoot "E:\Music" -Jobs 6
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $LibraryRoot,
    [int]    $Jobs        = 6,
    [string] $Repo        = "E:\Git\projectoverseer\mtx",
    [string] $StemsCache  = "",
    [switch] $KeepStems
)

$ErrorActionPreference = "Stop"

$mtx = Join-Path $Repo ".venv\Scripts\mtx.exe"
if (-not (Test-Path -LiteralPath $mtx)) { throw "not found: $mtx" }
if (-not (Test-Path -LiteralPath $LibraryRoot)) { throw "no library at $LibraryRoot" }

if ($StemsCache -eq "") { $StemsCache = Join-Path $LibraryRoot "_mtx_stems" }
$env:MTX_STEMS_CACHE = $StemsCache

$log = Join-Path $LibraryRoot ("_mtx_scan_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

function Write-Log([string] $Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Message
    Write-Host $line
    # -Encoding utf8 on both writers: Add-Content and Tee-Object disagree by
    # default, and a log half UTF-8 and half UTF-16 cannot be read by anything.
    Add-Content -LiteralPath $log -Value $line -Encoding utf8
}

function Get-FreeGB([string] $Path) {
    $qualifier = (Get-Item -LiteralPath $Path).PSDrive.Name
    return [math]::Round((Get-PSDrive $qualifier).Free / 1GB, 1)
}

$arguments = @("scan", $LibraryRoot, "--profile", "full", "--stems",
               "-j", "$Jobs")
if (-not $KeepStems) { $arguments += "--prune-stems" }

Write-Log ("library : {0}" -f $LibraryRoot)
Write-Log ("jobs    : {0}" -f $Jobs)
Write-Log ("cache   : {0}" -f $StemsCache)
Write-Log ("free    : {0} GB" -f (Get-FreeGB $LibraryRoot))
Write-Log ("log     : {0}" -f $log)
Write-Log ("run     : mtx {0}" -f ($arguments -join " "))

$started = Get-Date

# Preference dropped to Continue for the call, each record flattened to a
# string so it logs as ordinary text, and the exit code trusted as the only
# thing that says whether the program actually failed.
$previous = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & $mtx @arguments 2>&1 |
        ForEach-Object { "$_" } |
        Tee-Object -FilePath $log -Append -Encoding utf8 |
        Out-Host
    $code = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previous
}

$elapsed = (Get-Date) - $started
Write-Log ("done in {0:hh\:mm\:ss}, exit {1}, {2} GB free" -f `
    $elapsed, $code, (Get-FreeGB $LibraryRoot))
if ($code -ne 0) {
    Write-Log "mtx exited non-zero; re-running skips everything already measured"
}
Write-Log ("summary: {0}" -f (Join-Path $LibraryRoot "_mtx_out\summary.csv"))
