<#
.SYNOPSIS
    Measure a whole library overnight without filling the disk.

.DESCRIPTION
    `mtx scan` separates every track in its todo list before it measures any of
    them, and never evicts a stem afterwards. At about 165 MB a track that is
    fine for an album and impossible for a library: 1274 tracks want 210 GB of
    cache, and a run that big dies partway through with the disk full.

    This walks the library one artist at a time, so the cache only ever holds
    one artist's stems, and prunes the ones already measured before moving on.
    Peak usage becomes the largest single artist rather than the whole library.

    It is resumable. `mtx scan` skips tracks that already have a receipt, so
    re-running after a crash, a reboot or a Ctrl-C picks up where it stopped.
    Nothing is recomputed.

.PARAMETER LibraryRoot
    The library. Its immediate subdirectories are treated as the batches.

.PARAMETER Jobs
    Worker processes. Do not raise this above the physical core count: measured
    throughput is flat from there to twice that, and only per-file service time
    goes up. See PERFORMANCE.md.

.PARAMETER MinFreeGB
    Stop before an artist that would risk filling the disk, rather than dying
    halfway through one.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\scan_library.ps1 `
        -LibraryRoot "E:\Music" -Jobs 6
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $LibraryRoot,
    [int]    $Jobs        = 6,
    [int]    $MinFreeGB   = 25,
    [string] $Repo        = "E:\Git\projectoverseer\mtx",
    [string] $StemsCache  = "",
    [switch] $KeepStems
)

$ErrorActionPreference = "Stop"

$mtx    = Join-Path $Repo ".venv\Scripts\mtx.exe"
$python = Join-Path $Repo ".venv\Scripts\python.exe"
$pruner = Join-Path $Repo "tools\prune_stems.py"

foreach ($p in @($mtx, $python, $pruner)) {
    if (-not (Test-Path -LiteralPath $p)) { throw "not found: $p" }
}
if (-not (Test-Path -LiteralPath $LibraryRoot)) { throw "no library at $LibraryRoot" }

if ($StemsCache -eq "") { $StemsCache = Join-Path $LibraryRoot "_mtx_stems" }
$env:MTX_STEMS_CACHE = $StemsCache
$outDir = Join-Path $LibraryRoot "_mtx_out"
$log    = Join-Path $LibraryRoot ("_mtx_scan_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

function Write-Log([string] $Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -LiteralPath $log -Value $line -Encoding utf8
}

function Get-FreeGB([string] $Path) {
    $qualifier = (Get-Item -LiteralPath $Path).PSDrive.Name
    return [math]::Round((Get-PSDrive $qualifier).Free / 1GB, 1)
}

function Invoke-Native {
    <#
        Run a console program, show its output and log it, and report only a
        real failure.

        `mtx` writes its progress to stderr. Under PowerShell 5.1 a native
        program's stderr arrives as ErrorRecord objects, and with
        $ErrorActionPreference = "Stop" the very first progress line becomes a
        terminating NativeCommandError -- so a perfectly healthy scan looks
        like an immediate crash. Preference is dropped to Continue for the
        call, each record is flattened to a string so it logs and prints as
        ordinary text, and the exit code is the only thing trusted to say
        whether the program actually failed.
    #>
    param([string] $Exe, [string[]] $Arguments, [string] $LogFile)

    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Exe @Arguments 2>&1 |
            ForEach-Object { "$_" } |
            Tee-Object -FilePath $LogFile -Append |
            Out-Host
        return $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
}

# -LiteralPath throughout: PowerShell reads [ and ] in a -Path as a wildcard
# class, and album folders are full of them ("Play (Deluxe) [E]"). A -Path here
# silently matches nothing and the artist is skipped with no error at all.
$artists = Get-ChildItem -LiteralPath $LibraryRoot -Directory |
    Where-Object { $_.Name -notlike "_mtx*" } |
    Sort-Object Name

Write-Log ("library   : {0}" -f $LibraryRoot)
Write-Log ("artists   : {0}" -f $artists.Count)
Write-Log ("jobs      : {0}" -f $Jobs)
Write-Log ("cache     : {0}" -f $StemsCache)
Write-Log ("out       : {0}" -f $outDir)
Write-Log ("free      : {0} GB" -f (Get-FreeGB $LibraryRoot))
Write-Log ("log       : {0}" -f $log)

$n = 0
$started = Get-Date
foreach ($artist in $artists) {
    # Counted only once the artist is actually going to be scanned, so the
    # closing tally reports work done rather than artists looked at.
    $free = Get-FreeGB $LibraryRoot
    if ($free -lt $MinFreeGB) {
        Write-Log ("STOPPING: {0} GB free is under the {1} GB floor" -f $free, $MinFreeGB)
        Write-Log "re-run this script after making room; measured tracks are skipped"
        break
    }
    $n++

    Write-Log ("=== [{0}/{1}] {2}  ({3} GB free)" -f $n, $artists.Count, $artist.Name, $free)
    $code = Invoke-Native -Exe $mtx -LogFile $log -Arguments @(
        "scan", $artist.FullName, "--profile", "full", "--stems", "-j", "$Jobs")
    if ($code -ne 0) {
        Write-Log ("mtx exited {0} on {1}; continuing to the next artist" -f `
            $code, $artist.Name)
    }

    if (-not $KeepStems) {
        # Only entries whose track already has a corpus row are removed, so an
        # interrupted artist keeps the separations it has not yet measured.
        $null = Invoke-Native -Exe $python -LogFile $log -Arguments @(
            $pruner, $outDir, "--apply")
    }
}

$elapsed = (Get-Date) - $started
Write-Log ("done: {0} artist(s) in {1:hh\:mm\:ss}, {2} GB free" -f `
    $n, $elapsed, (Get-FreeGB $LibraryRoot))
Write-Log ("summary: {0}" -f (Join-Path $outDir "summary.csv"))
