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
    An upper bound on worker processes, not an instruction. Do not raise it
    above the physical core count: measured throughput is flat from there to
    twice that, and only per-file service time goes up.

    The run starts fewer than this whenever memory says so, and says which
    number it chose and why. A worker holds ~824 MB before it decodes a
    sample, and a 192 kHz master decodes to 9.7 GB against a 44.1 kHz one's
    2.8 GB, so six workers are right for one album and impossible for another.
    Seeing "2 process(es), not 6" on a hi-res album is the bound working.

    This matters more than it sounds: the budget is sized against the memory
    that is free when the scan starts, so anything left open does not slow the
    run, it narrows it for the whole night. See PERFORMANCE.md.

.PARAMETER KeepStems
    Keep every separation instead of dropping it once its track is measured.
    Four uncompressed wavs a track, about 165 MB: a library wants more disk
    for its stems than for itself, so this is for an album, not a library.

.PARAMETER StopIndexers
    Stop media-library indexers before starting, and say what was stopped.
    Without this they are only reported.

    A scan reading a whole music library is exactly what wakes them. One
    measured run shared the machine with Apple Music's AMPLibraryAgent, which
    held 45.4 GB of commit -- leaving 0.8 GB for everything else -- and 2.3 of
    6 cores, continuously, for the whole night. The scan showed no error for
    it: just MemoryErrors on 54 MiB allocations and a third of the expected
    throughput. They are COM-activated and relaunch on demand, so stopping one
    costs nothing but the indexing pass it was in.

.PARAMETER AllowSlowCpu
    Scan even though this machine is measurably slower than it has been.

    tools\bench_cpu.py records the best single-core FFT rate this machine has
    ever shown and compares against it. Well below that is a clamped chip, not
    expensive audio, and the two are indistinguishable from inside a scan: one
    morning here went to an i7-9750H pinned at 778 MHz -- 30% of nominal, on
    AC, cooler and slower than the same machine on battery an hour before --
    while the scan reported honest eighty-hour ETAs and looked broken.

.PARAMETER AllowBattery
    Run even though the machine is on battery. It refuses by default.

    A laptop on battery clocks its cores to roughly a third, and nothing about
    that looks like throttling from inside the scan -- it looks like slow code,
    on a machine that stays cold and silent while it works. Measured here at
    58% of nominal, 1.5 GHz against 3.9 all-core on AC: a 7-hour run becomes a
    20-hour one, on a battery that would not last either.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\scan_library.ps1 `
        -LibraryRoot "E:\Music" -Jobs 6 -StopIndexers
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $LibraryRoot,
    [int]    $Jobs        = 6,
    [string] $Repo        = "E:\Git\projectoverseer\mtx",
    [string] $StemsCache  = "",
    [switch] $KeepStems,
    [switch] $StopIndexers,
    [switch] $AllowBattery,
    [switch] $AllowSlowCpu
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

# Reported whether or not they are stopped: a run that quietly shares the
# machine with one of these looks like slow code, and looked like it for a
# whole night once.
$indexers = Get-Process -ErrorAction SilentlyContinue |
    Where-Object { $_.ProcessName -match "AMPLibraryAgent|iTunesLibrary|MusicBee|WMPNetworkSvc" }
foreach ($p in $indexers) {
    $gb = [math]::Round($p.PrivateMemorySize64 / 1GB, 1)
    $cpuh = [math]::Round($p.CPU / 3600, 1)
    if ($StopIndexers) {
        Write-Log ("stopping {0} (pid {1}): {2} GB committed, {3} CPU-hours" -f `
            $p.ProcessName, $p.Id, $gb, $cpuh)
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    } else {
        Write-Log ("WARNING: {0} (pid {1}) is running: {2} GB committed, {3} CPU-hours." -f `
            $p.ProcessName, $p.Id, $gb, $cpuh)
        Write-Log "         It will take memory and cores from this scan. Re-run with -StopIndexers."
    }
}

# A laptop on battery clocks its cores to roughly a third, which does not look
# like throttling from inside the scan: it looks like slow code, on a machine
# that is cold and silent while it works. Measured here at 58% of nominal --
# 1.5 GHz against 3.9 all-core on AC -- turning a 7-hour run into a 20-hour
# one, on a battery that would not last either.
$battery = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
$perf = $null
try {
    $perf = (Get-Counter '\Processor Information(_Total)\% Processor Performance' `
             -ErrorAction Stop).CounterSamples[0].CookedValue
} catch { }
if ($battery -and $battery.BatteryStatus -eq 1) {
    Write-Log ("power   : ON BATTERY ({0}%){1}" -f $battery.EstimatedChargeRemaining,
               $(if ($perf) { ", cores at {0:N0}% of nominal" -f $perf } else { "" }))
    if (-not $AllowBattery) {
        Write-Log "Plug the machine in. On battery this chip runs at about a third"
        Write-Log "of its speed, which turns a 7-hour scan into a 20-hour one and"
        Write-Log "looks like a bug rather than a power setting. Pass -AllowBattery"
        Write-Log "to run anyway."
        exit 3
    }
    Write-Log "-AllowBattery given; continuing at reduced clocks."
} elseif ($perf) {
    Write-Log ("power   : AC, cores at {0:N0}% of nominal" -f $perf)
}

# Is the machine running at the speed it usually does? A clamped CPU is
# indistinguishable from expensive audio from inside the scan -- both look
# like low utilisation and long tracks -- and a morning went to exactly that
# confusion, with this chip pinned at 778 MHz while the scan reported honest
# eighty-hour ETAs. Two seconds here against seven hours of that.
$bench = Join-Path $Repo "tools\bench_cpu.py"
if (Test-Path -LiteralPath $bench) {
    $py = Join-Path $Repo ".venv\Scripts\python.exe"
    $prev2 = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $out = & $py $bench --json 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev2
    $line = ($out | Where-Object { $_ -match '^\s*\{' } | Select-Object -First 1)
    if ($line) {
        $b = $line | ConvertFrom-Json
        Write-Log ("cpu     : {0} FFT/s, {1:P0} of this machine's best ({2})" -f `
            $b.now, $b.ratio, $b.best)
        if ($code -eq 4 -and -not $AllowSlowCpu) {
            Write-Log "The machine is well below its own best speed. That is not the"
            Write-Log "audio and not the scan: check the charger first, then a full"
            Write-Log "power-drain reset, then cooling. Run tools\bench_cpu.py after"
            Write-Log "each step. Pass -AllowSlowCpu to scan anyway."
            exit 4
        }
    }
}

$os = Get-CimInstance Win32_OperatingSystem
Write-Log ("memory  : {0:N1} GB free of {1:N1} GB; {2:N1} GB commit free" -f `
    ($os.FreePhysicalMemory / 1MB), ($os.TotalVisibleMemorySize / 1MB), `
    ($os.FreeVirtualMemory / 1MB))

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
#
# The tee is written by hand because Tee-Object has no -Encoding before
# PowerShell 6, and without one it writes UTF-16 into a log the rest of this
# script writes as UTF-8. A StreamWriter takes the encoding, and AutoFlush
# keeps the log current enough to watch a running scan from another window.
$previous = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$writer = New-Object System.IO.StreamWriter(
    $log, $true, (New-Object System.Text.UTF8Encoding($false)))
$writer.AutoFlush = $true
try {
    & $mtx @arguments 2>&1 |
        ForEach-Object {
            $line = "$_"
            Write-Host $line
            $writer.WriteLine($line)
        }
    $code = $LASTEXITCODE
} finally {
    $writer.Close()
    $ErrorActionPreference = $previous
}

$elapsed = (Get-Date) - $started
Write-Log ("done in {0:hh\:mm\:ss}, exit {1}, {2} GB free" -f `
    $elapsed, $code, (Get-FreeGB $LibraryRoot))
if ($code -ne 0) {
    Write-Log "mtx exited non-zero; re-running skips everything already measured"
}
Write-Log ("summary: {0}" -f (Join-Path $LibraryRoot "_mtx_out\summary.csv"))
