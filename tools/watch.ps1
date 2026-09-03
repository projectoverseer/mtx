<#
.SYNOPSIS
    Watch a long-running mtx job: progress, rate, ETA, and what it is finding.

.DESCRIPTION
    The long jobs -- transcribe, scan, enrich -- print one line per item in the
    shape "[tool] [i/n] status: what happened".  That is fine to read after the
    fact and useless to watch: 1,321 lines scroll past and none of them says
    how far along the run is or when it will end.

    This reads the same log and answers the three questions worth asking of a
    job that takes a night: how far, how fast, and is it still moving.  The
    last one matters most -- a stalled run and a slow run look identical in a
    tail, and this one says which it is.

    Read-only.  It never touches the job, so closing it, or running six of
    them, costs nothing.

.PARAMETER Log
    The log to read.  Defaults to the transcription log in %TEMP%.

.PARAMETER Interval
    Seconds between refreshes.  Default 10.

.PARAMETER Tail
    How many recent lines to show.  Default 8.

.PARAMETER Once
    Print one snapshot and exit, instead of refreshing.  Use this from a
    script, or when you just want the number.

.PARAMETER Failures
    Show the items that failed rather than the most recent ones.

.EXAMPLE
    .\tools\watch.ps1
    Watch the transcription run.

.EXAMPLE
    .\tools\watch.ps1 -Once
    One snapshot, for when you only want to know the number.

.EXAMPLE
    .\tools\watch.ps1 -Log $env:TEMP\scan.log -Interval 30
    Watch a scan instead, refreshing every half minute.
#>
[CmdletBinding()]
param(
    [string]$Log = (Join-Path $env:TEMP 'transcribe.log'),
    [int]$Interval = 10,
    [int]$Tail = 8,
    [switch]$Once,
    [switch]$Failures
)

$ErrorActionPreference = 'Stop'

# One line per item, from every mtx tool that reports progress:
#   [transcribe] [453/1321] ok: Drake\ICEMAN [E]\01. Make Them Cry -- 945 words
#   [scan] [12/40] 03. Whisper My Name.flac
# The status group is optional because scan does not have one.
$ProgressLine = '^\[(?<tool>[a-z_]+)\]\s+\[(?<i>\d+)/(?<n>\d+)\]\s*(?:(?<status>[a-z]+):)?\s*(?<rest>.*)$'

# Green for a result, yellow for a result that is thin, red for no result at
# all.  Skips are grey because a skip is the job declining to spend time, not
# a problem.
$COLOUR = @{
    ok = 'Green'; thin = 'Yellow'; skip = 'DarkGray'
    fail = 'Red'; error = 'Red'
}

function Format-Span {
    param([double]$Seconds)
    if ($Seconds -lt 0 -or [double]::IsNaN($Seconds) -or [double]::IsInfinity($Seconds)) {
        return '?'
    }
    $ts = [TimeSpan]::FromSeconds([Math]::Round($Seconds))
    if ($ts.TotalDays -ge 1) { return ('{0}d {1}h' -f [int]$ts.TotalDays, $ts.Hours) }
    if ($ts.TotalHours -ge 1) { return ('{0}h {1:00}m' -f [int]$ts.TotalHours, $ts.Minutes) }
    if ($ts.TotalMinutes -ge 1) { return ('{0}m {1:00}s' -f [int]$ts.TotalMinutes, $ts.Seconds) }
    return ('{0:0}s' -f $ts.TotalSeconds)
}

function Get-JobProcess {
    # The python process running this tool, if it is still running.  Its start
    # time is the only honest clock: the log's own timestamps are the file
    # system's, and a resumed run would make them lie about the rate.
    param([string]$Tool)
    if (-not $Tool) { return $null }
    try {
        return Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction Stop |
            Where-Object { $_.CommandLine -and $_.CommandLine -like "*$Tool*" } |
            Sort-Object CreationDate |
            Select-Object -First 1
    } catch {
        return $null
    }
}

function Get-Gpu {
    $exe = (Get-Command nvidia-smi -ErrorAction SilentlyContinue).Source
    if (-not $exe) {
        $fallback = Join-Path $env:SystemRoot 'System32\nvidia-smi.exe'
        if (Test-Path -LiteralPath $fallback) { $exe = $fallback }
    }
    if (-not $exe) { return $null }
    try {
        # No 2>$null here: in PowerShell 5.1 redirecting a native command's
        # stderr wraps every line in an ErrorRecord and, under
        # ErrorActionPreference Stop, turns a clean run into a thrown error.
        $row = & $exe --query-gpu=name,utilization.gpu,memory.used,memory.total `
                      --format=csv,noheader,nounits | Select-Object -First 1
    } catch { return $null }
    if (-not $row) { return $null }
    $f = $row -split '\s*,\s*'
    if ($f.Count -lt 4) { return $null }
    return [pscustomobject]@{
        Name = $f[0]; Util = [int]$f[1]; Used = [int]$f[2]; Total = [int]$f[3]
    }
}

function Read-Progress {
    param([string]$Path)

    # -ReadCount 0 reads the file in one gulp, which matters when this runs
    # every ten seconds against a log that grows all night.
    $lines = Get-Content -LiteralPath $Path -ReadCount 0 -ErrorAction Stop

    $counts = [ordered]@{ ok = 0; thin = 0; skip = 0; fail = 0; error = 0 }
    $tool = ''; $i = 0; $n = 0
    $recent = New-Object System.Collections.ArrayList
    $failed = New-Object System.Collections.ArrayList
    $finished = $null

    foreach ($line in $lines) {
        $m = [regex]::Match($line, $ProgressLine)
        if (-not $m.Success) {
            if ($line -match '^\[(?<tool>[a-z_]+)\]\s+done:\s*(?<rest>.+)$') {
                $finished = $Matches['rest']
            }
            continue
        }
        $tool = $m.Groups['tool'].Value
        $i = [int]$m.Groups['i'].Value
        $n = [int]$m.Groups['n'].Value
        $status = $m.Groups['status'].Value
        if ($status) {
            if (-not $counts.Contains($status)) { $counts[$status] = 0 }
            $counts[$status] = $counts[$status] + 1
        }
        $entry = [pscustomobject]@{
            Index = $i; Status = $status; Text = $m.Groups['rest'].Value
        }
        [void]$recent.Add($entry)
        if ($status -eq 'fail' -or $status -eq 'error') { [void]$failed.Add($entry) }
    }

    return [pscustomobject]@{
        Tool = $tool; Index = $i; Total = $n; Counts = $counts
        Recent = $recent; Failed = $failed; Finished = $finished
        Modified = (Get-Item -LiteralPath $Path).LastWriteTime
    }
}

function Write-Bar {
    param([double]$Fraction, [int]$Width = 44)
    $done = [Math]::Max(0, [Math]::Min($Width, [int][Math]::Round($Fraction * $Width)))
    Write-Host '  [' -NoNewline
    Write-Host ('#' * $done) -NoNewline -ForegroundColor Cyan
    Write-Host ('-' * ($Width - $done)) -NoNewline -ForegroundColor DarkGray
    Write-Host ']' -NoNewline
}

function Write-Snapshot {
    param([string]$Path, [int]$TailCount, [bool]$ShowFailures)

    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Host ''
        Write-Host "  no log at $Path" -ForegroundColor Yellow
        Write-Host '  nothing is running, or it is writing somewhere else.'
        Write-Host ''
        return $false
    }

    $p = Read-Progress -Path $Path
    if (-not $p.Total) {
        Write-Host ''
        Write-Host "  $Path has no progress lines yet." -ForegroundColor DarkGray
        Write-Host ''
        return $true
    }

    $proc = Get-JobProcess -Tool $p.Tool
    $now = Get-Date
    # Items that actually cost time.  A skip is a directory read.
    $worked = $p.Counts['ok'] + $p.Counts['thin'] + $p.Counts['fail'] + $p.Counts['error']

    if ($proc) {
        $elapsed = ($now - $proc.CreationDate).TotalSeconds
    } else {
        $elapsed = ($p.Modified - (Get-Item -LiteralPath $Path).CreationTime).TotalSeconds
    }
    $rate = 0.0
    if ($elapsed -gt 0 -and $worked -gt 0) { $rate = $worked / $elapsed }

    $left = $p.Total - $p.Index
    $eta = -1.0
    if ($rate -gt 0) { $eta = $left / $rate }

    Write-Host ''
    Write-Host ('  mtx {0}' -f $p.Tool) -NoNewline -ForegroundColor White
    if ($p.Finished) {
        Write-Host '   finished' -ForegroundColor Green
    } elseif ($proc) {
        Write-Host ('   running, pid {0}' -f $proc.ProcessId) -ForegroundColor Cyan
    } else {
        Write-Host '   not running' -ForegroundColor Yellow
    }

    Write-Bar -Fraction ($p.Index / [double]$p.Total)
    Write-Host ('  {0,5:0.0}%   {1} / {2}' -f
        (100.0 * $p.Index / $p.Total), $p.Index, $p.Total)
    Write-Host ''

    Write-Host '  ' -NoNewline
    foreach ($k in $p.Counts.Keys) {
        if (-not $p.Counts[$k]) { continue }
        $c = $COLOUR[$k]
        if (-not $c) { $c = 'Gray' }
        Write-Host ('{0} {1}   ' -f $k, $p.Counts[$k]) -NoNewline -ForegroundColor $c
    }
    Write-Host ''
    Write-Host ''

    if ($p.Finished) {
        Write-Host ('  {0}' -f $p.Finished) -ForegroundColor Green
    } else {
        Write-Host ('  rate     {0,6:0.0} / min' -f ($rate * 60))
        Write-Host ('  elapsed  {0}' -f (Format-Span $elapsed))
        if ($eta -ge 0) {
            Write-Host ('  left     {0}   ({1} to go, done about {2:HH:mm} {3})' -f
                (Format-Span $eta), $left, $now.AddSeconds($eta),
                $now.AddSeconds($eta).ToString('ddd'))
        }
    }

    # A stalled job and a slow job look the same in a tail.  This is the
    # difference: the log stops growing while the process is still there.
    $quiet = ($now - $p.Modified).TotalSeconds
    $line = '  last log  {0} ago' -f (Format-Span $quiet)
    if (-not $p.Finished -and $quiet -gt 600) {
        Write-Host ($line + '   <- nothing for ten minutes; it may be stuck') -ForegroundColor Red
    } elseif (-not $p.Finished -and $quiet -gt 180) {
        Write-Host $line -ForegroundColor Yellow
    } else {
        Write-Host $line -ForegroundColor DarkGray
    }

    $gpu = Get-Gpu
    if ($gpu) {
        $pct = 0
        if ($gpu.Total) { $pct = 100 * $gpu.Used / $gpu.Total }
        $c = 'DarkGray'
        if ($pct -gt 95) { $c = 'Yellow' }
        Write-Host ('  gpu       {0}  {1}% busy, {2} / {3} MiB' -f
            $gpu.Name, $gpu.Util, $gpu.Used, $gpu.Total) -ForegroundColor $c
    }
    Write-Host ''

    $show = $p.Recent
    $label = 'recent'
    if ($ShowFailures) { $show = $p.Failed; $label = 'failures' }
    if ($show.Count) {
        Write-Host ("  $label") -ForegroundColor DarkGray
        $start = [Math]::Max(0, $show.Count - $TailCount)
        for ($k = $start; $k -lt $show.Count; $k++) {
            $e = $show[$k]
            $c = $COLOUR[$e.Status]
            if (-not $c) { $c = 'Gray' }
            $text = $e.Text
            if ($text.Length -gt 96) { $text = $text.Substring(0, 93) + '...' }
            Write-Host ('  {0,5}  ' -f $e.Index) -NoNewline -ForegroundColor DarkGray
            Write-Host ('{0,-5} ' -f $e.Status) -NoNewline -ForegroundColor $c
            Write-Host $text
        }
    } elseif ($ShowFailures) {
        Write-Host '  failures  none' -ForegroundColor Green
    }
    Write-Host ''
    return $true
}

if ($Once) {
    $found = Write-Snapshot -Path $Log -TailCount $Tail -ShowFailures $Failures.IsPresent
    # Exit 0 when there was a job to report on, 1 when there was no log at all,
    # so this is usable as a check and not only as something to look at.
    if ($found) { exit 0 } else { exit 1 }
}

Write-Host ''
Write-Host "  watching $Log -- Ctrl-C to stop watching (the job keeps running)" `
    -ForegroundColor DarkGray
try {
    while ($true) {
        Clear-Host
        [void](Write-Snapshot -Path $Log -TailCount $Tail -ShowFailures $Failures.IsPresent)
        Write-Host ('  refreshing every {0}s -- Ctrl-C stops watching, not the job' -f $Interval) `
            -ForegroundColor DarkGray
        Start-Sleep -Seconds $Interval
    }
} finally {
    Write-Host ''
}
