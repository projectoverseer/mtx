<#
.SYNOPSIS
    Close what is competing for memory, then optionally start the scan.

.DESCRIPTION
    `mtx scan` fixes its memory budget once, at launch, from the memory that
    is free at that moment. Anything running then does not merely slow the run
    down for a minute -- it permanently narrows the pool for the whole night.
    On this machine the difference between an idle desktop and a working one
    is the sixth measuring lane, worth about 20 per cent over seven hours.

    So this closes things, in three passes:

      1. Media-library indexers, which a scan of a music library wakes by
         itself. One of them once held 45 GB of commit and 2.3 of 6 cores
         through an entire unattended run without appearing in any log.
      2. A named list of the usual large desktop applications.
      3. Anything else in your session with a visible window over -MinimumMB,
         which catches the ones nobody thought to name.

    Windows and its services are never touched, nor is anything without a
    window in pass 3, nor this script's own console. Each application is asked
    to close politely first and only forced after -GraceSeconds.

    IT WILL DISCARD UNSAVED WORK. Save first. `-DryRun` shows the list without
    closing anything, and is the right first thing to run.

.PARAMETER MinimumMB
    In pass 3, close windowed applications holding at least this much. Below
    about 200 MB there is nothing worth reclaiming.

.PARAMETER Keep
    Process names to spare, without `.exe`. Matched case-insensitively.

.PARAMETER GraceSeconds
    How long an application has to close on its own before it is forced.

.PARAMETER DryRun
    List what would be closed and stop. Nothing is touched.

.PARAMETER AndScan
    Start the library scan once the memory has settled.

.PARAMETER LibraryRoot
    Passed to scan_library.ps1 when -AndScan is given.

.PARAMETER Jobs
    Passed to scan_library.ps1 when -AndScan is given. An upper bound: the
    scan starts fewer workers if memory says so, and says which and why.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\free_memory.ps1 -DryRun

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\free_memory.ps1 `
        -AndScan -LibraryRoot "E:\Music" -Jobs 6

.NOTES
    Run this from a standalone PowerShell window, NOT from an editor's
    integrated terminal -- it closes editors, and would take its own console
    with it. It refuses to run from one.
#>
[CmdletBinding()]
param(
    [int]      $MinimumMB    = 200,
    [string[]] $Keep         = @(),
    [int]      $GraceSeconds = 8,
    [switch]   $DryRun,
    [switch]   $AndScan,
    [string]   $LibraryRoot  = "E:\Music",
    [int]      $Jobs         = 6,
    [string]   $Repo         = "E:\Git\projectoverseer\mtx"
)

$ErrorActionPreference = "Stop"

# Closing the editor would close the console this is running in, so the script
# would be killed halfway through its own list.
if ($env:TERM_PROGRAM -eq "vscode" -or $env:VSCODE_INJECTION -eq "1") {
    Write-Host ""
    Write-Host "  This is running inside VS Code's terminal." -ForegroundColor Yellow
    Write-Host "  It closes editors, so it would kill its own console halfway through."
    Write-Host ""
    Write-Host "  Open a standalone PowerShell window (Win+R, 'powershell') and run it there:"
    Write-Host "      cd $Repo"
    Write-Host "      powershell -ExecutionPolicy Bypass -File tools\free_memory.ps1 -AndScan"
    Write-Host ""
    exit 2
}

function Get-FreeGB {
    return [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)
}

# Never touched. The Windows session cannot survive without these, and
# explorer is left alone because closing the shell is disruptive out of all
# proportion to the few hundred megabytes it holds.
$protected = @(
    "System", "Idle", "Registry", "Memory Compression",
    "csrss", "wininit", "winlogon", "services", "lsass", "smss", "svchost",
    "fontdrvhost", "dwm", "explorer", "sihost", "ctfmon", "RuntimeBroker",
    "SearchHost", "StartMenuExperienceHost", "ShellExperienceHost",
    "TextInputHost", "audiodg", "conhost", "WindowsTerminal",
    "powershell", "pwsh", "cmd", "WerFault"
)

# A scan reading a whole music library is exactly what wakes these, and they
# are the reason this script exists at all.
$indexers = @("AMPLibraryAgent", "AppleMusic", "iTunes", "iTunesHelper",
              "MusicBee", "WMPNetworkSvc", "Spotify", "SearchIndexer")

# The usual large desktop applications, closed whether or not they are over
# the pass-3 threshold, because several of them idle small and grow later.
$known = @(
    "chrome", "msedge", "firefox", "brave", "opera", "vivaldi", "arc",
    "Code", "Code - Insiders", "devenv", "rider64", "idea64", "pycharm64",
    "sublime_text", "notepad++", "atom",
    "Discord", "Slack", "Teams", "ms-teams", "WhatsApp", "Telegram", "Signal",
    "zoom", "OUTLOOK", "WINWORD", "EXCEL", "POWERPNT", "onenote",
    "OneDrive", "Dropbox", "GoogleDriveFS", "MEGAsync",
    "Steam", "steamwebhelper", "EpicGamesLauncher", "Battle.net",
    "Docker Desktop", "com.docker.backend", "vmmem", "vmmemWSL",
    "obs64", "Photoshop", "Illustrator", "AfterFX", "Premiere Pro",
    "vlc", "PotPlayerMini64", "mpc-hc64", "Adobe Desktop Service", "AdobeIPCBroker",
    "GitHubDesktop", "Postman", "Insomnia", "Notion", "Obsidian", "Todoist"
)

$protected += $Keep
$protected += "mtx"          # never close a scan that is already running

# This script's own ancestry: the console it prints to, and whatever launched
# it. Killing any of them ends the script mid-list.
$mine = @()
$walk = $PID
for ($i = 0; $i -lt 8 -and $walk; $i++) {
    $mine += $walk
    $p = Get-CimInstance Win32_Process -Filter "ProcessId=$walk" -ErrorAction SilentlyContinue
    if (-not $p) { break }
    $walk = $p.ParentProcessId
}

$session = (Get-Process -Id $PID).SessionId

function Should-Close($proc) {
    if ($mine -contains $proc.Id) { return $false }
    if ($protected -contains $proc.ProcessName) { return $false }
    if ($proc.SessionId -ne $session -and $indexers -notcontains $proc.ProcessName) {
        return $false          # another user's session, or a service
    }
    if ($indexers -contains $proc.ProcessName) { return $true }
    if ($known -contains $proc.ProcessName) { return $true }
    # Pass 3: anything else this user is looking at that is big enough to matter.
    if ($proc.MainWindowHandle -ne 0 -and
        $proc.PrivateMemorySize64 -ge ($MinimumMB * 1MB)) { return $true }
    return $false
}

$before = Get-FreeGB
Write-Host ""
Write-Host ("  free before : {0} GB" -f $before) -ForegroundColor Cyan

$targets = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { Should-Close $_ } |
             Sort-Object PrivateMemorySize64 -Descending)

if (-not $targets) {
    Write-Host "  nothing worth closing is running." -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host ("  {0,-28} {1,8}  {2}" -f "process", "MB", "why")
    foreach ($p in $targets) {
        $why = "large window"
        if ($known -contains $p.ProcessName)    { $why = "known app" }
        if ($indexers -contains $p.ProcessName) { $why = "MEDIA INDEXER" }
        Write-Host ("  {0,-28} {1,8:N0}  {2}" -f $p.ProcessName,
                    ($p.PrivateMemorySize64 / 1MB), $why)
    }
    Write-Host ""
}

if ($DryRun) {
    Write-Host "  -DryRun: nothing was closed." -ForegroundColor Yellow
    exit 0
}

# Ask politely first. An application given the chance to close saves its own
# state; one that is killed does not.
foreach ($p in $targets) {
    try { if ($p.MainWindowHandle -ne 0) { [void]$p.CloseMainWindow() } } catch { }
}
if ($targets) { Start-Sleep -Seconds $GraceSeconds }

foreach ($p in $targets) {
    try {
        $still = Get-Process -Id $p.Id -ErrorAction SilentlyContinue
        if ($still) {
            Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
            Write-Host ("  forced  {0}" -f $p.ProcessName) -ForegroundColor DarkYellow
        }
    } catch { }
}

# Memory is returned to the free list lazily; give Windows a moment before
# reading the number the scan is about to size itself against.
Start-Sleep -Seconds 4
$after = Get-FreeGB
Write-Host ""
Write-Host ("  free after  : {0} GB  ({1:+0.0;-0.0;0} GB)" -f $after, ($after - $before)) -ForegroundColor Green

Write-Host ""
Write-Host "  still holding the most:"
Get-Process | Sort-Object PrivateMemorySize64 -Descending | Select-Object -First 6 |
    ForEach-Object { Write-Host ("    {0,-28} {1,8:N0} MB" -f $_.ProcessName,
                                 ($_.PrivateMemorySize64 / 1MB)) }
Write-Host ""

if ($AndScan) {
    $launcher = Join-Path $Repo "tools\scan_library.ps1"
    if (-not (Test-Path -LiteralPath $launcher)) { throw "not found: $launcher" }
    Write-Host "  starting the scan..." -ForegroundColor Cyan
    Write-Host ""
    & powershell -ExecutionPolicy Bypass -File $launcher `
        -LibraryRoot $LibraryRoot -Jobs $Jobs -StopIndexers
} else {
    Write-Host "  now run:" -ForegroundColor Cyan
    Write-Host ("    powershell -ExecutionPolicy Bypass -File {0} -LibraryRoot `"{1}`" -Jobs {2} -StopIndexers" -f `
                (Join-Path $Repo "tools\scan_library.ps1"), $LibraryRoot, $Jobs)
    Write-Host ""
}
