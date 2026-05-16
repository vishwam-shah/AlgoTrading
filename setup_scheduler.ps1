# setup_scheduler.ps1
# Registers Windows Scheduled Tasks for the AlgoTrading daily pipeline.
# Run once as Administrator (right-click PowerShell -> "Run as Administrator").
#
# Three tasks:
#   1. Evening (6:00 PM IST, Mon-Fri): pipeline + signal publisher
#   2. Morning (9:00 AM IST, Mon-Fri): exit runner + order execution
#   3. Reconcile (3:45 PM IST, Mon-Fri): P&L reconciliation
#
# IST = UTC+5:30, so Windows Task Scheduler uses IST times directly
# (Windows Scheduler uses local system time).

$PYTHON    = "C:\Users\Home\AppData\Local\Programs\Python\Python313\python.exe"
$WORKDIR   = "C:\Users\Home\Documents\AI_IN_STOCK_V2"
$RUNNER    = "V3\05_live_trading\daily_runner.py"
$LOGDIR    = "C:\Users\Home\Documents\AI_IN_STOCK_V2\V3\05_live_trading\logs"

# Create log directory
New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null

# ── Task 1: Evening run (6:00 PM IST, Mon-Fri) ───────────────────────────────

$action1 = New-ScheduledTaskAction `
    -Execute $PYTHON `
    -Argument "$RUNNER --mode evening --capital 500000" `
    -WorkingDirectory $WORKDIR

$trigger1 = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At "18:00"

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -StartWhenAvailable `
    -WakeToRun $false `
    -DisallowStartIfOnBatteries $false `
    -StopIfGoingOnBatteries $false

Register-ScheduledTask `
    -TaskName "AlgoTrading_Evening" `
    -Description "Evening pipeline: fetch data, run models, generate orders (6 PM IST Mon-Fri)" `
    -Action $action1 `
    -Trigger $trigger1 `
    -Settings $settings `
    -RunLevel Highest `
    -Force

Write-Host "[ok] Registered: AlgoTrading_Evening (6:00 PM Mon-Fri)"

# ── Task 2: Morning run (9:00 AM IST, Mon-Fri) ───────────────────────────────

$action2 = New-ScheduledTaskAction `
    -Execute $PYTHON `
    -Argument "$RUNNER --mode morning --capital 500000" `
    -WorkingDirectory $WORKDIR

$trigger2 = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At "09:00"

Register-ScheduledTask `
    -TaskName "AlgoTrading_Morning" `
    -Description "Morning: exit stale lots + paper execute today's buy orders (9:00 AM IST Mon-Fri)" `
    -Action $action2 `
    -Trigger $trigger2 `
    -Settings $settings `
    -RunLevel Highest `
    -Force

Write-Host "[ok] Registered: AlgoTrading_Morning (9:00 AM Mon-Fri)"

# ── Task 3: Reconcile run (3:45 PM IST, Mon-Fri) ─────────────────────────────

$action3 = New-ScheduledTaskAction `
    -Execute $PYTHON `
    -Argument "$RUNNER --mode reconcile" `
    -WorkingDirectory $WORKDIR

$trigger3 = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At "15:45"

Register-ScheduledTask `
    -TaskName "AlgoTrading_Reconcile" `
    -Description "Evening reconcile: compare fills vs Angel One holdings, append to history (3:45 PM IST Mon-Fri)" `
    -Action $action3 `
    -Trigger $trigger3 `
    -Settings $settings `
    -RunLevel Highest `
    -Force

Write-Host "[ok] Registered: AlgoTrading_Reconcile (3:45 PM Mon-Fri)"

Write-Host ""
Write-Host "All tasks registered. Verify in Task Scheduler (taskschd.msc)."
Write-Host ""
Write-Host "IMPORTANT: The system is in PAPER mode by default (TRADING_MODE=paper in .env)."
Write-Host "To go live, set TRADING_MODE=live in .env ONLY after the promotion gate approves."
Write-Host ""
Write-Host "To test a task immediately:"
Write-Host "  Start-ScheduledTask -TaskName 'AlgoTrading_Evening'"
Write-Host "  Start-ScheduledTask -TaskName 'AlgoTrading_Morning'"
