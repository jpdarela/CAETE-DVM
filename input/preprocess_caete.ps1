# =============================================================================
# CAETE Parallel Preprocessing Script
# PowerShell 7.5.2+ with ForEach-Object -Parallel support
# =============================================================================

# GLOBAL CONFIGURATION - Change this to switch between scripts
$SCRIPT_NAME = "preprocess_caete.py"   # Options: "preprocess_caete.py" or "preprocess_caete_pbz2.py"

# Define all preprocessing tasks
$tasks = @(
    @{dataset = "20CRv3-ERA5"; mode = "obsclim"},
    @{dataset = "20CRv3-ERA5"; mode = "spinclim"},
    # @{dataset = "20CRv3-ERA5"; mode = "transclim"},
    # @{dataset = "20CRv3-ERA5"; mode = "counterclim"},
    @{dataset = "MPI-ESM1-2-HR"; mode = "historical"},
    @{dataset = "MPI-ESM1-2-HR"; mode = "piControl"},
    @{dataset = "MPI-ESM1-2-HR"; mode = "ssp370"},
    @{dataset = "MPI-ESM1-2-HR"; mode = "ssp585"}
)

# =============================================================================
# SYSTEM RESOURCE MONITORING
# =============================================================================
$separator = "=" * 80
Write-Host $separator -ForegroundColor Cyan
Write-Host "CAETE Parallel Preprocessing - System Information" -ForegroundColor Cyan
Write-Host $separator -ForegroundColor Cyan

# FIXED: Better memory detection method
$systemInfo = Get-ComputerInfo
$cpuCores = $systemInfo.CsNumberOfLogicalProcessors

# Alternative method for total memory (more reliable)
$totalMemoryBytes = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
$totalMemoryGB = [math]::Round($totalMemoryBytes/1GB, 2)

$availableMemoryGB = [math]::Round((Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue/1024, 2)

Write-Host "[CPU] CPU Cores: $cpuCores" -ForegroundColor Yellow
Write-Host "[MEM] Total Memory: $totalMemoryGB GB" -ForegroundColor Yellow
Write-Host "[MEM] Available Memory: $availableMemoryGB GB" -ForegroundColor Yellow
Write-Host "[PY]  Python Script: $SCRIPT_NAME" -ForegroundColor Green
Write-Host "[TASK] Total Tasks: $($tasks.Count)" -ForegroundColor White

$throttleLimit = 4

Write-Host "[THROTTLE] Throttle Limit: $throttleLimit (manually set)" -ForegroundColor Magenta
Write-Host ""

# Confirm script selection
if (-not (Test-Path $SCRIPT_NAME)) {
    Write-Host "[ERROR] Script '$SCRIPT_NAME' not found in current directory!" -ForegroundColor Red
    Write-Host "[DIR] Current directory: $(Get-Location)" -ForegroundColor Gray
    Write-Host "[FILES] Available Python scripts:" -ForegroundColor Gray
    Get-ChildItem -Name "*.py" | ForEach-Object { Write-Host "   - $_" -ForegroundColor Gray }
    exit 1
}

Write-Host "[OK] Using script: $SCRIPT_NAME" -ForegroundColor Green
Write-Host ""

# =============================================================================
# PARALLEL PROCESSING EXECUTION
# =============================================================================
Write-Host "[START] Starting parallel preprocessing tasks..." -ForegroundColor Cyan
Write-Host $separator -ForegroundColor DarkGray

$globalStartTime = Get-Date

# Run tasks in parallel with FIXED log file handling
$results = $tasks | ForEach-Object -Parallel {
    $dataset = $_.dataset
    $mode = $_.mode
    $script = $using:SCRIPT_NAME

    # Create unique identifier for this task
    $taskId = "$dataset-$mode"

    Write-Host "[STARTING] $taskId" -ForegroundColor Yellow
    $taskStartTime = Get-Date

    try {
        # FIXED: Use unique temp directory for each process to avoid file conflicts
        $tempDir = New-Item -ItemType Directory -Path "temp_$taskId" -Force
        $stdoutFile = Join-Path $tempDir "stdout.log"
        $stderrFile = Join-Path $tempDir "stderr.log"

        # Capture both stdout and stderr with unique files
        $process = Start-Process python -ArgumentList "$script --dataset $dataset --mode $mode" -NoNewWindow -PassThru -RedirectStandardOutput $stdoutFile -RedirectStandardError $stderrFile -Wait

        $taskEndTime = Get-Date
        $duration = $taskEndTime - $taskStartTime

        # Read output files with retry mechanism
        $stdout = ""
        $stderr = ""

        # Wait a moment for file handles to be released
        Start-Sleep -Milliseconds 1000

        if (Test-Path $stdoutFile) {
            try {
                $stdout = Get-Content $stdoutFile -Raw -ErrorAction SilentlyContinue
            } catch {
                $stdout = "[Could not read stdout]"
            }
        }

        if (Test-Path $stderrFile) {
            try {
                $stderr = Get-Content $stderrFile -Raw -ErrorAction SilentlyContinue
            } catch {
                $stderr = "[Could not read stderr]"
            }
        }

        # Clean up temp directory
        try {
            Remove-Item $tempDir -Recurse -Force -ErrorAction SilentlyContinue
        } catch {
            # Ignore cleanup errors
        }

        if ($process.ExitCode -eq 0) {
            Write-Host "[SUCCESS] $taskId (Duration: $($duration.ToString('hh\:mm\:ss')))" -ForegroundColor Green

            return @{
                TaskId = $taskId
                Dataset = $dataset
                Mode = $mode
                Success = $true
                Duration = $duration
                ExitCode = $process.ExitCode
                Output = $stdout
                Error = $stderr
            }
        } else {
            Write-Host "[FAILED] $taskId (Exit Code: $($process.ExitCode), Duration: $($duration.ToString('hh\:mm\:ss')))" -ForegroundColor Red

            return @{
                TaskId = $taskId
                Dataset = $dataset
                Mode = $mode
                Success = $false
                Duration = $duration
                ExitCode = $process.ExitCode
                Output = $stdout
                Error = $stderr
            }
        }
    }
    catch {
        $taskEndTime = Get-Date
        $duration = $taskEndTime - $taskStartTime

        Write-Host "[EXCEPTION] $taskId - $_" -ForegroundColor Red

        return @{
            TaskId = $taskId
            Dataset = $dataset
            Mode = $mode
            Success = $false
            Duration = $duration
            ExitCode = -1
            Output = ""
            Error = $_.ToString()
        }
    }
} -ThrottleLimit $throttleLimit

$globalEndTime = Get-Date
$totalDuration = $globalEndTime - $globalStartTime

# =============================================================================
# RESULTS SUMMARY
# =============================================================================
Write-Host ""
Write-Host $separator -ForegroundColor Cyan
Write-Host "PREPROCESSING RESULTS SUMMARY" -ForegroundColor Cyan
Write-Host $separator -ForegroundColor Cyan

# FIXED: Correct success/failure counting
$successful = @($results | Where-Object { $_.Success -eq $true })
$failed = @($results | Where-Object { $_.Success -eq $false })

Write-Host "[STATS] Total Tasks: $($results.Count)" -ForegroundColor White
Write-Host "[SUCCESS] Successful: $($successful.Count)" -ForegroundColor Green
Write-Host "[FAILED] Failed: $($failed.Count)" -ForegroundColor Red
Write-Host "[TIME] Total Duration: $($totalDuration.ToString('hh\:mm\:ss'))" -ForegroundColor Yellow
Write-Host "[SCRIPT] Script Used: $SCRIPT_NAME" -ForegroundColor Green
Write-Host ""

# Detailed results
if ($successful.Count -gt 0) {
    Write-Host "[SUCCESS] SUCCESSFUL TASKS:" -ForegroundColor Green
    foreach ($task in $successful) {
        Write-Host "   $($task.TaskId): $($task.Duration.ToString('hh\:mm\:ss'))" -ForegroundColor DarkGreen
    }
    Write-Host ""
}

if ($failed.Count -gt 0) {
    Write-Host "[FAILED] FAILED TASKS:" -ForegroundColor Red
    foreach ($task in $failed) {
        Write-Host "   $($task.TaskId): Exit Code $($task.ExitCode)" -ForegroundColor DarkRed
        if ($task.Error -and $task.Error.Length -gt 0) {
            $errorPreview = $task.Error.Substring(0, [Math]::Min(200, $task.Error.Length))
            Write-Host "      Error: $errorPreview..." -ForegroundColor Gray
        }
    }
    Write-Host ""
}

# Performance metrics
if ($results.Count -gt 0) {
    $avgDuration = ($results | Measure-Object -Property { $_.Duration.TotalSeconds } -Average).Average
    Write-Host "[PERF] Average Task Duration: $([TimeSpan]::FromSeconds($avgDuration).ToString('hh\:mm\:ss'))" -ForegroundColor Cyan
    Write-Host "[SPEEDUP] Speedup Factor: ~$([Math]::Round(($results | Measure-Object -Property { $_.Duration.TotalSeconds } -Sum).Sum / $totalDuration.TotalSeconds, 1))x" -ForegroundColor Magenta
}

Write-Host ""
Write-Host "[COMPLETE] All preprocessing tasks completed!" -ForegroundColor Cyan
Write-Host $separator -ForegroundColor DarkGray

# Exit with appropriate code
if ($failed.Count -gt 0) {
    exit 1
} else {
    exit 0
}