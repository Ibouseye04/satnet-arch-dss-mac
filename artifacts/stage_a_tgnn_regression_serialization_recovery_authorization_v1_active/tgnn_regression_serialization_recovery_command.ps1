[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $RepoRoot,
    [Parameter(Mandatory = $true)] [string] $OutputRoot,
    [Parameter(Mandatory = $true)] [string] $ReconstructionRoot,
    [Parameter(Mandatory = $true)] [string] $RecoveryRoot,
    [Parameter(Mandatory = $true)] [string] $AssessmentRoot,
    [Parameter(Mandatory = $true)] [string] $ExpectedActivationHead,
    [Parameter(Mandatory = $true)] [string] $ExpectedSourceToolingSha,
    [Parameter(Mandatory = $true)] [string] $ExpectedRecoveryToolingSha,
    [Parameter(Mandatory = $true)] [string] $RecoveryTimestamp,
    [Parameter(Mandatory = $true)] [string] $PlannedFinalTreeSha256,
    [Parameter(Mandatory = $true)] [int] $PlannedFinalFileCount,
    [Parameter(Mandatory = $true)] [int] $PlannedFinalBytes,
    [Parameter(Mandatory = $true)] [string] $PythonExecutable,
    [Parameter(Mandatory = $true)] [string] $TranscriptDirectory,
    [switch] $PreflightOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$transcriptPath = Join-Path $TranscriptDirectory ("tgnn-regression-serialization-recovery-{0}.log" -f (Get-Date -Format 'yyyyMMddTHHmmssfffK'))
$recoveryScript = Join-Path $RepoRoot 'scripts\restore_stage_a_tgnn_regression_serialization_recovery.py'
$completionMarker = "$RecoveryRoot.completed.json"
$exitCode = 1

function Get-GitValue {
    param([string[]] $Arguments)
    $value = & git -C $RepoRoot @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Git command failed: $($Arguments -join ' ')" }
    return ($value -join "`n").Trim()
}

function Assert-ActivationIdentity {
    if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) { throw "Dedicated Python executable missing: $PythonExecutable" }
    $pythonResolved = (Resolve-Path -LiteralPath $PythonExecutable).Path
    $expectedPythonResolved = (Resolve-Path -LiteralPath 'C:\Users\johns\satnet-stage-a-tgnn-runtime-v1\.venv\Scripts\python.exe').Path
    if ($pythonResolved -ne $expectedPythonResolved) { throw "Dedicated Python identity mismatch: $pythonResolved" }
    $head = Get-GitValue @('rev-parse', 'HEAD')
    if ($head -ne $ExpectedActivationHead) { throw "Activation HEAD mismatch: $head" }
    $branch = Get-GitValue @('branch', '--show-current')
    if ($branch -ne 'activate/stage-a-tgnn-regression-serialization-recovery-v1') { throw "Activation branch mismatch: $branch" }
    $status = Get-GitValue @('status', '--porcelain')
    if (-not [string]::IsNullOrWhiteSpace($status)) { throw 'Activation worktree is not clean' }
    $sourceSha = Get-GitValue @('rev-list', '-n', '1', 'HEAD', '--', 'scripts/run_stage_a_tgnn_regression_training.py')
    if ($sourceSha -ne $ExpectedSourceToolingSha) { throw "Overwrite-guard source SHA mismatch: $sourceSha" }
    $recoverySha = Get-GitValue @('rev-list', '-n', '1', 'HEAD', '--', 'scripts/restore_stage_a_tgnn_regression_serialization_recovery.py')
    if ($recoverySha -ne $ExpectedRecoveryToolingSha) { throw "Recovery-tooling SHA mismatch: $recoverySha" }
    if (Test-Path -LiteralPath $completionMarker) { throw "Recovery already completed: $completionMarker" }
    if (Test-Path -LiteralPath $RecoveryRoot) { throw "Recovery staging root is not absent: $RecoveryRoot" }
    if (Test-Path -LiteralPath "$RecoveryRoot.lock") { throw "Recovery lock exists: $RecoveryRoot.lock" }
}

try {
    New-Item -ItemType Directory -Force -Path $TranscriptDirectory | Out-Null
    Start-Transcript -Path $transcriptPath -Force | Out-Null
    Assert-ActivationIdentity
    $pythonArguments = [System.Collections.Generic.List[string]]::new()
    $pythonArguments.Add($recoveryScript)
    $pythonArguments.Add('--mode')
    if ($PreflightOnly) { $pythonArguments.Add('preflight') } else { $pythonArguments.Add('execute') }
    $pythonArguments.Add('--repo-root'); $pythonArguments.Add($RepoRoot)
    $pythonArguments.Add('--output-root'); $pythonArguments.Add($OutputRoot)
    $pythonArguments.Add('--reconstruction-root'); $pythonArguments.Add($ReconstructionRoot)
    $pythonArguments.Add('--recovery-root'); $pythonArguments.Add($RecoveryRoot)
    $pythonArguments.Add('--assessment-root'); $pythonArguments.Add($AssessmentRoot)
    $pythonArguments.Add('--expected-activation-head'); $pythonArguments.Add($ExpectedActivationHead)
    $pythonArguments.Add('--recovery-timestamp'); $pythonArguments.Add($RecoveryTimestamp)
    $pythonArguments.Add('--recovery-tooling-sha'); $pythonArguments.Add($ExpectedRecoveryToolingSha)
    $pythonArguments.Add('--planned-final-file-count'); $pythonArguments.Add([string]$PlannedFinalFileCount)
    $pythonArguments.Add('--planned-final-bytes'); $pythonArguments.Add([string]$PlannedFinalBytes)
    $pythonArguments.Add('--planned-final-tree-sha256'); $pythonArguments.Add($PlannedFinalTreeSha256)
    & $PythonExecutable @pythonArguments
    $exitCode = $LASTEXITCODE
}
catch {
    Write-Error $_
    $exitCode = 1
}
finally {
    try { Stop-Transcript | Out-Null } catch {}
}
exit $exitCode
