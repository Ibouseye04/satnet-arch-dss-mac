[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $RepoRoot,
    [Parameter(Mandatory = $true)] [string] $OutputRoot,
    [Parameter(Mandatory = $true)] [string] $ReconstructionRoot,
    [Parameter(Mandatory = $true)] [string] $RecoveryRoot,
    [Parameter(Mandatory = $true)] [string] $AssessmentRoot,
    [Parameter(Mandatory = $true)] [string] $ExpectedActivationHead,
    [Parameter(Mandatory = $true)] [string] $ExpectedSourceToolingCommit,
    [Parameter(Mandatory = $true)] [string] $ExpectedSourceScriptSha256,
    [Parameter(Mandatory = $true)] [string] $ExpectedRecoveryToolingCommit,
    [Parameter(Mandatory = $true)] [string] $ExpectedRecoveryScriptSha256,
    [Parameter(Mandatory = $true)] [string] $ExpectedPublicTreeSha256,
    [Parameter(Mandatory = $true)] [int] $ExpectedPublicFileCount,
    [Parameter(Mandatory = $true)] [int] $ExpectedPublicBytes,
    [Parameter(Mandatory = $true)] [string] $ExpectedPublicCandidateSha256,
    [Parameter(Mandatory = $true)] [string] $ExpectedReconstructionTreeSha256,
    [Parameter(Mandatory = $true)] [int] $ExpectedReconstructionFileCount,
    [Parameter(Mandatory = $true)] [int] $ExpectedReconstructionBytes,
    [Parameter(Mandatory = $true)] [string] $ExpectedReconstructedCandidateSha256,
    [Parameter(Mandatory = $true)] [string] $ExpectedPartialRecoveryTreeSha256,
    [Parameter(Mandatory = $true)] [int] $ExpectedPartialRecoveryFileCount,
    [Parameter(Mandatory = $true)] [int] $ExpectedPartialRecoveryBytes,
    [Parameter(Mandatory = $true)] [string] $ExpectedPartialLockSha256,
    [Parameter(Mandatory = $true)] [int] $ExpectedPartialLockPid,
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
$recoveryScript = Join-Path $RepoRoot 'scripts\restore_stage_a_tgnn_regression_serialization_recovery.py'
$recoveryLock = "$RecoveryRoot.lock"
$completionMarker = "$RecoveryRoot.completed.json"
$transcriptPath = Join-Path $TranscriptDirectory ("tgnn-regression-final-tree-corrected-resume-{0}.log" -f (Get-Date -Format 'yyyyMMddTHHmmssfffK'))
$exitCode = 1
function Get-GitValue { param([string[]] $Arguments); $value = & git -C $RepoRoot @Arguments; if ($LASTEXITCODE -ne 0) { throw "Git command failed: $($Arguments -join ' ')" }; return ($value -join "`n").Trim() }
function Get-FileSha256 { param([string] $Path); return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Get-TreeIdentity {
    param([string] $Root)
    $files = @(Get-ChildItem -LiteralPath $Root -Recurse -Force -File | Sort-Object -Property FullName)
    $rows = [System.Collections.Generic.List[string]]::new(); [int64]$bytes = 0
    foreach ($file in $files) { [void]$rows.Add(("{0}|{1}|{2}" -f $file.FullName, $file.Length, (Get-FileSha256 $file.FullName))); $bytes += $file.Length }
    $digest = [System.Security.Cryptography.SHA256]::Create().ComputeHash([System.Text.Encoding]::UTF8.GetBytes($rows -join "`n"))
    return [pscustomobject]@{ FileCount = $files.Count; Bytes = $bytes; TreeSha256 = (-join ($digest | ForEach-Object { $_.ToString('x2') })) }
}
function Assert-Hash { param([string]$Value, [int]$Length, [string]$Name); if ($Value -notmatch ("^[0-9a-f]{{{0}}}$" -f $Length)) { throw "$Name format mismatch" } }
function Assert-Identity {
    Assert-Hash $ExpectedActivationHead 40 'ExpectedActivationHead'; Assert-Hash $ExpectedSourceToolingCommit 40 'ExpectedSourceToolingCommit'; Assert-Hash $ExpectedRecoveryToolingCommit 40 'ExpectedRecoveryToolingCommit'
    Assert-Hash $ExpectedSourceScriptSha256 64 'ExpectedSourceScriptSha256'; Assert-Hash $ExpectedRecoveryScriptSha256 64 'ExpectedRecoveryScriptSha256'; Assert-Hash $ExpectedPartialLockSha256 64 'ExpectedPartialLockSha256'; Assert-Hash $PlannedFinalTreeSha256 64 'PlannedFinalTreeSha256'
    if ($PlannedFinalTreeSha256 -ne '95134dab5669ee4bc9515777d19f48449186324977d17981be856fde533e6111' -or $PlannedFinalFileCount -ne 22 -or $PlannedFinalBytes -ne 2070248) { throw 'Corrected final-tree identity mismatch' }
    if ((Resolve-Path -LiteralPath $PythonExecutable).Path -ne (Resolve-Path -LiteralPath 'C:\Users\johns\satnet-stage-a-tgnn-runtime-v1\.venv\Scripts\python.exe').Path) { throw 'Dedicated Python identity mismatch' }
    $head = Get-GitValue @('rev-parse','HEAD')
    if ($head -cne $ExpectedActivationHead) { throw "Activation HEAD mismatch: actual=$head expected=$ExpectedActivationHead" }
    if ((Get-GitValue @('branch','--show-current')) -ne 'activate/stage-a-tgnn-regression-serialization-replacement-resume-v1') { throw 'Recovery activation branch mismatch' }
    if (-not [string]::IsNullOrWhiteSpace((Get-GitValue @('status','--porcelain','--untracked-files=all')))) { throw 'Recovery activation worktree is not clean' }
    if ((Get-GitValue @('rev-list','-n','1','HEAD','--','scripts/run_stage_a_tgnn_regression_training.py')) -ne $ExpectedSourceToolingCommit) { throw 'Source commit mismatch' }
    if ((Get-GitValue @('rev-list','-n','1','HEAD','--','scripts/restore_stage_a_tgnn_regression_serialization_recovery.py')) -ne $ExpectedRecoveryToolingCommit) { throw 'Recovery commit mismatch' }
    if ((Get-FileSha256 (Join-Path $RepoRoot 'scripts\run_stage_a_tgnn_regression_training.py')) -ne $ExpectedSourceScriptSha256) { throw 'Source script SHA mismatch' }
    if ((Get-FileSha256 $recoveryScript) -ne $ExpectedRecoveryScriptSha256) { throw 'Recovery script SHA mismatch' }
    foreach ($spec in @(@($OutputRoot, $ExpectedPublicFileCount, $ExpectedPublicBytes, $ExpectedPublicTreeSha256), @($ReconstructionRoot, $ExpectedReconstructionFileCount, $ExpectedReconstructionBytes, $ExpectedReconstructionTreeSha256), @($RecoveryRoot, $ExpectedPartialRecoveryFileCount, $ExpectedPartialRecoveryBytes, $ExpectedPartialRecoveryTreeSha256))) {
        $identity = Get-TreeIdentity $spec[0]; if ($identity.FileCount -ne $spec[1] -or $identity.Bytes -ne $spec[2] -or $identity.TreeSha256 -ne $spec[3]) { throw "Bound tree identity mismatch: $($spec[0])" }
    }
    if ((Get-FileSha256 (Join-Path $OutputRoot 'checkpoints\candidate_1.pt')) -ne $ExpectedPublicCandidateSha256) { throw 'Public candidate SHA mismatch' }
    if ((Get-FileSha256 (Join-Path $ReconstructionRoot 'checkpoints\candidate_1.pt')) -ne $ExpectedReconstructedCandidateSha256) { throw 'Reconstructed candidate SHA mismatch' }
    $preserved = Join-Path $RecoveryRoot 'superseded_checkpoints\candidate_1_overwritten_by_final_seed_62005.pt'; if ((Get-FileSha256 $preserved) -ne $ExpectedPublicCandidateSha256) { throw 'Preserved superseded SHA mismatch' }
    if (@(Get-ChildItem -LiteralPath $RecoveryRoot -Recurse -Force -File | Where-Object { $_.Name -match '\.pending$|\.tmp$|\.recovery-pending$|\.replacement-pending$' }).Count -ne 0) { throw 'Temporary residue exists' }
    if ((Get-FileSha256 $recoveryLock) -ne $ExpectedPartialLockSha256) { throw 'Stale lock SHA mismatch' }
    if ([System.IO.File]::ReadAllText($recoveryLock) -ne "pid=$ExpectedPartialLockPid`r`n") { throw 'Stale lock contents mismatch' }
    if ($null -ne (Get-Process -Id $ExpectedPartialLockPid -ErrorAction SilentlyContinue)) { throw 'Stale lock owner is running' }
    if (Test-Path -LiteralPath $completionMarker) { throw 'Recovery already completed' }
    if (Test-Path -LiteralPath "$OutputRoot.lock") { throw 'Output lock exists' }
}
try {
    New-Item -ItemType Directory -Force -Path $TranscriptDirectory | Out-Null
    Start-Transcript -LiteralPath $transcriptPath -Force | Out-Null
    Assert-Identity
    $pythonArguments = [System.Collections.Generic.List[string]]::new()
    [void]$pythonArguments.Add($recoveryScript); [void]$pythonArguments.Add('--mode'); [void]$pythonArguments.Add($(if ($PreflightOnly) { 'preflight' } else { 'execute' }))
    foreach ($pair in @(@('--repo-root',$RepoRoot), @('--output-root',$OutputRoot), @('--reconstruction-root',$ReconstructionRoot), @('--recovery-root',$RecoveryRoot), @('--assessment-root',$AssessmentRoot), @('--expected-activation-head',$ExpectedActivationHead), @('--recovery-timestamp',$RecoveryTimestamp), @('--recovery-tooling-commit',$ExpectedRecoveryToolingCommit), @('--expected-partial-lock-sha256',$ExpectedPartialLockSha256), @('--planned-final-file-count',[string]$PlannedFinalFileCount), @('--planned-final-bytes',[string]$PlannedFinalBytes), @('--planned-final-tree-sha256',$PlannedFinalTreeSha256))) { [void]$pythonArguments.Add($pair[0]); [void]$pythonArguments.Add($pair[1]) }
    [void]$pythonArguments.Add('--resume-partial')
    & $PythonExecutable @pythonArguments
    $exitCode = $LASTEXITCODE
} catch { Write-Error $_; $exitCode = 1 } finally { try { Stop-Transcript | Out-Null } catch {} }
exit $exitCode
