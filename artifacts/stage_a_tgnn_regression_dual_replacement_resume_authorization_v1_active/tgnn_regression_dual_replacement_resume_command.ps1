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
    [Parameter(Mandatory = $true)] [string] $ExpectedPublicSelectedIdentitySha256,
    [Parameter(Mandatory = $true)] [string] $ExpectedReconstructionTreeSha256,
    [Parameter(Mandatory = $true)] [int] $ExpectedReconstructionFileCount,
    [Parameter(Mandatory = $true)] [int] $ExpectedReconstructionBytes,
    [Parameter(Mandatory = $true)] [string] $ExpectedReconstructedCandidateSha256,
    [Parameter(Mandatory = $true)] [string] $ExpectedPartialRecoveryTreeSha256,
    [Parameter(Mandatory = $true)] [int] $ExpectedPartialRecoveryFileCount,
    [Parameter(Mandatory = $true)] [int] $ExpectedPartialRecoveryBytes,
    [Parameter(Mandatory = $true)] [string] $ExpectedPartialLockSha256,
    [Parameter(Mandatory = $true)] [int] $ExpectedPartialLockPid,
    [Parameter(Mandatory = $true)] [string] $ExpectedStagedSelectedIdentitySha256,
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
$sourceScript = Join-Path $RepoRoot 'scripts\run_stage_a_tgnn_regression_training.py'
$recoveryLock = "$RecoveryRoot.lock"
$completionMarker = "$RecoveryRoot.completed.json"
$transcriptPath = Join-Path $TranscriptDirectory ("tgnn-regression-dual-replacement-resume-{0}.log" -f (Get-Date -Format 'yyyyMMddTHHmmssfffK'))
$exitCode = 1
function Get-GitValue { param([string[]] $Arguments); $value = & git -C $RepoRoot @Arguments; if ($LASTEXITCODE -ne 0) { throw "Git command failed: $($Arguments -join ' ')" }; return ($value -join "`n").Trim() }
function Get-FileSha256 { param([string] $Path); if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Missing file: $Path" }; return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Get-TreeIdentity {
    param([string] $Root)
    if (-not (Test-Path -LiteralPath $Root -PathType Container)) { throw "Missing tree: $Root" }
    $files = @(Get-ChildItem -LiteralPath $Root -Recurse -Force -File | Sort-Object -Property FullName)
    $rows = [System.Collections.Generic.List[string]]::new(); [int64] $bytes = 0
    foreach ($file in $files) { [void] $rows.Add(("{0}|{1}|{2}" -f $file.FullName, $file.Length, (Get-FileSha256 $file.FullName))); $bytes += $file.Length }
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try { $digest = $sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($rows -join "`n")) } finally { $sha.Dispose() }
    return [pscustomobject]@{ FileCount = $files.Count; Bytes = $bytes; TreeSha256 = (-join ($digest | ForEach-Object { $_.ToString('x2') })) }
}
function Assert-HashFormat { param([string] $Value, [int] $Length, [string] $Name); if ($Value -notmatch ("^[0-9a-f]{{{0}}}$" -f $Length)) { throw "$Name format mismatch" } }
function Assert-Tree {
    param([string] $Root, [int] $FileCount, [int] $Bytes, [string] $TreeSha256)
    $identity = Get-TreeIdentity $Root
    if ($identity.FileCount -ne $FileCount -or $identity.Bytes -ne $Bytes -or $identity.TreeSha256 -cne $TreeSha256) { throw "Bound tree identity mismatch: $Root ($($identity | ConvertTo-Json -Compress))" }
}
function Assert-StagedFile {
    param([string] $Relative, [int] $Bytes, [string] $Sha256)
    $path = Join-Path $RecoveryRoot ($Relative -replace '/', '\\')
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing staged artifact: $Relative" }
    $item = Get-Item -LiteralPath $path
    if ($item.Length -ne $Bytes -or (Get-FileSha256 $path) -cne $Sha256) { throw "Staged artifact identity mismatch: $Relative" }
}
function Assert-ActivationAndState {
    Assert-HashFormat $ExpectedActivationHead 40 'ExpectedActivationHead'; Assert-HashFormat $ExpectedSourceToolingCommit 40 'ExpectedSourceToolingCommit'; Assert-HashFormat $ExpectedRecoveryToolingCommit 40 'ExpectedRecoveryToolingCommit'; Assert-HashFormat $ExpectedSourceScriptSha256 64 'ExpectedSourceScriptSha256'; Assert-HashFormat $ExpectedRecoveryScriptSha256 64 'ExpectedRecoveryScriptSha256'; Assert-HashFormat $ExpectedPartialLockSha256 64 'ExpectedPartialLockSha256'
    if ((Resolve-Path -LiteralPath $PythonExecutable).Path -ne (Resolve-Path -LiteralPath 'C:\Users\johns\satnet-stage-a-tgnn-runtime-v1\.venv\Scripts\python.exe').Path) { throw 'Dedicated Python identity mismatch' }
    if ((Get-GitValue @('rev-parse','HEAD')) -cne $ExpectedActivationHead) { throw 'Activation HEAD mismatch' }
    if ((Get-GitValue @('branch','--show-current')) -cne 'activate/stage-a-tgnn-regression-dual-replacement-resume-v1') { throw 'Activation branch mismatch' }
    if (-not [string]::IsNullOrWhiteSpace((Get-GitValue @('status','--porcelain','--untracked-files=all')))) { throw 'Activation worktree is not clean' }
    if ((Get-GitValue @('rev-list','-n','1','HEAD','--','scripts/run_stage_a_tgnn_regression_training.py')) -cne $ExpectedSourceToolingCommit) { throw 'Source tooling commit mismatch' }
    if ((Get-GitValue @('rev-list','-n','1','HEAD','--','scripts/restore_stage_a_tgnn_regression_serialization_recovery.py')) -cne $ExpectedRecoveryToolingCommit) { throw 'Recovery tooling commit mismatch' }
    if ((Get-FileSha256 $sourceScript) -cne $ExpectedSourceScriptSha256) { throw 'Source script SHA mismatch' }; if ((Get-FileSha256 $recoveryScript) -cne $ExpectedRecoveryScriptSha256) { throw 'Recovery script SHA mismatch' }
    Assert-Tree $OutputRoot $ExpectedPublicFileCount $ExpectedPublicBytes $ExpectedPublicTreeSha256; Assert-Tree $ReconstructionRoot $ExpectedReconstructionFileCount $ExpectedReconstructionBytes $ExpectedReconstructionTreeSha256; Assert-Tree $RecoveryRoot $ExpectedPartialRecoveryFileCount $ExpectedPartialRecoveryBytes $ExpectedPartialRecoveryTreeSha256
    if ((Get-FileSha256 (Join-Path $OutputRoot 'checkpoints\candidate_1.pt')) -cne $ExpectedPublicCandidateSha256) { throw 'Public candidate historical SHA mismatch' }; if ((Get-FileSha256 (Join-Path $OutputRoot 'selected_checkpoint_identity.json')) -cne $ExpectedPublicSelectedIdentitySha256) { throw 'Public selected identity historical SHA mismatch' }; if ((Get-FileSha256 (Join-Path $ReconstructionRoot 'checkpoints\candidate_1.pt')) -cne $ExpectedReconstructedCandidateSha256) { throw 'Reconstructed candidate SHA mismatch' }; if ((Get-FileSha256 (Join-Path $RecoveryRoot 'superseded_checkpoints\candidate_1_overwritten_by_final_seed_62005.pt')) -cne $ExpectedPublicCandidateSha256) { throw 'Preserved superseded checkpoint SHA mismatch' }
    Assert-StagedFile 'checkpoints/candidate_1.pt' 150163 $ExpectedReconstructedCandidateSha256; Assert-StagedFile 'selected_checkpoint_identity.json' 1442 $ExpectedStagedSelectedIdentitySha256; Assert-StagedFile 'validation_bootstrap_intervals.json' 1250 'cc3dc99f0cfa0c95f3c4a366a9b88d99e74d9deebc48078985180154b12dd0b7'; Assert-StagedFile 'final_seed_training_results.json' 2220 '7e1cb790785e3123261bddef5c0a47525b9ee9cd9b59142969d5b0ff8247bb10'; Assert-StagedFile 'training_report.json' 628 'd6307cabfdae4f540365958e4d38f65f105688ae27c0e101d2134f71bdeb3b26'; Assert-StagedFile 'reporting_recovery_manifest.json' 3942 '5c9638088133b2aeb2bc396085d0bd26601f5e87793086b699dfba080c3de707'; Assert-StagedFile 'staged_state.json' 357 '6db5806f3e64c566ae527983d40c44b4409ec77d58fa680d77893e9e0a69cf6e'
    if (@(Get-ChildItem -LiteralPath $RecoveryRoot -Recurse -Force -File | Where-Object { $_.Name -match '\.pending$|\.tmp$|\.recovery-pending$|\.replacement-pending$' }).Count -ne 0) { throw 'Temporary residue exists' }
    if (-not (Test-Path -LiteralPath $recoveryLock -PathType Leaf)) { throw 'Stale recovery lock is missing' }; if ((Get-FileSha256 $recoveryLock) -cne $ExpectedPartialLockSha256) { throw 'Stale lock SHA mismatch' }; if ([System.IO.File]::ReadAllText($recoveryLock) -ne "pid=$ExpectedPartialLockPid`r`n") { throw 'Stale lock contents mismatch' }; if ($null -ne (Get-Process -Id $ExpectedPartialLockPid -ErrorAction SilentlyContinue)) { throw 'Stale lock owner is running' }; if (Test-Path -LiteralPath $completionMarker) { throw 'Recovery already completed' }; if (Test-Path -LiteralPath "$OutputRoot.lock") { throw 'Output lock exists' }
}
function Assert-NoScientificExecution {
    $source = [System.IO.File]::ReadAllText($recoveryScript)
    foreach ($pattern in @('(?m)^\s*(from|import).*run_stage_a_tgnn_regression_training', 'torch\.save\s*\(', 'torch\.optim', '\.backward\s*\(', '\.fit\s*\(', '\.predict\s*\(')) { if ($source -match $pattern) { throw "Forbidden scientific execution pattern in recovery script: $pattern" } }
}
function Invoke-FocusedTests {
    Push-Location $RepoRoot
    try { & $PythonExecutable -m unittest tests.scripts.test_stage_a_tgnn_regression_replacement -v; if ($LASTEXITCODE -ne 0) { throw 'Focused dual-replacement tests failed' } }
    finally { Pop-Location }
}
function Invoke-RecoveryTool {
    $arguments = [System.Collections.Generic.List[string]]::new(); [void]$arguments.Add($recoveryScript); [void]$arguments.Add('--mode'); [void]$arguments.Add($(if ($PreflightOnly) { 'preflight' } else { 'execute' }))
    foreach ($pair in @(@('--repo-root',$RepoRoot), @('--output-root',$OutputRoot), @('--reconstruction-root',$ReconstructionRoot), @('--recovery-root',$RecoveryRoot), @('--assessment-root',$AssessmentRoot), @('--expected-activation-head',$ExpectedActivationHead), @('--recovery-timestamp',$RecoveryTimestamp), @('--recovery-tooling-commit',$ExpectedRecoveryToolingCommit), @('--expected-partial-lock-sha256',$ExpectedPartialLockSha256), @('--planned-final-file-count',[string]$PlannedFinalFileCount), @('--planned-final-bytes',[string]$PlannedFinalBytes), @('--planned-final-tree-sha256',$PlannedFinalTreeSha256))) { [void]$arguments.Add($pair[0]); [void]$arguments.Add($pair[1]) }
    [void]$arguments.Add('--resume-partial'); & $PythonExecutable @arguments; if ($LASTEXITCODE -ne 0) { throw "Recovery tool failed with exit code $LASTEXITCODE" }
}
try { New-Item -ItemType Directory -Force -Path $TranscriptDirectory | Out-Null; Start-Transcript -LiteralPath $transcriptPath -Force | Out-Null; Assert-ActivationAndState; Assert-NoScientificExecution; Invoke-FocusedTests; Invoke-RecoveryTool; $exitCode = 0 } catch { Write-Error $_; $exitCode = 1 } finally { try { Stop-Transcript | Out-Null } catch {} }
exit $exitCode
