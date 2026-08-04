[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $RepoRoot,
    [Parameter(Mandatory = $true)] [string] $OutputRoot,
    [Parameter(Mandatory = $true)] [string] $RecoveryRoot,
    [Parameter(Mandatory = $true)] [string] $ExpectedCampaignHash,
    [Parameter(Mandatory = $true)] [string] $ExpectedPreFinalizationTreeHash,
    [Parameter(Mandatory = $true)] [string] $ExpectedRecoveryTreeHash,
    [Parameter(Mandatory = $true)] [string] $ExpectedPlannedPostPublicationTreeHash,
    [Parameter(Mandatory = $true)] [string] $ExpectedActivationHead,
    [Parameter(Mandatory = $true)] [string] $AcceptedCheckpointSha256,
    [Parameter(Mandatory = $true)] [string] $SupersededCheckpointSha256,
    [Parameter(Mandatory = $true)] [string] $PythonExecutable,
    [switch] $PreflightOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptPath = Join-Path $RepoRoot "scripts\resume_stage_a_tgnn_classification_reporting.py"
if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) {
    throw "Windows-safe resume script is missing: $scriptPath"
}
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw "Dedicated Python executable is missing: $PythonExecutable"
}

$pythonArgs = @(
    $scriptPath,
    "--repo-root", $RepoRoot,
    "--output-root", $OutputRoot,
    "--recovery-root", $RecoveryRoot,
    "--expected-campaign-hash", $ExpectedCampaignHash,
    "--expected-pre-finalization-tree-hash", $ExpectedPreFinalizationTreeHash,
    "--expected-recovery-tree-hash", $ExpectedRecoveryTreeHash,
    "--expected-planned-post-publication-tree-hash", $ExpectedPlannedPostPublicationTreeHash,
    "--expected-activation-head", $ExpectedActivationHead,
    "--accepted-checkpoint-sha256", $AcceptedCheckpointSha256,
    "--superseded-checkpoint-sha256", $SupersededCheckpointSha256,
    "--python-executable", $PythonExecutable
)
if ($PreflightOnly) {
    $pythonArgs += "--preflight-only"
}

& $PythonExecutable @pythonArgs
$exitCode = $LASTEXITCODE
exit $exitCode
