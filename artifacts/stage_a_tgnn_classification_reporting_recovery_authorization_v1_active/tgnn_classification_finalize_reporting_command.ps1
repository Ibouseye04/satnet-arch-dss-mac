[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $RepoRoot,
    [Parameter(Mandatory = $true)] [string] $OutputRoot,
    [Parameter(Mandatory = $true)] [string] $ExpectedCampaignHash,
    [Parameter(Mandatory = $true)] [string] $ExpectedPreRecoveryTreeHash,
    [Parameter(Mandatory = $true)] [string] $ExpectedActivationHead,
    [Parameter(Mandatory = $true)] [string] $PythonExecutable,
    [switch] $PreflightOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptPath = Join-Path $RepoRoot "scripts\finalize_stage_a_tgnn_classification_reporting.py"
if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) {
    throw "Finalize-only recovery script is missing: $scriptPath"
}

$pythonArgs = @(
    $scriptPath,
    "--repo-root", $RepoRoot,
    "--output-root", $OutputRoot,
    "--expected-campaign-hash", $ExpectedCampaignHash,
    "--expected-pre-recovery-tree-hash", $ExpectedPreRecoveryTreeHash,
    "--expected-activation-head", $ExpectedActivationHead,
    "--python-executable", $PythonExecutable,
    "--finalize-reporting-only"
)
if ($PreflightOnly) {
    $pythonArgs += "--preflight-only"
}

& $PythonExecutable @pythonArgs
if ($LASTEXITCODE -ne 0) {
    throw "Finalize-only reporting command failed with exit code $LASTEXITCODE"
}
