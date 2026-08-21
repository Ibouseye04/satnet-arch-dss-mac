[CmdletBinding()]
param(
    [string]$RepoRoot = 'C:\Users\johns\satnet-stage-a-rf-regression-training-v1-activation',
    [string]$OutputRoot = 'C:\Users\johns\satnet-stage-a-rf-regression-training-v1',
    [string]$ExpectedActivationHead = 'bcad45c5edf07bdea4c0eab322ee1cbf7bf5472d',
    [string]$CommittedTrainingToolingSha = 'bcad45c5edf07bdea4c0eab322ee1cbf7bf5472d',
    [string]$PythonExecutable = 'python',
    [AllowNull()][string]$Command
)

$ErrorActionPreference = 'Stop'
$transcriptDirectory = Join-Path (Split-Path -Parent $OutputRoot) 'satnet-stage-a-rf-regression-training-v1-transcripts'
New-Item -ItemType Directory -Force -Path $transcriptDirectory | Out-Null
$timestamp = Get-Date -Format 'yyyyMMddTHHmmssfffK'
$transcriptPath = Join-Path $transcriptDirectory "rf-regression-training-$timestamp.log"
$exitCode = 1

function Add-OptionalArgument {
    param(
        [System.Collections.Generic.List[string]]$Arguments,
        [string]$Name,
        [AllowNull()][string]$Value
    )
    if (-not [string]::IsNullOrWhiteSpace($Value)) {
        $Arguments.Add($Name)
        $Arguments.Add($Value)
    }
}

try {
    Start-Transcript -Path $transcriptPath -Force | Out-Null
    $head = (& git -C $RepoRoot rev-parse HEAD).Trim()
    if ($head -ne $ExpectedActivationHead) { throw "Activation HEAD mismatch: $head" }
    $branch = (& git -C $RepoRoot branch --show-current).Trim()
    if ($branch -ne 'activate/stage-a-rf-regression-training-v1') { throw "Activation branch mismatch: $branch" }
    $status = (& git -C $RepoRoot status --porcelain) -join "`n"
    if (-not [string]::IsNullOrWhiteSpace($status)) { throw 'Activation worktree is not clean' }
    $tooling = (& git -C $RepoRoot rev-list -n 1 HEAD -- scripts/run_stage_a_rf_regression_training.py).Trim()
    if ($tooling -ne $CommittedTrainingToolingSha) { throw "Training-tooling identity mismatch: $tooling" }
    if (Test-Path -LiteralPath $OutputRoot) { throw "Output root already exists: $OutputRoot" }
    $lockPath = "$OutputRoot.lock"
    if (Test-Path -LiteralPath $lockPath) { throw "Training lock already exists: $lockPath" }
    $runnerPath = Join-Path $RepoRoot 'scripts/run_stage_a_rf_regression_training.py'
    $runnerArgs = [System.Collections.Generic.List[string]]::new()
    $runnerArgs.Add('--repo-root'); $runnerArgs.Add($RepoRoot)
    $runnerArgs.Add('--output-root'); $runnerArgs.Add($OutputRoot)
    $runnerArgs.Add('--mode'); $runnerArgs.Add('execute')
    $runnerArgs.Add('--expected-activation-head'); $runnerArgs.Add($ExpectedActivationHead)
    $runnerArgs.Add('--committed-tooling-sha'); $runnerArgs.Add($CommittedTrainingToolingSha)
    Add-OptionalArgument $runnerArgs '--command' $Command
    & $PythonExecutable $runnerPath @runnerArgs
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
