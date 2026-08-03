[CmdletBinding()]
param(
    [string]$RepoRoot = 'C:\Users\johns\satnet-stage-a-tgnn-classification-training-v1-activation',
    [string]$DatasetRoot = 'C:\Users\johns\Developer\satnet-arch-dss-mac\artifacts\stage_a_tgnn_dataset_v1',
    [string]$OutputRoot = 'C:\Users\johns\satnet-stage-a-tgnn-classification-training-v1',
    [Parameter(Mandatory = $true)]
    [string]$ExpectedActivationHead,
    [string]$CommittedTrainingToolingSha = 'fa314c07cdb3c60186365888e9c726a0e4a567e3',
    [string]$PythonExecutable = 'C:\Users\johns\satnet-stage-a-tgnn-runtime-v1\.venv\Scripts\python.exe',
    [AllowNull()][string]$CommandLine
)

$ErrorActionPreference = 'Stop'
$transcriptDirectory = Join-Path (Split-Path -Parent $OutputRoot) 'satnet-stage-a-tgnn-classification-training-v1-transcripts'
New-Item -ItemType Directory -Force -Path $transcriptDirectory | Out-Null
$timestamp = Get-Date -Format 'yyyyMMddTHHmmssfffK'
$transcriptPath = Join-Path $transcriptDirectory "tgnn-classification-training-$timestamp.log"
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
    if ($branch -ne 'activate/stage-a-tgnn-classification-training-v1') { throw "Activation branch mismatch: $branch" }
    $status = (& git -C $RepoRoot status --porcelain) -join "`n"
    if (-not [string]::IsNullOrWhiteSpace($status)) { throw 'Activation worktree is not clean' }
    $tooling = (& git -C $RepoRoot rev-list -n 1 HEAD -- scripts/run_stage_a_tgnn_classification_training.py).Trim()
    if ($tooling -ne $CommittedTrainingToolingSha) { throw "Training-tooling identity mismatch: $tooling" }
    if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) { throw "Dedicated Python executable missing: $PythonExecutable" }
    if (Test-Path -LiteralPath $OutputRoot) { throw "Output root already exists: $OutputRoot" }
    $lockPath = "$OutputRoot.lock"
    if (Test-Path -LiteralPath $lockPath) { throw "Training lock already exists: $lockPath" }
    $runnerPath = Join-Path $RepoRoot 'scripts/run_stage_a_tgnn_classification_training.py'
    $runnerArgs = [System.Collections.Generic.List[string]]::new()
    $runnerArgs.Add('--mode'); $runnerArgs.Add('execute')
    $runnerArgs.Add('--repo-root'); $runnerArgs.Add($RepoRoot)
    $runnerArgs.Add('--dataset-root'); $runnerArgs.Add($DatasetRoot)
    $runnerArgs.Add('--output-root'); $runnerArgs.Add($OutputRoot)
    $runnerArgs.Add('--expected-activation-head'); $runnerArgs.Add($ExpectedActivationHead)
    $runnerArgs.Add('--committed-tooling-sha'); $runnerArgs.Add($CommittedTrainingToolingSha)
    Add-OptionalArgument $runnerArgs '--command-line' $CommandLine
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
