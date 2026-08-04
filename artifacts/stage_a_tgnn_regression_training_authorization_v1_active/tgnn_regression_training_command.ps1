[CmdletBinding()]
param(
    [string]$RepoRoot = 'C:\Users\johns\satnet-stage-a-tgnn-regression-training-v1-activation',
    [string]$DatasetRoot = 'C:\Users\johns\Developer\satnet-arch-dss-mac\artifacts\stage_a_tgnn_dataset_v1',
    [string]$OutputRoot = 'C:\Users\johns\satnet-stage-a-tgnn-regression-training-v1',
    [string]$ExpectedActivationHead = 'd9cb035abff3668ccdba80039523110043ecf6ac',
    [string]$CommittedTrainingToolingSha = 'd9cb035abff3668ccdba80039523110043ecf6ac',
    [string]$PythonExecutable = 'C:\Users\johns\satnet-stage-a-tgnn-runtime-v1\.venv\Scripts\python.exe',
    [AllowNull()][string]$CommandLine
)

$ErrorActionPreference = 'Stop'
$transcriptDirectory = Join-Path (Split-Path -Parent $OutputRoot) 'satnet-stage-a-tgnn-regression-training-v1-transcripts'
New-Item -ItemType Directory -Force -Path $transcriptDirectory | Out-Null
$timestamp = Get-Date -Format 'yyyyMMddTHHmmssfffK'
$transcriptPath = Join-Path $transcriptDirectory "tgnn-regression-training-$timestamp.log"
$exitCode = 1
$expectedTrainIndexSha = 'f590fd975950bd6ab1b9cac26cd01e8d8826751774cf48ddb02f301a7dbe6271'
$expectedValidationIndexSha = 'cfc3d0c13898b0d24f2c273bfc9d15c11bbf20607900146e8fa5aa0d0815d361'

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
    if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) { throw "Dedicated Python executable missing: $PythonExecutable" }
    $pythonResolved = (Resolve-Path -LiteralPath $PythonExecutable).Path
    $expectedPythonResolved = (Resolve-Path -LiteralPath 'C:\Users\johns\satnet-stage-a-tgnn-runtime-v1\.venv\Scripts\python.exe').Path
    if ($pythonResolved -ne $expectedPythonResolved) { throw "Dedicated runtime identity mismatch: $pythonResolved" }
    $head = (& git -C $RepoRoot rev-parse HEAD).Trim()
    if ($head -ne $ExpectedActivationHead) { throw "Activation HEAD mismatch: $head" }
    $branch = (& git -C $RepoRoot branch --show-current).Trim()
    if ($branch -ne 'activate/stage-a-tgnn-regression-training-v1') { throw "Activation branch mismatch: $branch" }
    $status = (& git -C $RepoRoot status --porcelain) -join "`n"
    if (-not [string]::IsNullOrWhiteSpace($status)) { throw 'Activation worktree is not clean' }
    $tooling = (& git -C $RepoRoot rev-list -n 1 HEAD -- scripts/run_stage_a_tgnn_regression_training.py).Trim()
    if ($tooling -ne $CommittedTrainingToolingSha) { throw "Training-tooling identity mismatch: $tooling" }
    $trainIndex = Join-Path $DatasetRoot 'tgnn_train_index.jsonl'
    $validationIndex = Join-Path $DatasetRoot 'tgnn_validation_index.jsonl'
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $trainIndex).Hash.ToLowerInvariant() -ne $expectedTrainIndexSha) { throw 'Train index identity mismatch' }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $validationIndex).Hash.ToLowerInvariant() -ne $expectedValidationIndexSha) { throw 'Validation index identity mismatch' }
    if (Test-Path -LiteralPath $OutputRoot) { throw "Output root already exists: $OutputRoot" }
    $lockPath = "$OutputRoot.lock"
    if (Test-Path -LiteralPath $lockPath) { throw "Training lock already exists: $lockPath" }
    $runnerPath = Join-Path $RepoRoot 'scripts/run_stage_a_tgnn_regression_training.py'
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
