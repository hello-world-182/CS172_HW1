# Generate the matched train/validation sets, train, evaluate test0-test4 once,
# and write the PDF with the per-folder results.
#
#   conda activate pytorch
#   powershell -ExecutionPolicy Bypass -File scripts/run_final.ps1
#
# The test folders are only read by the final evaluation step.
param(
    [int]$TrainImages = 200000,
    [int]$ValImages = 10000,
    [int]$Epochs = 15,
    [int]$BatchSize = 64,
    [int]$Workers = 4,
    [string]$Model = "captcha_seq"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Invoke-Python([string[]]$Arguments) {
    Write-Host ">> python $($Arguments -join ' ')" -ForegroundColor Cyan
    & python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "step failed: python $($Arguments -join ' ')" }
}

Invoke-Python @("scripts/smoke_test.py")

$train = "data/training_matched.zip"
$val = "data/validation_matched.zip"

# 16 preview images so the rendering can be checked by eye.
if (-not (Test-Path "outputs/matched_preview")) {
    Invoke-Python @("scripts/generate_data.py", "outputs/matched_preview", "-n", "16", "--seed", "7", "--matched", "--jpeg-quality-range", "60", "95")
}
if (-not (Test-Path $train)) {
    Invoke-Python @("scripts/generate_data.py", $train, "--zip", "-n", "$TrainImages", "--seed", "42", "--matched", "--jpeg-quality-range", "60", "95")
}
if (-not (Test-Path $val)) {
    Invoke-Python @("scripts/generate_data.py", $val, "--zip", "-n", "$ValImages", "--seed", "1000042", "--matched", "--jpeg-quality-range", "60", "95")
}

Invoke-Python @(
    "scripts/train.py", "--data-dir", $train, "--val-data-dir", $val, "--output-dir", "outputs/matched",
    "--model", $Model, "--epochs", "$Epochs", "--warmup-epochs", "1", "--batch-size", "$BatchSize",
    "--learning-rate", "0.001", "--weight-decay", "0.0001", "--augment", "--seed", "42", "--workers", "$Workers"
)

$run = Get-ChildItem "outputs/matched" -Directory | Sort-Object Name | Select-Object -Last 1
$checkpoint = Join-Path $run.FullName "checkpoints/best.pt"
Invoke-Python @(
    "scripts/evaluate.py", "--checkpoint", $checkpoint,
    "--data", "data/test/test0", "data/test/test1", "data/test/test2", "data/test/test3", "data/test/test4",
    "--device", "auto", "--output-json", "final_test_results.json"
)
Invoke-Python @("scripts/make_test_report.py", (Join-Path $run.FullName "final_test_results.json"), "--output", "output/pdf/test_results.pdf")
Write-Host "Done. Run directory: $($run.FullName)" -ForegroundColor Green
