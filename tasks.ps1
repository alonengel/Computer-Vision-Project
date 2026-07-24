# One-word task runner. Usage: .\tasks.ps1 <task>
param([Parameter(Mandatory = $true)][string]$Task)

$PY = "C:\Users\Alon\Desktop\cv-ex2\rocm_win312\Scripts\python.exe"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
Set-Location $PSScriptRoot

switch ($Task) {
    "setup"    { & $PY scripts\setup_check.py }
    "data"     { & $PY scripts\prepare_data.py }
    "extract"  { & $PY scripts\extract_features.py }
    "run"      { & $PY scripts\run_experiments.py }
    "smoke"    { & $PY scripts\run_experiments.py --smoke }
    "tables"   { & $PY scripts\make_tables.py }
    "figures"  { & $PY scripts\make_figures.py }
    "notebook" {
        & $PY notebooks\build_notebook.py
        & $PY -m jupyter nbconvert --to notebook --execute --inplace notebooks\stage1_presentation.ipynb
    }
    "check"    { & $PY scripts\repro_check.py }
    default    { Write-Host "Unknown task '$Task'. Tasks: setup data extract run smoke tables figures notebook check" }
}
