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
    "run2"     { & $PY scripts\run_stage2.py }
    "run2raw"  { & $PY scripts\run_stage2.py --raw }
    "smoke2"   { & $PY scripts\run_stage2.py --smoke }
    "tables2"  { & $PY scripts\make_tables_stage2.py }
    "tests"    { & $PY tests\test_flow_matching.py }
    "figures2" { & $PY scripts\make_figures_stage2.py }
    "notebook2" {
        & $PY notebooks\build_notebook.py 2
        & $PY -m jupyter nbconvert --to notebook --execute --inplace notebooks\stage2_presentation.ipynb
    }
    "run3"     { & $PY scripts\run_stage3.py }
    "smoke3"   { & $PY scripts\run_stage3.py --smoke }
    "tables3"  { & $PY scripts\make_tables_stage3.py }
    "figures3" { & $PY scripts\make_figures_stage3.py }
    "notebook3" {
        & $PY notebooks\build_notebook.py 3
        & $PY -m jupyter nbconvert --to notebook --execute --inplace notebooks\stage3_presentation.ipynb
    }
    default    { Write-Host "Unknown task '$Task'. Tasks: setup data extract run smoke tables figures notebook check tests run2 run2raw smoke2 tables2 figures2 notebook2 run3 smoke3 tables3 figures3 notebook3" }
}
