# New_KG_Benchamark

A lightweight setup to manage Python dependencies with a single script. Run these steps after cloning the repo.

## Quick Start

1) Install prerequisites
- Python 3.8+ installed and on PATH
- Git (optional, for cloning)

2) Create a virtual environment (once)
```powershell
# PowerShell
python -m venv .\venv
```
```cmd
:: Command Prompt (CMD)
python -m venv .\venv
```

3) Activate the virtual environment
```powershell
# PowerShell
.\venv\Scripts\Activate.ps1
```
```cmd
:: CMD
.\venv\Scripts\activate.bat
```

4) Sync dependencies (after activating the venv)
- If you already have a `requirements.txt`, install exactly those:
```powershell
.\sync_deps.bat install
```
```cmd
sync_deps.bat install
```
- If you don’t have requirements yet, either add packages then freeze:
```powershell
.\sync_deps.bat add requests uvicorn
.\sync_deps.bat freeze
```

## Dependency Sync Script
Use `sync_deps.bat` from an active venv. It works in both PowerShell and CMD.

- Install from requirements:
  - PowerShell: `.\sync_deps.bat install`
  - CMD: `sync_deps.bat install`
- Freeze current environment to `requirements.txt`:
  - PowerShell: `.\sync_deps.bat freeze`
  - CMD: `sync_deps.bat freeze`
- Sync to match requirements (install missing, uninstall extras):
  - PowerShell: `.\sync_deps.bat sync`
  - CMD: `sync_deps.bat sync`
- Add packages and update requirements:
  - PowerShell: `.\sync_deps.bat add <pkg> [more...]`
  - CMD: `sync_deps.bat add <pkg> [more...]`
- Remove packages and update requirements:
  - PowerShell: `.\sync_deps.bat remove <pkg> [more...]`
  - CMD: `sync_deps.bat remove <pkg> [more...]`
- Upgrade all listed in requirements and re-freeze:
  - PowerShell: `.\sync_deps.bat upgrade`
  - CMD: `sync_deps.bat upgrade`
- Environment diagnostics:
  - PowerShell: `.\sync_deps.bat check`
  - CMD: `sync_deps.bat check`

Notes
- The script prefers `requirements.txt` and falls back to `requirment.txt` if present.
- Always activate your venv before running the script, so it targets the correct Python.

## Running the Project
This repository doesn’t define a specific entry-point yet. Common patterns:
- Module entry point: `python -m your_package`
- Script: `python main.py`

Adjust the command to your code structure once available.

## KG Pattern Statistics
Detect symmetry, anti-symmetry, and composition patterns for any KG triples.

Files supported:
- A folder with `train.txt`, `valid.txt`, `test.txt` (whitespace `h r t` per line)
- Or pass explicit files with `--files`
 - Or load a PyKEEN dataset with `--pykeen`

Examples:
- Folder (e.g., FB15k-237):
  - PowerShell: `python .\kg_pattern_stats.py --data path\to\dataset_folder`
  - CMD: `python kg_pattern_stats.py --data path\to\dataset_folder`
- Explicit files:
  - PowerShell: `python .\kg_pattern_stats.py --files .\train.txt .\valid.txt .\test.txt`
  - CMD: `python kg_pattern_stats.py --files .\train.txt .\valid.txt .\test.txt`
- PyKEEN dataset (requires `pip install pykeen`):
  - PowerShell: `python .\kg_pattern_stats.py --pykeen biokg`
  - CMD: `python kg_pattern_stats.py --pykeen biokg`
  - Also works: `--pykeen BioKG` or `--pykeen pykeen.datasets.biokg:BioKG`

Optional thresholds:
- `--sym-min-ratio 0.5`   minimum fraction of edges that have reverse for symmetry
- `--anti-max-ratio 0.0`  maximum reciprocal fraction (off-diagonal) for anti-symmetry
- `--comp-min-support 10` minimum composed edges explaining a relation
- `--comp-min-ratio 0.1`  minimum coverage of a relation by a composition

Output example:
```
pattern	dataset	anti-symmetry	composition	symmetry
fb15k237	205	147	3
```

## Troubleshooting
- PowerShell execution policy blocks activation
  - Run in the current session: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
  - Then activate: `./venv/Scripts/Activate.ps1`
- No active venv detected
  - Make sure you activated the venv (see step 3) before calling `sync_deps.bat`.
- Pip not found or wrong Python
  - Ensure `python` resolves to the interpreter inside `venv` after activation: `where python`
