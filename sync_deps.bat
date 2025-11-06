@echo off
setlocal ENABLEDELAYEDEXPANSION

rem ------------------------------------------------------------------
rem Sync Python project libraries for an active virtual environment.
rem Usage examples (run AFTER activating your venv):
rem   sync_deps.bat install        => install from requirements
rem   sync_deps.bat freeze         => write current env to requirements
rem   sync_deps.bat sync           => install + uninstall extras to match requirements
rem   sync_deps.bat add <pkgs...>  => pip install pkgs, then freeze
rem   sync_deps.bat remove <pkgs>  => pip uninstall pkgs, then freeze
rem   sync_deps.bat upgrade        => upgrade all from requirements, then freeze
rem   sync_deps.bat check          => environment diagnostics
rem   sync_deps.bat help           => show help
rem ------------------------------------------------------------------

set "REQ=requirements.txt"
if not exist "%REQ%" (
  if exist "requirment.txt" (
    set "REQ=requirment.txt"
  )
)

set "PYCMD=python"
set "PIPCMD=python -m pip"

if "%~1"=="" goto :help

rem Ensure we're in an active venv (VIRTUAL_ENV is set by venv activation scripts)
if not defined VIRTUAL_ENV (
  echo [ERROR] No active virtual environment detected.
  echo Activate your venv first, then re-run this script.
  exit /b 1
)

rem Basic sanity checks
%PYCMD% --version >NUL 2>&1
if errorlevel 1 (
  echo [ERROR] python is not available on PATH.
  exit /b 1
)
%PIPCMD% --version >NUL 2>&1
if errorlevel 1 (
  echo [ERROR] pip is not available in this environment.
  exit /b 1
)

set "CMD=%~1"
shift

if /I "%CMD%"=="help"    goto :help
if /I "%CMD%"=="install" goto :install
if /I "%CMD%"=="freeze"  goto :freeze
if /I "%CMD%"=="sync"    goto :sync
if /I "%CMD%"=="add"     goto :add
if /I "%CMD%"=="remove"  goto :remove
if /I "%CMD%"=="upgrade" goto :upgrade
if /I "%CMD%"=="check"   goto :check

echo [ERROR] Unknown command: %CMD%
goto :help

:require_file_guard
if not exist "%REQ%" (
  echo [WARN] "%REQ%" not found in %CD%
  echo        Create it with ^`%~nx0 freeze^` or choose the correct file name.
  exit /b 2
)
goto :eof

:install
call :require_file_guard || exit /b %errorlevel%
echo [INFO] Upgrading pip to latest...
%PIPCMD% install --upgrade pip >NUL
echo [INFO] Installing from "%REQ%"...
%PIPCMD% install -r "%REQ%"
exit /b %errorlevel%

:freeze
echo [INFO] Freezing current environment to "%REQ%"...
%PIPCMD% freeze > "%REQ%"
if errorlevel 1 exit /b %errorlevel%
echo [OK] Wrote %REQ%
exit /b 0

:upgrade
call :require_file_guard || exit /b %errorlevel%
echo [INFO] Upgrading all packages listed in "%REQ%"...
%PIPCMD% install --upgrade -r "%REQ%"
if errorlevel 1 exit /b %errorlevel%
echo [INFO] Re-freezing lock file...
%PIPCMD% freeze > "%REQ%"
exit /b %errorlevel%

:add
rem Robustly extract package args (ignore leading keyword if present)
set "PKGS="
for /f "tokens=1* delims= " %%A in ("%*") do (
  if /I "%%~A"=="add" (
    set "PKGS=%%~B"
  ) else (
    set "PKGS=%*"
  )
)
if not defined PKGS (
  echo [ERROR] No packages specified. Example: %~nx0 add requests uvicorn
  exit /b 1
)
echo [INFO] Installing: !PKGS!
%PIPCMD% install !PKGS!
if errorlevel 1 exit /b %errorlevel%
echo [INFO] Updating "%REQ%"...
%PIPCMD% freeze > "%REQ%"
exit /b %errorlevel%

:remove
rem Robustly extract package args (ignore leading keyword if present)
set "PKGS="
for /f "tokens=1* delims= " %%A in ("%*") do (
  if /I "%%~A"=="remove" (
    set "PKGS=%%~B"
  ) else (
    set "PKGS=%*"
  )
)
if not defined PKGS (
  echo [ERROR] No packages specified. Example: %~nx0 remove requests uvicorn
  exit /b 1
)
echo [INFO] Uninstalling: !PKGS!
%PIPCMD% uninstall -y !PKGS!
if errorlevel 1 exit /b %errorlevel%
echo [INFO] Updating "%REQ%"...
%PIPCMD% freeze > "%REQ%"
exit /b %errorlevel%

:sync
call :require_file_guard || exit /b %errorlevel%
echo [INFO] Ensuring installed packages exactly match "%REQ%"...
echo        1) Install any missing/older packages
%PIPCMD% install -r "%REQ%"
if errorlevel 1 exit /b %errorlevel%

echo        2) Compute packages not in requirements to uninstall
set "TMPPY=%TEMP%\pip_sync_%RANDOM%_%TIME:~6,2%%TIME:~3,2%%TIME:~0,2%.py"
set "TMPUN=%TEMP%\pip_uninstall_%RANDOM%.txt"
(
  echo import sys, re
  echo from importlib import metadata
  echo req = sys.argv[1]
  echo keep = set()
  echo pat = re.compile(r"\[|==|>=|<=|~=|!=|>|<")
  echo with open(req, 'r', encoding='utf-8', errors='ignore') as f:
  echo 	for line in f:
  echo 		s = line.strip()
  echo 		if not s or s.startswith('#') or s.startswith('-'):
  echo 			continue
  echo 		s = s.split(';')[0].strip()
  echo 		name = pat.split(s)[0].strip()
  echo 		if name:
  echo 			keep.add(name.lower().replace('_','-'))
  echo installed = set()
  echo for d in metadata.distributions():
  echo 	try:
  echo 		name = d.metadata['Name']
  echo 	except Exception:
  echo 		name = d.metadata.get('Name') if hasattr(d,'metadata') else None
  echo 	if name:
  echo 		installed.add(name.lower().replace('_','-'))
  echo base = {'pip','setuptools','wheel'}
  echo extras = sorted(p for p in installed - keep - base)
  echo sys.stdout.write("\n".join(extras))
) > "%TMPPY%"

%PYCMD% "%TMPPY%" "%REQ%" > "%TMPUN%"
set "PYERR=%ERRORLEVEL%"
del /q "%TMPPY%" 2>NUL
if not "%PYERR%"=="0" (
  echo [ERROR] Failed computing uninstall set. Aborting sync.
  del /q "%TMPUN%" 2>NUL
  exit /b %PYERR%
)

for /f "usebackq delims=" %%P in ("%TMPUN%") do (
  if not "%%~P"=="" (
    echo [UNINSTALL] %%~P
    %PIPCMD% uninstall -y "%%~P"
    if errorlevel 1 (
      echo [WARN] Uninstall failed for %%~P
    )
  )
)
del /q "%TMPUN%" 2>NUL

echo [INFO] Re-freezing "%REQ%" to reflect final state...
%PIPCMD% freeze > "%REQ%"
exit /b %errorlevel%

:check
echo [INFO] Diagnostics
echo ----------------------------------------
echo Working dir: %CD%
echo Virtualenv : %VIRTUAL_ENV%
echo Python     :
%PYCMD% --version
echo Path to python:
where python
echo Pip        :
%PIPCMD% --version
echo Path to pip:
where pip
echo Requirements file: %REQ%
if exist "%REQ%" (
  for %%A in ("%REQ%") do echo Requirements mtime: %%~tA ^| size: %%~zA bytes
) else (
  echo Requirements file not found.
)
echo Installed summary:
%PIPCMD% list
exit /b 0

:help
echo Usage: %~nx0 ^<command^> [args]
echo.
echo Commands:
echo   install           Install packages from requirements file
echo   freeze            Write current venv packages to requirements file
echo   sync              Install from requirements AND uninstall extras
echo   add ^<pkgs...^>    Install package(s) and update requirements
echo   remove ^<pkgs...^> Uninstall package(s) and update requirements
echo   upgrade           Upgrade all from requirements and update file
echo   check             Show environment diagnostics
echo   help              Show this help
echo.
echo Notes:
echo - Looks for "requirements.txt" first; falls back to "requirment.txt" if present.
echo - Run this after activating your virtualenv so it targets the right Python.
echo - Sync will attempt to uninstall any installed package not in requirements.
exit /b 1
