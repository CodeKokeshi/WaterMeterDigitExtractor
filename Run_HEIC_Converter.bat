@echo off
setlocal
TITLE HEIC Batch Converter

cd /d "%~dp0"

echo ========================================================
echo             HEIC Batch Converter Launcher
echo ========================================================
echo.

set "PYTHON_CMD="
py -3.14 -c "import sys" >nul 2>&1 && set "PYTHON_CMD=py -3.14"
if not defined PYTHON_CMD (
    py -3 -c "import sys" >nul 2>&1 && set "PYTHON_CMD=py -3"
)
if not defined PYTHON_CMD (
    python -c "import sys" >nul 2>&1 && set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    echo [ERROR] Python is not installed or not available in PATH.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [*] Creating local virtual environment .venv ...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create .venv
        pause
        exit /b 1
    )
)

echo [*] Activating .venv ...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate .venv
    pause
    exit /b 1
)

echo [*] Installing required packages...
python -m pip install --upgrade pip --quiet
if errorlevel 1 goto :install_error

python -m pip install -r requirements.txt --quiet
if errorlevel 1 goto :install_error

echo [*] Launching HEIC Batch Converter...
python heic_batch_converter.py
set "APP_EXIT=%ERRORLEVEL%"
deactivate >nul 2>&1

if not "%APP_EXIT%"=="0" (
    echo.
    echo [ERROR] Converter closed with an error.
    pause
)

exit /b %APP_EXIT%

:install_error
echo.
echo [ERROR] Failed to install requirements.
deactivate >nul 2>&1
pause
exit /b 1
