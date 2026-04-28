@echo off
setlocal
TITLE DigitExtractor Launcher

cd /d "%~dp0"

echo ========================================================
echo        DigitExtractor - Desktop Launcher (Windows)
echo ========================================================
echo.

set "PYTHON_CMD="
py -3.14 -c "import sys" >nul 2>&1 && set "PYTHON_CMD=py -3.14"
if not defined PYTHON_CMD (
    py -c "import sys" >nul 2>&1 && set "PYTHON_CMD=py"
)
if not defined PYTHON_CMD (
    python -c "import sys" >nul 2>&1 && set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    echo [ERROR] Python is not installed or not available in PATH.
    echo Please install Python 3.10 or newer.
    echo Python 3.14 is recommended for running main.py.
    echo https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

%PYTHON_CMD% -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] DigitExtractor needs Python 3.10 or newer to run main.py.
    echo [INFO] Python 3.14 is recommended for the app runtime.
    echo [INFO] Training remains separate in training_requirements.txt if needed later.
    echo.
    pause
    exit /b 1
)

for /f "delims=" %%V in ('%PYTHON_CMD% -c "import sys; print(sys.version.split()[0])"') do set "PYTHON_VERSION=%%V"
echo [*] Using Python %PYTHON_VERSION%

if not exist ".venv\Scripts\python.exe" (
    echo [*] First time setup: Creating local virtual environment .venv ...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create .venv
        echo.
        pause
        exit /b 1
    )
)

echo [*] Installing runtime dependencies for main.py...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate .venv
    echo.
    pause
    exit /b 1
)

python -m pip install --upgrade pip --quiet
if errorlevel 1 goto :install_error

python -m pip install -r requirements.txt --quiet
if errorlevel 1 goto :install_error

echo [*] Launching application...
python main.py
set "APP_EXIT=%ERRORLEVEL%"
deactivate >nul 2>&1

if not "%APP_EXIT%"=="0" (
    echo.
    echo [ERROR] Application crashed. See error above.
    pause
)

exit /b %APP_EXIT%

:install_error
echo.
echo [ERROR] Failed to install runtime requirements from requirements.txt
echo [INFO] Training dependencies are not installed by this launcher.
deactivate >nul 2>&1
pause
exit /b 1
