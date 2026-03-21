@echo off
TITLE DigitExtractor Launcher

:: Ensure we are running in the directory where the script is located
cd /d "%~dp0"

echo ========================================================
echo        DigitExtractor - Desktop Launcher (Windows)      
echo ========================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in your system PATH!
    echo Please download and install Python 3.10 or newer from https://www.python.org/
    echo IMPORTANT: Make sure to check the box "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

:: Check if virtual environment exists, if not, create it
IF NOT EXIST ".venv" (
    echo [*] First time setup: Creating hidden local virtual environment .venv ...
    python -m venv .venv
)

:: Activate environment and install/update requirements
echo [*] Verifying dependencies...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

:: Launch the main application
echo [*] Launching application...
python main.py

:: If the app crashes, this pause stops the window from immediately disappearing
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application crashed. See error above.
    pause
)
