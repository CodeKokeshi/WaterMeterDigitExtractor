#!/usr/bin/env bash

# Remove set -e to allow catching errors properly
# Change to the directory where the script is located
cd "$(dirname "$0")"

echo "========================================================"
echo "      DigitExtractor - Desktop Launcher (macOS/Linux)   "
echo "========================================================"
echo ""

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not in your PATH."
    echo "Please install Python 3. (e.g., download from python.org or run 'brew install python')"
    read -p "Press [Enter] to exit..."
    exit 1
fi

# Check if the .venv folder exists, if not, create it
if [ ! -d ".venv" ]; then
    echo "[*] First time setup: Creating hidden local virtual environment .venv ..."
    python3 -m venv .venv
fi

# Activate the environment
echo "[*] Verifying dependencies..."
source .venv/bin/activate

# Upgrade pip and install requirements quietly
python3 -m pip install --upgrade pip --quiet
python3 -m pip install -r requirements.txt --quiet

# Launch the Application
echo "[*] Launching application..."
python3 main.py

# Check for crash
if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Application crashed. See error above."
    read -p "Press [Enter] to exit..."
fi

# Deactivate gracefully when main.py exits
deactivate
