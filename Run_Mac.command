#!/usr/bin/env bash

cd "$(dirname "$0")"

echo "========================================================"
echo "      DigitExtractor - Desktop Launcher (macOS/Linux)   "
echo "========================================================"
echo ""

PYTHON_CMD=""
for candidate in python3.14 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
            PYTHON_CMD="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "[ERROR] Python 3.10 or newer is required to run main.py."
    echo "[INFO] Python 3.14 is recommended for the app runtime."
    echo "[INFO] Install Python from python.org or your package manager."
    read -p "Press [Enter] to exit..."
    exit 1
fi

PYTHON_VERSION=$("$PYTHON_CMD" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
echo "[*] Using Python $PYTHON_VERSION"

if [ ! -d ".venv" ]; then
    echo "[*] First time setup: Creating local virtual environment .venv ..."
    "$PYTHON_CMD" -m venv .venv || {
        echo "[ERROR] Failed to create .venv"
        read -p "Press [Enter] to exit..."
        exit 1
    }
fi

echo "[*] Installing runtime dependencies for main.py..."
source .venv/bin/activate || {
    echo "[ERROR] Failed to activate .venv"
    read -p "Press [Enter] to exit..."
    exit 1
}

python -m pip install --upgrade pip --quiet || {
    echo ""
    echo "[ERROR] Failed to upgrade pip."
    deactivate >/dev/null 2>&1 || true
    read -p "Press [Enter] to exit..."
    exit 1
}

python -m pip install -r requirements.txt --quiet || {
    echo ""
    echo "[ERROR] Failed to install runtime requirements from requirements.txt"
    echo "[INFO] Training dependencies are not installed by this launcher."
    deactivate >/dev/null 2>&1 || true
    read -p "Press [Enter] to exit..."
    exit 1
}

echo "[*] Launching application..."
python main.py
APP_EXIT=$?

deactivate >/dev/null 2>&1 || true

if [ "$APP_EXIT" -ne 0 ]; then
    echo ""
    echo "[ERROR] Application crashed. See error above."
    read -p "Press [Enter] to exit..."
fi

exit "$APP_EXIT"
