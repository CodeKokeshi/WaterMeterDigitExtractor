#!/bin/bash
# Build script for macOS
# Creates DigitExtractor.app
# 
# Usage: chmod +x build_mac.sh && ./build_mac.sh

echo "============================================"
echo " Building DigitExtractor for macOS (.app)"
echo "============================================"
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "ERROR: This script must be run on macOS to build .app bundles"
    echo "Current OS: $OSTYPE"
    exit 1
fi

# Activate virtual environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "ERROR: Virtual environment not found."
    echo "Please create one first: python3 -m venv .venv"
    echo "Then install dependencies: pip install -r requirements.txt"
    exit 1
fi

# Clean previous builds
if [ -d "build" ]; then
    echo "Cleaning previous build artifacts..."
    rm -rf build
fi
if [ -d "dist" ]; then
    echo "Cleaning previous distribution..."
    rm -rf dist
fi

# Build with PyInstaller
echo ""
echo "Building .app bundle with PyInstaller..."
python -m PyInstaller DigitExtractor.spec

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo " Build successful!"
    echo "============================================"
    echo ""
    echo "Your app is ready in:"
    echo "  ./dist/DigitExtractor.app"
    echo ""
    echo "To run: open ./dist/DigitExtractor.app"
    echo "Or double-click DigitExtractor.app in Finder"
    echo ""
    echo "To distribute: Zip the .app file or create a DMG"
    echo ""
    echo "Optional: Code sign the app (requires Apple Developer ID):"
    echo "  codesign --deep --force --verify --verbose --sign \"Developer ID\" ./dist/DigitExtractor.app"
else
    echo ""
    echo "Build failed. Check the error messages above."
    exit 1
fi
