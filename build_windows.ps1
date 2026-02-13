# Build script for Windows
# Creates DigitExtractor.exe
# 
# Usage: Run this script in PowerShell
#        .\build_windows.ps1

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Building DigitExtractor for Windows (.exe)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
$venvPath = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & $venvPath
} else {
    Write-Host "Virtual environment not found. Please run setup first." -ForegroundColor Red
    exit 1
}

# Clean previous builds
if (Test-Path ".\build") {
    Write-Host "Cleaning previous build artifacts..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force .\build
}
if (Test-Path ".\dist") {
    Write-Host "Cleaning previous distribution..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force .\dist
}

# Build with PyInstaller
Write-Host ""
Write-Host "Building executable with PyInstaller..." -ForegroundColor Green
& python -m PyInstaller DigitExtractor.spec

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Green
    Write-Host " Build successful!" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Your app is ready in:" -ForegroundColor Yellow
    Write-Host "  .\dist\DigitExtractor\" -ForegroundColor White
    Write-Host ""
    Write-Host "To run: .\dist\DigitExtractor\DigitExtractor.exe" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To distribute: Zip the entire 'DigitExtractor' folder" -ForegroundColor Magenta
} else {
    Write-Host ""
    Write-Host "Build failed. Check the error messages above." -ForegroundColor Red
    exit 1
}
