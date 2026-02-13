# Quick Git Setup Script
# Run this to initialize and push to GitHub

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Git Setup for DigitExtractor" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is installed
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Git is not installed!" -ForegroundColor Red
    Write-Host "Download from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

# Check if already initialized
if (Test-Path ".git") {
    Write-Host "Git repository already initialized." -ForegroundColor Green
} else {
    Write-Host "Initializing git repository..." -ForegroundColor Yellow
    git init
    git branch -M main
}

# Add files
Write-Host ""
Write-Host "Adding files to git..." -ForegroundColor Yellow
git add .gitignore
git add main.py requirements.txt DigitExtractor.spec
git add build_windows.ps1 build_mac.sh
git add *.md
git add .github/

# Show status
Write-Host ""
Write-Host "Files to be committed:" -ForegroundColor Cyan
git status --short

# Commit
Write-Host ""
$commitMsg = Read-Host "Enter commit message (or press Enter for default)"
if ([string]::IsNullOrWhiteSpace($commitMsg)) {
    $commitMsg = "Initial commit: DigitExtractor with GitHub Actions"
}

git commit -m $commitMsg

# GitHub setup
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " GitHub Repository Setup" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Go to https://github.com/new" -ForegroundColor White
Write-Host "2. Create a new repository named: DigitExtractor" -ForegroundColor White
Write-Host "3. Do NOT initialize with README (we have files already)" -ForegroundColor White
Write-Host "4. Copy the repository URL (it looks like: https://github.com/YOUR-USERNAME/DigitExtractor.git)" -ForegroundColor White
Write-Host ""
$repoUrl = Read-Host "Paste your GitHub repository URL here"

if ([string]::IsNullOrWhiteSpace($repoUrl)) {
    Write-Host ""
    Write-Host "Skipping remote setup. You can add it later with:" -ForegroundColor Yellow
    Write-Host "  git remote add origin YOUR-REPO-URL" -ForegroundColor White
    Write-Host "  git push -u origin main" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "Adding remote..." -ForegroundColor Yellow
    git remote remove origin 2>$null
    git remote add origin $repoUrl
    
    Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
    git push -u origin main
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host " SUCCESS!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Your code is now on GitHub!" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "To build for Windows + Mac:" -ForegroundColor Yellow
        Write-Host "1. Go to: $repoUrl" -ForegroundColor White
        Write-Host "2. Click 'Actions' tab" -ForegroundColor White
        Write-Host "3. Click 'Build DigitExtractor for Windows & Mac'" -ForegroundColor White
        Write-Host "4. Click 'Run workflow' -> 'Run workflow'" -ForegroundColor White
        Write-Host "5. Wait 5-10 minutes" -ForegroundColor White
        Write-Host "6. Download artifacts (both Windows and Mac builds)" -ForegroundColor White
        Write-Host ""
        Write-Host "See GITHUB_BUILD_GUIDE.md for detailed instructions." -ForegroundColor Magenta
    } else {
        Write-Host ""
        Write-Host "Push failed. You may need to authenticate." -ForegroundColor Red
        Write-Host "Try using GitHub Desktop or configure Git credentials:" -ForegroundColor Yellow
        Write-Host "  https://docs.github.com/get-started/getting-started-with-git/set-up-git" -ForegroundColor White
    }
}
