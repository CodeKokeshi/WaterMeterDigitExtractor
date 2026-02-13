# 🚀 DigitExtractor - Build Instructions

Create standalone executables for **Windows (.exe)** and **macOS (.app)** so users can run the app with **one click** — no Python installation required!

---

## 📦 Building for Windows

### Prerequisites
- Windows 10/11
- Python 3.8+ installed

### Steps

1. **Clone/download the project**
   ```powershell
   cd D:\OS2025Dev\DigitExtractor
   ```

2. **Set up virtual environment** (if not already done)
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. **Build the executable**
   ```powershell
   .\build_windows.ps1
   ```

4. **Find your app**
   ```
   .\dist\DigitExtractor\DigitExtractor.exe
   ```

5. **Distribute**
   - Zip the entire `dist\DigitExtractor\` folder
   - Send to users — they just unzip and double-click `DigitExtractor.exe`
   - No Python needed on their machine! ✨

---

## 🍎 Building for macOS

### Prerequisites
- macOS 10.13+ (High Sierra or later)
- Python 3.8+ installed
- **Important:** You MUST be on a Mac to build .app bundles

### Steps

1. **Clone/download the project**
   ```bash
   cd ~/DigitExtractor
   ```

2. **Set up virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Make build script executable**
   ```bash
   chmod +x build_mac.sh
   ```

4. **Build the app**
   ```bash
   ./build_mac.sh
   ```

5. **Find your app**
   ```
   ./dist/DigitExtractor.app
   ```

6. **Test it**
   ```bash
   open ./dist/DigitExtractor.app
   ```

7. **Distribute**
   - **Option A:** Zip the `.app` file
     ```bash
     cd dist
     zip -r DigitExtractor.zip DigitExtractor.app
     ```
   - **Option B:** Create a DMG (more professional)
     ```bash
     # Install create-dmg if needed
     brew install create-dmg
     
     # Create DMG
     create-dmg \
       --volname "DigitExtractor" \
       --window-pos 200 120 \
       --window-size 600 400 \
       --icon-size 100 \
       --app-drop-link 450 185 \
       DigitExtractor.dmg \
       dist/DigitExtractor.app
     ```

---

## 🔒 Code Signing (Optional but Recommended)

### Windows
Users may see "Windows protected your PC" warning. To avoid this:
- Get a code signing certificate from a Certificate Authority
- Sign the .exe:
  ```powershell
  signtool sign /tr http://timestamp.digicert.com /td sha256 /fd sha256 /a DigitExtractor.exe
  ```

### macOS
Users may see "App is damaged and can't be opened" on Catalina+. To avoid this:
- Get an Apple Developer ID ($99/year)
- Sign and notarize:
  ```bash
  # Sign
  codesign --deep --force --verify --verbose \
    --sign "Developer ID Application: Your Name (TEAMID)" \
    --options runtime \
    DigitExtractor.app
  
  # Notarize (required for macOS 10.15+)
  xcrun notarytool submit DigitExtractor.zip \
    --apple-id "your@email.com" \
    --team-id "TEAMID" \
    --wait
  
  # Staple
  xcrun stapler staple DigitExtractor.app
  ```

---

## 🐛 Troubleshooting

### Windows: "VCRUNTIME140.dll not found"
Install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### macOS: "App can't be opened because Apple cannot check it"
Users should right-click → Open → Open (bypasses Gatekeeper first time)

### Build takes too long
- Disable UPX compression in `DigitExtractor.spec`: change `upx=True` to `upx=False`

### App is too large
- Use single-file mode (slower startup but single executable):
  In `DigitExtractor.spec`, change `EXE()` parameter:
  ```python
  exe = EXE(
      pyz,
      a.scripts,
      a.binaries,      # Add this
      a.zipfiles,      # Add this
      a.datas,         # Add this
      [],
      exclude_binaries=False,  # Change to False
      ...
  )
  # Remove the COLLECT section
  ```

---

## 📏 File Sizes (Approximate)

| Platform | Size | Notes |
|----------|------|-------|
| Windows .exe + deps | ~150 MB | Folder with DLLs |
| macOS .app | ~180 MB | Bundle with frameworks |
| Single-file .exe | ~180 MB | Slower startup |

Sizes are large because they include:
- Qt GUI framework (~80 MB)
- OpenCV libraries (~60 MB)
- Python runtime (~30 MB)

---

## 🎯 Cross-Platform Building

**Important:** To build for macOS, you need a Mac. To build for Windows, you need Windows.

**No Mac?** Options:
1. Use GitHub Actions (free CI/CD) to build automatically
2. Rent a cloud Mac (e.g., MacStadium, AWS EC2 Mac)
3. Ask a friend with a Mac to run the build script

**No Windows?** Same options apply.

---

## ✅ Testing Your Build

### Auto-test script (add this if needed)
```bash
# Test the built app automatically
pytest tests/  # If you add tests later
```

### Manual testing checklist
- ✅ App launches without errors
- ✅ Load folder works
- ✅ Image displays correctly
- ✅ Zoom/pan works
- ✅ 4-point selection works
- ✅ Extract & preview works
- ✅ Save segments creates proper folders
- ✅ App works on a different computer (without Python)

---

## 🔄 Updating the App

1. Make changes to `main.py`
2. Re-run the build script
3. Test thoroughly
4. Redistribute the new build
5. Update version number in `DigitExtractor.spec` if using semantic versioning

---

## 📝 Adding an Icon (Optional)

1. **Create icons:**
   - Windows: `icon.ico` (256x256, .ico format)
   - macOS: `icon.icns` (1024x1024 source, convert with `iconutil`)

2. **Update spec file:**
   ```python
   # In DigitExtractor.spec
   exe = EXE(
       ...,
       icon='icon.ico',  # Uncomment and set path
   )
   
   app = BUNDLE(
       ...,
       icon='icon.icns',  # Uncomment and set path
   )
   ```

3. **Rebuild**

---

## 🎉 That's It!

Your users can now run **DigitExtractor** on any Windows or Mac computer without installing anything!

**Questions?** Check the [PyInstaller docs](https://pyinstaller.org) for advanced configuration.
