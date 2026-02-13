# 🚀 Build Mac Apps from Windows (FREE!)

Use **GitHub Actions** to build for both Windows AND Mac automatically — no Mac needed!

---

## ⚡ Quick Setup (5 minutes)

### 1. Create a GitHub Repository

```bash
# In your project folder:
cd D:\OS2025Dev\DigitExtractor

# Initialize git (if not already)
git init

# Create .gitignore
echo "# See .gitignore file" > .gitignore

# Add all files
git add .
git commit -m "Initial commit: DigitExtractor app"

# Create repo on GitHub.com, then:
git remote add origin https://github.com/YOUR-USERNAME/DigitExtractor.git
git branch -M main
git push -u origin main
```

### 2. Trigger the Build

**Option A: Automatic** — just push any commit:
```bash
git add .
git commit -m "Build both platforms"
git push
```

**Option B: Manual** — go to GitHub:
1. Go to your repo on GitHub.com
2. Click **"Actions"** tab
3. Click **"Build DigitExtractor for Windows & Mac"**
4. Click **"Run workflow"** → **"Run workflow"**

### 3. Download Your Apps

1. Wait 5-10 minutes for builds to complete
2. Go to **Actions** tab
3. Click the completed workflow run
4. Scroll to **"Artifacts"** section
5. Download:
   - `DigitExtractor-Windows.zip` (for PC)
   - `DigitExtractor-macOS.zip` (for Mac)

**Done!** 🎉 You just built a Mac app from Windows!

---

## 🏷️ Create a Release (Optional)

Want to distribute both versions together?

```bash
# Tag a version
git tag v1.0.0
git push origin v1.0.0
```

GitHub will automatically:
- Build both Windows and Mac versions
- Create a **Release** page
- Attach both zip files for download

Your users can download from:
```
https://github.com/YOUR-USERNAME/DigitExtractor/releases
```

---

## 💰 Pricing

| Plan | Windows builds | Mac builds | Cost |
|------|---------------|------------|------|
| **Public repo** | ✅ Unlimited | ✅ Unlimited | **FREE** |
| **Private repo** | 2,000 min/month | 2,000 min/month | **FREE** |
| **Extra (if needed)** | $0.008/min | $0.08/min | Pay-as-you-go |

**Each build takes ~5-8 minutes**, so you get LOTS of free builds!

---

## 🎯 Workflow Explained

The GitHub Actions workflow ([.github/workflows/build.yml](.github/workflows/build.yml)) does this:

```
When you push code:
├── Spin up Windows VM in the cloud
│   ├── Install Python
│   ├── Install dependencies
│   ├── Build DigitExtractor.exe
│   └── Upload as artifact
│
└── Spin up macOS VM in the cloud
    ├── Install Python
    ├── Install dependencies
    ├── Build DigitExtractor.app
    └── Upload as artifact
```

Both builds run **in parallel** — super fast! ⚡

---

## 🔄 Update Workflow

Every time you push changes, GitHub automatically:
1. Tests your code compiles on both platforms
2. Creates fresh builds
3. Makes them available for download

**Never worry about Mac builds again!**

---

## 🛠️ Advanced Options

### Build only on demand
Change in [.github/workflows/build.yml](.github/workflows/build.yml):
```yaml
on:
  workflow_dispatch:  # Manual only
```

### Build for specific Python versions
```yaml
- name: Set up Python
  uses: actions/setup-python@v5
  with:
    python-version: '3.11'  # Change version here
```

### Add automated tests
```yaml
- name: Run tests
  run: |
    pytest tests/  # If you add tests
```

### Build on schedule
```yaml
on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday
```

---

## 📊 Monitor Builds

Check build status:
1. Go to your repo → **Actions** tab
2. See all workflow runs
3. Click any run to see logs
4. Green ✅ = success
5. Red ❌ = failed (check logs for errors)

---

## 🐛 Troubleshooting

### Build fails with "No such file"
- Check [requirements.txt](requirements.txt) includes all dependencies
- Verify [DigitExtractor.spec](DigitExtractor.spec) paths are correct

### Artifacts not appearing
- Wait for build to complete (check green checkmark)
- Scroll down on the workflow run page to "Artifacts" section

### Mac build works but Windows fails (or vice versa)
- Check the specific job logs
- Each platform builds independently

### Want faster builds?
- **Cache dependencies** (already configured with `cache: 'pip'`)
- **Matrix strategy** to test multiple Python versions:
  ```yaml
  strategy:
    matrix:
      python-version: ['3.10', '3.11', '3.12']
  ```

---

## 🎉 Benefits

✅ **No Mac needed** — build from Windows  
✅ **Free for public repos** — unlimited builds  
✅ **Automatic** — push code, get builds  
✅ **Both platforms** — Windows + Mac in parallel  
✅ **CI/CD integrated** — test before release  
✅ **Release management** — auto-attach builds to releases  

---

## 📦 Example Release URL

After tagging v1.0.0:
```
https://github.com/YOUR-USERNAME/DigitExtractor/releases/tag/v1.0.0
```

Users see:
- 📝 Release notes (auto-generated from commits)
- 💾 DigitExtractor-Windows.zip
- 🍎 DigitExtractor-macOS.zip

Professional distribution! 🚀

---

## 🔗 Resources

- [GitHub Actions Documentation](https://docs.github.com/actions)
- [PyInstaller GitHub Actions Example](https://github.com/pyinstaller/pyinstaller/wiki/GitHub-Actions)
- [Managing Releases](https://docs.github.com/repositories/releasing-projects-on-github)

---

**Now you can build Mac apps without owning a Mac!** 🎊
