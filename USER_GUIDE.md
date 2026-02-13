# 🎯 DigitExtractor - User Guide

**One-click image dataset extractor — no installation required!**

---

## 🚀 Getting Started (Windows)

1. **Unzip** the folder you received
2. **Double-click** `DigitExtractor.exe`
3. That's it! 🎉

> ⚠️ First launch might take 10-15 seconds while Windows loads the app.

---

## 🚀 Getting Started (Mac)

1. **Unzip** the folder you received
2. **Double-click** `DigitExtractor.app`
3. If you see "can't be opened because Apple cannot check it":
   - **Right-click** the app → **Open** → **Open** again
   - This only happens the first time

---

## 📖 How to Use

### Step 1: Load Your Images
1. Click **File → Open Folder** (or press `Ctrl+O`)
2. Select a folder containing your images
3. You'll see all images in the left sidebar

### Step 2: Select an Image
- Click any image in the sidebar to view it
- Use your **mouse wheel** to zoom in/out
- **Drag** to pan around

### Step 3: Mark the Region (4 Points)
1. Click **"Select 4 Points"** button
2. Click **4 corners** on the image to define your digit strip
3. The corners will automatically be labeled:
   - **TL** = Top-Left
   - **TR** = Top-Right
   - **BR** = Bottom-Right  
   - **BL** = Bottom-Left
4. **Drag the circles** to fine-tune the position
5. Press **Esc** to cancel and start over

### Step 4: Extract & Preview
1. Click **"Extract & Preview"**
2. Wait a moment — you'll see:
   - The full processed strip (140×28 pixels)
   - Five separate segments (each 28×28 pixels)

### Step 5: Label & Save
1. Type a **5-character label** (e.g., `A8B3Z`, `12345`, `HELLO`)
2. Click **"Set Output Dir"** and choose where to save (only need to do this once)
3. Click **"Save Segments"**
4. Done! Your segments are saved to:
   ```
   YourOutputFolder/
   ├── A/
   │   └── segment_abc123.png
   ├── 8/
   │   └── segment_def456.png
   ├── B/
   │   └── segment_ghi789.png
   ...
   ```

### Repeat!
- Select another image from the sidebar
- Mark 4 points again
- Extract and save

---

## ⌨️ Keyboard Shortcuts

| Key | What it does |
|-----|-------------|
| `F` | Fit the image to the window |
| `Esc` | Cancel point selection |
| `Ctrl+O` | Open a folder |
| **Mouse Wheel** | Zoom in and out |

---

## 💡 Tips & Tricks

### For Best Results
- ✅ Make sure all 4 corners are visible in the image
- ✅ Zoom in before placing points for precision
- ✅ Fine-tune by dragging the corner handles
- ✅ The app handles perspective distortion — angled photos work great!

### Understanding the Output
- Each digit strip is split into **5 equal segments**
- Each segment becomes a **28×28 image** (perfect for ML training)
- Segments are saved in **folders named after the label character**
- Each file gets a unique name so you can process many images

### What Gets Captured?
The app captures **exactly what's inside the 4 points** — including:
- Shadows
- Borders  
- Background texture
- Perspective distortion

This is intentional! It helps ML models handle real-world imperfections.

---

## 🐛 Troubleshooting

### App won't start on Windows
**"Windows protected your PC" warning:**
1. Click **"More info"**
2. Click **"Run anyway"**

This happens because the app isn't code-signed. It's safe to run.

### App won't start on Mac
**"App is damaged and can't be opened":**
1. **Right-click** the app
2. Choose **"Open"**
3. Click **"Open"** in the dialog

This only needs to be done once.

### Can't see my images in the folder
Supported formats:
- PNG
- JPG / JPEG
- BMP
- TIFF
- WebP

Make sure your files have the correct extension!

### Extract button is grayed out
You need to place all **4 corner points** first.

### Segments look distorted
- Make sure the 4 corners are placed correctly
- Try zooming in and fine-tuning the handle positions
- The selected area should be a **strip** shape (5:1 ratio roughly)

### App is running slow
- Large images take longer to process
- The first extraction might be slower (Windows loads libraries)
- Processing happens in the background — you can still zoom/pan

---

## 📊 Output Structure

After processing several images with labels like `A8B3Z`, `12345`, `HELLO`:

```
YourOutputFolder/
├── A/
│   ├── segment_abc123.png
│   └── segment_xyz789.png
├── 1/
│   └── segment_def456.png
├── 2/
│   └── segment_ghi789.png
├── 3/
│   └── segment_jkl012.png
...
```

Perfect for training machine learning models! Each folder contains all examples of that character.

---

## 🎓 Advanced Usage

### Processing Many Images Quickly
1. Open folder with all images
2. Click first image
3. **Workflow:**
   - Select 4 points
   - Extract & Preview
   - Type label
   - Save
   - Click next image in sidebar
   - Repeat!

### Keyboard-Focused Workflow
1. `Ctrl+O` — open folder
2. Click image in sidebar
3. Select 4 Points button (or Space)
4. Click corners
5. Extract button (or Enter)
6. Type label
7. Save button (or Enter)
8. Arrow Down to next image

### Batch Processing Tips
- Use consistent naming for your source images
- Set the output directory once at the start
- Process similar images together
- Review segments before moving to next set

---

## ❓ FAQ

**Q: Do I need Python installed?**  
A: Nope! This is a standalone app.

**Q: Does this work offline?**  
A: Yes! No internet needed.

**Q: How accurate should my corner points be?**  
A: Pretty accurate! But you can adjust them by dragging after placing.

**Q: Can I process images at an angle?**  
A: Absolutely! That's the whole point — perspective correction is built-in.

**Q: What happens if my label isn't 5 characters?**  
A: The app will ask you to enter exactly 5. Use spaces or symbols if needed.

**Q: Can I use the same label multiple times?**  
A: Yes! Each segment gets a unique filename and is added to the folder.

**Q: Where are my images saved?**  
A: In the folder you chose with "Set Output Dir". They're organized by character.

---

## 🤝 Need Help?

If something isn't working:
1. Make sure you've followed the steps exactly
2. Try restarting the app
3. Check that your images are in a supported format
4. Make sure you have write permission to the output folder

---

## 🎉 That's It!

You're ready to build amazing datasets! Happy extracting! 🚀

---

*DigitExtractor v1.0 — Built for ML practitioners who need clean training data*
