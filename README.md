# DigitExtractor

**High-Precision Image Dataset Extractor** for creating 28×28 digit image datasets from perspective-distorted sources.

## 🎯 What It Does

Extract and segment digits from images (even at angles) into clean 28×28 training images:

1. **Load** a folder of images
2. **Select** 4 corner points to define the region
3. **Warp** perspective to 5:1 ratio (500×100 → 140×28)
4. **Binarize** using adaptive threshold + median blur
5. **Segment** into five 28×28 cells
6. **Save** to labeled folders for ML training

## 🚀 Quick Start

### Running from Source
```bash
python main.py
```

### Running Standalone (No Python Needed)
- **Windows:** Double-click `DigitExtractor.exe`
- **macOS:** Double-click `DigitExtractor.app`

## 📖 Usage

1. **File → Open Folder** — select a folder with images
2. Click an image in the sidebar to view it
3. **Select 4 Points** — click 4 corners of your digit strip
   - Points auto-sort to Top-Left, Top-Right, Bottom-Right, Bottom-Left
   - Drag handles to fine-tune
   - Press `Esc` to cancel
4. **Extract & Preview** — see the processed 140×28 strip and 5 segments
5. Enter a **5-character label** (e.g., `A8B3Z`)
6. **Set Output Dir** — choose where to save
7. **Save Segments** — saves each digit to `/output/<char>/segment_*.png`

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `F` | Fit image to view |
| `Esc` | Cancel selection |
| `Ctrl+O` | Open folder |
| Mouse wheel | Zoom in/out |

## 🏗️ Technical Details

- **Perspective Transform:** `cv2.getPerspectiveTransform` with auto-sorted corners
- **High-Res Processing:** Warps to 500×100 before binarization
- **Binarization:** Adaptive Gaussian threshold (block=11, C=2) + median blur (3×3)
- **Downscaling:** `INTER_AREA` interpolation to 140×28
- **Segmentation:** 5 equal 28×28 cells

## 🛠️ Building from Source

See [BUILD_README.md](BUILD_README.md) for creating standalone executables.

### Dependencies
```bash
pip install -r requirements.txt
```

- PyQt6 — GUI framework
- OpenCV — image processing
- NumPy — array operations
- PyInstaller — executable builder

## 📁 Project Structure

```
DigitExtractor/
├── main.py                 # Main application
├── requirements.txt        # Python dependencies
├── DigitExtractor.spec     # PyInstaller configuration
├── build_windows.ps1       # Windows build script
├── build_mac.sh            # macOS build script
├── BUILD_README.md         # Build instructions
└── README.md               # This file
```

## 🎨 UI Features

- **Sidebar:** Image list with quick switching
- **Viewer:** QGraphicsView with zoom/pan
- **Handles:** Draggable corner points with labels
- **Preview:** Real-time display of processed strip + segments
- **Dark Theme:** Easy on the eyes

## 💡 Use Cases

- Creating MNIST-style datasets from real photos
- Extracting serial numbers from images
- Processing distorted text/numbers from photos
- Building custom digit recognition training data
- Handling 3D perspective distortion in captured images

## ⚠️ Important Notes

- The extractor is **"blind"** — it captures exactly what's in the 4 points
- Bars, shadows, borders stay in the data (real-world parallax)
- Points are automatically sorted to prevent flipping
- Processing runs in background thread (UI stays responsive)
- Each segment saved with unique UUID filename

## 🐛 Troubleshooting

**Image won't load?**
- Supports: PNG, JPG, JPEG, BMP, TIFF, WebP
- Check file isn't corrupted

**Can't click points?**
- Click "Select 4 Points" button first
- Make sure image is loaded

**Extract button disabled?**
- Place all 4 points first

**Segments look wrong?**
- Adjust corner handles for better alignment
- Ensure selection covers only the digit strip

## 📄 License

MIT License - feel free to use in your projects!

## 🤝 Contributing

Built with ❤️ for ML practitioners who need clean training data.
