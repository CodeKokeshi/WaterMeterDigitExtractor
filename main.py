"""
High-Precision Image Dataset Extractor
PyQt6 + OpenCV application for extracting and segmenting digit images.
"""

import sys
import os
import uuid
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import (
    Qt, QPointF, QRectF, pyqtSignal, QThread, QObject
)
from PyQt6.QtGui import (
    QImage, QPixmap, QPen, QBrush, QColor, QPainter, QFont,
    QAction, QKeySequence, QWheelEvent
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsEllipseItem, QGraphicsLineItem,
    QGraphicsTextItem, QFileDialog, QListWidget, QListWidgetItem,
    QDockWidget, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QLabel, QInputDialog, QMessageBox, QStatusBar, QSplitter,
    QGroupBox, QLineEdit, QProgressBar, QSizePolicy
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HANDLE_RADIUS = 7          # pixels on screen
HANDLE_COLOR = QColor(0, 200, 255, 220)
HANDLE_HOVER_COLOR = QColor(255, 100, 0, 220)
LINE_COLOR = QColor(0, 255, 100, 180)
LINE_WIDTH = 2
WARP_HI_W, WARP_HI_H = 500, 100       # high-res buffer
FINAL_W, FINAL_H = 140, 28             # final strip size
SEGMENT_SIZE = 28                       # each digit cell
NUM_SEGMENTS = 5
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}


# ---------------------------------------------------------------------------
# Utility: order 4 points as TL, TR, BR, BL
# ---------------------------------------------------------------------------
def order_points(pts: np.ndarray) -> np.ndarray:
    """Sort 4 points into Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]   # TL has smallest x+y
    rect[2] = pts[np.argmax(s)]   # BR has largest  x+y
    rect[1] = pts[np.argmin(d)]   # TR has smallest x-y
    rect[3] = pts[np.argmax(d)]   # BL has largest  x-y
    return rect


# ---------------------------------------------------------------------------
# CV Processing Worker (runs on a QThread to keep UI responsive)
# ---------------------------------------------------------------------------
class ProcessingSignals(QObject):
    finished = pyqtSignal(object)   # emits the 140x28 strip (numpy)
    error = pyqtSignal(str)


class WarpWorker(QThread):
    """Perspective-warp → binarize → downscale in a background thread."""

    def __init__(self, image: np.ndarray, points: np.ndarray):
        super().__init__()
        self.signals = ProcessingSignals()
        self.image = image
        self.points = points

    def run(self):
        try:
            src = order_points(self.points)
            dst = np.array([
                [0, 0],
                [WARP_HI_W - 1, 0],
                [WARP_HI_W - 1, WARP_HI_H - 1],
                [0, WARP_HI_H - 1]
            ], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(self.image, M, (WARP_HI_W, WARP_HI_H))

            # Convert to grayscale for binarization
            if len(warped.shape) == 3:
                gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            else:
                gray = warped

            # "Mirror" step — adaptive threshold + median blur while high-res
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=11,
                C=2,
            )
            binary = cv2.medianBlur(binary, 3)

            # Downscale to final 140×28
            strip = cv2.resize(binary, (FINAL_W, FINAL_H),
                               interpolation=cv2.INTER_AREA)

            self.signals.finished.emit(strip)
        except Exception as exc:
            self.signals.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Draggable Handle (QGraphicsEllipseItem)
# ---------------------------------------------------------------------------
class DraggableHandle(QGraphicsEllipseItem):
    """A circular handle the user can drag to fine-tune corner position."""

    def __init__(self, x: float, y: float, index: int, parent_view):
        r = HANDLE_RADIUS
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.setPos(x, y)
        self.index = index
        self.parent_view = parent_view
        self.setBrush(QBrush(HANDLE_COLOR))
        self.setPen(QPen(Qt.GlobalColor.white, 1))
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setZValue(20)

        # Label text
        labels = ["TL", "TR", "BR", "BL"]
        self._label = QGraphicsTextItem(labels[index], self)
        self._label.setDefaultTextColor(Qt.GlobalColor.yellow)
        font = QFont("Consolas", 8, QFont.Weight.Bold)
        self._label.setFont(font)
        self._label.setPos(r + 2, -r - 2)

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.GraphicsItemChange.ItemPositionHasChanged:
            self.parent_view.update_lines()
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event):
        self.setBrush(QBrush(HANDLE_HOVER_COLOR))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setBrush(QBrush(HANDLE_COLOR))
        super().hoverLeaveEvent(event)


# ---------------------------------------------------------------------------
# Image Viewer (QGraphicsView)
# ---------------------------------------------------------------------------
class ImageViewer(QGraphicsView):
    """Zoomable / pannable image viewer with 4-point polygon selection."""

    points_ready = pyqtSignal()   # emitted when 4 points placed

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._cv_image: np.ndarray | None = None

        # Selection state
        self._handles: list[DraggableHandle] = []
        self._lines: list[QGraphicsLineItem] = []
        self._placing = False          # True while user is clicking corners

    # -- public API ----------------------------------------------------------

    def load_image(self, path: str):
        """Load an image from disk and display it."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.warning(self, "Load Error", f"Cannot read:\n{path}")
            return
        self._cv_image = img
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self._scene.clear()
        self._handles.clear()
        self._lines.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect().toRectF()))
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._placing = False

    def start_selection(self):
        """Enter point-placement mode (clear old selection)."""
        self._clear_selection()
        self._placing = True
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def get_points(self) -> np.ndarray | None:
        """Return ordered 4×2 float32 array or None."""
        if len(self._handles) != 4:
            return None
        raw = np.array(
            [[h.pos().x(), h.pos().y()] for h in self._handles],
            dtype=np.float32,
        )
        return order_points(raw)

    def get_cv_image(self) -> np.ndarray | None:
        return self._cv_image

    # -- internal helpers ----------------------------------------------------

    def _clear_selection(self):
        for h in self._handles:
            self._scene.removeItem(h)
        for l in self._lines:
            self._scene.removeItem(l)
        self._handles.clear()
        self._lines.clear()

    def _add_handle(self, scene_pos: QPointF):
        idx = len(self._handles)
        h = DraggableHandle(scene_pos.x(), scene_pos.y(), idx, self)
        self._scene.addItem(h)
        self._handles.append(h)

        if len(self._handles) == 4:
            self._placing = False
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self._create_lines()
            self._reorder_handles()
            self.points_ready.emit()

    def _reorder_handles(self):
        """Re-label handles after auto-sorting so the labels match."""
        pts = np.array(
            [[h.pos().x(), h.pos().y()] for h in self._handles],
            dtype=np.float32,
        )
        ordered = order_points(pts)
        labels = ["TL", "TR", "BR", "BL"]
        for i, h in enumerate(self._handles):
            pos = np.array([h.pos().x(), h.pos().y()])
            for j in range(4):
                if np.allclose(pos, ordered[j], atol=0.5):
                    h.index = j
                    h._label.setPlainText(labels[j])
                    break
        # Sort handles list by index so get_points returns them in order
        self._handles.sort(key=lambda h: h.index)

    def _create_lines(self):
        pen = QPen(LINE_COLOR, LINE_WIDTH)
        pen.setCosmetic(True)
        for i in range(4):
            line = QGraphicsLineItem()
            line.setPen(pen)
            line.setZValue(10)
            self._scene.addItem(line)
            self._lines.append(line)
        self.update_lines()

    def update_lines(self):
        if len(self._handles) != 4 or len(self._lines) != 4:
            return
        pts = [h.pos() for h in self._handles]
        for i in range(4):
            j = (i + 1) % 4
            self._lines[i].setLine(
                pts[i].x(), pts[i].y(),
                pts[j].x(), pts[j].y(),
            )

    # -- events --------------------------------------------------------------

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if self._placing and event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            self._add_handle(scene_pos)
            return
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._clear_selection()
            self._placing = False
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif event.key() == Qt.Key.Key_F:
            if self._pixmap_item:
                self.fitInView(self._pixmap_item,
                               Qt.AspectRatioMode.KeepAspectRatio)
        super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# Preview Widget — shows the processed 140×28 strip and segments
# ---------------------------------------------------------------------------
class PreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._strip_label = QLabel("No preview yet")
        self._strip_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._strip_label.setStyleSheet(
            "background: #222; border: 1px solid #555; padding: 4px;"
        )
        self._strip_label.setMinimumHeight(56)
        layout.addWidget(self._strip_label)

        seg_box = QGroupBox("Segments (28×28)")
        seg_layout = QHBoxLayout(seg_box)
        self._seg_labels: list[QLabel] = []
        for i in range(NUM_SEGMENTS):
            lbl = QLabel()
            lbl.setFixedSize(56, 56)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background: #1a1a1a; border: 1px solid #444;")
            seg_layout.addWidget(lbl)
            self._seg_labels.append(lbl)
        layout.addWidget(seg_box)

        self._strip_img: np.ndarray | None = None
        self._segments: list[np.ndarray] = []

    def set_strip(self, strip: np.ndarray):
        self._strip_img = strip
        self._segments = []
        # Show strip (scale up for visibility)
        disp = cv2.resize(strip, (FINAL_W * 3, FINAL_H * 3),
                          interpolation=cv2.INTER_NEAREST)
        qimg = QImage(disp.data, disp.shape[1], disp.shape[0],
                       disp.shape[1], QImage.Format.Format_Grayscale8)
        self._strip_label.setPixmap(QPixmap.fromImage(qimg))

        # Segment into 5 cells
        for i in range(NUM_SEGMENTS):
            x0 = i * SEGMENT_SIZE
            seg = strip[:, x0:x0 + SEGMENT_SIZE]
            self._segments.append(seg)
            disp_seg = cv2.resize(seg, (56, 56),
                                  interpolation=cv2.INTER_NEAREST)
            qimg_s = QImage(disp_seg.data, 56, 56, 56,
                            QImage.Format.Format_Grayscale8)
            self._seg_labels[i].setPixmap(QPixmap.fromImage(qimg_s))

    def get_segments(self) -> list[np.ndarray]:
        return self._segments

    def clear(self):
        self._strip_img = None
        self._segments = []
        self._strip_label.setText("No preview yet")
        self._strip_label.setPixmap(QPixmap())
        for lbl in self._seg_labels:
            lbl.setPixmap(QPixmap())


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DigitExtractor — Image Dataset Extractor")
        self.resize(1280, 800)
        self._output_dir = ""
        self._worker: WarpWorker | None = None

        self._build_ui()
        self._build_menu()
        self._connect_signals()

    # -- UI construction -----------------------------------------------------

    def _build_ui(self):
        # Central splitter
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.setCentralWidget(splitter)

        # --- Left: file list sidebar ---
        sidebar = QWidget()
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(4, 4, 4, 4)

        btn_open = QPushButton("Open Folder…")
        btn_open.setObjectName("btnOpenFolder")
        sb_layout.addWidget(btn_open)

        self._file_list = QListWidget()
        self._file_list.setMinimumWidth(180)
        sb_layout.addWidget(self._file_list)
        splitter.addWidget(sidebar)

        # --- Centre: viewer + bottom panel ---
        centre = QWidget()
        c_layout = QVBoxLayout(centre)
        c_layout.setContentsMargins(0, 0, 0, 0)

        self._viewer = ImageViewer()
        c_layout.addWidget(self._viewer, stretch=5)

        # Bottom controls
        ctrl = QWidget()
        ctrl_layout = QHBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(6, 2, 6, 2)

        self._btn_select = QPushButton("Select 4 Points")
        self._btn_select.setEnabled(False)
        ctrl_layout.addWidget(self._btn_select)

        self._btn_extract = QPushButton("Extract && Preview")
        self._btn_extract.setEnabled(False)
        ctrl_layout.addWidget(self._btn_extract)

        self._label_entry = QLineEdit()
        self._label_entry.setPlaceholderText("5-char label (e.g. A8B3Z)")
        self._label_entry.setMaxLength(5)
        self._label_entry.setFixedWidth(160)
        ctrl_layout.addWidget(self._label_entry)

        self._btn_save = QPushButton("Save Segments")
        self._btn_save.setEnabled(False)
        ctrl_layout.addWidget(self._btn_save)

        self._btn_output = QPushButton("Set Output Dir…")
        ctrl_layout.addWidget(self._btn_output)

        ctrl_layout.addStretch()

        self._status_info = QLabel("")
        ctrl_layout.addWidget(self._status_info)

        c_layout.addWidget(ctrl)

        # Preview panel
        self._preview = PreviewWidget()
        c_layout.addWidget(self._preview, stretch=1)

        splitter.addWidget(centre)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 5)

        # Status bar
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("Ready — open a folder to begin.")

        # Styling
        self.setStyleSheet("""
            QMainWindow { background: #2b2b2b; }
            QLabel, QGroupBox, QListWidget, QPushButton, QLineEdit, QStatusBar {
                color: #ddd;
                font-size: 13px;
            }
            QGroupBox { border: 1px solid #555; border-radius: 4px;
                        margin-top: 6px; padding-top: 14px; }
            QGroupBox::title { subcontrol-origin: margin;
                               left: 10px; padding: 0 4px; }
            QPushButton {
                background: #3a3f47; border: 1px solid #555;
                border-radius: 4px; padding: 5px 14px;
            }
            QPushButton:hover { background: #50565e; }
            QPushButton:disabled { color: #666; }
            QLineEdit {
                background: #222; border: 1px solid #555;
                border-radius: 3px; padding: 4px 8px;
            }
            QListWidget {
                background: #1e1e1e; border: 1px solid #444;
            }
            QListWidget::item:selected {
                background: #264f78;
            }
        """)

    def _build_menu(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")

        act_open = QAction("Open Folder…", self)
        act_open.setShortcut(QKeySequence("Ctrl+O"))
        act_open.triggered.connect(self._on_open_folder)
        file_menu.addAction(act_open)

        file_menu.addSeparator()

        act_quit = QAction("Quit", self)
        act_quit.setShortcut(QKeySequence("Ctrl+Q"))
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        view_menu = menu.addMenu("&View")
        act_fit = QAction("Fit Image", self)
        act_fit.setShortcut(QKeySequence("F"))
        act_fit.triggered.connect(
            lambda: self._viewer._pixmap_item and self._viewer.fitInView(
                self._viewer._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio
            )
        )
        view_menu.addAction(act_fit)

    def _connect_signals(self):
        self.findChild(QPushButton, "btnOpenFolder").clicked.connect(
            self._on_open_folder
        )
        self._file_list.currentRowChanged.connect(self._on_file_selected)
        self._btn_select.clicked.connect(self._on_start_select)
        self._btn_extract.clicked.connect(self._on_extract)
        self._btn_save.clicked.connect(self._on_save_segments)
        self._btn_output.clicked.connect(self._on_set_output)
        self._viewer.points_ready.connect(self._on_points_ready)

    # -- Slots ---------------------------------------------------------------

    def _on_open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        self._file_list.clear()
        files = sorted(
            p for p in Path(folder).iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            QMessageBox.information(self, "No Images",
                                    "No supported image files found.")
            return
        for f in files:
            item = QListWidgetItem(f.name)
            item.setData(Qt.ItemDataRole.UserRole, str(f))
            self._file_list.addItem(item)
        self._statusbar.showMessage(f"Loaded {len(files)} images from {folder}")
        self._file_list.setCurrentRow(0)

    def _on_file_selected(self, row: int):
        if row < 0:
            return
        item = self._file_list.item(row)
        path = item.data(Qt.ItemDataRole.UserRole)
        self._viewer.load_image(path)
        self._btn_select.setEnabled(True)
        self._btn_extract.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._preview.clear()
        self._statusbar.showMessage(f"Viewing: {item.text()}")

    def _on_start_select(self):
        self._viewer.start_selection()
        self._btn_extract.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._preview.clear()
        self._statusbar.showMessage(
            "Click 4 corners on the image. Press Esc to cancel."
        )

    def _on_points_ready(self):
        self._btn_extract.setEnabled(True)
        self._statusbar.showMessage(
            "4 points placed (auto-sorted). "
            "Drag handles to fine-tune, then click Extract."
        )

    def _on_extract(self):
        pts = self._viewer.get_points()
        img = self._viewer.get_cv_image()
        if pts is None or img is None:
            return
        self._btn_extract.setEnabled(False)
        self._statusbar.showMessage("Processing…")
        self._worker = WarpWorker(img, pts)
        self._worker.signals.finished.connect(self._on_warp_done)
        self._worker.signals.error.connect(self._on_warp_error)
        self._worker.start()

    def _on_warp_done(self, strip: np.ndarray):
        self._preview.set_strip(strip)
        self._btn_extract.setEnabled(True)
        self._btn_save.setEnabled(True)
        self._statusbar.showMessage(
            "Extraction complete — enter a 5-char label and save."
        )

    def _on_warp_error(self, msg: str):
        QMessageBox.critical(self, "Processing Error", msg)
        self._btn_extract.setEnabled(True)
        self._statusbar.showMessage("Error during processing.")

    def _on_set_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self._output_dir = folder
            self._statusbar.showMessage(f"Output directory: {folder}")

    def _on_save_segments(self):
        label = self._label_entry.text().strip()
        if len(label) != 5:
            QMessageBox.warning(
                self, "Invalid Label",
                "Please enter exactly 5 characters for the label."
            )
            return
        segments = self._preview.get_segments()
        if len(segments) != NUM_SEGMENTS:
            QMessageBox.warning(self, "No Segments",
                                "Extract an image first.")
            return
        if not self._output_dir:
            self._on_set_output()
            if not self._output_dir:
                return

        saved = 0
        for i, seg in enumerate(segments):
            char = label[i]
            char_dir = os.path.join(self._output_dir, char)
            os.makedirs(char_dir, exist_ok=True)
            fname = f"segment_{uuid.uuid4().hex[:8]}.png"
            cv2.imwrite(os.path.join(char_dir, fname), seg)
            saved += 1

        self._statusbar.showMessage(
            f"Saved {saved} segments for label '{label}' → {self._output_dir}"
        )
        QMessageBox.information(
            self, "Saved",
            f"Saved {saved} segment(s) into:\n{self._output_dir}\n\n"
            f"Folders: {', '.join(sorted(set(label)))}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette for Fusion
    from PyQt6.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(43, 43, 43))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(50, 50, 50))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
