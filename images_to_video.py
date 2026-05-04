"""
Images → Video Converter  (with optional background music)
===========================================================
Opens a folder of images, sorts them, renders to MP4.
Optionally embeds a music track with fade-in / fade-out.
Music is trimmed to match video length; the cut-off fades out smoothly.

Requirements (install once):
    pip install opencv-python PyQt6 moviepy
    # moviepy bundles its own ffmpeg — no separate install needed
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import cv2

from PyQt6.QtCore import QSettings, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Optional moviepy import (needed only when music is used)
# ---------------------------------------------------------------------------

try:
    from moviepy.editor import AudioFileClip, VideoFileClip          # v1.x
    _MOVIEPY_V2     = False
    MOVIEPY_AVAILABLE = True
except ImportError:
    try:
        from moviepy import AudioFileClip, VideoFileClip              # v2.x
        _MOVIEPY_V2     = True
        MOVIEPY_AVAILABLE = True
    except ImportError:
        AudioFileClip = VideoFileClip = None                          # type: ignore
        _MOVIEPY_V2     = False
        MOVIEPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SETTINGS_ORG            = "DigitExtractor"
SETTINGS_APP            = "ImagesToVideo"
SETTINGS_LAST_IMAGE_DIR = "paths/last_image_dir"
SETTINGS_LAST_OUT_DIR   = "paths/last_out_dir"
SETTINGS_LAST_MUSIC_DIR = "paths/last_music_dir"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a"}

SORT_OPTIONS = [
    "Filename  (A → Z)",
    "File Date  (oldest first)",
    "File Date  (newest first)",
]
FPS_PRESETS = [1, 2, 5, 10, 15, 24, 30, 60]

DEFAULT_FADE_IN  = 2.0   # seconds
DEFAULT_FADE_OUT = 3.0   # seconds


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

class VideoWorker(QThread):
    """Runs the full render pipeline off the main thread."""

    progress   = pyqtSignal(int)    # 0–100
    phase      = pyqtSignal(str)    # short phase label
    status_msg = pyqtSignal(str)    # detail line
    finished   = pyqtSignal(str)    # output path on success
    error      = pyqtSignal(str)

    def __init__(
        self,
        image_paths : list[Path],
        output_path : str,
        fps         : int,
        music_path  : str  = "",
        fade_in     : float = DEFAULT_FADE_IN,
        fade_out    : float = DEFAULT_FADE_OUT,
        parent=None,
    ):
        super().__init__(parent)
        self._image_paths = image_paths
        self._output_path = output_path
        self._fps         = fps
        self._music_path  = music_path
        self._fade_in     = fade_in
        self._fade_out    = fade_out
        self._cancelled   = False

    def cancel(self) -> None:
        self._cancelled = True

    # ------------------------------------------------------------------
    def run(self) -> None:
        paths = self._image_paths
        if not paths:
            self.error.emit("No images to encode.")
            return

        # ── Resolve output and temp paths ──────────────────────────────
        use_music   = bool(self._music_path) and MOVIEPY_AVAILABLE
        temp_path   = self._output_path + ".__temp__.mp4"
        video_dest  = temp_path if use_music else self._output_path

        # ── Phase 1: render frames ─────────────────────────────────────
        self.phase.emit("Rendering frames…")

        first_frame = None
        for p in paths:
            img = cv2.imread(str(p))
            if img is not None:
                first_frame = img
                break

        if first_frame is None:
            self.error.emit("Could not read any image from the folder.")
            return

        h, w   = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_dest, fourcc, self._fps, (w, h))

        if not writer.isOpened():
            self.error.emit(
                f"Could not open video writer.\n"
                f"Path: {video_dest}\n"
                f"Check the folder exists and is writable."
            )
            return

        total = len(paths)
        for i, path in enumerate(paths):
            if self._cancelled:
                writer.release()
                self._cleanup(temp_path)
                self.error.emit("Cancelled.")
                return

            frame = cv2.imread(str(path))
            if frame is None:
                self.status_msg.emit(f"Skipping unreadable: {path.name}")
                continue

            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

            writer.write(frame)

            pct = int((i + 1) / total * (60 if use_music else 100))
            self.progress.emit(pct)
            self.status_msg.emit(f"Frame {i + 1} / {total}  —  {path.name}")

        writer.release()

        if self._cancelled:
            self._cleanup(temp_path)
            self.error.emit("Cancelled.")
            return

        # ── Phase 2: add music (if requested) ─────────────────────────
        if use_music:
            self.phase.emit("Mixing audio…")
            self.progress.emit(65)
            self.status_msg.emit("Loading video and audio clips…")
            try:
                self._mix_audio(temp_path)
            except Exception as exc:
                self._cleanup(temp_path)
                self.error.emit(f"Audio mix failed: {exc}")
                return
            finally:
                self._cleanup(temp_path)

        elif not MOVIEPY_AVAILABLE and self._music_path:
            # Music was requested but moviepy isn't installed
            self.status_msg.emit(
                "⚠  Video saved WITHOUT music — moviepy not installed.\n"
                "   Run:  pip install moviepy"
            )

        self.progress.emit(100)
        self.finished.emit(self._output_path)

    # ------------------------------------------------------------------
    def _mix_audio(self, temp_video_path: str) -> None:
        """Combine silent video with music track, apply fades, write final file."""

        video = VideoFileClip(temp_video_path)
        audio = AudioFileClip(self._music_path)

        vid_duration = video.duration   # seconds

        # ── Trim audio to video length ─────────────────────────────────
        if audio.duration > vid_duration:
            self.status_msg.emit(
                f"Music ({audio.duration:.1f}s) is longer than video "
                f"({vid_duration:.1f}s) — trimming and fading out at cut."
            )
            if _MOVIEPY_V2:
                audio = audio.subclipped(0, vid_duration)
            else:
                audio = audio.subclip(0, vid_duration)

        # ── Apply fade-in ──────────────────────────────────────────────
        fade_in  = min(self._fade_in,  audio.duration / 2)
        fade_out = min(self._fade_out, audio.duration / 2)

        if _MOVIEPY_V2:
            import moviepy.audio.fx as afx
            if fade_in > 0:
                audio = audio.with_effects([afx.AudioFadeIn(fade_in)])
            if fade_out > 0:
                audio = audio.with_effects([afx.AudioFadeOut(fade_out)])
            final = video.with_audio(audio)
        else:
            from moviepy.audio.fx.all import audio_fadein, audio_fadeout
            if fade_in > 0:
                audio = audio_fadein(audio, fade_in)
            if fade_out > 0:
                audio = audio_fadeout(audio, fade_out)
            final = video.set_audio(audio)

        self.progress.emit(75)
        self.status_msg.emit("Encoding final video with audio…")

        final.write_videofile(
            self._output_path,
            codec       = "libx264",
            audio_codec = "aac",
            logger      = None,
        )

        self.progress.emit(95)
        self.status_msg.emit("Finalising…")

        final.close()
        video.close()
        audio.close()

    # ------------------------------------------------------------------
    @staticmethod
    def _cleanup(path: str) -> None:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

def _stylesheet() -> str:
    return """
    QWidget {
        background: #14181d;
        color: #eef2f8;
        font-family: "Segoe UI";
        font-size: 11pt;
    }
    QMainWindow { background: #14181d; }
    QGroupBox {
        background: #1b2129;
        border: 1px solid #2d3744;
        border-radius: 14px;
        margin-top: 14px;
        padding: 14px 16px 16px 16px;
        font-weight: 600;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px; padding: 0 6px;
        color: #9fb2c8;
    }
    QPushButton {
        background: #27313d;
        border: 1px solid #334151;
        border-radius: 10px;
        padding: 9px 18px;
        font-weight: 600;
    }
    QPushButton:hover   { background: #2f3b49; }
    QPushButton:pressed { background: #1d2630; }
    QPushButton:disabled {
        background: #1a2027; color: #6a7787; border-color: #222b35;
    }
    QPushButton#renderBtn {
        background: #1e4d8c; border-color: #2563eb;
        font-size: 12pt; padding: 12px 24px;
    }
    QPushButton#renderBtn:hover   { background: #2563eb; }
    QPushButton#renderBtn:pressed { background: #1a3f7a; }
    QPushButton#renderBtn:disabled {
        background: #1a2535; color: #5a7080; border-color: #1f2d3d;
    }
    QPushButton#musicClear {
        background: #3d1515; border-color: #7a2020;
        padding: 6px 12px; font-size: 9pt;
    }
    QPushButton#musicClear:hover { background: #7a2020; }
    QListWidget {
        background: #10151b; border: 1px solid #2d3744;
        border-radius: 10px; padding: 4px;
    }
    QListWidget::item { padding: 3px 6px; border-radius: 6px; }
    QListWidget::item:selected { background: #1e3a5f; color: #e8f4ff; }
    QComboBox, QDoubleSpinBox {
        background: #1e2730; border: 1px solid #334151;
        border-radius: 8px; padding: 6px 10px; min-width: 80px;
    }
    QComboBox::drop-down { border: none; width: 22px; }
    QProgressBar {
        background: #1b2129; border: none;
        border-radius: 5px; min-height: 10px;
    }
    QProgressBar::chunk {
        background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
            stop:0 #2563eb, stop:1 #7c3aed);
        border-radius: 5px;
    }
    QLabel#title    { font-size: 18pt; font-weight: 700; color: #ffffff; }
    QLabel#subtitle { color: #8ea1b7; font-size: 10pt; }
    QLabel#meta     { color: #8ea1b7; font-size: 9pt; }
    QLabel#warn     { color: #f59e0b; font-size: 9pt; }
    QStatusBar {
        background: #0f1319; color: #aab8c7;
        border-top: 1px solid #212833;
    }
    """


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class ImagesToVideoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Images → Video Converter")
        self.resize(880, 760)
        self.setStyleSheet(_stylesheet())

        self._image_paths : list[Path] = []
        self._music_path  : str        = ""
        self._worker      : VideoWorker | None = None
        self._settings    = QSettings(SETTINGS_ORG, SETTINGS_APP)
        self._last_img_dir   = self._load_dir(SETTINGS_LAST_IMAGE_DIR)
        self._last_out_dir   = self._load_dir(SETTINGS_LAST_OUT_DIR)
        self._last_music_dir = self._load_dir(SETTINGS_LAST_MUSIC_DIR)

        self._build_ui()
        self._refresh()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root  = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(18, 18, 18, 10)
        outer.setSpacing(14)

        # ── Header ────────────────────────────────────────────────────
        hero   = QGroupBox("Converter")
        hero_l = QVBoxLayout(hero)
        title  = QLabel("Images → Video Converter")
        title.setObjectName("title")
        sub = QLabel(
            "Pick a folder of images, optionally add background music, and render to MP4."
        )
        sub.setObjectName("subtitle")
        hero_l.addWidget(title)
        hero_l.addWidget(sub)
        outer.addWidget(hero)

        # ── Source folder ─────────────────────────────────────────────
        folder_group = QGroupBox("Source Folder")
        folder_l     = QVBoxLayout(folder_group)

        row1 = QHBoxLayout()
        self._btn_folder = QPushButton("Open Folder…")
        self._lbl_folder = QLabel("No folder selected")
        self._lbl_folder.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        row1.addWidget(self._btn_folder)
        row1.addWidget(self._lbl_folder, stretch=1)
        folder_l.addLayout(row1)

        row2 = QHBoxLayout()
        sort_lbl = QLabel("Sort by:")
        sort_lbl.setObjectName("meta")
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(SORT_OPTIONS)
        row2.addWidget(sort_lbl)
        row2.addWidget(self._sort_combo)
        row2.addStretch()
        folder_l.addLayout(row2)
        outer.addWidget(folder_group)

        # ── Image list ────────────────────────────────────────────────
        list_group = QGroupBox("Images Found")
        list_l     = QVBoxLayout(list_group)
        self._image_list = QListWidget()
        self._image_list.setFixedHeight(160)
        list_l.addWidget(self._image_list)
        outer.addWidget(list_group)

        # ── Music ─────────────────────────────────────────────────────
        music_group = QGroupBox("Background Music  (optional)")
        music_l     = QVBoxLayout(music_group)

        if not MOVIEPY_AVAILABLE:
            warn = QLabel(
                "⚠  moviepy not installed — music will be skipped.\n"
                "   Install with:  pip install moviepy"
            )
            warn.setObjectName("warn")
            music_l.addWidget(warn)

        row_music = QHBoxLayout()
        self._btn_music = QPushButton("Select Music File…")
        self._lbl_music = QLabel("No music selected  (video will be silent)")
        self._lbl_music.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._btn_music_clear = QPushButton("✕ Clear")
        self._btn_music_clear.setObjectName("musicClear")
        self._btn_music_clear.setVisible(False)
        row_music.addWidget(self._btn_music)
        row_music.addWidget(self._lbl_music, stretch=1)
        row_music.addWidget(self._btn_music_clear)
        music_l.addLayout(row_music)

        row_fade = QHBoxLayout()
        fade_in_lbl = QLabel("Fade in (s):")
        fade_in_lbl.setObjectName("meta")
        self._spin_fade_in = QDoubleSpinBox()
        self._spin_fade_in.setRange(0.0, 30.0)
        self._spin_fade_in.setSingleStep(0.5)
        self._spin_fade_in.setValue(DEFAULT_FADE_IN)

        fade_out_lbl = QLabel("Fade out (s):")
        fade_out_lbl.setObjectName("meta")
        self._spin_fade_out = QDoubleSpinBox()
        self._spin_fade_out.setRange(0.0, 30.0)
        self._spin_fade_out.setSingleStep(0.5)
        self._spin_fade_out.setValue(DEFAULT_FADE_OUT)

        row_fade.addWidget(fade_in_lbl)
        row_fade.addWidget(self._spin_fade_in)
        row_fade.addSpacing(20)
        row_fade.addWidget(fade_out_lbl)
        row_fade.addWidget(self._spin_fade_out)
        row_fade.addStretch()
        music_l.addLayout(row_fade)

        music_note = QLabel(
            "Music plays once.  If it's longer than the video, it's trimmed — the cut fades out smoothly."
        )
        music_note.setObjectName("subtitle")
        music_l.addWidget(music_note)
        outer.addWidget(music_group)

        # ── Output settings ───────────────────────────────────────────
        out_group = QGroupBox("Output Settings")
        out_l     = QVBoxLayout(out_group)

        row3 = QHBoxLayout()
        fps_lbl = QLabel("FPS:")
        fps_lbl.setObjectName("meta")
        self._fps_combo = QComboBox()
        for fps in FPS_PRESETS:
            self._fps_combo.addItem(str(fps), fps)
        self._fps_combo.setCurrentIndex(FPS_PRESETS.index(10))
        row3.addWidget(fps_lbl)
        row3.addWidget(self._fps_combo)
        row3.addSpacing(24)
        row3.addStretch()
        out_l.addLayout(row3)

        row4 = QHBoxLayout()
        self._btn_out = QPushButton("Set Output File…")
        self._lbl_out = QLabel("No output file set")
        self._lbl_out.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        row4.addWidget(self._btn_out)
        row4.addWidget(self._lbl_out, stretch=1)
        out_l.addLayout(row4)
        outer.addWidget(out_group)

        # ── Render ────────────────────────────────────────────────────
        render_group = QGroupBox("Render")
        render_l     = QVBoxLayout(render_group)

        self._btn_render = QPushButton("▶  Render Video")
        self._btn_render.setObjectName("renderBtn")
        self._btn_render.setFixedHeight(52)
        render_l.addWidget(self._btn_render)

        self._lbl_phase = QLabel("")
        self._lbl_phase.setObjectName("meta")
        self._lbl_phase.setAlignment(Qt.AlignmentFlag.AlignCenter)
        render_l.addWidget(self._lbl_phase)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        render_l.addWidget(self._progress)

        self._lbl_status = QLabel("Ready.")
        self._lbl_status.setObjectName("subtitle")
        self._lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        render_l.addWidget(self._lbl_status)

        outer.addWidget(render_group)
        self.setStatusBar(QStatusBar())

        # ── Wire signals ──────────────────────────────────────────────
        self._btn_folder.clicked.connect(self._pick_folder)
        self._btn_music.clicked.connect(self._pick_music)
        self._btn_music_clear.clicked.connect(self._clear_music)
        self._btn_out.clicked.connect(self._pick_output)
        self._btn_render.clicked.connect(self._toggle_render)
        self._sort_combo.currentIndexChanged.connect(self._reload_sorted)

    # ------------------------------------------------------------------
    # Settings helpers
    # ------------------------------------------------------------------

    def _load_dir(self, key: str) -> str:
        v = str(self._settings.value(key, "", type=str) or "")
        return v if v and Path(v).exists() else ""

    # ------------------------------------------------------------------
    # Pickers
    # ------------------------------------------------------------------

    def _pick_folder(self) -> None:
        start = self._last_img_dir or str(Path.cwd())
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", start)
        if not folder:
            return
        self._last_img_dir = folder
        self._settings.setValue(SETTINGS_LAST_IMAGE_DIR, folder)
        self._lbl_folder.setText(folder)

        if self._lbl_out.text() in ("", "No output file set"):
            default_out = str(Path(folder) / "output_video.mp4")
            self._lbl_out.setText(default_out)
            self._last_out_dir = folder
            self._settings.setValue(SETTINGS_LAST_OUT_DIR, folder)

        self._load_images(folder)

    def _pick_music(self) -> None:
        start = self._last_music_dir or self._last_img_dir or str(Path.cwd())
        exts  = " ".join(f"*{e}" for e in sorted(AUDIO_EXTENSIONS))
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Music File", start,
            f"Audio Files ({exts});;All Files (*)"
        )
        if not path:
            return
        self._music_path = path
        self._lbl_music.setText(Path(path).name)
        self._btn_music_clear.setVisible(True)
        self._last_music_dir = str(Path(path).parent)
        self._settings.setValue(SETTINGS_LAST_MUSIC_DIR, self._last_music_dir)
        self._refresh()

    def _clear_music(self) -> None:
        self._music_path = ""
        self._lbl_music.setText("No music selected  (video will be silent)")
        self._btn_music_clear.setVisible(False)
        self._refresh()

    def _pick_output(self) -> None:
        start = self._last_out_dir or self._last_img_dir or str(Path.cwd())
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Video As",
            str(Path(start) / "output_video.mp4"),
            "MP4 Video (*.mp4)",
        )
        if not path:
            return
        if not path.lower().endswith(".mp4"):
            path += ".mp4"
        self._lbl_out.setText(path)
        self._last_out_dir = str(Path(path).parent)
        self._settings.setValue(SETTINGS_LAST_OUT_DIR, self._last_out_dir)
        self._refresh()

    # ------------------------------------------------------------------
    # Image list
    # ------------------------------------------------------------------

    def _load_images(self, folder: str) -> None:
        self._image_paths = [
            p for p in Path(folder).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        self._reload_sorted()

    def _sorted_paths(self) -> list[Path]:
        idx = self._sort_combo.currentIndex()
        if idx == 0:
            return sorted(self._image_paths, key=lambda p: p.name.lower())
        elif idx == 1:
            return sorted(self._image_paths, key=lambda p: p.stat().st_mtime)
        else:
            return sorted(self._image_paths, key=lambda p: p.stat().st_mtime, reverse=True)

    def _reload_sorted(self) -> None:
        self._image_list.clear()
        for i, p in enumerate(self._sorted_paths(), 1):
            self._image_list.addItem(f"{i:>4}.  {p.name}")
        self._refresh()

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        ready = (
            bool(self._image_paths)
            and self._lbl_out.text() not in ("", "No output file set")
            and self._worker is None
        )
        self._btn_render.setEnabled(ready)
        count = len(self._image_paths)
        fps   = self._fps_combo.currentData() or 10
        if count:
            secs = count / fps
            music_hint = (
                f"  |  🎵 {Path(self._music_path).name}"
                if self._music_path else "  |  no music"
            )
            self.statusBar().showMessage(
                f"{count} image(s)  —  {fps} fps  —  ≈ {secs:.1f}s video{music_hint}"
            )

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _toggle_render(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        else:
            self._start_render()

    def _start_render(self) -> None:
        output = self._lbl_out.text().strip()
        fps    = int(self._fps_combo.currentData())
        images = self._sorted_paths()
        if not images:
            self._lbl_status.setText("No images found.")
            return

        self._progress.setValue(0)
        self._lbl_phase.setText("")
        self._lbl_status.setText("Starting…")
        self._btn_render.setText("■  Cancel")
        self._btn_folder.setEnabled(False)
        self._btn_out.setEnabled(False)
        self._btn_music.setEnabled(False)

        self._worker = VideoWorker(
            image_paths = images,
            output_path = output,
            fps         = fps,
            music_path  = self._music_path,
            fade_in     = self._spin_fade_in.value(),
            fade_out    = self._spin_fade_out.value(),
            parent      = self,
        )
        self._worker.progress.connect(self._progress.setValue)
        self._worker.phase.connect(self._lbl_phase.setText)
        self._worker.status_msg.connect(self._lbl_status.setText)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, path: str) -> None:
        self._progress.setValue(100)
        self._lbl_phase.setText("Complete!")
        self._lbl_status.setText(f"✓  Done!  Saved to: {path}")
        self.statusBar().showMessage(f"Video saved: {path}")
        self._cleanup_worker()

    def _on_error(self, msg: str) -> None:
        self._lbl_phase.setText("")
        self._lbl_status.setText(f"✗  {msg}")
        self.statusBar().showMessage(f"Error: {msg}")
        self._progress.setValue(0)
        self._cleanup_worker()

    def _cleanup_worker(self) -> None:
        self._worker = None
        self._btn_render.setText("▶  Render Video")
        self._btn_folder.setEnabled(True)
        self._btn_out.setEnabled(True)
        self._btn_music.setEnabled(True)
        self._refresh()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = ImagesToVideoWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
