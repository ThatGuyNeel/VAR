import sys, os, subprocess
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QSlider, QLabel, QComboBox,
    QStackedWidget, QSizePolicy, QMessageBox,
)
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QPainter, QPen, QFont, QColor
from PyQt5.QtCore import Qt, QUrl, QTimer, QRectF, pyqtSignal, QThread


NAVY = "#0F0F65"
TRACKING_DIR = str(Path(__file__).resolve().parent)
BTN_STYLE = "background:#DBDBDB; color:rgb(0,0,0);"
FONT = QFont("Arial", 10)


class Spinner(QWidget):
    def __init__(self, parent=None, size=36, thickness=4):
        super().__init__(parent)
        self._angle = 0
        self._size = size
        self._thick = thickness
        self.setFixedSize(size, size)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def start(self):
        self._timer.start(30)

    def stop(self):
        self._timer.stop()
        self.update()

    def _tick(self):
        self._angle = (self._angle + 12) % 360
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        s, t = self._size, self._thick
        rect = QRectF(t, t, s - 2 * t, s - 2 * t)
        pen = QPen(QColor(80, 80, 80), t, Qt.SolidLine, Qt.RoundCap)
        p.setPen(pen)
        p.drawEllipse(rect)
        pen.setColor(QColor(200, 200, 200))
        p.setPen(pen)
        p.drawArc(rect, (90 - self._angle) * 16, -90 * 16)
        p.end()


class TrackingWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, input_path, parent=None):
        super().__init__(parent)
        self.input_path = input_path

    def run(self):
        try:
            inp = Path(self.input_path)
            out_path = str(inp.parent / f"tracked_{inp.stem}{inp.suffix}")

            driver = f"""
import sys, warnings
warnings.filterwarnings('ignore')
import short_track as st
from pathlib import Path
import cv2
from ultralytics import YOLO
from collections import defaultdict

model = YOLO(st.MODEL_PATH)
model.to('cuda')
cap = cv2.VideoCapture({repr(self.input_path)})
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out_path = {repr(out_path)}
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
stabilizer  = st.TrackStabilizer()
det_filter  = st.DetectionFilter()
visualizer  = st.Visualizer()
fc = 0; active = set()
while True:
    ret, frame = cap.read()
    if not ret: break
    fc += 1; active.clear()
    results = model.track(frame, persist=True,
        tracker=st.TRACKER_CONFIG,
        conf=st.CONFIDENCE_THRESHOLD,
        iou=st.IOU_THRESHOLD,
        classes=[st.PLAYER_CLASS,st.GK_CLASS,st.REF_CLASS,st.STAFF_CLASS,st.BALL_CLASS],
        verbose=False, device='cuda')
    for r in results:
        if r.boxes is None or r.boxes.id is None: continue
        for box,tid,cid,conf in zip(r.boxes.xyxy,r.boxes.id,r.boxes.cls,r.boxes.conf):
            bbox=tuple(map(int,box.cpu().numpy()))
            tid=int(tid); cid=int(cid); cf=float(conf)
            if not det_filter.is_valid_detection(bbox,cid): continue
            stabilizer.add_detection(tid,bbox,cf,cid,fc)
            active.add(tid)
    for tid in set(stabilizer.track_classes)-active:
        if tid not in stabilizer.lost_tracks:
            stabilizer.mark_lost_track(tid,fc)
    stabilizer.cleanup_old_tracks(fc)
    cc = defaultdict(int)
    for tid in active:
        sb = stabilizer.get_smoothed_bbox(tid)
        if sb is None: continue
        cid2 = stabilizer.track_classes[tid]
        cf2  = stabilizer.get_smoothed_confidence(tid)
        ist  = stabilizer.is_stable_track(tid)
        visualizer.draw_detection(frame,sb,tid,cid2,cf2,ist)
        cc[cid2]+=1
    visualizer.draw_stats(frame,{{'active_tracks':len(active),
        'total_tracks':len(stabilizer.track_classes),'class_counts':cc}},fc)
    out.write(frame)
cap.release(); out.release()
print('DONE', flush=True)
"""
            proc = subprocess.Popen(
                [sys.executable, "-c", driver],
                cwd=TRACKING_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            proc.wait()
            if proc.returncode != 0:
                self.error.emit("Tracking failed. Check model path / CUDA.")
                return
            if os.path.exists(out_path):
                self.finished.emit(out_path)
            else:
                self.error.emit(f"Output not found: {out_path}")
        except Exception as e:
            self.error.emit(str(e))


class PlayerTrackingWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Player Tracking")
        self._input_path = None
        self._output_path = None
        self._worker = None

        self.setStyleSheet(f"background:{NAVY};")

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._pages = QStackedWidget()
        layout.addWidget(self._pages, stretch=1)

        self._page_open = self._build_page_open()
        self._page_preview = self._build_page_preview()
        self._page_loading = self._build_page_loading()
        self._page_result = self._build_page_result()

        self._pages.addWidget(self._page_open)
        self._pages.addWidget(self._page_preview)
        self._pages.addWidget(self._page_loading)
        self._pages.addWidget(self._page_result)

        self._pages.setCurrentIndex(0)

    # ── Page 0: Open file landing ────────────────────────────────────────────

    def _build_page_open(self):
        page = QWidget()
        page.setStyleSheet(f"background:{NAVY};")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addStretch(1)

        bar = QWidget()
        bar.setStyleSheet(f"background:{NAVY};")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(12, 8, 12, 8)

        btn = QPushButton("Open File")
        btn.setFont(FONT)
        btn.setFixedWidth(120)
        btn.setStyleSheet(BTN_STYLE)
        btn.clicked.connect(self._open_file)
        bar_layout.addWidget(btn)
        bar_layout.addStretch(1)

        layout.addWidget(bar)
        return page

    # ── Page 1: Video preview + Run Tracking ───────────────────────────────

    def _build_page_preview(self):
        page = QWidget()
        page.setStyleSheet(f"background:{NAVY};")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._preview_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self._preview_widget = QVideoWidget()
        self._preview_widget.setStyleSheet("background:#000;")
        self._preview_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._preview_player.setVideoOutput(self._preview_widget)
        layout.addWidget(self._preview_widget, stretch=1)

        self._preview_slider = QSlider(Qt.Horizontal)
        self._preview_slider.setRange(0, 0)
        self._preview_slider.sliderMoved.connect(
            lambda v: self._preview_player.setPosition(v)
        )
        layout.addWidget(self._preview_slider)

        bar = QWidget()
        bar.setStyleSheet(f"background:{NAVY};")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(8, 6, 8, 6)
        bar_layout.setSpacing(8)

        self._preview_play_btn = QPushButton("Play")
        self._preview_play_btn.setFont(FONT)
        self._preview_play_btn.setFixedWidth(72)
        self._preview_play_btn.setStyleSheet(BTN_STYLE)
        self._preview_play_btn.clicked.connect(self._toggle_preview)
        bar_layout.addWidget(self._preview_play_btn)

        open_btn = QPushButton("Open File")
        open_btn.setFont(FONT)
        open_btn.setFixedWidth(90)
        open_btn.setStyleSheet(BTN_STYLE)
        open_btn.clicked.connect(self._open_file)
        bar_layout.addWidget(open_btn)

        bar_layout.addStretch(1)

        run_btn = QPushButton("Run Tracking")
        run_btn.setFont(FONT)
        run_btn.setFixedWidth(110)
        run_btn.setStyleSheet(BTN_STYLE)
        run_btn.clicked.connect(self._run_tracking)
        bar_layout.addWidget(run_btn)

        layout.addWidget(bar)

        self._preview_player.stateChanged.connect(
            lambda s: self._preview_play_btn.setText(
                "Pause" if s == QMediaPlayer.PlayingState else "Play"
            )
        )
        self._preview_player.positionChanged.connect(
            lambda p: self._preview_slider.setValue(p)
        )
        self._preview_player.durationChanged.connect(
            lambda d: self._preview_slider.setRange(0, d)
        )

        return page

    # ── Page 2: Loading spinner ─────────────────────────────────────────────

    def _build_page_loading(self):
        page = QWidget()
        page.setStyleSheet(f"background:{NAVY};")
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignCenter)

        self._spinner = Spinner(size=40, thickness=4)
        layout.addWidget(self._spinner, alignment=Qt.AlignCenter)

        return page

    # ── Page 3: Result with speed controls ──────────────────────────────────

    def _build_page_result(self):
        page = QWidget()
        page.setStyleSheet(f"background:{NAVY};")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._result_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self._result_widget = QVideoWidget()
        self._result_widget.setStyleSheet("background:#000;")
        self._result_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._result_player.setVideoOutput(self._result_widget)
        layout.addWidget(self._result_widget, stretch=1)

        self._result_slider = QSlider(Qt.Horizontal)
        self._result_slider.setRange(0, 0)
        self._result_slider.sliderMoved.connect(
            lambda v: self._result_player.setPosition(v)
        )
        layout.addWidget(self._result_slider)

        bar = QWidget()
        bar.setStyleSheet(f"background:{NAVY};")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(8, 6, 8, 6)
        bar_layout.setSpacing(8)

        self._result_play_btn = QPushButton("Play")
        self._result_play_btn.setFont(FONT)
        self._result_play_btn.setFixedWidth(72)
        self._result_play_btn.setStyleSheet(BTN_STYLE)
        self._result_play_btn.clicked.connect(self._toggle_result)
        bar_layout.addWidget(self._result_play_btn)

        self._result_time = QLabel("0:00 / 0:00")
        self._result_time.setStyleSheet("color:#ccc; font-size:11px;")
        bar_layout.addWidget(self._result_time)

        bar_layout.addStretch(1)

        speed_lbl = QLabel("Speed:")
        speed_lbl.setStyleSheet("color:#ccc; font-size:11px;")
        bar_layout.addWidget(speed_lbl)

        self._speed_combo = QComboBox()
        for label in ["0.25x", "0.5x", "0.75x", "1x", "1.25x", "1.5x", "2x"]:
            self._speed_combo.addItem(label)
        self._speed_combo.setCurrentIndex(3)
        self._speed_combo.setFixedWidth(70)
        self._speed_combo.currentIndexChanged.connect(self._change_speed)
        bar_layout.addWidget(self._speed_combo)

        bar_layout.addSpacing(16)

        open_btn = QPushButton("Open File")
        open_btn.setFont(FONT)
        open_btn.setFixedWidth(90)
        open_btn.setStyleSheet(BTN_STYLE)
        open_btn.clicked.connect(self._open_file)
        bar_layout.addWidget(open_btn)

        layout.addWidget(bar)

        self._result_player.stateChanged.connect(
            lambda s: self._result_play_btn.setText(
                "Pause" if s == QMediaPlayer.PlayingState else "Play"
            )
        )
        self._result_player.positionChanged.connect(self._result_pos_changed)
        self._result_player.durationChanged.connect(
            lambda d: self._result_slider.setRange(0, d)
        )

        return page

    # ── Actions ─────────────────────────────────────────────────────────────

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", str(Path.home()),
            "Video Files (*.mp4 *.avi *.mov *.mkv *.m4v)",
        )
        if not path:
            return
        self._input_path = path
        self._output_path = None
        self._preview_player.stop()
        self._preview_player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
        self._pages.setCurrentIndex(1)
        self._preview_player.play()

    def _toggle_preview(self):
        if self._preview_player.state() == QMediaPlayer.PlayingState:
            self._preview_player.pause()
        else:
            self._preview_player.play()

    def _toggle_result(self):
        if self._result_player.state() == QMediaPlayer.PlayingState:
            self._result_player.pause()
        else:
            self._result_player.play()

    def _change_speed(self, idx):
        rates = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        self._result_player.setPlaybackRate(rates[idx])

    def _run_tracking(self):
        if not self._input_path or (self._worker and self._worker.isRunning()):
            return
        self._preview_player.stop()
        self._pages.setCurrentIndex(2)
        self._spinner.start()

        self._worker = TrackingWorker(self._input_path, parent=self)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, out_path):
        self._spinner.stop()
        self._output_path = out_path
        self._result_player.setMedia(QMediaContent(QUrl.fromLocalFile(out_path)))
        self._speed_combo.setCurrentIndex(3)
        self._pages.setCurrentIndex(3)
        self._result_player.play()

    def _on_error(self, msg):
        self._spinner.stop()
        self._pages.setCurrentIndex(1)
        QMessageBox.critical(self, "Tracking Error", msg)

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt(ms):
        s = ms // 1000
        return f"{s // 60}:{s % 60:02d}"

    def _result_pos_changed(self, pos):
        self._result_slider.setValue(pos)
        dur = self._result_player.duration()
        self._result_time.setText(f"{self._fmt(pos)} / {self._fmt(dur)}")

    def keyPressEvent(self, e):
        idx = self._pages.currentIndex()
        if e.key() == Qt.Key_Space:
            if idx == 1:
                self._toggle_preview()
            elif idx == 3:
                self._toggle_result()
        elif e.text() == "o":
            self._open_file()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PlayerTrackingWindow()
    win.showMaximized()
    sys.exit(app.exec_())
