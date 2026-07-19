import sys, os, json, subprocess, tempfile, cv2
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QFileDialog, QSlider, QLabel, QComboBox,
    QStackedWidget, QStackedLayout, QSizePolicy, QMessageBox, QDialog,
    QShortcut,
)
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QPainter, QPen, QFont, QColor, QPixmap, QKeySequence
from PyQt5.QtCore import Qt, QUrl, QTimer, QRectF, pyqtSignal, QThread, QDir

try:
    import torch
    import torch.nn as nn
    from torchvision.io.video import read_video
    from torchvision.models.video import MViT_V2_S_Weights
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

VAR_DIR = Path(__file__).resolve().parent
FOUL_DIR = str(VAR_DIR / "Foul Detection" / "VAR interface")
OFFSIDE_DIR = str(VAR_DIR / "Offside Detection")
TRACKING_DIR = str(VAR_DIR / "Player Tracking")
LOGO_PATH = str(VAR_DIR / "Foul Detection" / "VAR interface" / "interface" / "var_logo.png")

NAVY = "#0F0F65"
BTN = "background:#DBDBDB; color:rgb(0,0,0);"
FONT = QFont("Arial", 10)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

sys.path.insert(0, FOUL_DIR)
sys.path.insert(0, OFFSIDE_DIR)
sys.path.insert(0, TRACKING_DIR)


# ── Spinner ─────────────────────────────────────────────────────────────────

class Spinner(QWidget):
    def __init__(self, parent=None, size=40, thickness=4):
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


# ── Workers ─────────────────────────────────────────────────────────────────

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
                self.error.emit("Tracking failed. Check model / CUDA.")
                return
            if os.path.exists(out_path):
                self.finished.emit(out_path)
            else:
                self.error.emit(f"Output not found: {out_path}")
        except Exception as e:
            self.error.emit(str(e))


class ClassifyWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, frame_path, parent=None):
        super().__init__(parent)
        self.frame_path = frame_path

    def run(self):
        try:
            saved = os.getcwd()
            from model.teamClassification.team_classification import team_classification
            from model.sportsfield_release.calculateHomography import calculateOptimHomography
            # team_classification expects cwd to be the Offside Detection dir
            os.chdir(OFFSIDE_DIR)
            dictPlayers, colors, _ = team_classification(self.frame_path)
            # calculateHomography uses paths relative to the Offside Detection dir
            homography = calculateOptimHomography(self.frame_path)
            os.chdir(saved)
            self.finished.emit({
                "dictPlayers": dictPlayers,
                "colors": colors,
                "homography": homography,
                "frame_path": self.frame_path,
            })
        except Exception as e:
            os.chdir(str(VAR_DIR))
            self.error.emit(str(e))


class OffsideWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, data, team, parent=None):
        super().__init__(parent)
        self.data = data
        self.team = team

    def run(self):
        try:
            from offside import drawOffside
            d = self.data
            if self.team == "A":
                atk, def_ = d["dictPlayers"]["Team B"], d["dictPlayers"]["Team A"]
            else:
                atk, def_ = d["dictPlayers"]["Team A"], d["dictPlayers"]["Team B"]
            gk = d["dictPlayers"].get("goalkeeper")
            args = [d["frame_path"], self.team, d["colors"], d["homography"], atk, def_]
            if gk:
                args.append(gk)
            saved = os.getcwd()
            result_dir = os.path.join(OFFSIDE_DIR, "result")
            os.makedirs(result_dir, exist_ok=True)
            # drawOffside internally does os.chdir('result') then os.chdir('..')
            # so cwd must be OFFSIDE_DIR for that to resolve correctly
            os.chdir(OFFSIDE_DIR)
            offside = drawOffside(*args)
            os.chdir(saved)
            self.finished.emit({
                "offside": offside,
                "team": self.team,
                "result3D": os.path.join(result_dir, "result3D.jpg"),
                "result2D": os.path.join(result_dir, "result2D.png"),
            })
        except Exception as e:
            self.error.emit(str(e))


class FoulPredictionWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, files, model, softmax, parent=None):
        super().__init__(parent)
        self.files = files
        self.model = model
        self.softmax = softmax

    def run(self):
        try:
            from interface.config.classes import (
                INVERSE_EVENT_DICTIONARY_offence_severity_class,
                INVERSE_EVENT_DICTIONARY_action_class,
            )
            factor = (85 - 65) / (((85 - 65) / 25) * 21)
            for num_view in range(len(self.files)):
                video, _, _ = read_video(self.files[num_view], output_format="THWC")
                frames = video[65:85, :, :, :]
                final_frames = None
                transforms_model = MViT_V2_S_Weights.KINETICS400_V1.transforms()
                for j in range(len(frames)):
                    if j % factor < 1:
                        if final_frames is None:
                            final_frames = frames[j, :, :, :].unsqueeze(0)
                        else:
                            final_frames = torch.cat(
                                (final_frames, frames[j, :, :, :].unsqueeze(0)), 0
                            )
                final_frames = final_frames.permute(0, 3, 1, 2)
                final_frames = transforms_model(final_frames)
                if num_view == 0:
                    videos = final_frames.unsqueeze(0)
                else:
                    final_frames = final_frames.unsqueeze(0)
                    videos = torch.cat((videos, final_frames), 0)
            videos = videos.unsqueeze(0)
            pred = self.model(videos)
            pred_action = pred[1].unsqueeze(0)
            prediction_action = self.softmax(pred_action)
            values_a, index_a = torch.topk(prediction_action, 2)
            pred_offence = pred[0].unsqueeze(0)
            prediction_offence = self.softmax(pred_offence)
            values_o, index_o = torch.topk(prediction_offence, 2)
            result = {
                "action_top1": f"{INVERSE_EVENT_DICTIONARY_action_class[index_a[0][0].item()]}: {values_a[0][0].item():.2f}",
                "action_top2": f"{INVERSE_EVENT_DICTIONARY_action_class[index_a[0][1].item()]}: {values_a[0][1].item():.2f}",
                "offence_top1": f"{INVERSE_EVENT_DICTIONARY_offence_severity_class[index_o[0][0].item()]}: {values_o[0][0].item():.2f}",
                "offence_top2": f"{INVERSE_EVENT_DICTIONARY_offence_severity_class[index_o[0][1].item()]}: {values_o[0][1].item():.2f}",
                "files": self.files,
            }
            path1 = self.files[0].rsplit("/", 1)[0]
            val = ""
            idx = ""
            for i in range(1, 5):
                val = path1[-i]
                if val == "_":
                    break
                idx += val
            idx = idx[::-1]
            parent_path = path1.rsplit("/", 1)[0]
            anno_path = os.path.join(parent_path, "annotations.json")
            if os.path.exists(anno_path):
                with open(anno_path) as f:
                    data_json = json.load(f)
                result["gt_action"] = data_json["Actions"][idx]["Action class"]
                severity = data_json["Actions"][idx]["Severity"]
                sev_map = {"1.0": "+ No card", "2.0": "+ Borderline NC/YC",
                           "3.0": "+ Yellow card", "4.0": "+ Borderline YC/RC",
                           "5.0": "+ Red card"}
                result["gt_offence"] = data_json["Actions"][idx]["Offence"] + sev_map.get(severity, "")
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ── Offside Dialogs ─────────────────────────────────────────────────────────

class TeamSelectionDialog(QDialog):
    ACCENT = "#E8C438"
    TEXT = "#E8E8F5"
    PANEL = "#16166E"
    _BASE = ("background:#16166E;color:#E8E8F5;border:2px solid #E8C43844;"
            "font-family:'Courier New';font-size:13px;font-weight:bold;"
            "padding:8px 0;border-radius:4px;")
    _ACTIVE = ("background:#16166E;color:#E8E8F5;border:2px solid #E8C438;"
               "font-family:'Courier New';font-size:13px;font-weight:bold;"
               "padding:8px 0;border-radius:4px;")

    def __init__(self, img_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Attacking Team")
        self.setModal(True)
        self.setMinimumSize(860, 560)
        self.setStyleSheet(f"background:{NAVY};color:{self.TEXT};")
        self.selected_team = "A"
        self._px = QPixmap(img_path) if os.path.exists(img_path) else None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        self.imgLbl = QLabel()
        self.imgLbl.setAlignment(Qt.AlignCenter)
        self.imgLbl.setStyleSheet(f"background:{self.PANEL};")
        self.imgLbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if self._px:
            self.imgLbl.setPixmap(self._px.scaled(820, 460, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.imgLbl.setText("Classification image not available")
        layout.addWidget(self.imgLbl, stretch=1)
        legend = QLabel("RED = Team A  |  BLUE = Team B  |  BLACK = Goalkeeper")
        legend.setAlignment(Qt.AlignCenter)
        legend.setStyleSheet(f"font-family:'Courier New';font-size:11px;color:{self.ACCENT};")
        layout.addWidget(legend)
        prompt = QLabel("Which team is attacking?")
        prompt.setAlignment(Qt.AlignCenter)
        prompt.setStyleSheet(f"font-family:'Courier New';font-size:13px;color:{self.ACCENT};")
        layout.addWidget(prompt)
        self.btnA = QPushButton("Team A")
        self.btnB = QPushButton("Team B")
        self.btnA.setStyleSheet(self._ACTIVE)
        self.btnB.setStyleSheet(self._BASE)
        self.btnA.clicked.connect(lambda: self._sel("A"))
        self.btnB.clicked.connect(lambda: self._sel("B"))
        brow = QHBoxLayout()
        brow.setSpacing(16)
        brow.addWidget(self.btnA)
        brow.addWidget(self.btnB)
        layout.addLayout(brow)
        confirm = QPushButton("Confirm & Process")
        confirm.setCursor(Qt.PointingHandCursor)
        confirm.setStyleSheet(
            f"background:{self.ACCENT};color:#0F0F65;border:none;"
            f"font-family:'Courier New';font-size:12px;font-weight:bold;"
            f"padding:10px 0;border-radius:4px;")
        confirm.clicked.connect(self.accept)
        layout.addWidget(confirm)

    def _sel(self, team):
        self.selected_team = team
        self.btnA.setStyleSheet(self._ACTIVE if team == "A" else self._BASE)
        self.btnB.setStyleSheet(self._ACTIVE if team == "B" else self._BASE)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._px:
            self.imgLbl.setPixmap(self._px.scaled(
                self.imgLbl.width(), self.imgLbl.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation))


class ResultsDialog(QDialog):
    ACCENT = "#E8C438"
    PANEL = "#16166E"
    TEXT = "#E8E8F5"

    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Offside Result")
        self.setModal(True)
        self.setStyleSheet(f"background:{NAVY};color:{self.TEXT};")
        self._result = result
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        img_row = QHBoxLayout()
        self.label3d = self._img_label()
        self.label2d = self._img_label()
        img_row.addWidget(self.label3d, 65)
        img_row.addWidget(self.label2d, 35)
        root.addLayout(img_row, stretch=1)
        offside = result["offside"]
        team = result["team"]
        vtext = "NO OFFSIDE" if offside == 0 else f"OFFSIDE — Players: {offside}"
        vcol = "#38C47A" if offside == 0 else "#E84040"
        vrow = QHBoxLayout()
        vrow.addWidget(self._badge(f"Attacking: Team {team}", self.ACCENT))
        vrow.addWidget(self._badge(vtext, vcol), stretch=1)
        close_btn = QPushButton("Close")
        close_btn.setFixedHeight(40)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet(
            f"background:{self.ACCENT};color:#0F0F65;border:none;"
            f"font-family:'Courier New';font-size:13px;font-weight:bold;"
            f"padding:0 28px;border-radius:4px;")
        close_btn.clicked.connect(self.accept)
        vrow.addWidget(close_btn)
        root.addLayout(vrow)
        self.showMaximized()
        QTimer.singleShot(60, self._fill_images)

    def _img_label(self):
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(f"background:{self.PANEL};")
        lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return lbl

    def _badge(self, text, color):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(
            f"background:{self.PANEL};border:2px solid {color};"
            f"font-family:'Courier New';font-size:14px;font-weight:bold;"
            f"color:{color};padding:8px 20px;border-radius:4px;")
        return lbl

    def _fill_images(self):
        self._set_img(self.label3d, self._result["result3D"])
        self._set_img(self.label2d, self._result["result2D"])

    def _set_img(self, lbl, path):
        if not os.path.exists(path):
            lbl.setText(f"[Not found] {path}")
            return
        px = QPixmap(path)
        if px.isNull():
            lbl.setText(f"[Load error] {path}")
            return
        lbl.setProperty("src", path)
        lbl.setPixmap(px.scaled(lbl.width(), lbl.height(),
                                Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        for lbl in (self.label3d, self.label2d):
            path = lbl.property("src")
            if path:
                px = QPixmap(path)
                lbl.setPixmap(px.scaled(lbl.width(), lbl.height(),
                                        Qt.KeepAspectRatio, Qt.SmoothTransformation))


# ── Back Button helper ─────────────────────────────────────────────────────

def _back_button(text="Back", callback=None):
    btn = QPushButton(text)
    btn.setFont(FONT)
    btn.setFixedWidth(80)
    btn.setStyleSheet(BTN)
    if callback:
        btn.clicked.connect(callback)
    return btn


# ── Main Window ─────────────────────────────────────────────────────────────

class VARWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI-Based VAR")
        self.setStyleSheet(f"background:{NAVY};")

        self._video_path = None
        self._image_mode = False
        self._temp_frame_path = None
        self._foul_files = []
        self._foul_model = None
        self._soft = None

        self._workers = []

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._pages = QStackedWidget()
        layout.addWidget(self._pages, stretch=1)

        self.pg_landing = self._build_landing()
        self.pg_preview = self._build_preview()
        self.pg_pt_load = self._build_pt_loading()
        self.pg_pt_result = self._build_pt_result()
        self.pg_foul = self._build_foul()
        self.pg_offside = self._build_offside()

        self._pages.addWidget(self.pg_landing)   # 0
        self._pages.addWidget(self.pg_preview)   # 1
        self._pages.addWidget(self.pg_pt_load)   # 2
        self._pages.addWidget(self.pg_pt_result) # 3
        self._pages.addWidget(self.pg_foul)      # 4
        self._pages.addWidget(self.pg_offside)   # 5

        self._pages.setCurrentIndex(0)

    # ── Page 0: Landing ────────────────────────────────────────────────────

    def _build_landing(self):
        page = QWidget()
        page.setStyleSheet(f"background:{NAVY};")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addStretch(1)

        lbl = QLabel()
        lbl.setAlignment(Qt.AlignCenter)
        px = QPixmap(LOGO_PATH)
        if not px.isNull():
            lbl.setPixmap(px.scaled(600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        layout.addWidget(lbl, alignment=Qt.AlignCenter)

        layout.addStretch(1)

        bar = QWidget()
        bar.setStyleSheet(f"background:{NAVY};")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(12, 8, 12, 8)
        ob = QPushButton("Open File")
        ob.setFont(FONT)
        ob.setFixedWidth(120)
        ob.setStyleSheet(BTN)
        ob.clicked.connect(self._open_file)
        bar_layout.addWidget(ob)
        bar_layout.addStretch(1)
        layout.addWidget(bar)

        self._landing_lbl = lbl
        return page

    # ── Page 1: Video Preview + Component Buttons ──────────────────────────

    def _build_preview(self):
        page = QWidget()
        page.setStyleSheet(f"background:{NAVY};")
        outer = QHBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(0)

        self._prev_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self._prev_widget = QVideoWidget()
        self._prev_widget.setStyleSheet("background:#000;")
        self._prev_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._prev_player.setVideoOutput(self._prev_widget)
        left.addWidget(self._prev_widget, stretch=1)

        self._prev_slider = QSlider(Qt.Horizontal)
        self._prev_slider.setRange(0, 0)
        self._prev_slider.sliderMoved.connect(lambda v: self._prev_player.setPosition(v))
        left.addWidget(self._prev_slider)

        cbar = QWidget()
        cbar.setStyleSheet(f"background:{NAVY};")
        cb_lay = QHBoxLayout(cbar)
        cb_lay.setContentsMargins(8, 6, 8, 6)
        cb_lay.setSpacing(8)

        self._prev_play_btn = QPushButton("Play")
        self._prev_play_btn.setFont(FONT)
        self._prev_play_btn.setFixedWidth(72)
        self._prev_play_btn.setStyleSheet(BTN)
        self._prev_play_btn.clicked.connect(self._toggle_preview)
        cb_lay.addWidget(self._prev_play_btn)

        ob = QPushButton("Open File")
        ob.setFont(FONT)
        ob.setFixedWidth(90)
        ob.setStyleSheet(BTN)
        ob.clicked.connect(self._open_file)
        cb_lay.addWidget(ob)
        cb_lay.addStretch(1)
        left.addWidget(cbar)

        self._prev_player.stateChanged.connect(
            lambda s: self._prev_play_btn.setText(
                "Pause" if s == QMediaPlayer.PlayingState else "Play"))
        self._prev_player.positionChanged.connect(
            lambda p: self._prev_slider.setValue(p))
        self._prev_player.durationChanged.connect(
            lambda d: self._prev_slider.setRange(0, d))

        outer.addLayout(left, stretch=1)

        right = QWidget()
        right.setStyleSheet(f"background:{NAVY};")
        right.setFixedWidth(160)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(12, 12, 12, 12)
        rl.setSpacing(12)
        rl.addStretch(1)

        pt_btn = QPushButton("Player Tracking")
        pt_btn.setFont(FONT)
        pt_btn.setFixedHeight(36)
        pt_btn.setStyleSheet(BTN)
        pt_btn.clicked.connect(self._go_tracking)
        rl.addWidget(pt_btn)

        fd_btn = QPushButton("Foul Detection")
        fd_btn.setFont(FONT)
        fd_btn.setFixedHeight(36)
        fd_btn.setStyleSheet(BTN)
        fd_btn.clicked.connect(self._go_foul)
        rl.addWidget(fd_btn)

        od_btn = QPushButton("Offside Detection")
        od_btn.setFont(FONT)
        od_btn.setFixedHeight(36)
        od_btn.setStyleSheet(BTN)
        od_btn.clicked.connect(self._go_offside)
        rl.addWidget(od_btn)

        rl.addStretch(1)
        outer.addWidget(right)
        return page

    # ── Page 2: Player Tracking Loading ────────────────────────────────────

    def _build_pt_loading(self):
        page = QWidget()
        page.setStyleSheet(f"background:{NAVY};")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        top_bar = QWidget()
        top_bar.setStyleSheet(f"background:{NAVY};")
        top_lay = QHBoxLayout(top_bar)
        top_lay.setContentsMargins(8, 8, 8, 0)
        top_lay.addWidget(_back_button(callback=self._back_to_preview))
        top_lay.addStretch(1)
        layout.addWidget(top_bar)

        layout.addStretch(1)
        self._pt_spinner = Spinner()
        layout.addWidget(self._pt_spinner, alignment=Qt.AlignCenter)
        layout.addStretch(1)
        return page

    # ── Page 3: Player Tracking Result ────────────────────────────────────

    def _build_pt_result(self):
        page = QWidget()
        page.setStyleSheet(f"background:{NAVY};")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        top_bar = QWidget()
        top_bar.setStyleSheet(f"background:{NAVY};")
        top_lay = QHBoxLayout(top_bar)
        top_lay.setContentsMargins(8, 8, 8, 0)
        top_lay.addWidget(_back_button(callback=self._back_to_preview))
        top_lay.addStretch(1)
        layout.addWidget(top_bar)

        self._pt_result_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self._pt_result_widget = QVideoWidget()
        self._pt_result_widget.setStyleSheet("background:#000;")
        self._pt_result_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pt_result_player.setVideoOutput(self._pt_result_widget)
        layout.addWidget(self._pt_result_widget)

        self._pt_result_slider = QSlider(Qt.Horizontal)
        self._pt_result_slider.setRange(0, 0)
        self._pt_result_slider.sliderMoved.connect(
            lambda v: self._pt_result_player.setPosition(v))
        layout.addWidget(self._pt_result_slider)

        bar = QWidget()
        bar.setStyleSheet(f"background:{NAVY};")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(8, 6, 8, 6)
        bl.setSpacing(8)

        self._pt_play_btn = QPushButton("Play")
        self._pt_play_btn.setFont(FONT)
        self._pt_play_btn.setFixedWidth(72)
        self._pt_play_btn.setStyleSheet(BTN)
        self._pt_play_btn.clicked.connect(self._toggle_pt_result)
        bl.addWidget(self._pt_play_btn)

        self._pt_time = QLabel("0:00 / 0:00")
        self._pt_time.setStyleSheet("color:#ccc; font-size:11px;")
        bl.addWidget(self._pt_time)

        bl.addStretch(1)

        sl = QLabel("Speed:")
        sl.setStyleSheet("color:#ccc; font-size:11px;")
        bl.addWidget(sl)

        self._pt_speed = QComboBox()
        for label in ["0.25x", "0.5x", "0.75x", "1x", "1.25x", "1.5x", "2x"]:
            self._pt_speed.addItem(label)
        self._pt_speed.setCurrentIndex(3)
        self._pt_speed.setFixedWidth(70)
        self._pt_speed.currentIndexChanged.connect(self._pt_change_speed)
        bl.addWidget(self._pt_speed)

        layout.addWidget(bar)

        self._pt_result_player.stateChanged.connect(
            lambda s: self._pt_play_btn.setText(
                "Pause" if s == QMediaPlayer.PlayingState else "Play"))
        self._pt_result_player.positionChanged.connect(self._pt_pos_changed)
        self._pt_result_player.durationChanged.connect(
            lambda d: self._pt_result_slider.setRange(0, d))

        return page

    # ── Page 4: Foul Detection ────────────────────────────────────────────

    def _build_foul(self):
        page = QWidget()
        page.setStyleSheet(f"background:{NAVY};")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        top_bar = QWidget()
        top_bar.setStyleSheet(f"background:{NAVY};")
        top_lay = QHBoxLayout(top_bar)
        top_lay.setContentsMargins(8, 8, 8, 0)
        top_lay.addWidget(_back_button(callback=self._back_to_preview))
        top_lay.addStretch(1)
        layout.addWidget(top_bar)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self._foul_players = []
        self._foul_widgets = []
        for _ in range(4):
            mp = QMediaPlayer(None, QMediaPlayer.VideoSurface)
            vw = QVideoWidget()
            vw.setStyleSheet("background:#000;")
            vw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            mp.setVideoOutput(vw)
            self._foul_players.append(mp)
            self._foul_widgets.append(vw)

        up = QHBoxLayout()
        up.setContentsMargins(0, 0, 0, 0)
        up.addWidget(self._foul_widgets[0])
        up.addWidget(self._foul_widgets[1])

        dn = QHBoxLayout()
        dn.setContentsMargins(0, 0, 0, 0)
        dn.addWidget(self._foul_widgets[2])
        dn.addWidget(self._foul_widgets[3])

        vid_col = QVBoxLayout()
        vid_col.setContentsMargins(0, 0, 0, 0)
        vid_col.addLayout(up)
        vid_col.addLayout(dn)
        body.addLayout(vid_col, stretch=1)

        sidebar = QVBoxLayout()
        sidebar.setContentsMargins(8, 8, 8, 8)
        sidebar.insertSpacing(0, 60)

        font_title = QFont("Arial", 20)
        font_title.setBold(True)
        font_text = QFont("Arial", 14)

        self._foul_gt_title = QLabel("Groundtruth")
        self._foul_gt_title.setAlignment(Qt.AlignCenter)
        self._foul_gt_title.setFont(font_title)
        self._foul_gt_title.setStyleSheet("color:white;")
        self._foul_offence = QLabel("")
        self._foul_offence.setAlignment(Qt.AlignCenter)
        self._foul_offence.setFont(font_text)
        self._foul_offence.setStyleSheet("color:white;")
        self._foul_action = QLabel("")
        self._foul_action.setAlignment(Qt.AlignCenter)
        self._foul_action.setFont(font_text)
        self._foul_action.setStyleSheet("color:white;")

        self._foul_pred_title = QLabel("VARS Prediction")
        self._foul_pred_title.setAlignment(Qt.AlignCenter)
        self._foul_pred_title.setFont(font_title)
        self._foul_pred_title.setStyleSheet("color:white;")
        self._foul_pred1 = QLabel("")
        self._foul_pred1.setAlignment(Qt.AlignCenter)
        self._foul_pred1.setFont(font_text)
        self._foul_pred1.setStyleSheet("color:white;")
        self._foul_pred2 = QLabel("")
        self._foul_pred2.setAlignment(Qt.AlignCenter)
        self._foul_pred2.setFont(font_text)
        self._foul_pred2.setStyleSheet("color:white;")
        self._foul_pred3 = QLabel("")
        self._foul_pred3.setAlignment(Qt.AlignCenter)
        self._foul_pred3.setFont(font_text)
        self._foul_pred3.setStyleSheet("color:white;")
        self._foul_pred4 = QLabel("")
        self._foul_pred4.setAlignment(Qt.AlignCenter)
        self._foul_pred4.setFont(font_text)
        self._foul_pred4.setStyleSheet("color:white;")

        spacer = QLabel("")
        sidebar.addWidget(self._foul_gt_title)
        sidebar.addWidget(spacer)
        sidebar.addWidget(self._foul_offence)
        sidebar.addWidget(self._foul_action)
        sidebar.addWidget(spacer)
        sidebar.addWidget(spacer)
        sidebar.addWidget(self._foul_pred_title)
        sidebar.addWidget(self._foul_pred1)
        sidebar.addWidget(self._foul_pred2)
        sidebar.addWidget(spacer)
        sidebar.addWidget(self._foul_pred3)
        sidebar.addWidget(self._foul_pred4)

        self._foul_gt_title.hide()
        self._foul_pred_title.hide()

        show_btns = []
        for i in range(4):
            b = QPushButton(f"Show video {i + 1}")
            b.setFont(FONT)
            b.setStyleSheet(BTN)
            idx = i
            b.clicked.connect(lambda _, x=idx: self._foul_enlarge(x))
            b.hide()
            show_btns.append(b)
            sidebar.addWidget(b)

        self._foul_all_btn = QPushButton("Show all videos")
        self._foul_all_btn.setFont(FONT)
        self._foul_all_btn.setStyleSheet(BTN)
        self._foul_all_btn.clicked.connect(self._foul_show_all)
        self._foul_all_btn.hide()
        sidebar.addWidget(self._foul_all_btn)

        sidebar_widget = QWidget()
        sidebar_widget.setStyleSheet(f"background:{NAVY};")
        sidebar_widget.setLayout(sidebar)
        sidebar_widget.setMinimumWidth(300)
        body.addWidget(sidebar_widget, stretch=0)

        self._foul_show_btns = show_btns
        layout.addLayout(body, stretch=1)

        self._foul_status = QLabel("")
        self._foul_status.setAlignment(Qt.AlignCenter)
        self._foul_status.setStyleSheet("color:#E8C438;font-family:'Courier New';font-size:11px;")
        layout.addWidget(self._foul_status)

        cbar = QWidget()
        cbar.setStyleSheet(f"background:{NAVY};")
        cl = QHBoxLayout(cbar)
        cl.setContentsMargins(8, 6, 8, 6)
        cl.setSpacing(8)

        self._foul_play_btn = QPushButton("Play")
        self._foul_play_btn.setFont(FONT)
        self._foul_play_btn.setFixedWidth(72)
        self._foul_play_btn.setStyleSheet(BTN)
        self._foul_play_btn.clicked.connect(self._foul_toggle_play)
        cl.addWidget(self._foul_play_btn)

        self._foul_slider = QSlider(Qt.Horizontal)
        self._foul_slider.setRange(0, 0)
        self._foul_slider.sliderMoved.connect(
            lambda v: [mp.setPosition(v) for mp in self._foul_players])
        cl.addWidget(self._foul_slider)

        add_btn = QPushButton("Add Files")
        add_btn.setFont(FONT)
        add_btn.setFixedWidth(90)
        add_btn.setStyleSheet(BTN)
        add_btn.clicked.connect(self._foul_add_files)
        cl.addWidget(add_btn)

        run_btn = QPushButton("Run Prediction")
        run_btn.setFont(FONT)
        run_btn.setFixedWidth(120)
        run_btn.setStyleSheet(BTN)
        run_btn.clicked.connect(self._foul_run_prediction)
        cl.addWidget(run_btn)

        layout.addWidget(cbar)

        for mp in self._foul_players:
            mp.stateChanged.connect(self._foul_media_state)
            mp.positionChanged.connect(self._foul_pos_changed)
            mp.durationChanged.connect(lambda d: self._foul_slider.setRange(0, d))

        for vw in self._foul_widgets:
            vw.hide()

        return page

    # ── Page 5: Offside Detection ─────────────────────────────────────────

    def _build_offside(self):
        page = QWidget()
        page.setStyleSheet(f"background:{NAVY};")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        top_bar = QWidget()
        top_bar.setStyleSheet(f"background:{NAVY};")
        top_lay = QHBoxLayout(top_bar)
        top_lay.setContentsMargins(8, 8, 8, 0)
        top_lay.addWidget(_back_button(callback=self._back_to_preview))
        top_lay.addStretch(1)
        layout.addWidget(top_bar)

        self._od_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self._od_video = QVideoWidget()
        self._od_video.setStyleSheet("background:#000;")
        self._od_video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._od_player.setVideoOutput(self._od_video)

        self._od_image = QLabel()
        self._od_image.setAlignment(Qt.AlignCenter)
        self._od_image.setStyleSheet("background:#000;")
        self._od_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._od_image.hide()

        stack = QStackedLayout()
        stack.addWidget(self._od_video)
        stack.addWidget(self._od_image)
        self._od_stack = stack
        sw = QWidget()
        sw.setLayout(stack)
        sw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(sw, stretch=1)

        self._od_status = QLabel("")
        self._od_status.setAlignment(Qt.AlignCenter)
        self._od_status.setStyleSheet("color:#E8C438;font-family:'Courier New';font-size:11px;")
        layout.addWidget(self._od_status)

        self._od_error = QLabel("")
        self._od_error.setStyleSheet("color:#E84040;")
        self._od_error.hide()
        layout.addWidget(self._od_error)

        cbar = QWidget()
        cbar.setStyleSheet(f"background:{NAVY};")
        cl = QHBoxLayout(cbar)
        cl.setContentsMargins(8, 6, 8, 6)
        cl.setSpacing(8)

        self._od_play_btn = QPushButton("Play")
        self._od_play_btn.setFont(FONT)
        self._od_play_btn.setFixedWidth(72)
        self._od_play_btn.setStyleSheet(BTN)
        self._od_play_btn.clicked.connect(self._od_toggle)
        cl.addWidget(self._od_play_btn)

        self._od_slider = QSlider(Qt.Horizontal)
        self._od_slider.setRange(0, 0)
        self._od_slider.sliderMoved.connect(lambda v: self._od_player.setPosition(v))
        cl.addWidget(self._od_slider, stretch=1)

        ob = QPushButton("Open File")
        ob.setFont(FONT)
        ob.setFixedWidth(90)
        ob.setStyleSheet(BTN)
        ob.clicked.connect(self._open_file)
        cl.addWidget(ob)

        cap_btn = QPushButton("Capture Frame")
        cap_btn.setFont(FONT)
        cap_btn.setFixedWidth(110)
        cap_btn.setStyleSheet(BTN)
        cap_btn.setEnabled(False)
        cap_btn.clicked.connect(self._od_capture)
        self._od_capture_btn = cap_btn
        cl.addWidget(cap_btn)

        self._od_analyse_btn = QPushButton("Analyse")
        self._od_analyse_btn.setFont(FONT)
        self._od_analyse_btn.setStyleSheet(BTN)
        self._od_analyse_btn.setEnabled(False)
        self._od_analyse_btn.hide()
        self._od_analyse_btn.clicked.connect(self._od_analyse_image)
        cl.addWidget(self._od_analyse_btn)

        layout.addWidget(cbar)

        self._od_player.stateChanged.connect(
            lambda s: self._od_play_btn.setText(
                "Pause" if s == QMediaPlayer.PlayingState else "Play"))
        self._od_player.positionChanged.connect(
            lambda p: self._od_slider.setValue(p))
        self._od_player.durationChanged.connect(
            lambda d: self._od_slider.setRange(0, d))

        return page

    # ── Shared Actions ─────────────────────────────────────────────────────

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", QDir.homePath(),
            "Media (*.mp4 *.avi *.mov *.mkv *.m4v *.jpg *.jpeg *.png *.bmp *.tiff)")
        if not path:
            return
        self._video_path = path
        self._image_mode = os.path.splitext(path)[1].lower() in IMAGE_EXTS

        if self._image_mode:
            self._go_offside_with_image(path)
        else:
            self._prev_player.stop()
            self._prev_player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
            self._pages.setCurrentIndex(1)
            self._prev_player.play()

    def _toggle_preview(self):
        if self._prev_player.state() == QMediaPlayer.PlayingState:
            self._prev_player.pause()
        else:
            self._prev_player.play()

    def _back_to_preview(self):
        self._pt_result_player.stop()
        for mp in self._foul_players:
            mp.stop()
        self._od_player.stop()
        if self._video_path and not self._image_mode:
            self._prev_player.setMedia(QMediaContent(QUrl.fromLocalFile(self._video_path)))
            self._pages.setCurrentIndex(1)
            self._prev_player.play()
        else:
            self._pages.setCurrentIndex(0)

    # ── Player Tracking Flow ───────────────────────────────────────────────

    def _go_tracking(self):
        if not self._video_path or self._image_mode:
            return
        self._prev_player.pause()
        self._pages.setCurrentIndex(2)
        self._pt_spinner.start()

        w = TrackingWorker(self._video_path, parent=self)
        w.finished.connect(self._pt_on_done)
        w.error.connect(self._pt_on_error)
        self._workers.append(w)
        w.start()

    def _pt_on_done(self, out_path):
        self._pt_spinner.stop()
        self._pt_result_player.setMedia(QMediaContent(QUrl.fromLocalFile(out_path)))
        self._pt_speed.setCurrentIndex(3)
        self._pages.setCurrentIndex(3)
        self._pt_result_player.play()

    def _pt_on_error(self, msg):
        self._pt_spinner.stop()
        QMessageBox.critical(self, "Tracking Error", msg)
        self._pages.setCurrentIndex(1)

    def _toggle_pt_result(self):
        if self._pt_result_player.state() == QMediaPlayer.PlayingState:
            self._pt_result_player.pause()
        else:
            self._pt_result_player.play()

    def _pt_change_speed(self, idx):
        rates = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        self._pt_result_player.setPlaybackRate(rates[idx])

    @staticmethod
    def _fmt(ms):
        s = ms // 1000
        return f"{s // 60}:{s % 60:02d}"

    def _pt_pos_changed(self, pos):
        self._pt_result_slider.setValue(pos)
        dur = self._pt_result_player.duration()
        self._pt_time.setText(f"{self._fmt(pos)} / {self._fmt(dur)}")

    # ── Foul Detection Flow ────────────────────────────────────────────────

    def _go_foul(self):
        if not self._video_path or self._image_mode:
            return
        self._prev_player.pause()
        self._foul_files = [self._video_path]
        self._foul_reset_ui()
        self._pages.setCurrentIndex(4)
        self._foul_set_views(self._foul_files)

    def _foul_add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select additional camera angles", QDir.homePath())
        if not files:
            return
        combined = [self._video_path] + files[:3]
        self._foul_files = combined
        self._foul_set_views(combined)

    def _foul_set_views(self, files):
        for vw in self._foul_widgets:
            vw.hide()
        for i, f in enumerate(files[:4]):
            self._foul_players[i].setMedia(QMediaContent(QUrl.fromLocalFile(f)))
            self._foul_widgets[i].show()
        self._foul_play_btn.setEnabled(True)
        self._foul_slider.setValue(2500)
        for mp in self._foul_players:
            mp.setPosition(2500)
            mp.play()

    def _foul_run_prediction(self):
        if not HAS_TORCH:
            self._foul_status.setText("PyTorch not available — cannot predict")
            return
        self._foul_status.setText("Running prediction...")
        try:
            if self._foul_model is None:
                from interface.model import MVNetwork
                self._foul_model = MVNetwork(net_name="mvit_v2_s", agr_type="attention")
                path = os.path.join(FOUL_DIR, "interface", "14_model.pth.tar").replace("\\", "/")
                load = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
                self._foul_model.load_state_dict(load["state_dict"])
                self._foul_model.eval()
                self._soft = nn.Softmax(dim=1)
            w = FoulPredictionWorker(self._foul_files, self._foul_model, self._soft, parent=self)
            w.finished.connect(self._foul_on_pred)
            w.error.connect(self._foul_on_error)
            self._workers.append(w)
            w.start()
        except Exception as e:
            self._foul_status.setText("")
            QMessageBox.critical(self, "Model Error", str(e))

    def _foul_on_pred(self, result):
        self._foul_status.setText("")
        self._foul_pred1.setText(result.get("offence_top1", ""))
        self._foul_pred2.setText(result.get("offence_top2", ""))
        self._foul_pred3.setText(result.get("action_top1", ""))
        self._foul_pred4.setText(result.get("action_top2", ""))
        self._foul_gt_title.show()
        self._foul_pred_title.show()

        if "gt_offence" in result:
            self._foul_offence.setText(result["gt_offence"])
        if "gt_action" in result:
            self._foul_action.setText(result["gt_action"])

        n = len(self._foul_files)
        if n >= 2:
            self._foul_show_btns[0].show()
            self._foul_show_btns[1].show()
            self._foul_all_btn.show()
        if n >= 3:
            self._foul_show_btns[2].show()
        if n >= 4:
            self._foul_show_btns[3].show()

    def _foul_on_error(self, msg):
        self._foul_status.setText("")
        QMessageBox.critical(self, "Prediction Error", msg)

    def _foul_toggle_play(self):
        for mp in self._foul_players:
            if mp.state() == QMediaPlayer.PlayingState:
                mp.pause()
            else:
                mp.play()

    def _foul_media_state(self, state):
        if self._foul_players[0].state() == QMediaPlayer.PlayingState:
            self._foul_play_btn.setText("Pause")
        else:
            self._foul_play_btn.setText("Play")

    def _foul_pos_changed(self, pos):
        self._foul_slider.setValue(pos)

    def _foul_enlarge(self, idx):
        for vw in self._foul_widgets:
            vw.hide()
        if idx < len(self._foul_files):
            self._foul_widgets[idx].show()
        for mp in self._foul_players:
            mp.setPosition(2500)
        self._foul_play_btn.setEnabled(True)
        for mp in self._foul_players[:len(self._foul_files)]:
            mp.play()

    def _foul_show_all(self):
        for i, vw in enumerate(self._foul_widgets):
            if i < len(self._foul_files):
                vw.show()
            else:
                vw.hide()

    def _foul_reset_ui(self):
        for s in self._foul_show_btns:
            s.hide()
        self._foul_all_btn.hide()
        self._foul_gt_title.hide()
        self._foul_pred_title.hide()
        for lbl in (self._foul_offence, self._foul_action,
                    self._foul_pred1, self._foul_pred2,
                    self._foul_pred3, self._foul_pred4):
            lbl.setText("")
        self._foul_status.setText("")

    # ── Offside Detection Flow ─────────────────────────────────────────────

    def _go_offside(self):
        if not self._video_path:
            return
        self._prev_player.pause()
        self._pages.setCurrentIndex(5)
        if self._image_mode:
            self._od_stack.setCurrentIndex(1)
            self._od_image.show()
            self._od_video.hide()
            self._show_od_pixmap(self._video_path)
            self._od_play_btn.setEnabled(False)
            self._od_slider.setEnabled(False)
            self._od_capture_btn.setEnabled(False)
            self._od_analyse_btn.setText("Analyse Image")
            self._od_analyse_btn.setEnabled(True)
            self._od_analyse_btn.show()
        else:
            self._od_stack.setCurrentIndex(0)
            self._od_image.hide()
            self._od_video.show()
            self._od_player.setMedia(QMediaContent(QUrl.fromLocalFile(self._video_path)))
            self._od_capture_btn.setEnabled(True)
            self._od_analyse_btn.hide()
            self._od_player.play()

    def _go_offside_with_image(self, path):
        self._image_mode = True
        self._pages.setCurrentIndex(5)
        self._od_stack.setCurrentIndex(1)
        self._od_video.hide()
        self._od_image.show()
        self._show_od_pixmap(path)
        self._od_play_btn.setEnabled(False)
        self._od_slider.setEnabled(False)
        self._od_capture_btn.setEnabled(False)
        self._od_analyse_btn.setText("Analyse Image")
        self._od_analyse_btn.setEnabled(True)
        self._od_analyse_btn.show()

    def _od_toggle(self):
        if self._od_player.state() == QMediaPlayer.PlayingState:
            self._od_player.pause()
        else:
            self._od_player.play()

    def _od_capture(self):
        if not self._video_path or self._image_mode:
            return
        if self._workers and self._workers[-1].isRunning():
            return

        self._od_player.pause()

        pos_ms = self._od_player.position()
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            self._od_error.setText("Could not open video file.")
            self._od_error.show()
            return
        # Millisecond-based seeking 
        cap.set(cv2.CAP_PROP_POS_MSEC, float(pos_ms))
        ret, frame = cap.read()
        cap.release()

        if not ret:
            self._od_error.setText("Failed to read frame.")
            self._od_error.show()
            return

        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        cv2.imwrite(tmp.name, frame)
        self._temp_frame_path = tmp.name

        self._od_stack.setCurrentIndex(1)
        self._od_image.show()
        self._show_od_pixmap(tmp.name)
        self._od_start_classify(tmp.name)

    def _od_analyse_image(self):
        if self._image_mode and self._video_path:
            self._od_start_classify(self._video_path)

    def _od_start_classify(self, path):
        self._od_capture_btn.setEnabled(False)
        self._od_analyse_btn.setEnabled(False)
        self._od_status.setText("Classifying players...")
        self._od_error.hide()
        w = ClassifyWorker(path, parent=self)
        w.finished.connect(self._od_on_classify)
        w.error.connect(self._od_on_worker_error)
        self._workers.append(w)
        QTimer.singleShot(10, w.start)

    def _od_on_classify(self, data):
        self._od_status.setText("")
        cls_img = os.path.join(OFFSIDE_DIR, "result", "teamClassification.png")
        dlg = TeamSelectionDialog(cls_img, self)
        if dlg.exec_() != QDialog.Accepted:
            self._od_reset_buttons()
            self._od_cleanup_temp()
            return
        self._od_start_offside(data, dlg.selected_team)

    def _od_start_offside(self, data, team):
        self._od_status.setText("Detecting offside...")
        w = OffsideWorker(data, team, parent=self)
        w.finished.connect(self._od_on_offside)
        w.error.connect(self._od_on_worker_error)
        self._workers.append(w)
        w.start()

    def _od_on_offside(self, result):
        self._od_status.setText("")
        self._od_cleanup_temp()
        self._od_reset_buttons()
        ResultsDialog(result, parent=self).exec_()

    def _od_on_worker_error(self, msg):
        self._od_status.setText("")
        self._od_cleanup_temp()
        self._od_reset_buttons()
        self._od_error.setText(f"Error: {msg}")
        self._od_error.show()

    def _od_reset_buttons(self):
        if self._image_mode:
            self._od_analyse_btn.setEnabled(True)
        else:
            self._od_capture_btn.setEnabled(True)
            self._od_stack.setCurrentIndex(0)

    def _od_cleanup_temp(self):
        if self._temp_frame_path and os.path.exists(self._temp_frame_path):
            try:
                os.remove(self._temp_frame_path)
            except OSError:
                pass
        self._temp_frame_path = None

    def _show_od_pixmap(self, path):
        px = QPixmap(path)
        if not px.isNull():
            w = self._od_image.width() or 1280
            h = self._od_image.height() or 720
            self._od_image.setPixmap(px.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self._od_image.setText(f"[Cannot load] {path}")

    # ── Keyboard ────────────────────────────────────────────────────────────

    def keyPressEvent(self, e):
        idx = self._pages.currentIndex()
        if e.key() == Qt.Key_Space:
            if idx == 1:
                self._toggle_preview()
            elif idx == 3:
                self._toggle_pt_result()
            elif idx == 4:
                self._foul_toggle_play()
            elif idx == 5:
                self._od_toggle()
        elif e.text() == "o":
            self._open_file()
        elif e.text() == "b":
            self._back_to_preview()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VARWindow()
    win.showMaximized()
    sys.exit(app.exec_())
