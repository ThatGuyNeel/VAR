import sys, os, cv2, tempfile, numpy as np
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from offside import drawOffside
from model.sportsfield_release.calculateHomography import calculateOptimHomography
from model.teamClassification.team_classification import team_classification

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


# worker 1
class ClassifyWorker(QThread):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, frame_path, parent=None):
        super().__init__(parent)
        self.frame_path = frame_path

    def run(self):
        try:
            dictPlayers, colors, _ = team_classification(self.frame_path)
            homography = calculateOptimHomography(self.frame_path)
            self.finished.emit({
                'dictPlayers': dictPlayers,
                'colors':      colors,
                'homography':  homography,
                'frame_path':  self.frame_path,
            })
        except Exception as e:
            self.error.emit(str(e))


# worker 2: drawOffside 
class OffsideWorker(QThread):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, data: dict, team: str, parent=None):
        super().__init__(parent)
        self.data = data
        self.team = team

    def run(self):
        try:
            d = self.data
            if self.team == 'A':
                atk, def_ = d['dictPlayers']['Team B'], d['dictPlayers']['Team A']
            else:
                atk, def_ = d['dictPlayers']['Team A'], d['dictPlayers']['Team B']
            gk   = d['dictPlayers'].get('goalkeeper')
            args = [d['frame_path'], self.team, d['colors'], d['homography'], atk, def_]
            if gk:
                args.append(gk)
            offside = drawOffside(*args)
            self.finished.emit({
                'offside':  offside,
                'team':     self.team,
                'result3D': 'result/result3D.jpg',
                'result2D': 'result/result2D.png',
            })
        except Exception as e:
            self.error.emit(str(e))


# results dialog 
class ResultsDialog(QDialog):
    NAVY = "#0F0F65"; ACCENT = "#E8C438"; PANEL = "#16166E"; TEXT = "#E8E8F5"

    def __init__(self, result: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Offside Result")
        self.setModal(True)
        self.setStyleSheet(f"background:{self.NAVY};color:{self.TEXT};")
        self._result = result

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # title = QLabel("OFFSIDE ANALYSIS")
        # title.setAlignment(Qt.AlignCenter)
        # title.setStyleSheet(
        #     f"font-family:'Courier New';font-size:20px;font-weight:bold;"
        #     f"letter-spacing:6px;color:{self.ACCENT};")
        # root.addWidget(title)

        img_row = QHBoxLayout()
        self.label3d = self._make_img_label()
        self.label2d = self._make_img_label()
        img_row.addWidget(self.label3d, 65)
        img_row.addWidget(self.label2d, 35)
        root.addLayout(img_row, stretch=1)

        # verdict
        offside = result['offside']
        team    = result['team']
        vtext   = "NO OFFSIDE" if offside == 0 else f"OFFSIDE — Players: {offside}"
        vcol    = "#38C47A"    if offside == 0 else "#E84040"

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

    def _make_img_label(self):
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
        self._set_img(self.label3d, self._result['result3D'])
        self._set_img(self.label2d, self._result['result2D'])

    def _set_img(self, lbl, path):
        if not os.path.exists(path):
            lbl.setText(f"[Not found] {path}"); return
        px = QPixmap(path)
        if px.isNull():
            lbl.setText(f"[Load error] {path}"); return
        lbl.setProperty('src', path)
        lbl.setPixmap(px.scaled(lbl.width(), lbl.height(),
                                Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        for lbl in (self.label3d, self.label2d):
            path = lbl.property('src')
            if path:
                px = QPixmap(path)
                lbl.setPixmap(px.scaled(lbl.width(), lbl.height(),
                                        Qt.KeepAspectRatio, Qt.SmoothTransformation))


# team selection dialog : shows classification image 
class TeamSelectionDialog(QDialog):
    NAVY = "#0F0F65"; ACCENT = "#E8C438"; TEXT = "#E8E8F5"; PANEL = "#16166E"
    _BASE   = ("background:#16166E;color:#E8E8F5;border:2px solid #E8C43844;"
               "font-family:'Courier New';font-size:13px;font-weight:bold;"
               "padding:8px 0;border-radius:4px;")
    _ACTIVE = ("background:#16166E;color:#E8E8F5;border:2px solid #E8C438;"
               "font-family:'Courier New';font-size:13px;font-weight:bold;"
               "padding:8px 0;border-radius:4px;")

    def __init__(self, img_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Attacking Team")
        self.setModal(True)
        self.setMinimumSize(860, 560)
        self.setStyleSheet(f"background:{self.NAVY};color:{self.TEXT};")
        self.selected_team = "A"
        self._px = QPixmap(img_path) if os.path.exists(img_path) else None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # classification image
        self.imgLbl = QLabel()
        self.imgLbl.setAlignment(Qt.AlignCenter)
        self.imgLbl.setStyleSheet(f"background:{self.PANEL};")
        self.imgLbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if self._px:
            self.imgLbl.setPixmap(self._px.scaled(
                820, 460, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.imgLbl.setText("Classification image not available")
        layout.addWidget(self.imgLbl, stretch=1)

        legend = QLabel("RED = Team A   |   BLUE = Team B   |   BLACK = Goalkeeper")
        legend.setAlignment(Qt.AlignCenter)
        legend.setStyleSheet(
            f"font-family:'Courier New';font-size:11px;color:{self.ACCENT};")
        layout.addWidget(legend)

        prompt = QLabel("Which team is attacking?")
        prompt.setAlignment(Qt.AlignCenter)
        prompt.setStyleSheet(
            f"font-family:'Courier New';font-size:13px;color:{self.ACCENT};")
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


# main window
class VideoWindow(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Offside Detection System")
        self.setStyleSheet("background:#0F0F65;")

        self.current_filepath = None
        self._temp_frame_path = None
        self._worker          = None
        self._image_mode      = False

        font2 = QFont('Arial', 10)

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.videoWidget = QVideoWidget()
        self.videoWidget.setMinimumSize(1, 1)
        self.mediaPlayer.setVideoOutput(self.videoWidget)
        self.videoWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.imageLabel = QLabel()
        self.imageLabel.setMinimumSize(1, 1)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setStyleSheet("background:#000;")
        self.imageLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.imageLabel.hide()

        # stacked layout swaps between video and image
        stack = QStackedLayout()
        stack.addWidget(self.videoWidget)   # 0
        stack.addWidget(self.imageLabel)    # 1
        self._stack = stack
        stackWidget = QWidget()
        stackWidget.setLayout(stack)
        stackWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # controls
        self.playButton = QPushButton("Play")
        self.playButton.setEnabled(False)
        self.playButton.setFont(font2)
        self.playButton.setStyleSheet("background:#DBDBDB;color:rgb(0,0,0)")
        self.playButton.clicked.connect(self.play)
        QShortcut(QKeySequence("Space"), self).activated.connect(self.play)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.openButton = QPushButton("Open File")
        self.openButton.setFont(font2)
        self.openButton.setStyleSheet("background:#DBDBDB;color:rgb(0,0,0)")
        self.openButton.clicked.connect(self.openFile)

        self.captureButton = QPushButton("Capture Frame")
        self.captureButton.setFont(font2)
        self.captureButton.setStyleSheet("background:#DBDBDB;color:rgb(0,0,0)")
        self.captureButton.setEnabled(False)
        self.captureButton.clicked.connect(self.capture_frame)

        self.analyseButton = QPushButton("Analyse")
        self.analyseButton.setFont(font2)
        self.analyseButton.setStyleSheet(
            "background:#DBDBDB;color:rgb(0,0,0)")
        self.analyseButton.setEnabled(False)
        self.analyseButton.hide()
        self.analyseButton.clicked.connect(self._on_analyse_clicked)

        self.statusLabel = QLabel("")
        self.statusLabel.setAlignment(Qt.AlignCenter)
        self.statusLabel.setStyleSheet(
            "color:#E8C438;font-family:'Courier New';font-size:11px;")

        self.errorLabel = QLabel()
        self.errorLabel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.errorLabel.setStyleSheet("color:#E84040;")
        self.errorLabel.hide()

        wid = QWidget(self)
        self.setCentralWidget(wid)

        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(4, 4, 4, 4)
        ctrl.addWidget(self.playButton)
        ctrl.addWidget(self.positionSlider, stretch=1)
        ctrl.addWidget(self.openButton)
        ctrl.addWidget(self.captureButton)
        ctrl.addWidget(self.analyseButton)

        main = QVBoxLayout(wid)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)
        main.addWidget(stackWidget, stretch=1)
        main.addWidget(self.statusLabel)
        main.addLayout(ctrl)
        main.addWidget(self.errorLabel)

        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)

    # open file 
    def openFile(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Open File", QDir.homePath(),
            "Media (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp *.tiff)")
        if not file:
            return
        self._reset_state()
        self.current_filepath = file
        if os.path.splitext(file)[1].lower() in IMAGE_EXTS:
            self._load_image(file)
        else:
            self._load_video(file)

    def _load_image(self, path):
        self._image_mode = True
        self._stack.setCurrentIndex(1)
        self.imageLabel.show()
        self._show_pixmap(path)
        self.playButton.setEnabled(False)
        self.positionSlider.setEnabled(False)
        self.captureButton.setEnabled(False)
        self.analyseButton.setText("Analyze Image")
        self.analyseButton.setEnabled(True)
        self.analyseButton.show()
        self.statusLabel.setText(f"Image: {os.path.basename(path)}")

    def _load_video(self, path):
        self._image_mode = False
        self._stack.setCurrentIndex(0)
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
        self.playButton.setEnabled(True)
        self.positionSlider.setEnabled(True)
        self.captureButton.setEnabled(True)
        self.analyseButton.hide()
        self.mediaPlayer.play()

    def _show_pixmap(self, path):
        px = QPixmap(path)
        if not px.isNull():
            w = self.imageLabel.width()  or 1280
            h = self.imageLabel.height() or 720
            self.imageLabel.setPixmap(
                px.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.imageLabel.setText(f"[Cannot load] {path}")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._image_mode and self.current_filepath:
            self._show_pixmap(self.current_filepath)

    # capture 
    def capture_frame(self):
        if not self.current_filepath or (self._worker and self._worker.isRunning()):
            return
        self.mediaPlayer.pause()

        pos_ms = self.mediaPlayer.position()
        cap = cv2.VideoCapture(self.current_filepath)
        if not cap.isOpened():
            self._show_error("Could not open video file.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # fallback

        # calculate exact frame number
        frame_number = int(round((pos_ms / 1000.0) * fps))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # clamp to valid range
        frame_number = max(0, min(frame_number, total_frames - 1))

        # seek to that exact frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            self._show_error("Failed to read frame.")
            return

        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        cv2.imwrite(tmp.name, frame)
        self._temp_frame_path = tmp.name

        self._stack.setCurrentIndex(1)
        self.imageLabel.show()
        self._show_pixmap(tmp.name)
        self._start_classify(tmp.name)

    def _on_analyse_clicked(self):
        if self._image_mode and self.current_filepath:
            self._start_classify(self.current_filepath)

    # pipeline stage 1 
    def _start_classify(self, path):
        self.captureButton.setEnabled(False)
        self.analyseButton.setEnabled(False)
        self.statusLabel.setText("Classifying players…")
        self.errorLabel.hide()
        self._worker = ClassifyWorker(path, parent=self)
        self._worker.finished.connect(self._on_classify_done)
        self._worker.error.connect(self._on_worker_error)
        QTimer.singleShot(10, self._worker.start)

    def _on_classify_done(self, data: dict):
        self.statusLabel.setText("")
        classification_img = 'result/teamClassification.png'
        # if os.path.exists(classification_img):
        #     self._show_pixmap(classification_img)
        # else:
        #     # fallback if file not found 
        #     self._show_pixmap(data['frame_path'])
        
        dlg = TeamSelectionDialog(classification_img, self)
        if dlg.exec_() != QDialog.Accepted:
            self._reset_buttons()
            self._cleanup_temp()
            return
        self._start_offside(data, dlg.selected_team)

    # pipeline stage 2 
    def _start_offside(self, data: dict, team: str):
        self.statusLabel.setText("Detecting offside…")
        self._worker = OffsideWorker(data, team, parent=self)
        self._worker.finished.connect(self._on_offside_done)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

    def _on_offside_done(self, result: dict):
        self.statusLabel.setText("")
        self._cleanup_temp()
        self._reset_buttons()
        ResultsDialog(result, parent=self).exec_()

    def _on_worker_error(self, msg: str):
        self.statusLabel.setText("")
        self._cleanup_temp()
        self._reset_buttons()
        self._show_error(f"Error: {msg}")

    def _reset_buttons(self):
        if self._image_mode:
            self.analyseButton.setEnabled(True)
        else:
            self.captureButton.setEnabled(True)
            self._stack.setCurrentIndex(0)

    def _reset_state(self):
        self.mediaPlayer.stop()
        self._image_mode = False
        self._cleanup_temp()
        self.errorLabel.hide()
        self.statusLabel.setText("")

    def _cleanup_temp(self):
        if self._temp_frame_path and os.path.exists(self._temp_frame_path):
            try: os.remove(self._temp_frame_path)
            except OSError: pass
        self._temp_frame_path = None

    def _show_error(self, msg):
        self.errorLabel.setText(msg); self.errorLabel.show()

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self, _):
        self.playButton.setText(
            "Pause" if self.mediaPlayer.state() == QMediaPlayer.PlayingState else "Play")

    def positionChanged(self, pos): self.positionSlider.setValue(pos)
    def durationChanged(self, dur): self.positionSlider.setRange(0, dur)
    def setPosition(self, pos):     self.mediaPlayer.setPosition(pos)
    def handleError(self):
        self.playButton.setEnabled(False)
        self._show_error("Media error: " + self.mediaPlayer.errorString())
    def keyPressEvent(self, e):
        if e.text() == "o": self.openFile()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoWindow()
    player.showMaximized()
    sys.exit(app.exec_())