"""
Patched copy of the Foul Detection video_window.py with Python 3.12 fixes:
1. Removes the unused `from moviepy.editor import *` and `from moviepy.config import get_setting`
   (moviepy 1.0.3 crashes on Python 3.12 because the `imghdr` module was removed)
2. Changes `== None` to `is None` (SyntaxWarning on Python 3.12)

To use the standalone Foul Detection GUI on Python 3.12, run this file instead of the original:
    python video_window_compat.py

This file is a clean re-derivative. The original file at
C:/Users/neeld/VAR/Foul Detection/VARS interface/interface/video_window.py
is NOT modified.
"""

import sys
import os

from PyQt5 import QtCore
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pandas as pd
import json
import torch
from torchvision.io.video import read_video
import torch.nn as nn
from interface.model import MVNetwork
from torchvision.models.video import MViT_V2_S_Weights
from interface.config.classes import EVENT_DICTIONARY, INVERSE_EVENT_DICTIONARY_action_class, INVERSE_EVENT_DICTIONARY_offence_severity_class


class VideoWindow(QMainWindow):

    def __init__(self, parent=None):
        super(VideoWindow, self).__init__(parent)

        self.show_prediction = True
        rootdir = os.getcwd()

        self.model = MVNetwork(net_name="mvit_v2_s", agr_type="attention")
        path = os.path.join(rootdir, 'interface')
        path = os.path.join(path, '14_model.pth.tar')
        path = path.replace('\\', '/')

        load = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        self.model.load_state_dict(load['state_dict'])
        self.model.eval()
        self.softmax = nn.Softmax(dim=1)

        self.setWindowTitle("Foul Detection")

        path = os.path.join(rootdir, 'interface')
        path_image = os.path.join(path, 'var_logo.png')
        path_image = path_image.replace('\\', '/')

        self.setStyleSheet("background: #0F0F65;")

        self.mediaPlayers = []
        self.videoWidgets = []
        self.frame_duration_ms = 40

        self.files = []

        for i in range(4):
            self.mediaPlayers.append(QMediaPlayer(
                None, QMediaPlayer.VideoSurface))
            self.videoWidgets.append(QVideoWidget())
            self.mediaPlayers[i].setVideoOutput(self.videoWidgets[i])

        upperLayout = QHBoxLayout()
        upperLayout.setContentsMargins(0, 0, 0, 0)
        upperLayout.addWidget(self.videoWidgets[0])
        upperLayout.addWidget(self.videoWidgets[1])

        bottomLayout = QHBoxLayout()
        bottomLayout.setContentsMargins(0, 0, 0, 0)
        bottomLayout.addWidget(self.videoWidgets[2])
        bottomLayout.addWidget(self.videoWidgets[3])

        finalLayout = QVBoxLayout()
        finalLayout.setContentsMargins(0, 0, 0, 0)
        finalLayout.addLayout(upperLayout)
        finalLayout.addLayout(bottomLayout)

        sidebar = QVBoxLayout()
        sidebar.insertSpacing(0, 100)

        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        font2 = QFont('Arial', 10)
        self.playButton.setFont(font2)
        self.playButton.setStyleSheet("background:#DBDBDB;"
            "color: rgb(0,0,0)")
        self.playButton.setText("Play")
        self.playButton.clicked.connect(self.play)

        playShortcut = QShortcut(QKeySequence("Space"), self)
        playShortcut.activated.connect(self.play)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.errorLabel = QLabel()
        self.errorLabel.setSizePolicy(QSizePolicy.Preferred,
                                      QSizePolicy.Maximum)
        self.errorLabel.hide()

        openButton = QPushButton("Open files", self)
        openButton.setFont(font2)
        openButton.setStyleSheet("background:#DBDBDB;"
            "color: rgb(0,0,0)")
        openButton.clicked.connect(self.openFile)

        wid = QWidget(self)
        self.setCentralWidget(wid)

        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)
        controlLayout.addWidget(openButton)

        decisionLayout = QVBoxLayout()
        font = QFont('Arial', 20)
        font.setBold(True)
        self.decisionTitle = QLabel("Groundtruth")
        self.decisionTitle.setAlignment(Qt.AlignCenter)
        self.decisionTitle.setFont(font)
        self.decisionTitle.setStyleSheet("color: rgb(255,255,255)")

        fontText = QFont('Arial', 14)

        self.spacetext = QLabel("")
        self.actionText = QLabel("")
        self.actionText.setStyleSheet("color: rgb(255,255,255)")
        self.actionText.setAlignment(Qt.AlignCenter)
        self.actionText.setFont(fontText)

        self.offenceText = QLabel("")
        self.offenceText.setStyleSheet("color: rgb(255,255,255)")
        self.offenceText.setAlignment(Qt.AlignCenter)
        self.offenceText.setFont(fontText)

        self.prediction1Text = QLabel("")
        self.prediction1Text.setStyleSheet("color: rgb(255,255,255)")
        self.prediction1Text.setAlignment(Qt.AlignCenter)
        self.prediction1Text.setFont(fontText)

        self.prediction2Text = QLabel("")
        self.prediction2Text.setStyleSheet("color: rgb(255,255,255)")
        self.prediction2Text.setAlignment(Qt.AlignCenter)
        self.prediction2Text.setFont(fontText)

        self.prediction3Text = QLabel("")
        self.prediction3Text.setStyleSheet("color: rgb(255,255,255)")
        self.prediction3Text.setAlignment(Qt.AlignCenter)
        self.prediction3Text.setFont(fontText)

        self.prediction4Text = QLabel("")
        self.prediction4Text.setStyleSheet("color: rgb(255,255,255)")
        self.prediction4Text.setAlignment(Qt.AlignCenter)
        self.prediction4Text.setFont(fontText)

        self.predictionTitle = QLabel("VARS Prediction")
        self.predictionTitle.setFont(font)
        self.predictionTitle.setAlignment(Qt.AlignCenter)
        self.predictionTitle.setStyleSheet("color: rgb(255,255,255)")

        decisionLayout.addWidget(self.decisionTitle)
        decisionLayout.addWidget(self.spacetext)
        decisionLayout.addWidget(self.offenceText)
        decisionLayout.addWidget(self.actionText)

        decisionLayout.addWidget(self.spacetext)
        decisionLayout.addWidget(self.spacetext)
        decisionLayout.addWidget(self.predictionTitle)
        decisionLayout.addWidget(self.spacetext)
        decisionLayout.addWidget(self.prediction1Text)
        decisionLayout.addWidget(self.prediction2Text)
        decisionLayout.addWidget(self.spacetext)
        decisionLayout.addWidget(self.prediction3Text)
        decisionLayout.addWidget(self.prediction4Text)

        holder = QVBoxLayout()
        self.holdertext = QLabel("")
        holder.addWidget(self.holdertext)

        showVideoLayout = QVBoxLayout()

        self.showVid1 = QPushButton("Show video 1", self)
        self.showVid1.setFont(font2)
        self.showVid1.setStyleSheet("background:#DBDBDB;"
            "color: rgb(0,0,0)")
        self.showVid1.clicked.connect(self.enlargeV1)

        self.showVid2 = QPushButton("Show video 2", self)
        self.showVid2.setFont(font2)
        self.showVid2.setStyleSheet("background:#DBDBDB;"
            "color: rgb(0,0,0)")
        self.showVid2.clicked.connect(self.enlargeV2)

        self.showVid3 = QPushButton("Show video 3", self)
        self.showVid3.setFont(font2)
        self.showVid3.setStyleSheet("background:#DBDBDB;"
            "color: rgb(0,0,0)")
        self.showVid3.clicked.connect(self.enlargeV3)

        self.showVid4 = QPushButton("Show video 4", self)
        self.showVid4.setFont(font2)
        self.showVid4.setStyleSheet("background:#DBDBDB;"
            "color: rgb(0,0,0)")
        self.showVid4.clicked.connect(self.enlargeV4)

        self.showAllVid = QPushButton("Show all videos", self)
        self.showAllVid.setFont(font2)
        self.showAllVid.setStyleSheet("background:#DBDBDB;"
            "color: rgb(0,0,0)")
        self.showAllVid.clicked.connect(self.allVideos)

        showVideoLayout.addWidget(self.showVid1)
        showVideoLayout.addWidget(self.showVid2)
        showVideoLayout.addWidget(self.showVid3)
        showVideoLayout.addWidget(self.showVid4)
        showVideoLayout.addWidget(self.showAllVid)

        self.decisionTitle.hide()
        self.predictionTitle.hide()
        self.showVid1.hide()
        self.showVid2.hide()
        self.showVid3.hide()
        self.showVid4.hide()
        self.showAllVid.hide()

        sidebar.addLayout(decisionLayout, 0)
        sidebar.addLayout(holder, 1)
        sidebar.addLayout(showVideoLayout, 2)

        mainScreen = QGridLayout()
        mainScreen.addLayout(finalLayout, 0, 0)
        mainScreen.addLayout(controlLayout, 3, 0)
        mainScreen.addWidget(self.errorLabel, 4, 0)
        mainScreen.addLayout(sidebar, 0, 1)

        wid.setLayout(mainScreen)

        self.label = QLabel(self)
        self.label.setGeometry(QtCore.QRect(500, 0, 1000, 900))
        self.pixmap = QPixmap(path_image)
        self.label.setPixmap(self.pixmap)

        for i in self.mediaPlayers:
            i.stateChanged.connect(self.mediaStateChanged)
            i.positionChanged.connect(self.positionChanged)
            i.durationChanged.connect(self.durationChanged)
            i.error.connect(self.handleError)

        for i in self.videoWidgets:
            i.hide()

    def enlargeV1(self):
        for i in self.videoWidgets:
            i.hide()
        cou = 0
        index = 1
        for w in self.videoWidgets:
            cou += 1
            if cou > len(self.files):
                continue
            if index == cou:
                w.show()
        for m, f in zip(self.mediaPlayers, self.files):
            m.setMedia(QMediaContent(QUrl.fromLocalFile(f)))
        self.playButton.setEnabled(True)
        self.setPosition(2500)
        self.play()

    def enlargeV2(self):
        for i in self.videoWidgets:
            i.hide()
        cou = 0
        index = 2
        for w in self.videoWidgets:
            cou += 1
            if cou > len(self.files):
                continue
            if index == cou:
                w.show()
        for m, f in zip(self.mediaPlayers, self.files):
            m.setMedia(QMediaContent(QUrl.fromLocalFile(f)))
        self.playButton.setEnabled(True)
        self.setPosition(2500)
        self.play()

    def enlargeV3(self):
        for i in self.videoWidgets:
            i.hide()
        cou = 0
        index = 3
        for w in self.videoWidgets:
            cou += 1
            if cou > len(self.files):
                continue
            if index == cou:
                w.show()
        for m, f in zip(self.mediaPlayers, self.files):
            m.setMedia(QMediaContent(QUrl.fromLocalFile(f)))
        self.playButton.setEnabled(True)
        self.setPosition(2500)
        self.play()

    def enlargeV4(self):
        for i in self.videoWidgets:
            i.hide()
        cou = 0
        index = 4
        for w in self.videoWidgets:
            cou += 1
            if cou > len(self.files):
                continue
            if index == cou:
                w.show()
        for m, f in zip(self.mediaPlayers, self.files):
            m.setMedia(QMediaContent(QUrl.fromLocalFile(f)))
        self.playButton.setEnabled(True)
        self.setPosition(2500)
        self.play()

    def allVideos(self):
        cou = 0
        for w in self.videoWidgets:
            cou += 1
            if cou > len(self.files):
                continue
            w.show()
        for m, f in zip(self.mediaPlayers, self.files):
            m.setMedia(QMediaContent(QUrl.fromLocalFile(f)))
        self.playButton.setEnabled(True)
        self.setPosition(2500)
        self.play()

    def keyPressEvent(self, event):
        if event.text() == "a" and self.mediaPlayers[0].state() != QMediaPlayer.PlayingState:
            position = self.mediaPlayers[0].position()
            if position > self.frame_duration_ms:
                for i in self.mediaPlayers:
                    i.setPosition(position-self.frame_duration_ms)
                    self.setFocus()

        if event.text() == "d" and self.mediaPlayers[0].state() != QMediaPlayer.PlayingState:
            position = self.mediaPlayers[0].position()
            duration = self.mediaPlayers[0].duration()
            if position < duration - self.frame_duration_ms:
                for i in self.mediaPlayers:
                    i.setPosition(position+self.frame_duration_ms)
                    self.setFocus()

        if event.key() == Qt.Key_F1:
            position = self.mediaPlayers[0].position()
            for i in self.mediaPlayers:
                i.setPlaybackRate(1)
                i.setPosition(position)
                self.setFocus()

        if event.key() == Qt.Key_F2:
            position = self.mediaPlayers[0].position()
            for i in self.mediaPlayers:
                i.setPlaybackRate(0.5)
                i.setPosition(position)
                self.setFocus()

        if event.key() == Qt.Key_F3:
            position = self.mediaPlayers[0].position()
            for i in self.mediaPlayers:
                i.setPlaybackRate(0.3)
                i.setPosition(position)
                self.setFocus()

        if event.key() == Qt.Key_F4:
            position = self.mediaPlayers[0].position()
            for i in self.mediaPlayers:
                i.setPlaybackRate(0.25)
                i.setPosition(position)
                self.setFocus()

        if event.key() == Qt.Key_F5:
            position = self.mediaPlayers[0].position()
            for i in self.mediaPlayers:
                i.setPlaybackRate(0.2)
                i.setPosition(position)
                self.setFocus()

        if event.text() == "s":
            for i in self.mediaPlayers:
                i.setPosition(2500)
                i.play()
                i.setMuted(True)

        if event.text() == "k":
            for i in self.mediaPlayers:
                i.setPosition(3000)

        if event.text() == "o":
            self.openFile()

    def openFile(self):
        for i in self.videoWidgets:
            i.hide()

        files, _ = QFileDialog.getOpenFileNames(
            self, "Select up to 4 files", QDir.homePath())

        if len(files) != 0:
            self.files = files
            self.predictionTitle.hide()
            self.decisionTitle.hide()
            self.showVid1.hide()
            self.showVid2.hide()
            self.showVid3.hide()
            self.showVid4.hide()
            self.showAllVid.hide()
            self.label.hide()

            if self.show_prediction:
                factor = (85 - 65) / (((85 - 65) / 25) * 21)

                for num_view in range(len(files)):
                    video, _, _ = read_video(files[num_view], output_format="THWC")
                    print(video.size())
                    frames = video[65:85,:,:,:]
                    final_frames = None
                    transforms_model = MViT_V2_S_Weights.KINETICS400_V1.transforms()

                    for j in range(len(frames)):
                        if j%factor<1:
                            if final_frames is None:   # FIXED: was == None
                                final_frames = frames[j,:,:,:].unsqueeze(0)
                            else:
                                final_frames = torch.cat((final_frames, frames[j,:,:,:].unsqueeze(0)), 0)

                    final_frames = final_frames.permute(0, 3, 1, 2)
                    final_frames = transforms_model(final_frames)

                    if num_view == 0:
                        videos = final_frames.unsqueeze(0)
                    else:
                        final_frames = final_frames.unsqueeze(0)
                        videos = torch.cat((videos, final_frames), 0)

                videos = videos.unsqueeze(0)
                pred = self.model(videos)

                pred1 = pred[1]
                pred1 = pred1.unsqueeze(0)
                prediction = self.softmax(pred1)
                values, index = torch.topk(prediction, 2)

                self.prediction3Text.setText(INVERSE_EVENT_DICTIONARY_action_class[index[0][0].item()] + ": " + "{:.2f}".format(values[0][0].item()))
                self.prediction4Text.setText(INVERSE_EVENT_DICTIONARY_action_class[index[0][1].item()] + ": " + "{:.2f}".format(values[0][1].item()))

                pred1 = pred[0]
                pred1 = pred1.unsqueeze(0)
                prediction = self.softmax(pred1)
                values, index = torch.topk(prediction, 2)

                self.prediction1Text.setText(INVERSE_EVENT_DICTIONARY_offence_severity_class[index[0][0].item()] + ": " + "{:.2f}".format(values[0][0].item()))
                self.prediction2Text.setText(INVERSE_EVENT_DICTIONARY_offence_severity_class[index[0][1].item()] + ": " + "{:.2f}".format(values[0][1].item()))

                path1 = files[0].rsplit("/", 1)[0]
                val = ''
                index = ''
                for i in range(4):
                    i+=1
                    val = path1[-i]
                    if val == "_":
                        break
                    index += val

                index = index[::-1]

                path = path1.rsplit("/", 1)[0]
                print(os.path.join(path, 'annotations.json'))
                if os.path.exists(os.path.join(path, 'annotations.json')):
                    json_file = open(os.path.join(path, 'annotations.json'))
                    data_json = json.load(json_file)

                    self.actionText.setText(data_json['Actions'][index]["Action class"])

                    severity = data_json['Actions'][index]["Severity"]
                    if severity == "1.0":
                        severity_text = "+ No card"
                    elif severity == "2.0":
                        severity_text = "+ Borderline NC/YC"
                    elif severity == "3.0":
                        severity_text = "+ Yellow card"
                    elif severity == "4.0":
                        severity_text = "+ Borderline YC/RC"
                    elif severity == "5.0":
                        severity_text = "+ Red card"
                    else:
                        severity_text = ""

                    offence_severity_text = data_json['Actions'][index]["Offence"] + severity_text
                    self.offenceText.setText(offence_severity_text)

            self.label.hide()
            cou = 0
            for w in self.videoWidgets:
                cou += 1
                if cou > len(files):
                    continue
                w.show()

            self.decisionTitle.show()
            self.offenceText.show()
            self.predictionTitle.show()

            if len(files) >= 2:
                self.showVid1.show()
                self.showVid2.show()
                self.showAllVid.show()

            if len(files) >= 3:
                self.showVid3.show()

            if len(files) >= 4:
                self.showVid4.show()

            for m, f in zip(self.mediaPlayers, files):
                m.setMedia(QMediaContent(QUrl.fromLocalFile(f)))
            self.playButton.setEnabled(True)
            self.setPosition(2500)
            self.play()
        else:
            self.allVideos()

    def play(self):
        for i in self.mediaPlayers:
            if i.state() == QMediaPlayer.PlayingState:
                i.pause()
            else:
                i.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayers[0].state() == QMediaPlayer.PlayingState:
            self.playButton.setText("Pause")
        else:
            self.playButton.setText("Play")

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        for i in self.mediaPlayers:
            i.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.errorLabel.setText("Error: " + self.mediaPlayers[0].errorString())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoWindow()
    player.showMaximized()
    sys.exit(app.exec_())
