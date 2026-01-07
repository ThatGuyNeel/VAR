from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans


MODEL_PATH = "C:/Users/neeld/SoccerNet/runs/detect/train/weights/best.pt"
VIDEO_PATH = "clips/action_8/clip_0.mp4"

PLAYER_CLASS = 0
GK_CLASS = 1
REF_CLASS = 2
STAFF_CLASS = 3
BALL_CLASS = 4

EARLY_FRAMES = 5   # frames needed before soft clustering

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

# Buffers
track_colors = defaultdict(list)   
track_team_soft = {}               


def extract_dominant_jersey_color(frame, x1, y1, x2, y2):
    h = y2 - y1
    jersey = frame[y1 : y1 + int(0.45 * h), x1:x2]

    if jersey.size == 0:
        return None

    hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)

    # Remove dark / unsaturated pixels
    pixels = pixels[
        (pixels[:,1] > 40) &
        (pixels[:,2] > 40)
    ]

    if len(pixels) < 50:
        return None

    # Supressing grass
    is_grass = (
        (pixels[:,0] > 35) & (pixels[:,0] < 70) &
        (pixels[:,1] > 60) &
        (pixels[:,2] > 60)
    )
    pixels = pixels[~is_grass]

    if len(pixels) < 30:
        return None

    # Dominant colour via clustering
    kmeans = KMeans(n_clusters=3, n_init=3, random_state=0)
    labels = kmeans.fit_predict(pixels)

    unique, counts = np.unique(labels, return_counts=True)
    dominant_label = unique[np.argmax(counts)]
    dominant = kmeans.cluster_centers_[dominant_label]

    return np.array([
        dominant[0] / 180.0,   # Hue (normalized)
        dominant[1] / 255.0    # Saturation (normalized)
    ])

def quick_cluster(features):
    kmeans = KMeans(n_clusters=2, n_init=5, random_state=0)
    return kmeans.fit_predict(features)
# ---------------------------------------- #

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack_early.yaml",
        conf=0.05,   
        iou=0.5,
        classes=[PLAYER_CLASS, GK_CLASS, REF_CLASS, STAFF_CLASS, BALL_CLASS]
    )

    for r in results:
        if r.boxes.id is None:
            continue

        for box, tid, cls in zip(
            r.boxes.xyxy,
            r.boxes.id,
            r.boxes.cls
        ):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            tid = int(tid)
            cls = int(cls)

            # Jersey features
            if cls in [PLAYER_CLASS, GK_CLASS] and tid not in track_team_soft:
                jersey_feat = extract_dominant_jersey_color(frame, x1, y1, x2, y2)
                if jersey_feat is not None:
                    track_colors[tid].append(jersey_feat)

            # Drawing bounding boxes
            color = (200, 200, 200)
            label = "UNKNOWN"

            if cls == PLAYER_CLASS:
                label = "Player"
            elif cls == GK_CLASS:
                label = "GK"
            elif cls == REF_CLASS:
                label = "Ref"
                color = (0, 255, 255)
            elif cls == STAFF_CLASS:
                label = "Staff"
                color = (255, 192, 203)
            elif cls == BALL_CLASS:
                label = "Ball"
                color = (0, 255, 0)

            # Treating soft team as tthe final team
            if tid in track_team_soft:
                color = (255, 0, 0) if track_team_soft[tid] == 0 else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label}-{tid}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    ready = [
        tid for tid in track_colors
        if tid not in track_team_soft and len(track_colors[tid]) >= EARLY_FRAMES
    ]

    if len(ready) >= 2:
        feats = [np.mean(track_colors[tid], axis=0) for tid in ready]
        labels = quick_cluster(np.array(feats))

        for tid, team in zip(ready, labels):
            track_team_soft[tid] = int(team)

    cv2.imshow("Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.imwrite('screenshot.png', frame)
        print("Screenshot saved as screenshot.png")
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
