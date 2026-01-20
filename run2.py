import os
import json
import shutil
import zipfile
import random
from tqdm import tqdm

#config
DATASET_ROOT = "C:/Users/neeld/SoccerNet/data"
OUTPUT_DIR = "C:/Users/neeld/SoccerNet/split"

TRAIN_MATCHES = 300
VAL_MATCHES = 60
TEST_MATCHES = 40
RANDOM_SEED = 42

CLASS_MAP = {
    "player": 0,
    "goalkeeper": 1,
    "referee": 2,
    "staff": 3,
    "ball": 4
}

IMG_W, IMG_H = 1920, 1080

def get_class_id_from_label(label):
    label = label.lower()
    if "goalkeeper" in label:
        return 1
    if "player" in label:
        return 0
    if "referee" in label:
        return 2
    if "staff" in label:
        return 3
    if "ball" in label:
        return 4
    return None


def convert_points_to_yolo(points):
    x1, y1 = points["x1"], points["y1"]
    x2, y2 = points["x2"], points["y2"]

    w = x2 - x1
    h = y2 - y1
    xc = x1 + w / 2
    yc = y1 + h / 2

    return (
        xc / IMG_W,
        yc / IMG_H,
        w / IMG_W,
        h / IMG_H
    )


def extract_frames(match_path):
    zip_path = os.path.join(match_path, "Frames-v3.zip")
    frames_dir = os.path.join(match_path, "Frames")

    if not os.path.exists(zip_path):
        return None

    if not os.path.isdir(frames_dir):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(frames_dir)

    return frames_dir


#main
def process_soccernet_v3():

    #output dirs
    train_img = os.path.join(OUTPUT_DIR, "train/images")
    train_lbl = os.path.join(OUTPUT_DIR, "train/labels")
    val_img = os.path.join(OUTPUT_DIR, "val/images")
    val_lbl = os.path.join(OUTPUT_DIR, "val/labels")
    test_img = os.path.join(OUTPUT_DIR, "test/images")
    test_lbl = os.path.join(OUTPUT_DIR, "test/labels")


    for d in [train_img, train_lbl, val_img, val_lbl,test_img,test_lbl]:
        os.makedirs(d, exist_ok=True)

    #all matches
    all_matches = []  # (league, season, match, match_path)

    for league in os.listdir(DATASET_ROOT):
        league_path = os.path.join(DATASET_ROOT, league)
        if not os.path.isdir(league_path):
            continue

        for season in os.listdir(league_path):
            season_path = os.path.join(league_path, season)
            if not os.path.isdir(season_path):
                continue

            for match in os.listdir(season_path):
                match_path = os.path.join(season_path, match)
                if not os.path.isdir(match_path):
                    continue

                if os.path.exists(os.path.join(match_path, "Labels-v3.json")):
                    all_matches.append((league, season, match, match_path))

    assert len(all_matches) >= TRAIN_MATCHES + VAL_MATCHES + TEST_MATCHES, \
        f"Not enough matches found: {len(all_matches)}"

    #split
    random.seed(RANDOM_SEED)
    random.shuffle(all_matches)

    train_set = set(all_matches[:TRAIN_MATCHES])
    val_set = set(all_matches[TRAIN_MATCHES:TRAIN_MATCHES + VAL_MATCHES])
    test_set = set(all_matches[TRAIN_MATCHES + VAL_MATCHES : TRAIN_MATCHES + VAL_MATCHES + TEST_MATCHES])

    print(f"Train matches: {len(train_set)}")
    print(f"Val matches:   {len(val_set)}")
    print(f"Test matches: {len(test_set)}")

    #process matches
    for league, season, match, match_path in tqdm(all_matches, desc="Processing matches"):

        if (league, season, match, match_path) in train_set:
            img_out, lbl_out = train_img, train_lbl
        elif (league, season, match, match_path) in val_set:
            img_out, lbl_out = val_img, val_lbl
        elif (league, season, match, match_path) in test_set:
            img_out, lbl_out = test_img, test_lbl
        else:
            continue  

        with open(os.path.join(match_path, "Labels-v3.json"), "r") as f:
            data = json.load(f)

        metadata = data.get("GameMetadata", {})
        list_replays = metadata.get("list_replays",[])
        list_actions = metadata.get("list_actions", [])
        actions = data.get("actions", {})
        replays = data.get("replays",{})

        image_dir = extract_frames(match_path)
        if image_dir is None:
            continue

        for img_name, frame_data in actions.items():

            src_img = os.path.join(image_dir, img_name)
            if not os.path.exists(src_img):
                continue

            yolo_lines = []

            for obj in frame_data.get("bboxes", []):
                cls_id = get_class_id_from_label(
                    obj.get("label", obj.get("class", ""))
                )
                if cls_id is None:
                    continue

                points = obj.get("points")
                if points is None:
                    continue

                bbox = convert_points_to_yolo(points)
                yolo_lines.append(
                    f"{cls_id} " + " ".join(f"{v:.6f}" for v in bbox)
                )

            if not yolo_lines:
                continue

            base = f"{league}_{season}_{match}_{img_name}".replace(" ", "_")

            with open(os.path.join(lbl_out, base + ".txt"), "w") as f:
                f.write("\n".join(yolo_lines))

            shutil.copy(
                src_img,
                os.path.join(img_out, base + ".jpg")
            )
        for img_name, frame_data in replays.items():

            src_img = os.path.join(image_dir, img_name)
            if not os.path.exists(src_img):
                continue

            yolo_lines = []

            for obj in frame_data.get("bboxes", []):
                cls_id = get_class_id_from_label(
                    obj.get("label", obj.get("class", ""))
                )
                if cls_id is None:
                    continue

                points = obj.get("points")
                if points is None:
                    continue

                bbox = convert_points_to_yolo(points)
                yolo_lines.append(
                    f"{cls_id} " + " ".join(f"{v:.6f}" for v in bbox)
                )

            if not yolo_lines:
                continue

            base = f"{league}_{season}_{match}_{img_name}".replace(" ", "_")

            with open(os.path.join(lbl_out, base + ".txt"), "w") as f:
                f.write("\n".join(yolo_lines))

            shutil.copy(
                src_img,
                os.path.join(img_out, base + ".jpg")
            )

#main entry
if __name__ == "__main__":
    process_soccernet_v3()
