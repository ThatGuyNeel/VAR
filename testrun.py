import os, csv, traceback
import cv2
import numpy as np
from model.sportsfield_release.calculateHomography import calculateOptimHomography
from model.teamClassification.team_classification import team_classification
from offside import drawOffside

IMG_DIR   = "C:/Users/neeld/Downloads/Offside.yolov8/train/images"
LABEL_DIR = "C:/Users/neeld/Downloads/Offside.yolov8/train/labels"
OUT_CSV   = "results.csv"

# dataset class ids
CLS_BALL = 0
CLS_GK   = 1
CLS_MANC = 2
CLS_MANU = 3




def image_dims(img_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    return w, h


def parse_labels(label_path, img_w, img_h):
    """Return GT centres per class."""
    data = {CLS_MANC: [], CLS_MANU: [], CLS_GK: [], CLS_BALL: []}
    if not os.path.exists(label_path):
        return data
    with open(label_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 5:
                continue
            cls = int(p[0])
            if cls not in data:
                continue
            data[cls].append((float(p[1]) * img_w, float(p[2]) * img_h))
    return data


def mean_centroid(points):
    return np.array(points).mean(axis=0) if points else None


def match_gt_to_inferred(gt, inferred_dict):
    
    def box_centres(boxes):
        return [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in boxes]

    def dist(a, b):
        return float(np.linalg.norm(np.array(a) - np.array(b))) \
               if a is not None and b is not None else float('inf')

    cA    = mean_centroid(box_centres(inferred_dict.get("Team A", [])))
    cB    = mean_centroid(box_centres(inferred_dict.get("Team B", [])))
    cManC = mean_centroid(gt[CLS_MANC])
    cManU = mean_centroid(gt[CLS_MANU])

    # try both assignments, pick lower total distance
    if dist(cManC, cA) + dist(cManU, cB) <= dist(cManC, cB) + dist(cManU, cA):
        return {"Team A": "Man_C", "Team B": "Man_U"}
    return {"Team A": "Man_U", "Team B": "Man_C"}


def run_pipeline(img_path, inferred_dict, colors, attacking_team):
    homography = calculateOptimHomography(img_path)
    has_gk = "goalkeeper" in inferred_dict
    if attacking_team == "A":
        atk, def_ = inferred_dict["Team B"], inferred_dict["Team A"]
    else:
        atk, def_ = inferred_dict["Team A"], inferred_dict["Team B"]
    if has_gk:
        return drawOffside(img_path, attacking_team, colors, homography,
                           atk, def_, inferred_dict["goalkeeper"])
    return drawOffside(img_path, attacking_team, colors, homography, atk, def_)


# main

def main():
    images = sorted(
        [e for e in os.scandir(IMG_DIR)
         if e.is_file() and e.name.lower().endswith(('.jpg', '.jpeg', '.png'))],
        key=lambda e: e.name
    )

    fieldnames = [
        "filename",
        "gt_offside",               # ground truth from filename prefix
        "pred_offside",             # pipeline binary result
        "offside_correct",          # 1 if match
        "gt_manc_count",            # GT player counts
        "gt_manu_count",
        "gt_gk_count",
        "inferred_teamA_count",     # inferred player counts
        "inferred_teamB_count",
        "inferred_gk_count",
        "manc_mapped_to",           # which inferred team Man_C matched to
        "manu_mapped_to",
        "teamA_count_match",        # inferred count == gt count for that team
        "teamB_count_match",
        "attacking_team",           # as decided by predictTeamAttacking
        "error",
    ]

    correct = total = 0

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for entry in images:
            name       = entry.name
            gt_offside = 1 if name[:3].lower() == "yes" else 0
            row        = dict.fromkeys(fieldnames, "")
            row.update({"filename": name, "gt_offside": gt_offside})

            try:
                img_w, img_h = image_dims(entry.path)

                label_path = os.path.join(LABEL_DIR,
                                          os.path.splitext(name)[0] + ".txt")
                gt = parse_labels(label_path, img_w, img_h)

                row["gt_manc_count"] = len(gt[CLS_MANC])
                row["gt_manu_count"] = len(gt[CLS_MANU])
                row["gt_gk_count"]   = len(gt[CLS_GK])

                # team_classification also runs predictTeamAttacking internally
                # Team A is whichever team it decided was attacking
                inferred_dict, colors, _ = team_classification(entry.path)

                inf_A  = inferred_dict.get("Team A", [])
                inf_B  = inferred_dict.get("Team B", [])
                inf_gk = inferred_dict.get("goalkeeper", [])
                row["inferred_teamA_count"] = len(inf_A)
                row["inferred_teamB_count"] = len(inf_B)
                row["inferred_gk_count"]    = len(inf_gk)

                # match GT class names to inferred team labels
                mapping = match_gt_to_inferred(gt, inferred_dict)
                row["manc_mapped_to"] = "Team A" if mapping["Team A"] == "Man_C" else "Team B"
                row["manu_mapped_to"] = "Team A" if mapping["Team A"] == "Man_U" else "Team B"

                gt_count_A = len(gt[CLS_MANC if mapping["Team A"] == "Man_C" else CLS_MANU])
                gt_count_B = len(gt[CLS_MANU if mapping["Team B"] == "Man_U" else CLS_MANC])
                row["teamA_count_match"] = int(len(inf_A) == gt_count_A)
                row["teamB_count_match"] = int(len(inf_B) == gt_count_B)

                # team_classification always places the attacking team as "Team A"
                attacking_team = "A"
                row["attacking_team"] = attacking_team

                offside_result        = run_pipeline(entry.path, inferred_dict,
                                                     colors, attacking_team)
                pred                  = 0 if offside_result == 0 else 1
                row["pred_offside"]   = pred
                row["offside_correct"] = int(gt_offside == pred)

                if gt_offside == pred:
                    correct += 1
                total += 1

            except Exception:
                row["error"] = traceback.format_exc(limit=3).replace("\n", " | ")
                total += 1

            writer.writerow(row)
            status = ("OK"   if row["offside_correct"] == 1 else
                      "FAIL" if row["offside_correct"] == 0 else "ERR")
            print(f"[{status}] {name}")

    acc = correct / total * 100 if total else 0
    print(f"\nAccuracy: {correct}/{total} ({acc:.1f}%)")
    print(f"Saved → {OUT_CSV}")


if __name__ == "__main__":
    main()
