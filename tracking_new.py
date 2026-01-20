from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from pathlib import Path


MODEL_PATH = "C:/Users/neeld/SoccerNet/runs/detect/train3/weights/best.pt"
VIDEO_PATH = "clips/action_8/clip_0.mp4"

PLAYER_CLASS = 0
GK_CLASS = 1
REF_CLASS = 2
STAFF_CLASS = 3
BALL_CLASS = 4

MIN_FRAMES_FOR_CLUSTERING = 5
TEAM_CONFIDENCE_THRESHOLD = 3  #consecutive frames needed to confirm team
COLOUR_HISTORY_SIZE = 10  #keep last N colour samples

#bounding box size constraints 
# MIN_BOX_AREA = 800      #minimum area (width*height) - filters out tiny false detections
# MAX_BOX_AREA = 150000   #maximum area - filters out very large false detections
# MIN_BOX_WIDTH = 20      #minimum width
# MIN_BOX_HEIGHT = 40     #minimum height (people are taller than wide)
# MAX_BOX_WIDTH = 400     #maximum width
# MAX_BOX_HEIGHT = 600    #maximum height


model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

# Buffers
track_colours = defaultdict(list)
track_team_votes = defaultdict(list)  #track team assignment history
track_team_final = {}  #final confirmed team assignment
global_team_colours = {0: [], 1: []}  #dtore representative colours for each team
track_last_seen = {}  #track last frame each ID was seen
track_jersey_signature = {}  #store stable jersey colour signature per ID


def extract_dominant_jersey_colour(frame, x1, y1, x2, y2):
    h = y2 - y1
    #focus on upper 50% of bounding box for jersey
    jersey = frame[y1:y1 + int(0.5 * h), x1:x2]

    if jersey.size == 0:
        return None

    hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)

    #filter: remove very dark, very bright, and low saturation pixels
    mask = (
        (pixels[:, 1] > 30) &  #saturation>30
        (pixels[:, 2] > 30) &  #value>30
        (pixels[:, 2] < 240)   #value<240 (avoiding white/bright)
    )
    pixels = pixels[mask]

    if len(pixels) < 50:
        return None

    #suppress grass
    is_grass = (
        (pixels[:, 0] >= 35) & (pixels[:, 0] <= 85) &  #green hue range
        (pixels[:, 1] > 70) &  #high saturation - purity
        (pixels[:, 2] > 80) & (pixels[:, 2] < 180)  #medium high brightness range
    )
    pixels = pixels[~is_grass]

    if len(pixels) < 30:
        return None

    #use more clusters for better colour extraction
    n_clusters = min(5, max(2, len(pixels) // 100))
    kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    labels = kmeans.fit_predict(pixels)

    #find dominant cluster
    unique, counts = np.unique(labels, return_counts=True)
    dominant_label = unique[np.argmax(counts)]
    dominant = kmeans.cluster_centers_[dominant_label]

    #return normalized HSV (H,S only - ignore brightness)
    return np.array([
        dominant[0] / 180.0,
        dominant[1] / 255.0
    ])


def colour_distance(c1, c2):
    #circular distance for hue
    h_diff = min(abs(c1[0] - c2[0]), 1 - abs(c1[0] - c2[0]))
    s_diff = abs(c1[1] - c2[1])
    return np.sqrt(h_diff**2 + s_diff**2)


def assign_team_by_similarity(jersey_colour, global_team_colours):
    if not global_team_colours[0] and not global_team_colours[1]:
        return None
    
    dist_to_team0 = float('inf')
    dist_to_team1 = float('inf')
    
    if global_team_colours[0]:
        avg_colour_0 = np.mean(global_team_colours[0], axis=0)
        dist_to_team0 = colour_distance(jersey_colour, avg_colour_0)
    
    if global_team_colours[1]:
        avg_colour_1 = np.mean(global_team_colours[1], axis=0)
        dist_to_team1 = colour_distance(jersey_colour, avg_colour_1)
    
    #if only one team has ucolours, assign to the other
    if not global_team_colours[0]:
        return 1
    if not global_team_colours[1]:
        return 0
    
    #assign to closer team
    return 0 if dist_to_team0 < dist_to_team1 else 1


def cluster_initial_teams(track_colours):
    track_ids = list(track_colours.keys())
    if len(track_ids) < 2:
        return {}
    
    #get average colour for each track
    avg_colours = []
    for tid in track_ids:
        avg_colours.append(np.mean(track_colours[tid], axis=0))
    
    #cluster into 2 teams
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = kmeans.fit_predict(np.array(avg_colours))
    
    assignments = {}
    for tid, label in zip(track_ids, labels):
        assignments[tid] = int(label)
    
    return assignments


#dynamic output path
input_path = Path(VIDEO_PATH)
output_filename = f"tracked_{input_path.stem}{input_path.suffix}"
output_path = input_path.parent / output_filename
OUTPUT_PATH = str(output_path)

#get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

#initialize video writer once outside loop
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print(f"Processing video: {VIDEO_PATH}")
print(f"Output will be saved to: {OUTPUT_PATH}")

frame_count = 0
initial_clustering_done = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack_early.yaml",
        conf=0.25,
        iou=0.5,
        classes=[PLAYER_CLASS, GK_CLASS, REF_CLASS, STAFF_CLASS, BALL_CLASS]
    )

    for r in results:
        if r.boxes.id is None:
            continue

        for box, tid, cls in zip(r.boxes.xyxy, r.boxes.id, r.boxes.cls):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            tid = int(tid)
            cls = int(cls)
            
            # #calculate bounding box dimensions
            # box_width = x2 - x1
            # box_height = y2 - y1
            # box_area = box_width * box_height
            
            # #filter based on bounding box size constraints
            # if (box_area < MIN_BOX_AREA or box_area > MAX_BOX_AREA or
            #     box_width < MIN_BOX_WIDTH or box_width > MAX_BOX_WIDTH or
            #     box_height < MIN_BOX_HEIGHT or box_height > MAX_BOX_HEIGHT):
            #     continue  # Skip this detection

            #extract jersey colour for players and goalkeepers
            if cls in [PLAYER_CLASS, GK_CLASS]:
                jersey_feat = extract_dominant_jersey_colour(frame, x1, y1, x2, y2)
                if jersey_feat is not None:
                    track_colours[tid].append(jersey_feat)
                    #keeping only recent colours
                    if len(track_colours[tid]) > COLOUR_HISTORY_SIZE:
                        track_colours[tid].pop(0)
                    
                    #creating stable jersey signature once confident
                    if tid not in track_jersey_signature and len(track_colours[tid]) >= MIN_FRAMES_FOR_CLUSTERING:
                        track_jersey_signature[tid] = np.mean(track_colours[tid], axis=0)
            
            #update last seen frame
            track_last_seen[tid] = frame_count

            #determine colour and label
            colour = (200, 200, 200)
            label = "UNKNOWN"

            if cls == PLAYER_CLASS:
                label = "Player"
            elif cls == GK_CLASS:
                label = "GK"
            elif cls == REF_CLASS:
                label = "Ref"
                colour = (0, 255, 255)  #yellow for ref
            elif cls == STAFF_CLASS:
                label = "Staff"
                colour = (255, 192, 203)  #pink for staff

            #apply team colour if assigned
            if tid in track_team_final:
                team = track_team_final[tid]
                colour = (255, 0, 0) if team == 0 else (0, 0, 255)
                label = f"Team{team}"

            #draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(
                frame,
                f"{label}-{tid}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                colour,
                2
            )

    #initial clustering: wait until we have enough data
    if not initial_clustering_done:
        ready_tracks = [
            tid for tid in track_colours
            if len(track_colours[tid]) >= MIN_FRAMES_FOR_CLUSTERING
        ]
        
        if len(ready_tracks) >= 4:  #at least 4 players
            assignments = cluster_initial_teams(
                {tid: track_colours[tid] for tid in ready_tracks}
            )
            
            #initialize team colours and assignments
            for tid, team in assignments.items():
                track_team_final[tid] = team
                avg_colour = np.mean(track_colours[tid], axis=0)
                global_team_colours[team].append(avg_colour)
            
            initial_clustering_done = True
            print(f"Initial team clustering done at frame {frame_count}")

    #continuous assignment for new players
    elif initial_clustering_done:
        for tid in track_colours:
            if tid not in track_team_final and len(track_colours[tid]) >= MIN_FRAMES_FOR_CLUSTERING:
                avg_colour = np.mean(track_colours[tid], axis=0)
                team = assign_team_by_similarity(avg_colour, global_team_colours)
                
                if team is not None:
                    #using voting mechanism for stability
                    track_team_votes[tid].append(team)
                    
                    if len(track_team_votes[tid]) >= TEAM_CONFIDENCE_THRESHOLD:
                        #only confirm team if votes are consistent
                        recent_votes = track_team_votes[tid][-TEAM_CONFIDENCE_THRESHOLD:]
                        if len(set(recent_votes)) == 1:  #all votes agree
                            track_team_final[tid] = team
                            global_team_colours[team].append(avg_colour)
                            #limit team colour history
                            if len(global_team_colours[team]) > 20:
                                global_team_colours[team].pop(0)

    #write frame
    # out.write(frame)
    
    #display
    cv2.imshow("Player Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

#cleanup
cap.release()
# out.release()
cv2.destroyAllWindows()

print(f"\nProcessing complete!")
print(f"Output saved to: {OUTPUT_PATH}")
print(f"Total frames processed: {frame_count}")
print(f"Players tracked: {len(track_team_final)}")
print(f"Unique IDs seen: {len(track_last_seen)}")