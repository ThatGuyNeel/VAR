import pandas as pd
import cv2
from ultralytics import YOLO

video_path = "clips/action_8/clip_0.mp4"
model_path = "runs/detect/train/weights/best.pt"
output_csv_path = "track/track.csv"
def track_to_csv(video_path, output_csv_path, model_path):
    # Load model
    model = YOLO(model_path)
    
    all_tracks_data = []
    frame_id = 0
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    print(f"Processing: {video_path}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Couldnt load video")
            break
            
        # Run tracking (Ultralytics handles ByteTrack automatically)
        results = model.track(
            frame, 
            persist=True,  # Maintain tracks across frames
            tracker="bytetrack.yaml",  # Built-in ByteTrack config
            verbose=False
        )
        
        # Process results for this frame
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # x1,y1,x2,y2
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
            
            for i, (bbox, track_id, conf, cls_id) in enumerate(zip(boxes, track_ids, confs, class_ids)):
                x1, y1, x2, y2 = bbox
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                class_name = model.names[int(cls_id)]
                
                track_data = {
                    'frame_id': frame_id,
                    'track_id': track_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                    'confidence': conf,
                    'class_name': class_name,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                }
                all_tracks_data.append(track_data)
        
        frame_id += 1
    
    cap.release()
    
    # Save CSV
    df = pd.DataFrame(all_tracks_data)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved {len(df)} tracks to {output_csv_path}")
    print("\nSample output:")
    print(df.head(10))
    
    return df

# Usage
df = track_to_csv(video_path,output_csv_path,model_path)
