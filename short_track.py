"""
Robust Multi-Object Tracking System
Optimized for stable tracking across short clips (4-6 seconds)
No team assignment - pure detection and tracking focus
"""

from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict, deque
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "C:/Users/neeld/SoccerNet/runs/detect/train/weights/best.pt"
VIDEO_PATH = "clips/action_8/clip_0.mp4"

# Class definitions
PLAYER_CLASS = 0
GK_CLASS = 1
REF_CLASS = 2
STAFF_CLASS = 3
BALL_CLASS = 4

CLASS_NAMES = {
    PLAYER_CLASS: "Player",
    GK_CLASS: "Goalkeeper",
    REF_CLASS: "Referee",
    STAFF_CLASS: "Staff",
    BALL_CLASS: "Ball"
}

CLASS_COLORS = {
    PLAYER_CLASS: (255, 0, 0),      # Blue
    GK_CLASS: (255, 0, 255),         # Magenta
    REF_CLASS: (0, 255, 255),        # Yellow
    STAFF_CLASS: (255, 192, 203),    # Pink
    BALL_CLASS: (255, 255, 255)      # White
}

# Tracking optimization parameters
CONFIDENCE_THRESHOLD = 0.25          # Detection confidence
IOU_THRESHOLD = 0.45                 # Lower IOU for better tracking continuity
TRACKER_CONFIG = "bytetrack.yaml"    # ByteTrack is robust for sports

# Stability features
POSITION_SMOOTHING_WINDOW = 5        # Smooth bbox positions over frames
CONFIDENCE_SMOOTHING_WINDOW = 3      # Smooth confidence scores
MIN_TRACK_LENGTH = 3                 # Minimum frames to display a track
LOST_TRACK_MEMORY = 30               # Remember lost tracks for this many frames

# Quality filtering
MIN_BBOX_AREA = 400                  # Minimum bbox area (filters noise)
MAX_BBOX_AREA = 150000               # Maximum bbox area
MIN_ASPECT_RATIO = 1.2               # Min height/width ratio
MAX_ASPECT_RATIO = 5.0               # Max height/width ratio


# ============================================================================
# TRACK STABILIZER
# ============================================================================

class TrackStabilizer:
    """Stabilizes track positions and manages track lifecycle"""
    
    def __init__(self):
        self.position_buffer = defaultdict(lambda: deque(maxlen=POSITION_SMOOTHING_WINDOW))
        self.confidence_buffer = defaultdict(lambda: deque(maxlen=CONFIDENCE_SMOOTHING_WINDOW))
        self.track_classes = {}
        self.track_first_seen = {}
        self.track_last_seen = {}
        self.track_history_length = defaultdict(int)
        self.lost_tracks = {}  # Store info about lost tracks
        
    def add_detection(self, track_id, bbox, confidence, class_id, frame_num):
        """Add new detection to track"""
        x1, y1, x2, y2 = bbox
        
        # Store class (use most recent)
        self.track_classes[track_id] = class_id
        
        # Track lifetime
        if track_id not in self.track_first_seen:
            self.track_first_seen[track_id] = frame_num
        self.track_last_seen[track_id] = frame_num
        
        # Add to buffers
        self.position_buffer[track_id].append((x1, y1, x2, y2))
        self.confidence_buffer[track_id].append(confidence)
        
        # Update history length
        self.track_history_length[track_id] += 1
        
        # Remove from lost tracks if it reappeared
        if track_id in self.lost_tracks:
            del self.lost_tracks[track_id]
    
    def get_smoothed_bbox(self, track_id):
        """Get temporally smoothed bounding box"""
        if track_id not in self.position_buffer or len(self.position_buffer[track_id]) == 0:
            return None
        
        positions = list(self.position_buffer[track_id])
        
        # Use weighted average (recent frames have more weight)
        weights = np.linspace(0.5, 1.0, len(positions))
        weights = weights / weights.sum()
        
        # Calculate weighted average for each coordinate
        avg_bbox = np.average(positions, axis=0, weights=weights)
        
        return tuple(map(int, avg_bbox))
    
    def get_smoothed_confidence(self, track_id):
        """Get smoothed confidence score"""
        if track_id not in self.confidence_buffer or len(self.confidence_buffer[track_id]) == 0:
            return 0.0
        
        return np.mean(list(self.confidence_buffer[track_id]))
    
    def is_stable_track(self, track_id):
        """Check if track is stable enough to display"""
        return self.track_history_length[track_id] >= MIN_TRACK_LENGTH
    
    def get_track_age(self, track_id, current_frame):
        """Get track age in frames"""
        if track_id in self.track_first_seen:
            return current_frame - self.track_first_seen[track_id]
        return 0
    
    def mark_lost_track(self, track_id, frame_num):
        """Mark a track as lost"""
        if track_id in self.track_classes:
            self.lost_tracks[track_id] = {
                'class': self.track_classes[track_id],
                'last_seen': frame_num,
                'last_bbox': self.get_smoothed_bbox(track_id)
            }
    
    def cleanup_old_tracks(self, current_frame):
        """Remove old lost tracks from memory"""
        to_remove = []
        for track_id, info in self.lost_tracks.items():
            if current_frame - info['last_seen'] > LOST_TRACK_MEMORY:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.lost_tracks[track_id]


# ============================================================================
# DETECTION QUALITY FILTER
# ============================================================================

class DetectionFilter:
    """Filter out low-quality detections"""
    
    @staticmethod
    def is_valid_detection(bbox, class_id):
        """Check if detection meets quality criteria"""
        x1, y1, x2, y2 = bbox
        
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # Size filtering
        if area < MIN_BBOX_AREA or area > MAX_BBOX_AREA:
            return False
        
        # Aspect ratio filtering (not applicable for ball)
        if class_id != BALL_CLASS:
            if width <= 0 or height <= 0:
                return False
            
            aspect_ratio = height / width
            
            if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
                return False
        
        return True
    
    @staticmethod
    def is_bbox_at_border(bbox, frame_shape, margin=5):
        """Check if bbox is at image border (likely truncated)"""
        x1, y1, x2, y2 = bbox
        h, w = frame_shape[:2]
        
        if x1 <= margin or y1 <= margin or x2 >= w - margin or y2 >= h - margin:
            return True
        return False


# ============================================================================
# VISUALIZATION
# ============================================================================

class Visualizer:
    """Handle all visualization"""
    
    @staticmethod
    def draw_detection(frame, bbox, track_id, class_id, confidence, is_stable=True):
        """Draw bounding box with label"""
        x1, y1, x2, y2 = bbox
        
        # Get color based on class
        color = CLASS_COLORS.get(class_id, (200, 200, 200))
        
        # Dim color for unstable tracks
        if not is_stable:
            color = tuple(int(c * 0.5) for c in color)
        
        # Draw bounding box
        thickness = 3 if is_stable else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label
        class_name = CLASS_NAMES.get(class_id, "Unknown")
        label = f"{class_name} {track_id}"
        
        # Add confidence if low
        if confidence < 0.5:
            label += f" ({confidence:.2f})"
        
        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Draw label background
        cv2.rectangle(
            frame,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w + 4, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - baseline - 2),
            font,
            font_scale,
            (0, 0, 0),  # Black text
            font_thickness
        )
        
        # Draw stability indicator (small circle)
        if is_stable:
            cv2.circle(frame, (x1 + 8, y1 + 8), 4, (255, 255, 255), -1)
    
    @staticmethod
    def draw_stats(frame, stats, frame_num):
        """Draw statistics overlay"""
        # Background panel
        panel_height = 120
        cv2.rectangle(frame, (0, 0), (400, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (400, panel_height), (255, 255, 255), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Frame number
        cv2.putText(frame, f"Frame: {frame_num}", (10, 25), font, 0.7, (255, 255, 255), 2)
        
        # Active tracks
        cv2.putText(frame, f"Active Tracks: {stats['active_tracks']}", (10, 50), font, 0.6, (0, 255, 0), 2)
        
        # Total tracks
        cv2.putText(frame, f"Total Tracks: {stats['total_tracks']}", (10, 75), font, 0.6, (255, 255, 255), 2)
        
        # Class breakdown
        class_info = " | ".join([f"{CLASS_NAMES.get(k, k)}: {v}" for k, v in stats['class_counts'].items()])
        cv2.putText(frame, class_info, (10, 100), font, 0.5, (200, 200, 200), 1)


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    # Initialize
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {VIDEO_PATH}")
        return
    
    # Dynamic output path
    input_path = Path(VIDEO_PATH)
    output_filename = f"tracked_{input_path.stem}{input_path.suffix}"
    output_path = input_path.parent / output_filename
    OUTPUT_PATH = str(output_path)
    
    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n{'='*70}")
    print(f"ROBUST MULTI-OBJECT TRACKING SYSTEM")
    print(f"{'='*70}")
    print(f"Video: {input_path.name}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total Frames: {total_frames}")
    print(f"Duration: {total_frames/fps:.2f}s")
    print(f"{'='*70}\n")
    
    # Initialize components
    stabilizer = TrackStabilizer()
    detector_filter = DetectionFilter()
    visualizer = Visualizer()
    
    # Video writer (uncomment to save)
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    
    frame_count = 0
    active_tracks_current_frame = set()
    
    print("Processing video...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        active_tracks_current_frame.clear()
        
        # Run tracking
        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER_CONFIG,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            classes=[PLAYER_CLASS, GK_CLASS, REF_CLASS, STAFF_CLASS, BALL_CLASS],
            verbose=False
        )
        
        # Process detections
        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue
            
            for box, track_id, class_id, conf in zip(
                r.boxes.xyxy, r.boxes.id, r.boxes.cls, r.boxes.conf
            ):
                bbox = tuple(map(int, box.cpu().numpy()))
                track_id = int(track_id)
                class_id = int(class_id)
                confidence = float(conf)
                
                # Quality filtering
                if not detector_filter.is_valid_detection(bbox, class_id):
                    continue
                
                # Add to stabilizer
                stabilizer.add_detection(track_id, bbox, confidence, class_id, frame_count)
                active_tracks_current_frame.add(track_id)
        
        # Mark lost tracks
        all_known_tracks = set(stabilizer.track_classes.keys())
        lost_this_frame = all_known_tracks - active_tracks_current_frame
        for track_id in lost_this_frame:
            if track_id not in stabilizer.lost_tracks:
                stabilizer.mark_lost_track(track_id, frame_count)
        
        # Cleanup old lost tracks
        stabilizer.cleanup_old_tracks(frame_count)
        
        # Visualization
        class_counts = defaultdict(int)
        
        for track_id in active_tracks_current_frame:
            # Get smoothed bbox
            smooth_bbox = stabilizer.get_smoothed_bbox(track_id)
            if smooth_bbox is None:
                continue
            
            # Get class and confidence
            class_id = stabilizer.track_classes[track_id]
            confidence = stabilizer.get_smoothed_confidence(track_id)
            
            # Check if stable
            is_stable = stabilizer.is_stable_track(track_id)
            
            # Draw
            visualizer.draw_detection(
                frame, smooth_bbox, track_id, class_id, confidence, is_stable
            )
            
            class_counts[class_id] += 1
        
        # Draw stats
        stats = {
            'active_tracks': len(active_tracks_current_frame),
            'total_tracks': len(stabilizer.track_classes),
            'class_counts': class_counts
        }
        visualizer.draw_stats(frame, stats, frame_count)
        
        # Write frame
        # out.write(frame)
        
        # Display
        cv2.imshow("Robust Tracking", frame)
        
        # Progress
        if frame_count % 10 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})", end='\r')
        
        # Exit on ESC
        if cv2.waitKey(100) & 0xFF == 27:
            print("\n\nStopped by user")
            break
    
    # Cleanup
    cap.release()
    # out.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    print(f"\n\n{'='*70}")
    print(f"TRACKING COMPLETE")
    print(f"{'='*70}")
    print(f"Frames Processed: {frame_count}/{total_frames}")
    print(f"Total Unique Tracks: {len(stabilizer.track_classes)}")
    print(f"\nTracks by Class:")
    
    class_track_counts = defaultdict(int)
    for track_id, class_id in stabilizer.track_classes.items():
        class_track_counts[class_id] += 1
    
    for class_id, count in sorted(class_track_counts.items()):
        class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
        print(f"  {class_name}: {count} tracks")
    
    print(f"\nOutput would be saved to: {OUTPUT_PATH}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()