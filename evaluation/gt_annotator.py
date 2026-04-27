"""
Ground Truth Annotation Tool
Interactive tool for creating ground truth annotations for MOT evaluation
"""

import cv2
import json
import numpy as np
from pathlib import Path
from collections import defaultdict


class GroundTruthAnnotator:
    """Interactive annotation tool for creating ground truth"""
    
    def __init__(self, video_path: str, output_path: str = None):
        self.video_path = video_path
        self.output_path = output_path or video_path.replace('.mp4', '_gt.json')
        
        self.cap = cv2.VideoCapture(video_path)
        self.current_frame = 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Annotations: {frame_id: {track_id: [x1, y1, x2, y2, team_id]}}
        self.annotations = defaultdict(dict)
        
        # Current annotation state
        self.current_track_id = 1
        self.current_team_id = 0  # 0 or 1
        self.drawing = False
        self.start_point = None
        self.temp_bbox = None
        
        # Display
        self.frame = None
        self.display_frame = None
        
        print("="*70)
        print("GROUND TRUTH ANNOTATION TOOL")
        print("="*70)
        print("\nControls:")
        print("  Left Click + Drag: Draw bounding box")
        print("  't': Change team (0 or 1)")
        print("  'n': Next track ID")
        print("  'd': Delete last annotation in current frame")
        print("  'Right Arrow': Next frame")
        print("  'Left Arrow': Previous frame")
        print("  's': Save annotations")
        print("  'q': Quit (will prompt to save)")
        print("  'h': Show help")
        print("="*70)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bboxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.temp_bbox = (self.start_point[0], self.start_point[1], x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point:
                x1, y1 = self.start_point
                x2, y2 = x, y
                
                # Ensure x1 < x2 and y1 < y2
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Add annotation
                self.annotations[self.current_frame][self.current_track_id] = [
                    x1, y1, x2, y2, self.current_team_id
                ]
                
                print(f"✓ Added Track {self.current_track_id} "
                      f"(Team {self.current_team_id}) at frame {self.current_frame}")
                
                self.start_point = None
                self.temp_bbox = None
    
    def draw_annotations(self):
        """Draw existing annotations on frame"""
        self.display_frame = self.frame.copy()
        
        # Draw existing annotations for current frame
        if self.current_frame in self.annotations:
            for track_id, bbox_team in self.annotations[self.current_frame].items():
                x1, y1, x2, y2, team_id = bbox_team
                
                # Color based on team
                color = (0, 255, 0) if team_id == 0 else (255, 0, 0)
                
                # Draw box
                cv2.rectangle(self.display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Label
                label = f"ID:{track_id} T:{team_id}"
                cv2.putText(self.display_frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw temporary bbox while drawing
        if self.temp_bbox:
            x1, y1, x2, y2 = self.temp_bbox
            color = (0, 255, 255)  # Yellow for temp
            cv2.rectangle(self.display_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw info text
        info = f"Frame: {self.current_frame}/{self.total_frames} | " \
               f"Track ID: {self.current_track_id} | Team: {self.current_team_id}"
        cv2.putText(self.display_frame, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw annotations count
        count_info = f"Annotations: {len(self.annotations[self.current_frame])}"
        cv2.putText(self.display_frame, count_info, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def load_frame(self):
        """Load current frame"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, self.frame = self.cap.read()
        
        if not ret:
            return False
        
        # Resize for easier annotation
        self.frame = cv2.resize(self.frame, (1280, 720))
        return True
    
    def save_annotations(self):
        """Save annotations to JSON file"""
        # Convert to serializable format
        output = {}
        for frame_id, detections in self.annotations.items():
            output[f"frame_{frame_id}"] = {
                str(track_id): bbox_team 
                for track_id, bbox_team in detections.items()
            }
        
        with open(self.output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✅ Annotations saved to: {self.output_path}")
        print(f"   Total frames annotated: {len(self.annotations)}")
        print(f"   Total detections: {sum(len(d) for d in self.annotations.values())}")
    
    def run(self):
        """Run annotation tool"""
        cv2.namedWindow('Ground Truth Annotation')
        cv2.setMouseCallback('Ground Truth Annotation', self.mouse_callback)
        
        if not self.load_frame():
            print("Error: Cannot read video")
            return
        
        while True:
            self.draw_annotations()
            cv2.imshow('Ground Truth Annotation', self.display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Quit
            if key == ord('q'):
                response = input("\nSave annotations before quitting? (y/n): ")
                if response.lower() == 'y':
                    self.save_annotations()
                break
            
            # Save
            elif key == ord('s'):
                self.save_annotations()
            
            # Next frame
            elif key == 83 or key == ord('d'):  # Right arrow or 'd'
                if self.current_frame < self.total_frames - 1:
                    self.current_frame += 1
                    self.load_frame()
            
            # Previous frame
            elif key == 81 or key == ord('a'):  # Left arrow or 'a'
                if self.current_frame > 0:
                    self.current_frame -= 1
                    self.load_frame()
            
            # Change team
            elif key == ord('t'):
                self.current_team_id = 1 - self.current_team_id
                print(f"→ Team changed to: {self.current_team_id}")
            
            # Next track ID
            elif key == ord('n'):
                self.current_track_id += 1
                print(f"→ Track ID changed to: {self.current_track_id}")
            
            # Delete last annotation
            elif key == ord('x'):
                if self.annotations[self.current_frame]:
                    last_id = max(self.annotations[self.current_frame].keys())
                    del self.annotations[self.current_frame][last_id]
                    print(f"✗ Deleted Track {last_id}")
            
            # Help
            elif key == ord('h'):
                print("\n" + "="*70)
                print("CONTROLS:")
                print("  Left Click + Drag: Draw bounding box")
                print("  't': Change team (0 or 1)")
                print("  'n': Next track ID")
                print("  'x': Delete last annotation in current frame")
                print("  'Right Arrow' or 'd': Next frame")
                print("  'Left Arrow' or 'a': Previous frame")
                print("  's': Save annotations")
                print("  'q': Quit")
                print("="*70 + "\n")
        
        self.cap.release()
        cv2.destroyAllWindows()


def convert_tracking_output_to_gt_format(tracking_results: dict, output_file: str):
    """
    Convert tracking system output to ground truth format
    
    Args:
        tracking_results: {frame_id: {track_id: {'bbox': [x1,y1,x2,y2], 'team': team_id}}}
        output_file: Path to save converted data
    """
    output = {}
    
    for frame_id, detections in tracking_results.items():
        output[f"frame_{frame_id}"] = {}
        
        for track_id, data in detections.items():
            bbox = data['bbox']
            team = data.get('team', -1)
            output[f"frame_{frame_id}"][str(track_id)] = bbox + [team]
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✅ Converted tracking results saved to: {output_file}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        annotator = GroundTruthAnnotator(video_path, output_path)
        annotator.run()
    else:
        print("\nUsage:")
        print("  python gt_annotator.py <video_path> [output_path]")
        print("\nExample:")
        print("  python gt_annotator.py clips/action_8/clip_0.mp4 ground_truth.json")