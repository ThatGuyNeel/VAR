"""
Tracking Evaluation Integration
Runs production tracking system and evaluates against ground truth
Modified to work with image sequences and MOT format (detection + tracking only)
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
import cv2
from typing import Union, List
import re

PLAYER_CLASS = 0
def get_image_sequence(images_path: Union[str, Path], pattern: str = None) -> List[Path]:
    """
    Get sorted list of image files from directory
    
    Args:
        images_path: Path to directory containing frames
        pattern: Optional regex pattern to filter files (e.g., r'frame_\d+\.jpg')
    
    Returns:
        Sorted list of image paths
    """
    images_path = Path(images_path)
    
    if not images_path.is_dir():
        raise ValueError(f"Path is not a directory: {images_path}")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        f for f in images_path.iterdir() 
        if f.suffix.lower() in image_extensions
    ]
    
    # Apply pattern filter if provided
    if pattern:
        regex = re.compile(pattern)
        image_files = [f for f in image_files if regex.match(f.name)]
    
    # Sort by filename (assumes numeric ordering)
    # Try to extract numbers for proper sorting
    def extract_number(filepath):
        numbers = re.findall(r'\d+', filepath.stem)
        return int(numbers[0]) if numbers else filepath.stem
    
    image_files.sort(key=extract_number)
    
    if not image_files:
        raise ValueError(f"No images found in {images_path}")
    
    print(f"Found {len(image_files)} images")
    print(f"  First: {image_files[0].name}")
    print(f"  Last:  {image_files[-1].name}")
    
    return image_files


def load_mot_ground_truth(gt_file: str) -> dict:
    """
    Load ground truth from MOT format file
    
    MOT Format: frame_id, track_id, x, y, w, h, conf, class, visibility, ...
    Example: 1,1,136,520,51,135,1,-1,-1,-1
    
    Args:
        gt_file: Path to gt.txt file
    
    Returns:
        Dictionary: {frame_id: {track_id: [x, y, w, h]}}
    """
    gt_data = defaultdict(dict)
    
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])

            # if w * h <300:
            #     continue
            
            # Store as [x, y, w, h] (MOT format)
            gt_data[frame_id][track_id] = [x, y, w, h]
    
    return dict(gt_data)


def save_mot_predictions(predictions: dict, output_file: str):
    """
    Save predictions in MOT format
    
    Args:
        predictions: Dictionary {frame_id: {track_id: [x, y, w, h]}}
        output_file: Path to save predictions
    """
    with open(output_file, 'w') as f:
        for frame_id in sorted(predictions.keys()):
            for track_id, bbox in predictions[frame_id].items():
                x, y, w, h = bbox
                # MOT format: frame, id, x, y, w, h, conf, -1, -1, -1
                f.write(f"{frame_id},{track_id},{x},{y},{w},{h},1,-1,-1,-1\n")


def run_tracking_and_extract_results(images_path: Union[str, Path], 
                                     model_path: str,
                                     output_txt: str = None,
                                     image_pattern: str = None,
                                     resize_to = None,
                                     conf_threshold: float = 0.3) -> dict:
    """
    Run tracking system on image sequence and extract results in MOT format
    
    Args:
        images_path: Path to directory containing frame images
        model_path: Path to YOLO model
        output_txt: Optional path to save tracking results (MOT format)
        image_pattern: Optional regex pattern to filter image files
        resize_to: Tuple (width, height) to resize frames, or None to keep original size
        conf_threshold: Confidence threshold for detections
    
    Returns:
        Dictionary: {frame_id: {track_id: [x, y, w, h]}}
    """
    print("\n" + "="*70)
    print("RUNNING TRACKING SYSTEM ON IMAGE SEQUENCE")
    print("="*70)
    
    # Get image sequence
    image_files = get_image_sequence(images_path, image_pattern)
    
    # Initialize
    model = YOLO(model_path)
    
    # Store results
    tracking_results = defaultdict(dict)
    
    frame_count = 0
    
    print(f"\nProcessing {len(image_files)} frames from: {images_path}")
    print(f"Confidence threshold: {conf_threshold}")
    
    for img_path in image_files:
        frame_count += 1
        
        # Read frame
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  ⚠️  Warning: Could not read {img_path.name}, skipping...")
            continue
        
        # Resize if requested
        if resize_to is not None:
            frame = cv2.resize(frame, (1920,1080))
        
        # Track
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf_threshold,
            iou=0.5,
            verbose=False
        )
        
        # Extract results for this frame
        for r in results:
            if r.boxes.id is None:
                continue
            
            for box, tid, cls, conf in zip(r.boxes.xywh, r.boxes.id, r.boxes.cls, r.boxes.conf):
                if int(cls) != PLAYER_CLASS:
                    continue
                # if conf < 0.4:
                #     continue
                track_id = int(tid)
                x, y, w, h = map(float, box.cpu().numpy())
                
                # Convert from center format (x_center, y_center, w, h) 
                # to top-left format (x, y, w, h) for MOT
                x_tl = x - w/2
                y_tl = y - h/2
                
                tracking_results[frame_count][track_id] = [x_tl, y_tl, w, h]
        
        # Progress
        if frame_count % 50 == 0:
            print(f"  Processed {frame_count}/{len(image_files)} frames...")
    
    print(f"\n✅ Tracking complete: {frame_count} frames processed")
    print(f"   Total unique tracks: {len(set(tid for frame in tracking_results.values() for tid in frame.keys()))}")
    print(f"   Total detections: {sum(len(frame) for frame in tracking_results.values())}")
    
    # Convert to regular dict
    tracking_results = dict(tracking_results)
    
    # Save if requested
    if output_txt:
        save_mot_predictions(tracking_results, output_txt)
        print(f"   Results saved to: {output_txt}")
    
    return tracking_results


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in [x, y, w, h] format"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to [x1, y1, x2, y2]
    box1_x2 = x1 + w1
    box1_y2 = y1 + h1
    box2_x2 = x2 + w2
    box2_y2 = y2 + h2
    
    # Intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def evaluate_mot_tracking(ground_truth: dict, predictions: dict, iou_threshold: float = 0.5):
    """
    Evaluate MOT metrics
    
    Args:
        ground_truth: {frame_id: {track_id: [x, y, w, h]}}
        predictions: {frame_id: {track_id: [x, y, w, h]}}
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary of metrics
    """
    print("\n" + "="*70)
    print("EVALUATING TRACKING METRICS")
    print("="*70)
    
    total_gt = 0
    total_pred = 0
    total_matches = 0
    total_fp = 0
    total_fn = 0
    total_id_switches = 0
    
    # Track matching history for ID switches
    gt_to_pred_history = {}  # {gt_id: last_matched_pred_id}
    
    all_frames = sorted(ground_truth.keys())
    
    for frame_id in all_frames:
        # if frame_id < 10:
        #     continue
        gt_frame = ground_truth.get(frame_id, {})
        pred_frame = predictions.get(frame_id, {})
        
        total_gt += len(gt_frame)
        total_pred += len(pred_frame)
        
        # Match detections using Hungarian algorithm (greedy approximation)
        matched_gt = set()
        matched_pred = set()
        frame_matches = []
        
        # Calculate IoU matrix
        iou_matrix = {}
        for gt_id, gt_box in gt_frame.items():
            for pred_id, pred_box in pred_frame.items():
                iou = calculate_iou(gt_box, pred_box)
                if iou >= iou_threshold:
                    iou_matrix[(gt_id, pred_id)] = iou
        
        # Greedy matching (highest IoU first)
        for (gt_id, pred_id), iou in sorted(iou_matrix.items(), key=lambda x: x[1], reverse=True):
            if gt_id not in matched_gt and pred_id not in matched_pred:
                matched_gt.add(gt_id)
                matched_pred.add(pred_id)
                frame_matches.append((gt_id, pred_id))
                
                # Check for ID switch
                if gt_id in gt_to_pred_history:
                    if gt_to_pred_history[gt_id] != pred_id:
                        total_id_switches += 1
                
                gt_to_pred_history[gt_id] = pred_id
        
        total_matches += len(frame_matches)
        total_fp += len(pred_frame) - len(matched_pred)
        total_fn += len(gt_frame) - len(matched_gt)
    
    # Calculate metrics
    precision = total_matches / total_pred if total_pred > 0 else 0
    recall = total_matches / total_gt if total_gt > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # MOTA = 1 - (FN + FP + ID_switches) / GT
    mota = 1 - (total_fn + total_fp + total_id_switches) / total_gt if total_gt > 0 else 0
    
    metrics = {
        'total_gt_detections': total_gt,
        'total_pred_detections': total_pred,
        'total_matches': total_matches,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'total_id_switches': total_id_switches,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'MOTA': mota,
        'num_frames': len(all_frames),
        'iou_threshold': iou_threshold
    }
    
    # Print results
    print(f"\n{'Metric':<25} {'Value':<15}")
    print("-" * 40)
    print(f"{'Frames Evaluated':<25} {metrics['num_frames']:<15}")
    print(f"{'GT Detections':<25} {total_gt:<15}")
    print(f"{'Pred Detections':<25} {total_pred:<15}")
    print(f"{'Matches':<25} {total_matches:<15}")
    print(f"{'False Positives':<25} {total_fp:<15}")
    print(f"{'False Negatives':<25} {total_fn:<15}")
    print(f"{'ID Switches':<25} {total_id_switches:<15}")
    print("-" * 40)
    print(f"{'Precision':<25} {precision:.4f}")
    print(f"{'Recall':<25} {recall:.4f}")
    print(f"{'F1 Score':<25} {f1_score:.4f}")
    print(f"{'MOTA':<25} {mota:.4f}")
    
    return metrics


def evaluate_tracking_system(images_path: Union[str, Path], 
                            model_path: str, 
                            ground_truth_path: str,
                            predictions_path: str = None,
                            results_path: str = None,
                            image_pattern: str = None,
                            resize_to: tuple = (1280, 720),
                            conf_threshold: float = 0.3,
                            iou_threshold: float = 0.5):
    """
    Complete evaluation pipeline: run tracking + evaluate on image sequence
    
    Args:
        images_path: Path to directory containing frame images
        model_path: Path to YOLO model
        ground_truth_path: Path to ground truth txt file (MOT format)
        predictions_path: Optional path to save predictions (MOT format)
        results_path: Optional path to save evaluation results (JSON)
        image_pattern: Optional regex pattern to filter image files
        resize_to: Tuple (width, height) to resize frames, or None
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for matching
    """
    # Set default paths
    if predictions_path is None:
        dir_name = Path(images_path).name
        predictions_path = f"predictions_{dir_name}.txt"
    
    if results_path is None:
        dir_name = Path(images_path).name
        results_path = f"evaluation_{dir_name}.json"
    
    print("\n" + "="*70)
    print("TRACKING EVALUATION PIPELINE (MOT FORMAT)")
    print("="*70)
    print(f"Images Dir:   {images_path}")
    print(f"Model:        {model_path}")
    print(f"Ground Truth: {ground_truth_path}")
    print(f"Predictions:  {predictions_path}")
    print(f"Results:      {results_path}")
    if image_pattern:
        print(f"Pattern:      {image_pattern}")
    print("="*70)
    
    # Step 1: Run tracking
    print("\n[1/3] Running tracking system...")
    predictions = run_tracking_and_extract_results(
        images_path=images_path,
        model_path=model_path,
        output_txt=predictions_path,
        image_pattern=image_pattern,
        resize_to=resize_to,
        conf_threshold=conf_threshold
    )
    
    # Step 2: Load ground truth
    print("\n[2/3] Loading ground truth...")
    ground_truth = load_mot_ground_truth(ground_truth_path)
    print(f"   Loaded {len(ground_truth)} frames")
    print(f"   Total GT detections: {sum(len(frame) for frame in ground_truth.values())}")
    
    # Step 3: Evaluate
    print("\n[3/3] Evaluating against ground truth...")
    metrics = evaluate_mot_tracking(ground_truth, predictions, iou_threshold)
    
    # Save results
    if results_path:
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✅ Results saved to: {results_path}")
    
    return metrics


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("EXAMPLE: MOT Format Tracking Evaluation")
    print("="*70)
    
    images_path = "img1"  # Directory with your ~750 images
    model_path = "C:/Users/neeld/SoccerNet/runs/detect/train/weights/best.pt"
    ground_truth_path = "gt.txt"  # Your MOT format ground truth
    
    # Check if paths exist
    if Path(images_path).exists() and Path(ground_truth_path).exists():
        metrics = evaluate_tracking_system(
            images_path=images_path,
            model_path=model_path,
            ground_truth_path=ground_truth_path,
            predictions_path="predictions.txt",
            results_path="evaluation_results.json",
            conf_threshold=0.5,
            iou_threshold=0.5,
            resize_to = None
        )
    else:
        print("\n⚠️  Setup Instructions:")
        print("\n1. Organize your frames:")
        print("   frames/action_8_clip_0/")
        print("     ├── frame_0001.jpg")
        print("     ├── frame_0002.jpg")
        print("     └── ...")
        print("\n2. Ensure gt.txt is in MOT format:")
        print("   1,1,136,520,51,135,1,-1,-1,-1")
        print("   1,2,250,100,45,120,1,-1,-1,-1")
        print("   ...")
        print("\n3. Run:")
        print("   python evaluate_tracking.py")