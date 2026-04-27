"""
Multi-Object Tracking Evaluation System
Comprehensive metrics for soccer player tracking evaluation

Implements metrics from major tracking papers:
- MOTA, MOTP (Bernardin & Stiefelhagen, 2008)
- GMOTA, GMME (Baysal & Duygulu, 2016)
- MT, PT, ML, IDS, Frag (Xing et al., 2011)
- Detection metrics: Precision, Recall, F1
- Tracking-specific: TDR, Fragmentation
"""

import numpy as np
from collections import defaultdict
from pathlib import Path
import json
from typing import Dict, List, Tuple, Set
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


class MOTEvaluator:
    """
    Multi-Object Tracking Evaluator
    
    Computes comprehensive tracking metrics against ground truth
    """
    
    def __init__(self, iou_threshold=0.5, distance_threshold=50):
        """
        Initialize evaluator
        
        Args:
            iou_threshold: IoU threshold for considering a match (default: 0.5)
            distance_threshold: Center distance threshold in pixels (default: 50)
        """
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        
        # Frame-level metrics
        self.tp_frames = []  # True positives per frame
        self.fp_frames = []  # False positives per frame
        self.fn_frames = []  # False negatives per frame
        self.idsw_frames = []  # ID switches per frame
        self.mota_frames = []  # MOTA per frame
        
        # Track-level metrics
        self.gt_tracks = defaultdict(list)  # Ground truth trajectories
        self.pred_tracks = defaultdict(list)  # Predicted trajectories
        
        # Global metrics
        self.total_gt_detections = 0
        self.total_pred_detections = 0
        self.total_matches = 0
        self.total_id_switches = 0
        self.total_fragmentations = 0
        
        # Track matching for ID consistency
        self.track_mappings = {}  # Maps predicted ID to GT ID per frame
        self.previous_mappings = {}  # Previous frame mappings
        
        # Distance errors for MOTP
        self.distance_errors = []
    
    @staticmethod
    def bbox_iou(bbox1, bbox2):
        """
        Calculate IoU between two bounding boxes
        
        Args:
            bbox1, bbox2: [x1, y1, x2, y2]
        
        Returns:
            IoU value [0, 1]
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-10)
    
    @staticmethod
    def bbox_center_distance(bbox1, bbox2):
        """Calculate Euclidean distance between bbox centers"""
        cx1 = (bbox1[0] + bbox1[2]) / 2
        cy1 = (bbox1[1] + bbox1[3]) / 2
        cx2 = (bbox2[0] + bbox2[2]) / 2
        cy2 = (bbox2[1] + bbox2[3]) / 2
        
        return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    
    def add_frame(self, frame_id: int, 
                  gt_detections: Dict[int, Tuple],
                  pred_detections: Dict[int, Tuple]):
        """
        Add frame for evaluation
        
        Args:
            frame_id: Frame number
            gt_detections: {track_id: (x1, y1, x2, y2, team_id)}
            pred_detections: {track_id: (x1, y1, x2, y2, team_id)}
        """
        # Store trajectories
        for tid, bbox_team in gt_detections.items():
            bbox = bbox_team[:4]
            team = bbox_team[4] if len(bbox_team) > 4 else -1
            self.gt_tracks[tid].append((frame_id, bbox, team))
        
        for tid, bbox_team in pred_detections.items():
            bbox = bbox_team[:4]
            team = bbox_team[4] if len(bbox_team) > 4 else -1
            self.pred_tracks[tid].append((frame_id, bbox, team))
        
        # Match detections using Hungarian algorithm
        matches, unmatched_gt, unmatched_pred = self._match_detections(
            gt_detections, pred_detections
        )
        
        # Count metrics for this frame
        tp = len(matches)
        fp = len(unmatched_pred)
        fn = len(unmatched_gt)
        
        self.tp_frames.append(tp)
        self.fp_frames.append(fp)
        self.fn_frames.append(fn)
        
        # Track ID switches
        id_switches = self._count_id_switches(matches, frame_id)
        self.idsw_frames.append(id_switches)
        
        # Store distance errors for MOTP
        for gt_id, pred_id in matches:
            dist = self.bbox_center_distance(
                gt_detections[gt_id][:4],
                pred_detections[pred_id][:4]
            )
            self.distance_errors.append(dist)
        
        # Update totals
        self.total_gt_detections += len(gt_detections)
        self.total_pred_detections += len(pred_detections)
        self.total_matches += tp
        self.total_id_switches += id_switches
        
        # Calculate MOTA for this frame
        num_gt = len(gt_detections)
        if num_gt > 0:
            mota = 1 - (fn + fp + id_switches) / num_gt
            self.mota_frames.append(mota)
        else:
            self.mota_frames.append(0.0)
    
    def _match_detections(self, gt_detections, pred_detections):
        """
        Match ground truth and predicted detections using Hungarian algorithm
        
        Returns:
            matches: List of (gt_id, pred_id) tuples
            unmatched_gt: List of unmatched GT IDs
            unmatched_pred: List of unmatched prediction IDs
        """
        if not gt_detections or not pred_detections:
            return [], list(gt_detections.keys()), list(pred_detections.keys())
        
        gt_ids = list(gt_detections.keys())
        pred_ids = list(pred_detections.keys())
        
        # Build cost matrix (negative IoU for maximization)
        cost_matrix = np.zeros((len(gt_ids), len(pred_ids)))
        
        for i, gt_id in enumerate(gt_ids):
            for j, pred_id in enumerate(pred_ids):
                iou = self.bbox_iou(
                    gt_detections[gt_id][:4],
                    pred_detections[pred_id][:4]
                )
                cost_matrix[i, j] = -iou  # Negative for minimization
        
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Filter matches by IoU threshold
        matches = []
        matched_gt = set()
        matched_pred = set()
        
        for i, j in zip(row_ind, col_ind):
            iou = -cost_matrix[i, j]
            if iou >= self.iou_threshold:
                matches.append((gt_ids[i], pred_ids[j]))
                matched_gt.add(gt_ids[i])
                matched_pred.add(pred_ids[j])
        
        unmatched_gt = [g for g in gt_ids if g not in matched_gt]
        unmatched_pred = [p for p in pred_ids if p not in matched_pred]
        
        return matches, unmatched_gt, unmatched_pred
    
    def _count_id_switches(self, matches, frame_id):
        """Count identity switches in current frame"""
        id_switches = 0
        current_mappings = {}
        
        for gt_id, pred_id in matches:
            current_mappings[pred_id] = gt_id
            
            # Check if this prediction was matched to different GT in previous frame
            if pred_id in self.previous_mappings:
                if self.previous_mappings[pred_id] != gt_id:
                    id_switches += 1
        
        self.track_mappings[frame_id] = current_mappings
        self.previous_mappings = current_mappings
        
        return id_switches
    
    def compute_metrics(self) -> Dict:
        """
        Compute all tracking metrics
        
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # ========================================
        # DETECTION METRICS
        # ========================================
        
        total_tp = sum(self.tp_frames)
        total_fp = sum(self.fp_frames)
        total_fn = sum(self.fn_frames)
        
        # Precision and Recall
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1_score
        
        # Detection metrics
        metrics['total_tp'] = total_tp
        metrics['total_fp'] = total_fp
        metrics['total_fn'] = total_fn
        metrics['detection_error'] = total_fp + total_fn
        
        # ========================================
        # MOTA (Multiple Object Tracking Accuracy)
        # ========================================
        
        total_idsw = sum(self.idsw_frames)
        
        if self.total_gt_detections > 0:
            mota = 1 - (total_fn + total_fp + total_idsw) / self.total_gt_detections
        else:
            mota = 0
        
        metrics['MOTA'] = mota
        metrics['total_id_switches'] = total_idsw
        
        # ========================================
        # MOTP (Multiple Object Tracking Precision)
        # ========================================
        
        if len(self.distance_errors) > 0:
            motp = np.mean(self.distance_errors)
        else:
            motp = float('inf')
        
        metrics['MOTP'] = motp
        
        # ========================================
        # TRACK-LEVEL METRICS
        # ========================================
        
        track_metrics = self._compute_track_metrics()
        metrics.update(track_metrics)
        
        # ========================================
        # GLOBAL METRICS
        # ========================================
        
        global_metrics = self._compute_global_metrics()
        metrics.update(global_metrics)
        
        # ========================================
        # FRAGMENTATION METRICS
        # ========================================
        
        frag_metrics = self._compute_fragmentation_metrics()
        metrics.update(frag_metrics)
        
        # ========================================
        # HIT/MISS MEASUREMENTS
        # ========================================
        
        hit_rate = total_tp / self.total_gt_detections if self.total_gt_detections > 0 else 0
        miss_rate = total_fn / self.total_gt_detections if self.total_gt_detections > 0 else 0
        
        metrics['hit_rate'] = hit_rate
        metrics['miss_rate'] = miss_rate
        
        # ========================================
        # SUCCESS/FAIL RATES
        # ========================================
        
        success_rate = (total_tp / (total_tp + total_fp + total_fn)) if (total_tp + total_fp + total_fn) > 0 else 0
        fail_rate = 1 - success_rate
        
        metrics['success_rate'] = success_rate
        metrics['fail_rate'] = fail_rate
        
        # ========================================
        # TRACKING ACCURACY
        # ========================================
        
        tracking_accuracy = total_tp / self.total_pred_detections if self.total_pred_detections > 0 else 0
        metrics['tracking_accuracy'] = tracking_accuracy
        
        # ========================================
        # RELIABILITY (Correct matches / Total predictions)
        # ========================================
        
        reliability = self.total_matches / self.total_pred_detections if self.total_pred_detections > 0 else 0
        metrics['reliability'] = reliability
        
        return metrics
    
    def _compute_track_metrics(self) -> Dict:
        """
        Compute trajectory-level metrics:
        - MT (Mostly Tracked): >80% trajectory covered
        - PT (Partially Tracked): 20-80% trajectory covered
        - ML (Mostly Lost): <20% trajectory covered
        """
        metrics = {}
        
        # Match GT tracks to predicted tracks
        mt_count = 0
        pt_count = 0
        ml_count = 0
        
        for gt_id, gt_trajectory in self.gt_tracks.items():
            gt_frames = set([frame_id for frame_id, _, _ in gt_trajectory])
            
            # Find best matching predicted track
            best_match_ratio = 0
            best_pred_id = None
            
            for pred_id, pred_trajectory in self.pred_tracks.items():
                pred_frames = set([frame_id for frame_id, _, _ in pred_trajectory])
                
                # Calculate overlap
                intersection = len(gt_frames & pred_frames)
                ratio = intersection / len(gt_frames) if len(gt_frames) > 0 else 0
                
                if ratio > best_match_ratio:
                    best_match_ratio = ratio
                    best_pred_id = pred_id
            
            # Classify based on coverage
            if best_match_ratio >= 0.8:
                mt_count += 1
            elif best_match_ratio >= 0.2:
                pt_count += 1
            else:
                ml_count += 1
        
        total_tracks = len(self.gt_tracks)
        
        metrics['MT'] = mt_count
        metrics['PT'] = pt_count
        metrics['ML'] = ml_count
        metrics['MT_ratio'] = mt_count / total_tracks if total_tracks > 0 else 0
        metrics['PT_ratio'] = pt_count / total_tracks if total_tracks > 0 else 0
        metrics['ML_ratio'] = ml_count / total_tracks if total_tracks > 0 else 0
        
        return metrics
    
    def _compute_global_metrics(self) -> Dict:
        """
        Compute GMOTA and GMME (Global metrics with team assignment)
        """
        metrics = {}
        
        # Count team assignment errors
        team_errors = 0
        team_comparisons = 0
        
        for frame_id, mappings in self.track_mappings.items():
            for pred_id, gt_id in mappings.items():
                # Find team assignments
                gt_team = None
                pred_team = None
                
                # Get GT team
                if gt_id in self.gt_tracks:
                    for fid, bbox, team in self.gt_tracks[gt_id]:
                        if fid == frame_id:
                            gt_team = team
                            break
                
                # Get predicted team
                if pred_id in self.pred_tracks:
                    for fid, bbox, team in self.pred_tracks[pred_id]:
                        if fid == frame_id:
                            pred_team = team
                            break
                
                # Count mismatches
                if gt_team is not None and pred_team is not None and gt_team != -1 and pred_team != -1:
                    team_comparisons += 1
                    if gt_team != pred_team:
                        team_errors += 1
        
        # GMOTA (MOTA with team assignment)
        total_fn = sum(self.fn_frames)
        total_fp = sum(self.fp_frames)
        total_idsw = sum(self.idsw_frames)
        
        if self.total_gt_detections > 0:
            gmota = 1 - (total_fn + total_fp + total_idsw + team_errors) / self.total_gt_detections
        else:
            gmota = 0
        
        # GMME (Global Identity Mismatch with team)
        gmme_rate = team_errors / team_comparisons if team_comparisons > 0 else 0
        
        metrics['GMOTA'] = gmota
        metrics['GMME'] = team_errors
        metrics['GMME_rate'] = gmme_rate
        metrics['team_assignment_accuracy'] = 1 - gmme_rate if team_comparisons > 0 else 0
        
        return metrics
    
    def _compute_fragmentation_metrics(self) -> Dict:
        """
        Compute track fragmentation metrics
        - Fragmentation: Number of times a track is interrupted
        - TDR (Track Detection Rate): Percentage of GT track detected
        """
        metrics = {}
        
        total_fragments = 0
        tdr_scores = []
        
        for gt_id, gt_trajectory in self.gt_tracks.items():
            gt_frames = sorted([frame_id for frame_id, _, _ in gt_trajectory])
            
            # Find matching predicted track segments
            matched_pred_tracks = defaultdict(list)
            
            for frame_id in gt_frames:
                if frame_id in self.track_mappings:
                    for pred_id, matched_gt_id in self.track_mappings[frame_id].items():
                        if matched_gt_id == gt_id:
                            matched_pred_tracks[pred_id].append(frame_id)
            
            # Count fragments (continuous segments)
            fragments = 0
            for pred_id, pred_frames in matched_pred_tracks.items():
                pred_frames = sorted(pred_frames)
                
                # Count breaks in sequence
                for i in range(1, len(pred_frames)):
                    if pred_frames[i] - pred_frames[i-1] > 1:
                        fragments += 1
                
                # At least 1 fragment per matched track
                if pred_frames:
                    fragments = max(1, fragments)
            
            total_fragments += max(0, fragments - 1)  # Don't count first fragment
            
            # TDR: ratio of detected frames
            detected_frames = sum(len(frames) for frames in matched_pred_tracks.values())
            tdr = detected_frames / len(gt_frames) if len(gt_frames) > 0 else 0
            tdr_scores.append(tdr)
        
        metrics['total_fragmentations'] = total_fragments
        metrics['avg_fragmentations'] = total_fragments / len(self.gt_tracks) if len(self.gt_tracks) > 0 else 0
        metrics['TDR'] = np.mean(tdr_scores) if tdr_scores else 0
        
        return metrics
    
    def print_summary(self, metrics: Dict = None):
        """Print formatted summary of all metrics"""
        if metrics is None:
            metrics = self.compute_metrics()
        
        print("\n" + "="*70)
        print("MULTI-OBJECT TRACKING EVALUATION SUMMARY")
        print("="*70)
        
        print("\n📊 DETECTION METRICS")
        print("-"*70)
        print(f"  Precision:              {metrics['precision']:.4f}")
        print(f"  Recall:                 {metrics['recall']:.4f}")
        print(f"  F1-Score:               {metrics['f1_score']:.4f}")
        print(f"  True Positives (TP):    {metrics['total_tp']}")
        print(f"  False Positives (FP):   {metrics['total_fp']}")
        print(f"  False Negatives (FN):   {metrics['total_fn']}")
        print(f"  Detection Error:        {metrics['detection_error']}")
        
        print("\n🎯 TRACKING ACCURACY METRICS")
        print("-"*70)
        print(f"  MOTA:                   {metrics['MOTA']:.4f}")
        print(f"  MOTP:                   {metrics['MOTP']:.2f} pixels")
        print(f"  ID Switches:            {metrics['total_id_switches']}")
        print(f"  Tracking Accuracy:      {metrics['tracking_accuracy']:.4f}")
        print(f"  Reliability:            {metrics['reliability']:.4f}")
        
        print("\n🏆 GLOBAL METRICS (with Team Assignment)")
        print("-"*70)
        print(f"  GMOTA:                  {metrics['GMOTA']:.4f}")
        print(f"  GMME (Team Errors):     {metrics['GMME']}")
        print(f"  GMME Rate:              {metrics['GMME_rate']:.4f}")
        print(f"  Team Assign Accuracy:   {metrics['team_assignment_accuracy']:.4f}")
        
        print("\n📈 TRAJECTORY METRICS")
        print("-"*70)
        print(f"  Mostly Tracked (MT):    {metrics['MT']} ({metrics['MT_ratio']:.2%})")
        print(f"  Partially Tracked (PT): {metrics['PT']} ({metrics['PT_ratio']:.2%})")
        print(f"  Mostly Lost (ML):       {metrics['ML']} ({metrics['ML_ratio']:.2%})")
        
        print("\n🔀 FRAGMENTATION METRICS")
        print("-"*70)
        print(f"  Total Fragmentations:   {metrics['total_fragmentations']}")
        print(f"  Avg Frags per Track:    {metrics['avg_fragmentations']:.2f}")
        print(f"  Track Detection Rate:   {metrics['TDR']:.4f}")
        
        print("\n✅ HIT/MISS & SUCCESS METRICS")
        print("-"*70)
        print(f"  Hit Rate:               {metrics['hit_rate']:.4f}")
        print(f"  Miss Rate:              {metrics['miss_rate']:.4f}")
        print(f"  Success Rate:           {metrics['success_rate']:.4f}")
        print(f"  Fail Rate:              {metrics['fail_rate']:.4f}")
        
        print("\n" + "="*70)
    
    def save_results(self, output_path: str, metrics: Dict = None):
        """Save results to JSON file"""
        if metrics is None:
            metrics = self.compute_metrics()
        
        # Convert numpy types to Python types for JSON serialization
        clean_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                clean_metrics[key] = float(value)
            else:
                clean_metrics[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(clean_metrics, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_path}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_ground_truth(gt_file: str) -> Dict:
    """
    Load ground truth from file
    
    Expected format (JSON):
    {
        "frame_1": {
            "1": [x1, y1, x2, y2, team_id],
            "2": [x1, y1, x2, y2, team_id],
            ...
        },
        "frame_2": {...},
        ...
    }
    
    Returns:
        Dictionary: {frame_id: {track_id: (x1, y1, x2, y2, team_id)}}
    """
    with open(gt_file, 'r') as f:
        data = json.load(f)
    
    ground_truth = {}
    for frame_key, detections in data.items():
        frame_id = int(frame_key.replace('frame_', ''))
        ground_truth[frame_id] = {}
        
        for track_id, bbox_team in detections.items():
            ground_truth[frame_id][int(track_id)] = tuple(bbox_team)
    
    return ground_truth


def load_predictions(pred_file: str) -> Dict:
    """
    Load predictions from file (same format as ground truth)
    """
    return load_ground_truth(pred_file)


def evaluate_from_files(gt_file: str, pred_file: str, 
                        iou_threshold: float = 0.5,
                        output_file: str = None) -> Dict:
    """
    Evaluate tracking from ground truth and prediction files
    
    Args:
        gt_file: Path to ground truth JSON
        pred_file: Path to predictions JSON
        iou_threshold: IoU threshold for matching
        output_file: Optional path to save results
    
    Returns:
        Dictionary of all metrics
    """
    print(f"\n📂 Loading data...")
    print(f"  Ground Truth: {gt_file}")
    print(f"  Predictions:  {pred_file}")
    
    ground_truth = load_ground_truth(gt_file)
    predictions = load_predictions(pred_file)
    
    # Initialize evaluator
    evaluator = MOTEvaluator(iou_threshold=iou_threshold)
    
    # Get all frame IDs
    all_frames = sorted(set(ground_truth.keys()) | set(predictions.keys()))
    
    print(f"\n⚙️  Processing {len(all_frames)} frames...")
    
    # Add frames
    for frame_id in all_frames:
        gt_dets = ground_truth.get(frame_id, {})
        pred_dets = predictions.get(frame_id, {})
        
        evaluator.add_frame(frame_id, gt_dets, pred_dets)
    
    # Compute metrics
    print(f"\n🔄 Computing metrics...")
    metrics = evaluator.compute_metrics()
    
    # Print summary
    evaluator.print_summary(metrics)
    
    # Save results if requested
    if output_file:
        evaluator.save_results(output_file, metrics)
    
    return metrics


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Evaluate from files
    gt_file = "ground_truth.json"
    pred_file = "predictions.json"
    
    # Check if files exist
    if Path(gt_file).exists() and Path(pred_file).exists():
        metrics = evaluate_from_files(
            gt_file=gt_file,
            pred_file=pred_file,
            iou_threshold=0.5,
            output_file="evaluation_results.json"
        )
    else:
        print("\n⚠️  Example files not found. Creating demo...")
        print("\nTo use this evaluator:")
        print("  1. Prepare ground_truth.json with format:")
        print('     {"frame_1": {"1": [x1, y1, x2, y2, team_id], ...}, ...}')
        print("  2. Prepare predictions.json with same format")
        print("  3. Run: python mot_evaluator.py")
        print("\nOr use programmatically:")
        print("  evaluator = MOTEvaluator()")
        print("  evaluator.add_frame(frame_id, gt_dict, pred_dict)")
        print("  metrics = evaluator.compute_metrics()")