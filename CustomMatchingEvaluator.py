import numpy as np
import torch
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.structures import Boxes, pairwise_iou
from detectron2.utils import comm

def perform_box_matching(gt_boxes: Boxes, pred_boxes: Boxes, iou_threshold: float = 0.5):
    """
    Perform a greedy matching between ground truth and predicted boxes.
    Args:
         gt_boxes (Boxes): ground truth boxes.
         pred_boxes (Boxes): predicted boxes.
         iou_threshold (float): minimum IoU required to consider a match.
    Returns:
         matches (list of tuples): each tuple is (gt_index, pred_index)
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return []
    
    # Move gt_boxes to the device of pred_boxes
    gt_boxes = Boxes(gt_boxes.tensor.to(pred_boxes.tensor.device))
    # Compute pairwise IoU matrix; shape: (num_pred, num_gt)
    iou_matrix = pairwise_iou(pred_boxes, gt_boxes)
    matches = []
    used_pred = set()
    num_gt = len(gt_boxes)
    for gt_idx in range(num_gt):
        best_iou = 0.0
        best_pred = -1
        for pred_idx in range(len(pred_boxes)):
            if pred_idx in used_pred:
                continue
            iou = iou_matrix[pred_idx, gt_idx]
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_pred = pred_idx
        if best_pred >= 0:
            matches.append((gt_idx, best_pred))
            used_pred.add(best_pred)
    return matches

class CustomMatchingEvaluator(DatasetEvaluator):
    """
    A custom evaluator that:
      1. Retrieves for each image the ground truth boxes and predicted boxes.
      2. Performs matching using a greedy IoU-based algorithm with a specified threshold.
      3. Returns a dictionary where each key is an image_id and the value is the list
         of matched pairs (gt_index, pred_index).
    """
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        # We'll use a dictionary to store per-image data.
        # For each image_id, we store a dict with:
        #   "gt_boxes" (Boxes) and "pred_boxes" (Boxes)
        self.data = {}

    def process(self, inputs, outputs):
        """
        Expects each input dict to have an "instances" field in ground truth (with gt_boxes)
        and each output to have an "instances" field in predictions (with pred_boxes).
        """
        for input_dict, output in zip(inputs, outputs):
            image_id = input_dict.get("image_id", input_dict.get("file_name", "unknown"))
            # Process ground truth
            gt_instances = input_dict.get("instances", None)
            if gt_instances is not None and hasattr(gt_instances, "gt_boxes"):
                gt_boxes = gt_instances.gt_boxes
            else:
                gt_boxes = Boxes(torch.empty((0, 4)))
            # Process predictions
            pred_instances = output.get("instances", None)
            if pred_instances is not None and hasattr(pred_instances, "pred_boxes"):
                pred_boxes = pred_instances.pred_boxes
            else:
                pred_boxes = Boxes(torch.empty((0, 4)))
            # Store data for this image
            self.data[image_id] = {
                "gt_boxes": gt_boxes,
                "pred_boxes": pred_boxes,
                # Optionally add apertures if needed:
                "gt_apertures": gt_instances.gt_aperture.cpu().numpy().flatten() if (gt_instances is not None and hasattr(gt_instances, "gt_aperture")) else np.array([]),
                "pred_apertures": pred_instances.aperture.cpu().numpy().flatten() if (pred_instances is not None and hasattr(pred_instances, "aperture")) else np.array([]),
            }

    def evaluate(self):
        """
        For each image in self.data, perform box matching and compute:
        - Matches (TP)
        - False Positives (FP)
        - False Negatives (FN)
        Also compute aperture errors on the matched pairs.
        
        Returns:
            A dictionary containing:
            - The matches per image.
            - The aggregated aperture evaluation (mean absolute error).
            - Detection metrics: total TP, FP, FN, Precision, and Recall.
        """

        all_data = comm.all_gather(self.data)
        if not comm.is_main_process():
            return {}
        # Merge all dictionaries
        merged_data = {}
        for data in all_data:
            merged_data.update(data)
        self.data = merged_data

        results = {}
        aperture_errors = []
        
        # Counters for detection metrics (across all images)
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for image_id, item in self.data.items():
            gt_boxes = item["gt_boxes"]
            pred_boxes = item["pred_boxes"]
            gt_apertures = item["gt_apertures"]
            pred_apertures = item["pred_apertures"]

            # Perform box matching based on IoU:
            matches = perform_box_matching(gt_boxes, pred_boxes, self.iou_threshold)
            #results[image_id] = matches

            # Detection metrics for this image:
            tp = len(matches)                  # True positives: number of matched boxes.
            fp = len(pred_boxes) - tp          # False positives: predictions that didn't match.
            fn = len(gt_boxes) - tp            # False negatives: ground truth objects with no match.
            
            total_tp += tp
            total_fp += fp
            total_fn += fn

            # Compute aperture errors for the matched boxes.
            for gt_idx, pred_idx in matches:
                # Make sure indices are within bounds.
                if gt_idx < len(gt_apertures) and pred_idx < len(pred_apertures):
                    # Denormalize: multiply by 80 to convert from [0, 1] to [0, 80] degrees.
                    gt_aperture = gt_apertures[gt_idx] * 80
                    pred_aperture = pred_apertures[pred_idx] * 80
                    # Debug print (optional):
                    error = abs(gt_aperture - pred_aperture)
                    aperture_errors.append(error)
        
        # Compute overall detection precision and recall.
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

        # Compute mean aperture error.
        mean_aperture_error = np.mean(aperture_errors) if aperture_errors else float("nan")

        # Aggregate the results.
        results["aperture_evaluation"] = {
            "mean_absolute_error": mean_aperture_error,
            "num_matches": len(aperture_errors),
        }
        results["detection_metrics"] = {
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "precision": precision,
            "recall": recall,
        }

        return results
