import torch
from .base import BaseMetric

class MAPMetric(BaseMetric):
    """
    Mean Average Precision metric for object detection
    """
    def __init__(self, iou_threshold=0.5):
        """
        Initializes the mAP metric.
        """
        self.iou_threshold = iou_threshold
        super().__init__()
    
    def reset(self):
        """
        Resets the accumulated predictions and targets.
        """
        self.all_predictions = []
        self.all_targets = []
    
    def update(self, predictions, targets):
        """
        Updates the accumulated predictions and targets.

        Args:
            predictions: List of dictionaries with 'boxes', 'labels', 'scores'.
            targets: List of dictionaries with 'boxes', 'labels'.
        """
        self.all_predictions.extend(predictions)
        self.all_targets.extend(targets)
    
    def __call__(self, predictions, targets):
        """Process a batch and return compatible values for accumulation"""
        self.update(predictions, targets)
        # Return dummy values for compatibility with accumulation pattern
        return 1, 1
    
    def result(self):
        """
        Computes and returns the mean Average Precision (mAP).
        """
        all_aps = {}
        for class_id in range(91):  # COCO has 80 classes, but IDs range from 1 to 90
            class_preds = []
            class_targets = []
            for pred, target in zip(self.all_predictions, self.all_targets):
                mask = pred['labels'] == class_id
                boxes = pred['boxes'][mask]
                scores = pred['scores'][mask]
                gt_mask = target['labels'] == class_id
                gt_boxes = target['boxes'][gt_mask]
                class_preds.append((boxes, scores))
                class_targets.append(gt_boxes)
            if not any((len(gt) > 0 for gt in class_targets)):
                continue
            ap = self._compute_ap(class_preds, class_targets, self.iou_threshold)
            all_aps[class_id] = ap

        # Compute mean AP
        if len(all_aps) == 0:
            return 0.0  # No valid predictions

        mean_ap = float(torch.tensor(list(all_aps.values())).mean())
        return mean_ap
    
    def _compute_ap(self, class_preds, class_targets, iou_threshold):
        """
        Computes Average Precision (AP) for a single class.

        Args:
            class_preds: List of tuples (boxes, scores) for the class.
            class_targets: List of ground truth boxes for the class.
            iou_threshold: IoU threshold for considering a detection as correct.

        Returns:
            float: AP score for the class.
        """
        boxes_all = []
        scores_all = []
        matches = []
        n_gt = 0
        for (boxes, scores), gt_boxes in zip(class_preds, class_targets):
            if len(gt_boxes) > 0:
                n_gt += len(gt_boxes)
            if len(boxes) == 0:
                continue
            iou = self._box_iou(boxes, gt_boxes)
            for i in range(len(boxes)):
                if len(gt_boxes) == 0:
                    matches.append(0)
                else:
                    matches.append(1 if iou[i].max() >= iou_threshold else 0)
                scores_all.append(scores[i])
        if not scores_all:
            return 0.0
        scores_all = torch.tensor(scores_all)
        matches = torch.tensor(matches)
        sorted_indices = torch.argsort(scores_all, descending=True)
        matches = matches[sorted_indices]
        tp = torch.cumsum(matches, dim=0)
        fp = torch.cumsum(~matches.bool(), dim=0)
        precision = tp / (tp + fp)
        recall = tp / n_gt if n_gt > 0 else torch.zeros_like(tp)
        ap = 0.0
        for r in torch.linspace(0, 1, 11):
            mask = recall >= r
            if mask.any():
                ap += precision[mask].max() / 11
        return ap
    
    def _box_iou(self, boxes1, boxes2):
        """
        Computes IoU between two sets of boxes.

        Args:
            boxes1: Tensor of shape (N, 4).
            boxes2: Tensor of shape (M, 4).

        Returns:
            Tensor: IoU matrix of shape (N, M).
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        union = area1[:, None] + area2 - inter
        return inter / union

# Function to create metric instance
def create_metric(out_shape=None):
    return MAPMetric()
