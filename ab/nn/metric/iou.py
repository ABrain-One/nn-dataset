import torch
from .base import BaseMetric

class MIoU(BaseMetric):
    """
    Computes mIoU metric scores
    """
    def __init__(self, out_shape):
        self.nclass = out_shape[0]
        super().__init__()
    
    def reset(self):
        """
        Resets the internal evaluation result to initial state.
        """
        self.total_inter = torch.zeros(self.nclass)
        self.total_union = torch.zeros(self.nclass)
    
    def update(self, preds, labels):
        """
        Updates the internal evaluation result.
        :param preds: Model predictions.
        :param labels: Ground truth labels.
        """
        inter, union = self._batch_intersection_union(preds, labels, self.nclass)
        if self.total_inter.device != inter.device:
            self.total_inter = self.total_inter.to(inter.device)
            self.total_union = self.total_union.to(union.device)
        self.total_inter += inter
        self.total_union += union
    
    def __call__(self, outputs, targets):
        """Process a batch and return compatible values for accumulation"""
        self.update(outputs, targets)
        # Return dummy values for compatibility with accumulation pattern
        return 1, 1
    
    def result(self):
        """
        Gets the current evaluation result.
        :return: mIoU
        """
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)
        mIoU = IoU.mean().item()
        return mIoU
    
    def _batch_intersection_union(self, output, target, nclass):
        """
        mIoU calculation for a batch
        """
        mini = 1
        maxi = nclass
        nbins = nclass
        predict = torch.argmax(output, 1) + 1
        target = target.float() + 1

        predict = predict.float() * (target > 0).float()
        intersection = predict * (predict == target).float()
        area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)
        area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
        area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)
        area_union = area_pred + area_lab - area_inter
        assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
        return area_inter.float(), area_union.float()

# Function to create metric instance
def create_metric(out_shape=None):
    return MIoU(out_shape)
