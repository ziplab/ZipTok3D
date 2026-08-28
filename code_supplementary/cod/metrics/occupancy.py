import torch
from torchmetrics import Metric


class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("accuracy", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        B = preds.size(0)
        accuracy = (preds == target).float().view(B, -1).mean(dim=-1)
        self.accuracy += accuracy.sum()
        self.total += B

    def compute(self):
        return self.accuracy.float() * 100 / self.total


class IoU(Metric):
    def __init__(self, eps=1e-5):
        super().__init__()

        self.eps = eps
        self.add_state("iou", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        intersection = (preds * target).sum(dim=1)
        union = (preds + target).gt(0).sum(dim=1)
        iou = intersection.float() / (union + self.eps)
        B = target.size(0)
        self.iou += iou.sum()
        self.total += B

    def compute(self):
        return self.iou * 100 / self.total
