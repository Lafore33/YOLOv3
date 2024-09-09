import torch


def iou_wh(boxes1, boxes2):

    """Calculates the intersection over union between boxes with only width and height,
        assuming they have the same center"""

    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(boxes1[..., 1], boxes2[..., 1])
    union = boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection

    return intersection / union

