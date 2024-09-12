import torch
from utils import intersection_over_union
from collections import Counter


def mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5
):
    """
    Calculates mean average precision
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, x1, y1, w, h]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    detections = pred_boxes[:]
    ground_truths = true_boxes[:]

    # find the amount of bboxes for each training example
    # Counter here finds how many ground truth bboxes we get
    # for each training example, so let's say img 0 has 3,
    # img 1 has 5 then we will obtain a dictionary with:
    # amount_bboxes = {0:3, 1:5}
    amount_bboxes = Counter([gt[0] for gt in ground_truths])

    # We then go through each key, val in this dictionary
    # and convert to the following (w.r.t same example):
    # amount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
    for key, val in amount_bboxes.items():
        amount_bboxes[key] = torch.zeros(val)

    # sort by box probabilities which is index 1
    detections.sort(key=lambda x: x[1], reverse=True)
    TP = torch.zeros((len(detections)))
    FP = torch.zeros((len(detections)))
    total_true_bboxes = len(ground_truths)

    # If none exists for this class then we can safely skip
    if total_true_bboxes == 0:
        return 0

    for detection_idx, detection in enumerate(detections):
        # Only take out the ground_truths that have the same
        # training idx as detection
        ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

        best_iou = 0
        best_gt_idx = -1

        for idx, gt in enumerate(ground_truth_img):

            iou = intersection_over_union(
                torch.tensor(detection[2:]),
                torch.tensor(gt[2:]),
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou > iou_threshold:
            # only detect ground truth detection once
            if amount_bboxes[detection[0]][best_gt_idx] == 0:
                # true positive and add this bounding box to seen
                TP[detection_idx] = 1
                amount_bboxes[detection[0]][best_gt_idx] = 1
            else:
                FP[detection_idx] = 1

        # if IOU is lower than the detection is a false positive
        else:
            FP[detection_idx] = 1

    TP_cumsum = torch.cumsum(TP, dim=0)
    FP_cumsum = torch.cumsum(FP, dim=0)

    recalls = TP_cumsum / (total_true_bboxes + epsilon)
    precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)

    recalls = torch.cat((torch.tensor([0]), recalls))
    precisions = torch.cat((torch.tensor([1]), precisions))

    # torch.trapz for numerical integration
    average_precisions.append(torch.trapz(precisions, recalls))

    return float(sum(average_precisions) / len(average_precisions))
