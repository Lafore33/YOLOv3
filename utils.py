import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def iou_wh(boxes1, boxes2):

    """Calculates the intersection over union between boxes with only width and height,
        assuming they have the same center"""

    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(boxes1[..., 1], boxes2[..., 1])
    union = boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection

    return intersection / union


def convert_to_corners(boxes):
    """ Converts boxes with midpoint format to corner format"""

    x1 = boxes[..., 0:1] - boxes[..., 2:3] / 2
    y1 = boxes[..., 1:2] - boxes[..., 3:4] / 2
    x2 = boxes[..., 0:1] + boxes[..., 2:3] / 2
    y2 = boxes[..., 1:2] + boxes[..., 3:4] / 2

    return x1, y1, x2, y2


def intersection_over_union(boxes_preds, boxes_labels):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
    Returns:
        tensor: Intersection over union for all examples
    """
    box1_x1, box1_y1, box1_x2, box1_y2 = convert_to_corners(boxes_preds)
    box2_x1, box2_y1, box2_x2, box2_y2 = convert_to_corners(boxes_labels)

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, x, y, w, h]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:])) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # Create a Rectangle patch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()


# функции вывода графиков
def show_losses(train_loss_hist, test_loss_hist=None):
    #     clear_output()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.title('Train Loss')
    plt.plot(np.arange(len(train_loss_hist)), train_loss_hist)
    plt.yscale('log')
    plt.grid()

    if test_loss_hist is not None:
        plt.subplot(1, 2, 2)
        plt.title('Test Loss')
        plt.plot(np.arange(len(test_loss_hist)), test_loss_hist)
        plt.yscale('log')
        plt.grid()

    plt.show()


def cells_to_boxes(boxes, anchors, split, is_pred=True):
    batch_size = boxes.shape[0]
    num_anchors = len(anchors)

    if is_pred:
        anchors = anchors.reshape(1, len(anchors), 1, 2)
        boxes[..., 2:4] = torch.sigmoid(boxes[..., 2:4])
        boxes[..., 4:6] = torch.exp(boxes[..., 4:6]) * anchors
        boxes[..., 0:1] = torch.sigmoid(boxes[..., 0:1])

    cell_indices = (
        torch.arange(split)
        .repeat(boxes.shape[0], 3, split, 1)
        .unsqueeze(-1)
        .to(boxes.device)
    )

    boxes[..., 2:3] = 1 / split * (boxes[..., 2:3] + cell_indices)
    boxes[..., 3:4] = 1 / split * (boxes[..., 3:4] + cell_indices.permute(0, 1, 3, 2, 4))
    boxes[..., 4:6] = 1 / split * (boxes[..., 4:6])
    boxes = boxes.reshape(batch_size, num_anchors * split * split, 6)
    return boxes.tolist()









