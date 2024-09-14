from tqdm import tqdm
import torch
from utils import cells_to_boxes, non_max_suppression, plot_image


def train_fn(model, loader, loss_fn, anchors, optimizer=None, device='cpu'):
    loop = tqdm(loader, leave=True)
    mean_loss = []

    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(device)
        y0, y1, y2 = y[0].to(device), y[1].to(device), y[2].to(device)

        if optimizer is not None:
            optimizer.zero_grad()

        raw_pred = model(x)
        loss = (
                loss_fn(raw_pred[0], y0, anchors[0])
                + loss_fn(raw_pred[1], y1, anchors[1])
                + loss_fn(raw_pred[2], y2, anchors[2])
        )

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        mean_loss.append(loss.item())
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(mean_loss)


def get_bboxes(model, loader, anchors, iou_threshold, class_threshold, device='cpu'):
    all_true_boxes = []
    all_pred_boxes = []
    train_idx = 0

    model.eval()

    for batch_idx, (x, y) in enumerate(loader):
        batch_size = x.size()[0]
        x = x.to(device)

        for idx in range(len(y)):
            y[idx] = y[idx].to(device)

        with torch.no_grad():
            raw_pred = model(x)

        predictions = [[] for _ in range(batch_size)]

        num_scales = len(y)
        for i in range(num_scales):
            cur_split = raw_pred[i].shape[2]
            cur_anchors = anchors[i]

            boxes = cells_to_boxes(raw_pred[i], cur_anchors, cur_split)
            for idx, box in enumerate(boxes):
                predictions[idx] += box

        # for ground truth object we do need to process each scale, 'cause each scale will give the same resul
        gt_boxes = cells_to_boxes(y[0], anchors[0], y[0].shape[2], False)

        for idx in range(batch_size):
            temp = []
            nms_boxes = non_max_suppression(predictions[idx], iou_threshold=iou_threshold, threshold=class_threshold)

            for box in nms_boxes:
                all_pred_boxes.append([train_idx] + box)
            plot_image(x[idx].permute(1, 2, 0).to("cpu"), nms_boxes)

            for box in gt_boxes[idx]:
                if box[0] == 1:
                    all_true_boxes.append([train_idx] + box)
                    temp.append(box)

            # plot_image(x[idx].permute(1, 2, 0).to("cpu"), temp)

            train_idx += 1

    return all_pred_boxes, all_true_boxes
