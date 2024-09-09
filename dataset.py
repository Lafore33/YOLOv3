import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from PIL import Image
from utils import iou_wh


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, image_root,
                 label_root,
                 anchors,
                 image_size=416,
                 cells_split=(13, 26, 52),
                 num_classes=1,
                 transform=None):

        self.image_root = image_root
        self.label_root = label_root
        self.size = len(os.listdir(image_root))
        self.transform = transform
        self.cells_split = cells_split
        self.num_classes = num_classes
        self.image_size = image_size

        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = len(self.anchors)
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        prefix = f"img{idx}"
        image_file = prefix + ".jpg"
        label_file = prefix + ".txt"

        image_path = os.path.join(self.image_root, image_file)
        img = Image.open(image_path)
        if self.transform is not None:
            img = self.transform(img)

        label_path = os.path.join(self.label_root, label_file)

        label_matrix = [torch.zeros((self.num_anchors // 3, split, split, 6)) for split in self.cells_split]

        if os.path.getsize(label_path) != 0:
            label = torch.tensor(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2))
        else:
            return img, label_matrix

        for box in label:
            anchors_iou = iou_wh(box[3:5], self.anchors)
            anchors_iou_indices = anchors_iou.argsort(descending=True, dim=0)
            class_label, x, y, width, height = box
            scale_used = [False] * 3

            for anchor_index in anchors_iou_indices:
                scale_index = anchor_index // self.num_anchors_per_scale
                anchor_on_scale = anchor_index % self.num_anchors_per_scale
                cur_split = self.cells_split[scale_index]
                hc, vc = int(x * cur_split), int(y * cur_split)
                anchor_in_cell = label_matrix[scale_index][anchor_on_scale, vc, hc, 0]

                if not anchor_in_cell and not scale_used[scale_index]:
                    label_matrix[scale_index][anchor_on_scale, vc, hc, 0] = 1
                    x_cell, y_cell = x * cur_split - hc, y * cur_split - vc
                    width_cell, height_cell = width * cur_split, height * cur_split
                    label_matrix[scale_index][anchor_on_scale, vc, hc, 1] = 1
                    label_matrix[scale_index][anchor_on_scale, vc, hc, 2:6] = torch.tensor([x_cell, y_cell,
                                                                                            width_cell, height_cell])
                    scale_used[scale_index] = True

                # here it mean we already have best anchor for this scale, but it's not the best for the current obj,
                # so we ignore it

                elif not anchor_in_cell and anchors_iou[anchor_index] > self.ignore_iou_thresh:
                    label_matrix[scale_index][anchor_on_scale, vc, hc, 0] = -1
        return img, label_matrix
