import torch
from torch import nn
from utils import intersection_over_union as iou


class Loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, predictions, target, anchors):
        epsilon = 1e-15
        anchors = anchors.reshape(1, 3, 1, 1, 2)

        # ignore -1
        exist_obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0

        no_obj_loss = self.mse(
            predictions[..., 0:1][no_obj], target[..., 0:1][no_obj]
        )

        predictions[..., 2:4] = self.sigmoid(predictions[..., 2:4])
        target[..., 4:6] = torch.log(
            epsilon + target[..., 4:6] / anchors
        )
        coord_loss = self.mse(predictions[..., 2:6][exist_obj], target[..., 2:6][exist_obj])

        class_loss = self.cross_entropy(predictions[..., 1:2], target[..., 1:2])

        # box_pred = torch.cat([self.sigmoid(predictions[..., 1:3]),
        # torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        # ious = iou(box_pred[exist_obj], target[..., 1:5][exist_obj]).detach()
        # object_loss = self.mse(self.sigmoid(predictions[..., 0:1][exist_obj]), ious * target[..., 0:1][exist_obj])
        object_loss = self.mse(predictions[..., 0:1][exist_obj], target[..., 0:1][exist_obj])

        total_loss = no_obj_loss + coord_loss + class_loss + object_loss
        return total_loss


