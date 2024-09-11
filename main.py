import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import cells_to_boxes, non_max_suppression
from model import YOLOv3
from loss import Loss
from train import train_fn
from dataset import ImageDataset

def main():
    anchors = [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ]

    LR = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLOv3().to(device)

    seed = 123
    torch.manual_seed(seed)
    train_images_path = "detection/images/train"
    train_labels_path = "detection/labels/train"
    val_images_path = "detection/images/val"
    val_labels_path = "detection/labels/val"

    transform_aug = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor()
    ])
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.0005)
    loss_function = Loss()

    train_dataset = ImageDataset(train_images_path, train_labels_path, anchors=anchors, transform=transform_aug)
    val_dataset = ImageDataset(val_images_path, val_labels_path, anchors=anchors, transform=transform_aug)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    iou_threshold = 0.4
    class_threshold = 0.5
    scaled_anchors = (
            torch.tensor(anchors)
            * torch.tensor([13, 26, 52]).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(DEVICE)

    train_loss_hist = []
    for epoch in range(NUM_EPOCHS):
        mean_loss = train_fn(model, train_loader, loss_function, scaled_anchors, optimizer, DEVICE)
        #     train_loss_hist.append(mean_loss)
        #     show_losses(train_loss_hist)


if __name__ == "__main__":
    main()