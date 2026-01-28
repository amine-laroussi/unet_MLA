import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from unet_model.model import unet
from dataset_normalizer.dataset import SegmentationDataset


# ============================================================
# Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Hyperparameters
# ============================================================
NUM_CLASSES = 2
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50


# ============================================================
# Loss function: Weighted Cross Entropy (U-Net 2015)
# ============================================================
class UNetWeightedCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, target, weight_map):
        ce = self.ce(logits, target)          # (B, H, W)
        return (ce * weight_map).mean()


# ============================================================
# Utility: center crop for tensors (B, H, W)
# ============================================================
def center_crop_tensor(tensor, target_h, target_w):
    _, h, w = tensor.shape
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return tensor[:, top:top + target_h, left:left + target_w]


# ============================================================
# Dataset & DataLoader
# ============================================================
img2mask = {
    r"C:\Users\amine\Desktop\Unet\images\image0.jpg":
    r"C:\Users\amine\Desktop\Unet\labels\image0.jpg",
}

train_dataset = SegmentationDataset(img2mask=img2mask, train=True)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# ============================================================
# Model, optimizer, loss
# ============================================================
model = unet(num_classes=NUM_CLASSES).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = UNetWeightedCrossEntropy()


# ============================================================
# Training sanity check (single iteration)
# ============================================================
model.train()

for images, masks, weight_maps in train_loader:
    images = images.to(device)
    masks = masks.to(device).long()
    weight_maps = weight_maps.to(device).float()

    optimizer.zero_grad()

    outputs = model(images)

    print("outputs shape:", outputs.shape)
    print("masks shape:", masks.shape)
    print("weight_maps shape:", weight_maps.shape)

    _, _, H, W = outputs.shape
    masks = center_crop_tensor(masks, H, W)
    weight_maps = center_crop_tensor(weight_maps, H, W)

    loss = criterion(outputs, masks, weight_maps)
    print("LOSS VALUE:", loss.item())

    loss.backward()
    optimizer.step()

    break
