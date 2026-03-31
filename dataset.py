import os
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import TrivialAugmentWide, RandAugment
from collections import Counter


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class ImageDataset(Dataset):

    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        classes = sorted(
            [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))],
            key=lambda x: int(x),
        )
        self.class_to_idx = {c: int(c) for c in classes}
        for cls in classes:
            for fname in os.listdir(os.path.join(root, cls)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(
                        (os.path.join(root, cls, fname), int(cls))
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_labels(self):
        return [s[1] for s in self.samples]


class TestDataset(Dataset):

    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.files = sorted(
            f for f in os.listdir(root)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(os.path.join(self.root, fname)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, os.path.splitext(fname)[0]


# ---------------------------------------------------------------------------
# Balanced Sampler
# ---------------------------------------------------------------------------

def make_balanced_sampler(dataset: ImageDataset) -> WeightedRandomSampler:
    labels = dataset.get_labels()
    counts = Counter(labels)
    weights = [1.0 / counts[lbl] for lbl in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def get_train_transform(img_size: int = 224, use_trivial: bool = True) -> T.Compose:
    auto_aug = TrivialAugmentWide() if use_trivial else RandAugment(num_ops=2, magnitude=9)
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.08, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(p=0.1),
        auto_aug,
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
        T.RandomErasing(p=0.3, scale=(0.02, 0.25)),
    ])


def get_val_transform(img_size: int = 224) -> T.Compose:
    resize_size = int(img_size * 256 / 224)
    return T.Compose([
        T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ])


def get_tta_transforms(img_size: int = 224) -> list:
    norm = T.Compose([T.ToTensor(), T.Normalize(_MEAN, _STD)])
    rs = int(img_size * 256 / 224)
    rs_large = int(img_size * 288 / 224)

    def _make(resize, crop, hflip=False, vflip=False):
        ops = [T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
               T.CenterCrop(crop)]
        if hflip:
            ops.append(T.RandomHorizontalFlip(p=1.0))
        if vflip:
            ops.append(T.RandomVerticalFlip(p=1.0))
        ops += [T.ToTensor(), T.Normalize(_MEAN, _STD)]
        return T.Compose(ops)

    return [
        _make(rs, img_size),
        _make(rs, img_size, hflip=True),
        _make(rs_large, img_size),
        _make(rs_large, img_size, hflip=True),
        _make(rs, img_size, vflip=True),
        _make(rs_large, img_size, hflip=True, vflip=True),
    ]


# ---------------------------------------------------------------------------
# MixUp / CutMix
# ---------------------------------------------------------------------------

def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, h, w = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(h * cut_rat), int(w * cut_rat)
    cx, cy = np.random.randint(w), np.random.randint(h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (w * h)
    return mixed, y, y[idx], lam


def mixup_criterion(criterion, pred, ya, yb, lam):
    return lam * criterion(pred, ya) + (1 - lam) * criterion(pred, yb)
