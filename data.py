"""Dataset and augmentation pipelines."""

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DeepfakeDataset(Dataset):
    """Loads face images and (optionally) their real/fake labels."""

    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)["image"]

        if self.labels is not None:
            return image, self.labels[idx]
        return image


def get_train_transforms():
    """Training-time augmentations.

    Kept moderate on purpose: blur / noise / dropout increase robustness to the
    dataset's artificial artefacts, but overly aggressive transforms would
    destroy the subtle GAN traces the model needs to detect.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
        ], p=0.3),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms():
    """Validation / plain inference transforms: normalisation only."""
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_tta_transforms():
    """Test-Time Augmentation set: original + horizontal flip + slight
    brightness up / down. Predictions are averaged across these passes to
    reduce variance without retraining."""
    def _norm(*extra):
        return A.Compose([*extra, A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])

    return [
        _norm(),                                                      # original
        _norm(A.HorizontalFlip(p=1.0)),                               # flip
        _norm(A.RandomBrightnessContrast(brightness_limit=(0.1, 0.1), contrast_limit=0, p=1.0)),   # brighter
        _norm(A.RandomBrightnessContrast(brightness_limit=(-0.1, -0.1), contrast_limit=0, p=1.0)), # darker
    ]
