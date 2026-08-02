"""Inference: validation-tuned threshold + Test-Time Augmentation (TTA).

Run: python inference.py
Loads best_model.pth, finds the F1-optimal decision threshold on the validation
split, runs TTA on the test set, and writes submission.csv.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from config import Config
from model import DeepfakeResNet18
from data import DeepfakeDataset, get_val_transforms, get_tta_transforms


def find_best_threshold(model, val_loader, device):
    """Search the decision threshold that maximises F1 on validation.

    The default 0.5 is rarely optimal under class imbalance and when the task
    penalises false negatives (missed fakes) more than false positives.
    """
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Threshold search"):
            outputs = model(images.to(device)).squeeze()
            all_probs.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs, all_labels = np.array(all_probs), np.array(all_labels)
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.30, 0.71, 0.01):
        f1 = f1_score(all_labels, (all_probs > t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"Best threshold: {best_t:.2f} | Val F1: {best_f1:.4f}")
    return best_t


def create_submission():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepfakeResNet18(pretrained=False).to(device)
    ckpt = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, Val F1 {ckpt['best_val_f1']:.4f}")

    # Rebuild the same validation split to tune the threshold
    train_df = pd.read_csv(Config.TRAIN_LABELS_PATH, header=None, names=["Id", "target_feature"])
    train_df["image_path"] = train_df["Id"].apply(
        lambda x: os.path.join(Config.TRAIN_IMG_PATH, f"{x}.jpg")
    )
    _, val_paths, _, val_labels = train_test_split(
        train_df["image_path"].values, train_df["target_feature"].values,
        test_size=Config.VAL_SIZE, random_state=Config.SEED,
        stratify=train_df["target_feature"].values,
    )
    val_loader = DataLoader(
        DeepfakeDataset(val_paths, val_labels, get_val_transforms()),
        batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2,
    )
    best_threshold = find_best_threshold(model, val_loader, device)

    # Collect test paths / ids
    test_images = sorted(f for f in os.listdir(Config.TEST_IMG_PATH) if f.endswith(".jpg"))
    test_paths = [os.path.join(Config.TEST_IMG_PATH, img) for img in test_images]
    test_ids = [int(img.split(".")[0]) for img in test_images]

    # TTA: average sigmoid probabilities over several light transforms
    tta = get_tta_transforms()
    print(f"Test images: {len(test_paths)}, TTA passes: {len(tta)}")
    accumulated = np.zeros(len(test_paths))

    for i, transform in enumerate(tta):
        loader = DataLoader(
            DeepfakeDataset(test_paths, labels=None, transform=transform),
            batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2,
        )
        pass_probs = []
        with torch.no_grad():
            for images in tqdm(loader, desc=f"TTA pass {i + 1}/{len(tta)}"):
                outputs = model(images.to(device)).squeeze()
                pass_probs.extend(torch.sigmoid(outputs).cpu().numpy())
        accumulated += np.array(pass_probs)

    avg_probs = accumulated / len(tta)
    predictions = (avg_probs > best_threshold).astype(int)

    submission = pd.DataFrame({"id": test_ids, "target_feature": predictions}).sort_values("id")
    submission.to_csv("submission.csv", index=False)
    print(f"Threshold used: {best_threshold:.2f}")
    print(f"Predicted distribution:\n{submission['target_feature'].value_counts()}")
    return submission


if __name__ == "__main__":
    create_submission()
