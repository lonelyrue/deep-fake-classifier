"""Training pipeline for the SE-ResNet18 deepfake classifier.

Run: python train.py
Produces best_model.pth (checkpoint with weights + experiment metadata) and
returns the training history for plotting.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score

from config import Config
from model import DeepfakeResNet18
from data import DeepfakeDataset, get_train_transforms, get_val_transforms


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # disabled for full reproducibility


def calculate_class_weights(labels):
    """Inverse-frequency class weights to counter class imbalance."""
    unique, counts = np.unique(labels, return_counts=True)
    weights = len(labels) / (len(unique) * counts)
    return torch.FloatTensor(weights)


def _run_epoch(model, dataloader, criterion, device, optimizer=None):
    """One epoch. If optimizer is given -> train mode, else eval mode."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss, all_preds, all_labels = 0.0, [], []
    desc = "Training" if is_train else "Validation"

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in tqdm(dataloader, desc=desc):
            images = images.to(device)
            labels = labels.float().to(device)

            if is_train:
                optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    return (
        epoch_loss,
        f1_score(all_labels, all_preds),
        recall_score(all_labels, all_preds),
        precision_score(all_labels, all_preds),
    )


def train_model():
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load labels and build image paths
    train_df = pd.read_csv(Config.TRAIN_LABELS_PATH, header=None, names=["Id", "target_feature"])
    train_df["image_path"] = train_df["Id"].apply(
        lambda x: os.path.join(Config.TRAIN_IMG_PATH, f"{x}.jpg")
    )
    print(f"Class distribution:\n{train_df['target_feature'].value_counts()}")

    # Stratified train/val split (preserves class ratio)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_df["image_path"].values,
        train_df["target_feature"].values,
        test_size=Config.VAL_SIZE,
        random_state=Config.SEED,
        stratify=train_df["target_feature"].values,
    )
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")

    train_loader = DataLoader(
        DeepfakeDataset(train_paths, train_labels, get_train_transforms()),
        batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        DeepfakeDataset(val_paths, val_labels, get_val_transforms()),
        batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True,
    )

    model = DeepfakeResNet18(pretrained=True, dropout_rate=0.5).to(device)

    # Weighted loss so the rare class is not ignored
    if Config.USE_WEIGHTED_LOSS:
        cw = calculate_class_weights(train_labels).to(device)
        pos_weight = cw[1] / cw[0]
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using weighted loss, pos_weight={pos_weight:.4f}")
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    best_f1 = 0.0
    history = {k: [] for k in [
        "train_loss", "val_loss", "train_f1", "val_f1",
        "train_recall", "val_recall", "train_precision", "val_precision", "lr",
    ]}

    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        tr = _run_epoch(model, train_loader, criterion, device, optimizer)
        vl = _run_epoch(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        for key, val in zip(
            ["train_loss", "train_f1", "train_recall", "train_precision"], tr
        ):
            history[key].append(val)
        for key, val in zip(
            ["val_loss", "val_f1", "val_recall", "val_precision"], vl
        ):
            history[key].append(val)
        history["lr"].append(current_lr)

        print(f"Train - Loss {tr[0]:.4f} F1 {tr[1]:.4f} Recall {tr[2]:.4f} Prec {tr[3]:.4f}")
        print(f"Val   - Loss {vl[0]:.4f} F1 {vl[1]:.4f} Recall {vl[2]:.4f} Prec {vl[3]:.4f}")

        # Save best checkpoint (weights + experiment metadata for reproducibility)
        if vl[1] > best_f1:
            best_f1 = vl[1]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_f1,
                "config": {
                    "seed": Config.SEED, "lr": Config.LEARNING_RATE,
                    "batch_size": Config.BATCH_SIZE, "epochs": Config.EPOCHS,
                    "dropout": 0.5, "architecture": "ResNet18 + SE blocks",
                },
            }, Config.MODEL_SAVE_PATH)
            print(f"  checkpoint saved (Val F1 {best_f1:.4f})")

    print(f"\nBest Val F1: {best_f1:.4f}")
    return model, history


if __name__ == "__main__":
    train_model()
