"""Diagnostic visualisations: training curves, sample predictions, and the
model's confidence distribution.

These are the plots referenced in the README. Run them after training
(`train.py` returns the history dict; a saved checkpoint is needed for the
prediction and confidence plots).
"""

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from model import DeepfakeResNet18
from data import DeepfakeDataset, get_val_transforms


def plot_training_history(history, save_path="training_curves.png"):
    """Four panels: Loss, F1, Recall & Precision, and the LR schedule."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SE-ResNet18 training curves", fontsize=16, fontweight="bold", y=1.01)

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train Loss")
    ax.plot(epochs, history["val_loss"], "r-o", markersize=4, label="Val Loss")
    ax.set_title("Loss (BCEWithLogitsLoss)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(True, alpha=0.3)

    # F1 (mark the best epoch)
    ax = axes[0, 1]
    ax.plot(epochs, history["train_f1"], "b-o", markersize=4, label="Train F1")
    ax.plot(epochs, history["val_f1"], "r-o", markersize=4, label="Val F1")
    best_epoch = int(np.argmax(history["val_f1"])) + 1
    best_f1 = max(history["val_f1"])
    ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best epoch {best_epoch}")
    ax.set_title(f"F1-score (best val: {best_f1:.4f})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1"); ax.legend(); ax.grid(True, alpha=0.3)

    # Recall & Precision together
    ax = axes[1, 0]
    ax.plot(epochs, history["train_recall"], "b-o", markersize=4, label="Train Recall")
    ax.plot(epochs, history["val_recall"], "r-o", markersize=4, label="Val Recall")
    ax.plot(epochs, history["train_precision"], "b--s", markersize=4, label="Train Precision")
    ax.plot(epochs, history["val_precision"], "r--s", markersize=4, label="Val Precision")
    ax.set_title("Recall & Precision")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Learning-rate schedule (cosine annealing)
    ax = axes[1, 1]
    ax.plot(epochs, history["lr"], "g-o", markersize=4)
    ax.set_title("Learning Rate (Cosine Annealing)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR"); ax.set_yscale("log"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


def show_predictions(device, num_samples=8):
    """Qualitative check: real/fake predictions on a few test images."""
    model = DeepfakeResNet18(pretrained=False).to(device)
    ckpt = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_images = sorted(f for f in os.listdir(Config.TEST_IMG_PATH) if f.endswith(".jpg"))[:num_samples]
    transform = get_val_transforms()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for idx, img_name in enumerate(test_images):
        image = Image.open(os.path.join(Config.TEST_IMG_PATH, img_name)).convert("RGB")
        tensor = transform(image=np.array(image))["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(tensor).squeeze()).item()
        pred = int(prob > 0.5)
        axes[idx].imshow(image); axes[idx].axis("off")
        label, color = ("FAKE", "red") if pred == 1 else ("REAL", "green")
        axes[idx].set_title(f"{label} ({prob:.2%})", color=color, fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_confidence_distribution(device, save_path="confidence_distribution.png"):
    """Confidence histogram over the test set.

    A well-calibrated classifier produces two peaks near 0 and 1 (confident
    decisions) rather than a smear of probabilities around 0.5.
    """
    model = DeepfakeResNet18(pretrained=False).to(device)
    ckpt = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_images = sorted(f for f in os.listdir(Config.TEST_IMG_PATH) if f.endswith(".jpg"))
    test_paths = [os.path.join(Config.TEST_IMG_PATH, img) for img in test_images]
    loader = DataLoader(
        DeepfakeDataset(test_paths, labels=None, transform=get_val_transforms()),
        batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2,
    )

    all_probs = []
    with torch.no_grad():
        for images in tqdm(loader, desc="Collecting probabilities"):
            all_probs.extend(torch.sigmoid(model(images.to(device)).squeeze()).cpu().numpy())
    all_probs = np.array(all_probs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(all_probs, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0].axvline(0.5, color="red", linestyle="--", linewidth=1.5, label="threshold=0.5")
    axes[0].set_title("Model confidence distribution")
    axes[0].set_xlabel("P(fake)"); axes[0].set_ylabel("Number of images")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    fake_count = int((all_probs > 0.5).sum())
    real_count = len(all_probs) - fake_count
    axes[1].bar(["REAL", "FAKE"], [real_count, fake_count],
                color=["#2ecc71", "#e74c3c"], edgecolor="white", width=0.5)
    axes[1].set_title("Predicted class distribution")
    axes[1].set_ylabel("Count")
    for i, v in enumerate([real_count, fake_count]):
        axes[1].text(i, v + 30, str(v), ha="center", fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"REAL: {real_count} | FAKE: {fake_count}")
