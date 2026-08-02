# Deepfake Face Detection — Real vs. StyleGAN

Binary image classifier distinguishing **real faces** (Flickr photographs) from
**synthetic faces** (generated with StyleGAN). Built as the final project of a
computer-vision course, in a Kaggle-style competition where using pre-built
architectures as the final solution was **not allowed** — the model had to be a
custom implementation with a reproducible training pipeline.

**Final result:** F1 ≈ 0.98 on the competition leaderboard.
**Role:** team lead (team of 3).

---

## Problem

- **Data:** 50,000 training images and 10,000 test images, 256×256 RGB.
  Real faces from Flickr; synthetic faces from StyleGAN.
- **Challenges (given in the task):** class imbalance and artificial artefacts/noise in the images.
- **Metric:** F1-score, with an explicit emphasis on **recall** — the task
  penalises false negatives (a deepfake classified as real) more than false
  positives, so the model must minimise missed fakes while keeping precision reasonable.

## Approach

### Architecture explored: F3-Net (frequency domain)
The first direction I studied was **F3-Net** (Frequency in Face Forgery Network),
a two-stream network that classifies from an image's **frequency representation**
rather than raw RGB. The motivation: under compression (JPEG/H.264), the subtle
spatial artefacts of GAN generation are smoothed out and become invisible to
standard spatial CNNs — but in the frequency domain (via DCT) forgery traces
survive compression, appearing as anomalously strong signals at specific
frequencies. F3-Net combines a Frequency-Aware Decomposition stream and a Local
Frequency Statistics stream, fused with cross-attention.

This was a valuable study of *why* frequency-domain detection is robust, and is
documented in the project report. For the final submission, however, a
spatial-attention model proved more stable and gave a better score on this
particular dataset.

### Final model: SE-ResNet18 (spatial domain, custom)
A **ResNet-18 backbone with Squeeze-and-Excitation (SE) channel-attention blocks**
inserted into every residual block, plus a regularised classification head.
- **SE blocks** let the network learn per-channel importance and suppress noisy
  channels — helpful given the artificial artefacts in the data.
- **Custom head:** dropout → FC(512) → ReLU → dropout → FC(1), for regularisation.
- Single-logit output, trained with `BCEWithLogitsLoss`.

### Handling imbalance and the recall objective
- **Weighted loss:** inverse-frequency class weights via `pos_weight`, so the
  rarer class is not ignored.
- **Threshold tuning:** the decision threshold is not fixed at 0.5 but searched
  on the validation set to maximise F1 (see `inference.py`).
- **Moderate augmentation:** flips, brightness/contrast, mild blur/noise and
  coarse dropout increase robustness to artefacts — kept deliberately moderate so
  the subtle GAN traces the model relies on are not destroyed.

### Training and inference details
- Optimiser **AdamW** + **CosineAnnealingLR** schedule.
- Stratified train/validation split; fixed seed and deterministic mode for reproducibility.
- Best checkpoint saved with weights **and** experiment metadata.
- **Test-Time Augmentation (TTA):** each test image is passed several times
  (original, horizontal flip, slight brightness up/down) and the sigmoid
  probabilities are averaged, reducing prediction variance without retraining.

## Diagnostics

The training script logs Loss / F1 / Recall / Precision per epoch. Additional
diagnostics (in the notebook): training curves, a confidence-distribution
histogram (a well-calibrated model shows two peaks near 0 and 1 rather than a
smear in the middle), and qualitative real/fake predictions on sample images.

## Repository structure

```
model.py        SE-ResNet18 architecture (SE block, block wrapper, model, head)
data.py         Dataset + train / val / TTA augmentation pipelines
train.py        Training loop (weighted loss, AdamW + cosine LR, checkpointing)
inference.py    Validation threshold search + TTA inference -> submission.csv
visualize.py    Diagnostics: training curves, sample predictions, confidence histogram
config.py       Paths and hyperparameters
requirements.txt
```

## Usage

```bash
pip install -r requirements.txt

# 1. Place the dataset under ./dataset (train_images/, test_images/, train_solution.csv)
#    and set paths in config.py.
# 2. Train:
python train.py         # produces best_model.pth
# 3. Generate predictions:
python inference.py      # produces submission.csv
```

**Note on data and weights:** the dataset (Flickr + StyleGAN images) is not
redistributed here for licensing reasons — the code reads it from the competition
source. Trained weights (`best_model.pth`) are excluded from version control due
to size and are available on request.

## What I would explore next
- Return to the frequency-domain direction and **fuse** spatial (SE-ResNet) and
  frequency (F3-Net) features, rather than choosing one.
- Evaluate **cross-generator generalisation** — training on StyleGAN and testing
  against other generators — which is the central open problem in deepfake
  detection: detectors that work on one generator degrade against unseen ones.
