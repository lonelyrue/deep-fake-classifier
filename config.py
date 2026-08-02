"""Central configuration for training and inference."""


class Config:
    # Data paths (adjust to your local / Colab layout)
    TRAIN_IMG_PATH = "dataset/train_images/"
    TEST_IMG_PATH = "dataset/test_images/"
    TRAIN_LABELS_PATH = "dataset/train_solution.csv"

    # Image / training hyperparameters
    IMG_SIZE = 256
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    VAL_SIZE = 0.15
    USE_WEIGHTED_LOSS = True  # weight the loss to handle class imbalance

    SEED = 42
    MODEL_SAVE_PATH = "best_model.pth"
