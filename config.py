"""
Central configuration for the BUS-BRA breast-ultrasound classification study.

Task: binary classification  benign (0)  vs  malignant (1)
Input: 128 x 128 x 1  (grayscale, as requested)
Models: EfficientNetB4, MobileNetV2, ResNet50, DenseNet121
"""
import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Project root = folder that contains this file
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset roots.  Override with environment variables when the datasets live
# elsewhere, e.g.  export BUSBRA_DIR=/path/to/BUS-BRA
#                  export FETAL_DIR=/path/to/FETAL_PLANES_DB
DATASET_DIR = os.environ.get(
    "BUSBRA_DIR", os.path.abspath(os.path.join(PROJECT_DIR, "..", "BUS-BRA")))
IMAGES_DIR = os.path.join(DATASET_DIR, "Images")
CSV_PATH = os.path.join(DATASET_DIR, "bus_data.csv")

# Second modality: FETAL_PLANES_DB (fetal plane identification, 6 classes)
FETAL_DIR = os.environ.get(
    "FETAL_DIR", os.path.abspath(os.path.join(PROJECT_DIR, "..", "Fetal US Dataset")))
FETAL_IMAGES_DIR = os.path.join(FETAL_DIR, "Images")
FETAL_CSV_PATH = os.path.join(FETAL_DIR, "FETAL_PLANES_DB_data.csv")

# All generated artefacts live here
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
SALIENCY_DIR = os.path.join(OUTPUT_DIR, "saliency")
DATA_CACHE = os.path.join(OUTPUT_DIR, "dataset_128x128x1.npz")
FETAL_CACHE = os.path.join(OUTPUT_DIR, "fetal_128x128x1.npz")
# UltraFaith benchmark artefacts
FAITH_DIR = os.path.join(OUTPUT_DIR, "faithfulness")

for _d in (OUTPUT_DIR, MODELS_DIR, PLOTS_DIR, RESULTS_DIR, SALIENCY_DIR, FAITH_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------- #
# Data / image settings
# --------------------------------------------------------------------------- #
IMG_SIZE = 128            # height = width
CHANNELS = 1              # grayscale input (128 x 128 x 1)
NUM_CLASSES = 1           # single sigmoid unit (binary)
CLASS_NAMES = ["benign", "malignant"]
LABEL_COLUMN = "Pathology"          # values: benign / malignant
GROUP_COLUMN = "Case"               # patient id  -> patient-level split
POSITIVE_LABEL = "malignant"        # mapped to 1

# Split ratios (patient-level, stratified)
TEST_FRACTION = 0.20
VAL_FRACTION = 0.15       # fraction of the *whole* dataset

# --------------------------------------------------------------------------- #
# Training hyper-parameters
# --------------------------------------------------------------------------- #
SEED = 42
BATCH_SIZE = 16
# Per-model batch overrides for a small GPU (GTX 1650, ~4 GB).  Larger
# backbones use a smaller batch during fine-tuning to avoid out-of-memory.
BATCH_OVERRIDE = {"EfficientNetB4": 8, "ResNet50": 8}


def batch_for(name):
    return BATCH_OVERRIDE.get(name, BATCH_SIZE)
HEAD_EPOCHS = 8           # phase 1 : frozen backbone, train new head
FINE_TUNE_EPOCHS = 30     # phase 2 : unfreeze top of backbone
HEAD_LR = 1e-3
FINE_TUNE_LR = 1e-5
EARLY_STOP_PATIENCE = 8
REDUCE_LR_PATIENCE = 4
FINE_TUNE_UNFREEZE = 60   # number of top backbone layers to unfreeze in phase 2
DROPOUT = 0.3

# Fetal fine-tuning is capped tighter (12,400 images, small GPU): the classifier
# only needs to be competitive so faithfulness is not confounded (paper Sec 5).
FETAL_HEAD_EPOCHS = 3
FETAL_FINE_TUNE_EPOCHS = 12
FETAL_EARLY_STOP_PATIENCE = 4

# Models to run (name -> builder key in models.py)
MODEL_NAMES = ["EfficientNetB4", "MobileNetV2", "ResNet50", "DenseNet121"]

# --------------------------------------------------------------------------- #
# Modality registry (cross-modality UltraFaith benchmark)
# --------------------------------------------------------------------------- #
# Each modality: number of classes, class names, whether pixel masks exist.
MODALITIES = {
    "BUS-BRA": {
        "num_classes": 1,                        # binary sigmoid (benign/malig)
        "class_names": ["benign", "malignant"],
        "has_masks": True,
        "cache": DATA_CACHE,
        "weight_suffix": "",                     # existing weights: <Model>.weights.h5
    },
    "FETAL": {
        "num_classes": 6,                        # softmax, 6 fetal planes
        "class_names": ["Fetal abdomen", "Fetal brain", "Fetal femur",
                        "Fetal thorax", "Maternal cervix", "Other"],
        "has_masks": False,
        "cache": FETAL_CACHE,
        "weight_suffix": "_FETAL",               # weights: <Model>_FETAL.weights.h5
    },
}
MODALITY_NAMES = ["BUS-BRA", "FETAL"]

# --------------------------------------------------------------------------- #
# UltraFaith faithfulness protocol  (paper, Experimental Setup)
# --------------------------------------------------------------------------- #
FAITH_STEPS = 20               # deletion/insertion removal steps
FAITH_K = 0.20                 # reference removal level for directional agreement
BLUR_SIGMA = 11                # Gaussian-blur baseline sigma
BLUR_KERNEL = 31               # Gaussian-blur kernel size (odd)
IG_STEPS_FAITH = 32            # Integrated Gradients interpolation steps
SHAP_SAMPLES = 32              # GradientSHAP Monte-Carlo samples
SCORECAM_MAX_N = 128           # Score-CAM top activation channels
ATTRIBUTION_METHODS = ["Grad-CAM", "Integrated Gradients", "SHAP", "Score-CAM"]
# Number of test images used for the (compute-heavy) faithfulness sweep per
# modality.  Paper uses the full test set; capped here for the local GPU.
N_FAITH_IMAGES = 120
BOOTSTRAP_RESAMPLES = 1000     # paired bootstrap for CIs

# --------------------------------------------------------------------------- #
# Plot settings
# --------------------------------------------------------------------------- #
PLOT_DPI = 300            # every figure is saved at 300 dpi (requested)

# Reproducibility helper -------------------------------------------------------
def set_global_seed(seed: int = SEED):
    import random
    import numpy as np
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def enable_gpu_memory_growth():
    """Avoid TF grabbing all VRAM up-front (important on small GPUs)."""
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
