"""
Two-phase transfer-learning training for the four backbones.

Phase 1 : backbone frozen, train the new classification head (fast, high LR).
Phase 2 : unfreeze the top `FINE_TUNE_UNFREEZE` conv layers, fine-tune (low LR).

Class weights compensate the benign/malignant imbalance.  For each model we
save: the trained weights, the full training history and a small JSON of
computational metrics (wall-clock training time, #params, #epochs).

Run:  python train.py                 # trains every model in config.MODEL_NAMES
      python train.py MobileNetV2     # train a single model
      python train.py --epochs 2      # quick smoke test (overrides both phases)
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")   # Keras 2 before TF import
import sys
import json
import time
import argparse
import numpy as np
import tensorflow as tf

import config as C
C.enable_gpu_memory_growth()
import data_loader as D
from models import build_model, compile_model, set_backbone_trainable


# --------------------------------------------------------------------------- #
# Optional geometric augmentation (rotation / zoom / translation) used when
# config.STRONG_AUG is on -- an effective overfitting curb on small datasets.
_geo_aug = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.08, fill_mode="reflect"),
    tf.keras.layers.RandomZoom(0.10, fill_mode="reflect"),
    tf.keras.layers.RandomTranslation(0.06, 0.06, fill_mode="reflect"),
], name="geo_aug")


def _augment(image, label):
    """Augmentation appropriate for grayscale breast ultrasound."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=15.0)     # [0,255] scale
    image = tf.image.random_contrast(image, 0.9, 1.1)
    if C.STRONG_AUG:
        image = _geo_aug(image, training=True)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label


def make_datasets(cache, batch_size=C.BATCH_SIZE):
    AUTOTUNE = tf.data.AUTOTUNE
    tr = (tf.data.Dataset.from_tensor_slices(
              (cache["X_train"], cache["y_train"].astype("float32")))
          .shuffle(len(cache["X_train"]), seed=C.SEED)
          .map(_augment, num_parallel_calls=AUTOTUNE)
          .batch(batch_size).prefetch(AUTOTUNE))
    va = (tf.data.Dataset.from_tensor_slices(
              (cache["X_val"], cache["y_val"].astype("float32")))
          .batch(batch_size).prefetch(AUTOTUNE))
    return tr, va


def _callbacks(monitor="val_auc", mode="max", patience=C.EARLY_STOP_PATIENCE):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, mode=mode, patience=patience,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, mode=mode, factor=0.3,
            patience=C.REDUCE_LR_PATIENCE, min_lr=1e-7, verbose=1),
    ]


def _merge_history(h1, h2):
    out = {}
    for k in set(list(h1.keys()) + list(h2.keys())):
        merged = list(h1.get(k, [])) + list(h2.get(k, []))
        out[k] = [float(v) for v in merged]      # ensure JSON-serializable
    return out


def train_one(name, cache, class_weight, head_epochs, ft_epochs,
              num_classes=1, weight_suffix="", patience=C.EARLY_STOP_PATIENCE):
    """
    Train one backbone.  num_classes=1 -> binary (BUS-BRA); >1 -> softmax (FETAL).
    weight_suffix distinguishes the two modalities' checkpoints/results.
    """
    tag = name + weight_suffix
    print("\n" + "=" * 78)
    print(f"  TRAINING  {tag}   (num_classes={num_classes})")
    print("=" * 78)
    C.set_global_seed()
    tf.keras.backend.clear_session()

    # binary ranks on val AUC; multiclass on val accuracy
    monitor, mode = (("val_auc", "max") if num_classes == 1
                     else ("val_accuracy", "max"))

    tr_ds, va_ds = make_datasets(cache, batch_size=C.batch_for(name))
    model, backbone, last_conv = build_model(name, num_classes=num_classes)

    t0 = time.time()

    # ---- phase 1 : frozen backbone -----------------------------------------
    set_backbone_trainable(backbone, trainable=False)
    compile_model(model, lr=C.HEAD_LR, num_classes=num_classes)
    print(f"[{tag}] Phase 1 (frozen backbone) - {head_epochs} epochs")
    h1 = model.fit(tr_ds, validation_data=va_ds, epochs=head_epochs,
                   class_weight=class_weight,
                   callbacks=_callbacks(monitor, mode, patience), verbose=2)

    # ---- phase 2 : fine-tune top of backbone -------------------------------
    # ft_epochs <= 0 -> head-only training (used when the backbone cannot be
    # back-propagated in the available VRAM, e.g. ResNet50/EfficientNet at 224
    # on a 2 GB GPU); the frozen-backbone head is still competitive on fetal.
    if ft_epochs and ft_epochs > 0:
        set_backbone_trainable(backbone, trainable=True,
                               unfreeze_top=C.FINE_TUNE_UNFREEZE)
        compile_model(model, lr=C.FINE_TUNE_LR, num_classes=num_classes)
        n_train = sum(int(tf.size(w)) for w in model.trainable_weights)
        print(f"[{tag}] Phase 2 (fine-tune top {C.FINE_TUNE_UNFREEZE}) - "
              f"{ft_epochs} epochs | trainable params ~ {n_train:,}")
        h2 = model.fit(tr_ds, validation_data=va_ds, epochs=ft_epochs,
                       class_weight=class_weight,
                       callbacks=_callbacks(monitor, mode, patience), verbose=2)
    else:
        print(f"[{tag}] Head-only (backbone frozen; VRAM-limited fine-tune skip)")
        h2 = type("H", (), {"history": {}})()   # empty history stand-in
        n_train = sum(int(tf.size(w)) for w in model.trainable_weights)

    train_time = time.time() - t0

    # ---- persist ------------------------------------------------------------
    w_path = C.weights_path(name, weight_suffix)
    model.save_weights(w_path)

    history = _merge_history(h1.history, h2.history)
    with open(os.path.join(C.RESULTS_DIR, f"{C.tagged(tag)}_history.json"), "w") as f:
        json.dump(history, f)

    comp = {
        "model": name, "modality_tag": tag, "num_classes": num_classes,
        "total_params": int(model.count_params()),
        "trainable_params_phase2": int(n_train),
        "epochs_run": len(history.get("loss", [])),
        "train_time_sec": round(train_time, 2),
        "last_conv_layer": last_conv,
        "weights_file": os.path.basename(w_path),
        "weights_size_mb": round(os.path.getsize(w_path) / 1e6, 2),
    }
    with open(os.path.join(C.RESULTS_DIR, f"{C.tagged(tag)}_compute.json"), "w") as f:
        json.dump(comp, f, indent=2)

    print(f"[{tag}] done in {train_time/60:.1f} min | weights -> {w_path}")
    tf.keras.backend.clear_session()
    return comp


def train_fetal(names=None):
    """Train the four backbones on FETAL_PLANES_DB (6-class softmax)."""
    names = names or C.MODEL_NAMES
    cache = D.build_or_load_fetal_cache()
    cw = D.compute_class_weights(cache["y_train"], num_classes=6)
    print("FETAL class weights:", {k: round(v, 3) for k, v in cw.items()})
    summary = []
    for name in names:
        summary.append(train_one(
            name, cache, cw, C.FETAL_HEAD_EPOCHS, C.FETAL_FINE_TUNE_EPOCHS,
            num_classes=6, weight_suffix="_FETAL",
            patience=C.FETAL_EARLY_STOP_PATIENCE))
    with open(os.path.join(C.RESULTS_DIR, "compute_summary_FETAL.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nAll FETAL training finished.")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=None,
                    help="subset of model names (default: all)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override both phase epoch counts (smoke test)")
    ap.add_argument("--modality", choices=["BUS-BRA", "FETAL"], default="BUS-BRA")
    args = ap.parse_args()

    names = args.models if args.models else C.MODEL_NAMES

    if args.modality == "FETAL":
        train_fetal(names)
        return

    head_e = args.epochs if args.epochs else C.HEAD_EPOCHS
    ft_e = args.epochs if args.epochs else C.FINE_TUNE_EPOCHS
    cache = D.build_or_load_cache()
    class_weight = D.compute_class_weights(cache["y_train"])
    print("Class weights:", class_weight)

    summary = []
    for name in names:
        summary.append(train_one(name, cache, class_weight, head_e, ft_e))

    with open(os.path.join(C.RESULTS_DIR, "compute_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nAll training finished. Compute summary saved.")


if __name__ == "__main__":
    main()
