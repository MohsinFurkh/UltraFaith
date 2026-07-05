"""
Post-hoc saliency / attribution analysis for every trained model.

Four complementary explainers are computed for the same set of test images so
the models can be compared qualitatively:

  * Grad-CAM              (gradient-weighted class activation map)      - tf-keras-vis
  * Score-CAM             (gradient-free, mask-perturbation CAM)        - tf-keras-vis
  * Integrated Gradients  (axiomatic input attribution, black baseline)- custom
  * SHAP                  (GradientExplainer expected-gradient values)  - shap

For every model a figure is produced with
      rows  = sample test images
      cols  = [Original | Grad-CAM | Score-CAM | Integrated Gradients | SHAP]
Additionally a cross-model Grad-CAM comparison figure is saved.
All figures are written at 300 dpi.

Run:  python saliency.py
"""
import os
# The saliency stage runs on CPU: Score-CAM / Integrated-Gradients / SHAP need
# large intermediate batches that do not fit on a small (2 GB) GPU, and speed is
# irrelevant for a handful of images.  Hiding the GPU also guarantees no OOM.
# (Must be set before TensorFlow is imported.)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# NOTE: shap pulls in torch; on Windows torch's DLLs must be loaded BEFORE
# TensorFlow or the load order clashes (WinError 127).  Import shap first.
import shap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

import config as C
C.enable_gpu_memory_growth()
import data_loader as D
from models import build_model

# tf-keras-vis (Grad-CAM, Score-CAM)
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import BinaryScore

plt.rcParams["savefig.dpi"] = C.PLOT_DPI
N_SAMPLES = 4          # sample test images used for the panels
IG_STEPS = 50
SHAP_BG = 40           # background images for SHAP GradientExplainer


# --------------------------------------------------------------------------- #
def _pick_samples(cache):
    """Deterministically pick a mix of malignant + benign test images."""
    y = cache["y_test"].astype(int)
    mal = np.where(y == 1)[0]
    ben = np.where(y == 0)[0]
    k = N_SAMPLES // 2
    idx = np.concatenate([mal[:k], ben[:N_SAMPLES - k]])
    return idx


def _overlay(gray, heat, cmap="jet", alpha=0.5):
    """Alpha-blend a [0,1] heatmap over a grayscale [0,255] image -> RGB."""
    import matplotlib.cm as cm
    g = gray.squeeze() / 255.0
    rgb = np.stack([g, g, g], axis=-1)
    hm = cm.get_cmap(cmap)(np.clip(heat, 0, 1))[..., :3]
    return np.clip((1 - alpha) * rgb + alpha * hm, 0, 1)


def _norm(a):
    a = np.asarray(a, dtype=np.float32)
    a = a - a.min()
    m = a.max()
    return a / m if m > 0 else a


# --------------------------------------------------------------------------- #
def integrated_gradients(model, x, target_is_positive, steps=IG_STEPS, chunk=8):
    """
    Integrated Gradients w.r.t. a black baseline (all-zero image).
    The interpolation path is processed in small chunks so peak memory stays
    bounded (a single 50-image batch OOMs the larger backbones).
    """
    x = tf.convert_to_tensor(x[None].astype("float32"))   # (1,H,W,1)
    baseline = tf.zeros_like(x)
    alphas = tf.linspace(0.0, 1.0, steps)
    grad_sum = tf.zeros_like(x)                           # (1,H,W,1)

    for start in range(0, steps, chunk):
        a = tf.reshape(alphas[start:start + chunk], (-1, 1, 1, 1))
        interp = baseline + a * (x - baseline)           # (c,H,W,1)
        with tf.GradientTape() as tape:
            tape.watch(interp)
            p = model(interp, training=False)[:, 0]       # malignant prob
            score = p if target_is_positive else (1.0 - p)
        g = tape.gradient(score, interp)                 # (c,H,W,1)
        grad_sum += tf.reduce_sum(g, axis=0, keepdims=True)

    avg_grads = grad_sum[0] / float(steps)               # (H,W,1)
    ig = (x[0] - baseline[0]) * avg_grads                # (H,W,1)
    return _norm(tf.reduce_sum(tf.abs(ig), axis=-1).numpy())


def shap_map(explainer, x, target_is_positive):
    """Expected-gradient (SHAP) attribution magnitude for one image."""
    sv = explainer.shap_values(x[None].astype("float32"))
    # single-output model -> list with one element, shape (1,H,W,1)
    vals = sv[0] if isinstance(sv, list) else sv
    vals = np.asarray(vals)[0]                            # (H,W,1)
    m = np.abs(vals).sum(axis=-1)                        # magnitude
    return _norm(m)


# --------------------------------------------------------------------------- #
def run_for_model(name, cache, sample_idx, gradcam_store):
    print(f"[saliency] {name} ...")
    tf.keras.backend.clear_session()
    model, _, last_conv = build_model(name)
    model.load_weights(C.weights_path(name))

    X_test, y_test = cache["X_test"], cache["y_test"].astype(int)
    samples = X_test[sample_idx]
    proba = model.predict(samples, verbose=0).ravel()
    preds = (proba >= 0.5).astype(int)

    # CAM explainers (ReplaceToLinear -> use the logit, not the sigmoid prob)
    gradcam = Gradcam(model, model_modifier=ReplaceToLinear(), clone=True)
    scorecam = Scorecam(model, model_modifier=ReplaceToLinear(), clone=True)

    # SHAP background from the training set
    rng = np.random.RandomState(C.SEED)
    bg_idx = rng.choice(len(cache["X_train"]),
                        size=min(SHAP_BG, len(cache["X_train"])), replace=False)
    background = cache["X_train"][bg_idx].astype("float32")
    explainer = shap.GradientExplainer(model, background)

    methods = ["Original", "Grad-CAM", "Score-CAM",
               "Integrated Grad.", "SHAP"]
    fig, axes = plt.subplots(len(sample_idx), len(methods),
                             figsize=(2.4 * len(methods), 2.6 * len(sample_idx)))
    axes = np.atleast_2d(axes)

    for r, i in enumerate(sample_idx):
        img = X_test[i]
        pred_pos = bool(preds[r])
        score = BinaryScore(pred_pos)             # explain the predicted class

        def _safe(fn, label):
            try:
                return _norm(fn())
            except Exception as e:                      # keep the panel, note it
                print(f"    [{name}] {label} failed: {str(e)[:80]}")
                return np.zeros((C.IMG_SIZE, C.IMG_SIZE), dtype=np.float32)

        gcam = _safe(lambda: gradcam(score, samples[r][None],
                                     penultimate_layer=last_conv)[0], "Grad-CAM")

        # max_N caps the number of activation channels Score-CAM perturbs (for
        # speed).  tf-keras-vis rejects a max_N above the layer's channel count,
        # so fall back to progressively smaller caps until one is accepted.
        def _run_scorecam():
            for mN in (64, 32, 16, None):
                try:
                    return scorecam(score, samples[r][None],
                                    penultimate_layer=last_conv, max_N=mN)[0]
                except ValueError:
                    continue
            raise ValueError("Score-CAM: no valid max_N")
        scam = _safe(_run_scorecam, "Score-CAM")
        ig = _safe(lambda: integrated_gradients(model, img, pred_pos), "IG")
        sh = _safe(lambda: shap_map(explainer, img, pred_pos), "SHAP")

        panels = [None, gcam, scam, ig, sh]
        for c, (title, heat) in enumerate(zip(methods, panels)):
            ax = axes[r, c]
            if heat is None:
                ax.imshow(img.squeeze(), cmap="gray")
            else:
                ax.imshow(_overlay(img, heat))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=10)
            if c == 0:
                ok = preds[r] == y_test[i]
                ax.set_ylabel(
                    f"T:{C.CLASS_NAMES[y_test[i]]}\n"
                    f"P:{C.CLASS_NAMES[preds[r]]} ({proba[r]:.2f})",
                    fontsize=8, color="green" if ok else "red")
        gradcam_store.setdefault(i, {})[name] = (_norm(gcam), img,
                                                 y_test[i], preds[r], proba[r])

    fig.suptitle(f"Post-hoc saliency maps - {name}", y=1.005, fontsize=13)
    fig.tight_layout()
    out = os.path.join(C.SALIENCY_DIR, f"saliency_{name}.png")
    fig.savefig(out, dpi=C.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  saved", os.path.basename(out))


def cross_model_gradcam(gradcam_store):
    """One figure: rows = sample images, cols = models (Grad-CAM overlays)."""
    if not gradcam_store:
        return
    img_ids = list(gradcam_store.keys())
    models = C.MODEL_NAMES
    fig, axes = plt.subplots(len(img_ids), len(models) + 1,
                             figsize=(2.4 * (len(models) + 1), 2.6 * len(img_ids)))
    axes = np.atleast_2d(axes)
    for r, iid in enumerate(img_ids):
        # original in first column
        any_model = next(iter(gradcam_store[iid].values()))
        _, img, ytrue, _, _ = any_model
        axes[r, 0].imshow(img.squeeze(), cmap="gray")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(f"True:\n{C.CLASS_NAMES[ytrue]}", fontsize=8)
        if r == 0:
            axes[r, 0].set_title("Original", fontsize=10)
        for c, name in enumerate(models, start=1):
            ax = axes[r, c]
            if name in gradcam_store[iid]:
                gcam, img, ytrue, pred, prob = gradcam_store[iid][name]
                ax.imshow(_overlay(img, gcam))
                ok = pred == ytrue
                ax.set_xlabel(f"{C.CLASS_NAMES[pred]} ({prob:.2f})",
                              fontsize=7, color="green" if ok else "red")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=10)
    fig.suptitle("Grad-CAM comparison across models", y=1.005, fontsize=13)
    fig.tight_layout()
    out = os.path.join(C.SALIENCY_DIR, "gradcam_cross_model.png")
    fig.savefig(out, dpi=C.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  saved", os.path.basename(out))


def main(models=None, do_cross=True):
    cache = D.build_or_load_cache()
    sample_idx = _pick_samples(cache)
    print("Sample test indices:", sample_idx.tolist())
    names = models if models else C.MODEL_NAMES
    gradcam_store = {}
    for name in names:
        if not os.path.exists(C.weights_path(name)):
            print(f"[saliency] skipping {name} (no weights)")
            continue
        run_for_model(name, cache, sample_idx, gradcam_store)
    if do_cross:
        cross_model_gradcam(gradcam_store)
    print(f"\nSaliency figures saved under: {C.SALIENCY_DIR}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help="subset of model names (default all)")
    ap.add_argument("--no-cross", action="store_true",
                    help="skip the cross-model Grad-CAM panel")
    a = ap.parse_args()
    main(models=a.models or None, do_cross=not a.no_cross)
