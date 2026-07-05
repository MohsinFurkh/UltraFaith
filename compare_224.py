"""
Direct 128 vs 224 comparison of Grad-CAM sharpness / lesion localisation.

Builds each backbone at BOTH input sizes in one process (the conv weights are
spatial-size agnostic, so the 128 and 224 checkpoints load into 128- and
224-input models respectively), computes Grad-CAM for the predicted class, and:
  * measures the energy-pointing-game rho (saliency mass inside the lesion mask)
    over a test sample -> quantifies whether the CAM is better localised;
  * renders a side-by-side qualitative panel (orig | 128 Grad-CAM | 224 Grad-CAM).
Also reports validation-tuned test accuracy (removing the label-smoothing
threshold artefact).
"""
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

import config as C
C.enable_gpu_memory_growth()
from tensorflow.keras.applications import (
    EfficientNetB4, MobileNetV2, ResNet50, DenseNet121)
from tensorflow.keras.applications.efficientnet import preprocess_input as pp_eff
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as pp_mob
from tensorflow.keras.applications.resnet50 import preprocess_input as pp_res
from tensorflow.keras.applications.densenet import preprocess_input as pp_dense
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import BinaryScore

_BB = {"EfficientNetB4": (EfficientNetB4, pp_eff),
       "MobileNetV2": (MobileNetV2, pp_mob),
       "ResNet50": (ResNet50, pp_res),
       "DenseNet121": (DenseNet121, pp_dense)}
N_RHO = 150            # test images for the rho estimate
N_SHOW = 4             # images in the qualitative panel
MODELS_DIR = C.MODELS_DIR


def build_at(name, size):
    cls, pre = _BB[name]
    inp = layers.Input((size, size, 1))
    x = layers.Concatenate()([inp, inp, inp])
    x = layers.Lambda(pre)(x)
    base = cls(include_top=False, weights=None, input_tensor=x)
    y = layers.GlobalAveragePooling2D()(base.output)
    y = layers.Dropout(0.0)(y)
    out = layers.Dense(1, activation="sigmoid", name="predictions")(y)
    m = Model(inp, out, name=name)
    last = [l.name for l in m.layers if len(l.output_shape) == 4][-1]
    return m, last


def load_cache(size, tag):
    p = os.path.join(C.OUTPUT_DIR, f"dataset_{size}x{size}x1{tag}.npz")
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}


def read_mask(img_id, size):
    fn = "mask_" + str(img_id).split("bus_")[-1] + ".png"
    path = os.path.join(C.DATASET_DIR, "Masks", fn)
    if not os.path.exists(path):
        return None
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def norm(a):
    a = np.asarray(a, np.float32); a -= a.min(); mx = a.max()
    return a / mx if mx > 0 else a


def rho(S, mask):
    s = S.astype(np.float64); d = s.sum()
    return float((s * (mask > 0)).sum() / d) if (d > 0 and mask is not None) else np.nan


def gradcam_maps(model, last, X, preds):
    gc = Gradcam(model, model_modifier=ReplaceToLinear(), clone=True)
    out = []
    for i in range(len(X)):
        cam = gc(BinaryScore(bool(preds[i])), X[i:i+1], penultimate_layer=last)[0]
        out.append(norm(cam))
    return out


def tuned_acc(model, cache):
    """Best-threshold (on val) test accuracy, plus AUROC/F1."""
    pv = model.predict(cache["X_val"], batch_size=8, verbose=0).ravel()
    yv = cache["y_val"].astype(int)
    ths = np.linspace(0.05, 0.95, 91)
    best = max(ths, key=lambda t: accuracy_score(yv, (pv >= t).astype(int)))
    pt = model.predict(cache["X_test"], batch_size=8, verbose=0).ravel()
    yt = cache["y_test"].astype(int)
    pred = (pt >= best).astype(int)
    return (accuracy_score(yt, pred) * 100, f1_score(yt, pred) * 100,
            roc_auc_score(yt, pt), best)


def main():
    c128 = load_cache(128, "")
    c224 = load_cache(224, "_224")
    ids = c128["ids_test"]
    rng = np.random.RandomState(C.SEED)
    idx_rho = rng.choice(len(ids), size=min(N_RHO, len(ids)), replace=False)
    # sample: 2 malignant + 2 benign, correctly localisable
    y = c128["y_test"].astype(int)
    show = np.concatenate([np.where(y == 1)[0][:2], np.where(y == 0)[0][:2]])

    summary = []
    panel = {}
    for name in C.MODEL_NAMES:
        row = {"backbone": name}
        for size, cache, tag in [(128, c128, ""), (224, c224, "_224")]:
            wp = os.path.join(MODELS_DIR, f"{name}{tag}.weights.h5")
            if not os.path.exists(wp):
                continue
            tf.keras.backend.clear_session()
            model, last = build_at(name, size)
            model.load_weights(wp)
            # rho over sample
            Xs = cache["X_test"][idx_rho].astype("float32")
            ps = (model.predict(Xs, batch_size=8, verbose=0).ravel() >= 0.5).astype(int)
            cams = gradcam_maps(model, last, Xs, ps)
            rhos = [rho(cams[j], read_mask(ids[idx_rho[j]], size))
                    for j in range(len(idx_rho))]
            row[f"rho_{size}"] = float(np.nanmean(rhos))
            acc, f1, auc, th = tuned_acc(model, cache)
            row[f"acc_{size}"] = acc; row[f"auc_{size}"] = auc
            # panel maps for the shown images
            Xsh = cache["X_test"][show].astype("float32")
            psh = (model.predict(Xsh, batch_size=8, verbose=0).ravel() >= 0.5).astype(int)
            panel[(name, size)] = (Xsh, gradcam_maps(model, last, Xsh, psh),
                                   [read_mask(ids[k], size) for k in show])
        summary.append(row)

    print("\n===== Grad-CAM localisation rho and tuned accuracy: 128 vs 224 =====")
    print("%-15s %8s %8s %8s | %7s %7s" %
          ("Backbone", "rho128", "rho224", "d_rho", "acc128", "acc224"))
    for r in summary:
        print("%-15s %8.3f %8.3f %+8.3f | %6.1f  %6.1f" %
              (r["backbone"], r.get("rho_128", np.nan), r.get("rho_224", np.nan),
               r.get("rho_224", np.nan) - r.get("rho_128", np.nan),
               r.get("acc_128", np.nan), r.get("acc_224", np.nan)))

    # ---- qualitative panel: rows=images, cols=[orig|128|224] per backbone ----
    _render_panel(panel, show, c128, c224)
    import json
    json.dump(summary, open(os.path.join(C.RESULTS_DIR,
              "compare_128_vs_224.json"), "w"), indent=2)


def _overlay(gray, heat, a=0.5):
    import matplotlib.cm as cm
    g = gray.squeeze() / 255.0
    rgb = np.stack([g, g, g], -1)
    hm = cm.get_cmap("jet")(np.clip(heat, 0, 1))[..., :3]
    return np.clip((1 - a) * rgb + a * hm, 0, 1)


def _render_panel(panel, show, c128, c224):
    names = C.MODEL_NAMES
    ncol = 1 + 2 * len(names)                      # orig + (128,224) per backbone
    fig, axes = plt.subplots(len(show), ncol,
                             figsize=(2.1 * ncol, 2.2 * len(show)))
    axes = np.atleast_2d(axes)
    for r in range(len(show)):
        g = c224["X_test"][show[r]].squeeze()
        axes[r, 0].imshow(g, cmap="gray"); axes[r, 0].set_ylabel(f"img {show[r]}", fontsize=8)
        if r == 0:
            axes[r, 0].set_title("input", fontsize=9)
        for bi, name in enumerate(names):
            for si, size in enumerate((128, 224)):
                col = 1 + bi * 2 + si
                ax = axes[r, col]
                if (name, size) in panel:
                    Xsh, cams, masks = panel[(name, size)]
                    ax.imshow(_overlay(Xsh[r], cams[r]))
                    if masks[r] is not None and masks[r].sum() > 0:
                        ax.contour(masks[r], levels=[0.5], colors="lime", linewidths=0.7)
                if r == 0:
                    ax.set_title(f"{name}\n{size}", fontsize=7)
        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Grad-CAM: 128 vs 224 (green = lesion mask)", y=1.01)
    fig.tight_layout()
    p = os.path.join(C.SALIENCY_DIR, "gradcam_128_vs_224.png")
    fig.savefig(p, dpi=C.PLOT_DPI, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    main()
