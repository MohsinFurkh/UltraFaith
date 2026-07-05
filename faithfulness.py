"""
UltraFaith - self-referential faithfulness measurement (paper Section 3).

For one (modality, backbone) this computes, for each of the four attribution
methods (Grad-CAM, Integrated Gradients, GradientSHAP, Score-CAM), the paper's
self-referential faithfulness quantities on a sample of the test set:

  * deletion / insertion trajectories D(t), I(t) against a Gaussian-blur baseline
  * AUC_del (lower better), AUC_ins (higher better)
  * F = AUC_ins - AUC_del                       (primary ranking metric, eq. 5)
  * signed  Delta_faith  at k=20%               (eq. 3)  and the per-image hit
    used for the directional-agreement rate DA  (eq. 4)
  * localisation rho  (energy pointing-game, eq. 6)  where lesion masks exist

Everything is self-referential: the perturbed region and the tracked class are
the model's *own* prediction on the clean image.

Outputs (per config, under outputs/faithfulness/):
  faith_<MOD>_<BB>.csv     per-image, per-method scores
  curves_<MOD>_<BB>.npz    mean D(t)/I(t) per method (for Fig 2)
  qual_<MOD>_<BB>.npz      example images + saliency maps + mask (for Fig 3)

Run:  python faithfulness.py <MODALITY> <BACKBONE>
      e.g. python faithfulness.py BUS-BRA EfficientNetB4
"""
import os
import sys
# shap pulls torch; must precede TensorFlow on Windows (WinError 127).
import shap
import numpy as np
import cv2

import config as C
C.enable_gpu_memory_growth()
import tensorflow as tf
import data_loader as D
from models import build_model

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import BinaryScore, CategoricalScore


# --------------------------------------------------------------------------- #
#  Model probability helper (unifies binary-sigmoid and multiclass-softmax)
# --------------------------------------------------------------------------- #
def model_probs(model, X, batch_size=32):
    """Return full class-probability matrix (N, C_eff). Binary -> [1-p, p]."""
    out = model.predict(X, batch_size=batch_size, verbose=0)
    if out.shape[1] == 1:                      # sigmoid binary
        return np.hstack([1.0 - out, out])
    return out                                  # softmax multiclass


def _norm(a):
    a = np.asarray(a, dtype=np.float32)
    a = a - a.min()
    m = a.max()
    return a / m if m > 0 else a


def gaussian_baseline(x):
    """Gaussian-blur baseline b (in-distribution reference), 128x128x1 [0,255]."""
    b = cv2.GaussianBlur(x[:, :, 0], (C.BLUR_KERNEL, C.BLUR_KERNEL), C.BLUR_SIGMA)
    return b[:, :, None].astype(np.float32)


# --------------------------------------------------------------------------- #
#  Saliency map for the model's predicted class (128x128, normalised [0,1])
# --------------------------------------------------------------------------- #
class Explainers:
    """Bundles the four attribution methods for a single model."""

    def __init__(self, model, last_conv, background):
        self.model = model
        self.last_conv = last_conv
        self.binary = model.output_shape[-1] == 1
        self.gradcam = Gradcam(model, model_modifier=ReplaceToLinear(), clone=True)
        self.scorecam = Scorecam(model, model_modifier=ReplaceToLinear(), clone=True)
        self.shap = shap.GradientExplainer(model, background)

    def _score(self, pred_class):
        if self.binary:
            return BinaryScore(bool(pred_class))
        return CategoricalScore(int(pred_class))

    def gradcam_map(self, x, pred_class):
        return _norm(self.gradcam(self._score(pred_class), x[None],
                                  penultimate_layer=self.last_conv)[0])

    def scorecam_map(self, x, pred_class, batch_size):
        for mN in (C.SCORECAM_MAX_N, 32, 16, None):
            try:
                m = self.scorecam(self._score(pred_class), x[None],
                                  penultimate_layer=self.last_conv,
                                  max_N=mN, batch_size=batch_size)[0]
                return _norm(m)
            except ValueError:
                continue
        return np.zeros((C.IMG_SIZE, C.IMG_SIZE), np.float32)

    def ig_map(self, x, pred_class, steps=C.IG_STEPS_FAITH, chunk=8):
        xt = tf.convert_to_tensor(x[None].astype("float32"))
        baseline = tf.zeros_like(xt)
        alphas = tf.linspace(0.0, 1.0, steps)
        grad_sum = tf.zeros_like(xt)
        for s in range(0, steps, chunk):
            a = tf.reshape(alphas[s:s + chunk], (-1, 1, 1, 1))
            interp = baseline + a * (xt - baseline)
            with tf.GradientTape() as tape:
                tape.watch(interp)
                out = self.model(interp, training=False)
                if self.binary:
                    p = out[:, 0] if pred_class == 1 else 1.0 - out[:, 0]
                else:
                    p = out[:, int(pred_class)]
            g = tape.gradient(p, interp)
            grad_sum += tf.reduce_sum(g, axis=0, keepdims=True)
        ig = (xt[0] - baseline[0]) * (grad_sum[0] / float(steps))
        return _norm(tf.reduce_sum(tf.abs(ig), axis=-1).numpy())

    def shap_map(self, x, pred_class):
        sv = self.shap.shap_values(x[None].astype("float32"),
                                   nsamples=C.SHAP_SAMPLES)
        if isinstance(sv, list):
            idx = int(pred_class) if len(sv) > int(pred_class) else 0
            vals = np.asarray(sv[idx])[0]
        else:
            vals = np.asarray(sv)[0]
        return _norm(np.abs(vals).sum(axis=-1))

    def all_maps(self, x, pred_class, batch_size):
        maps = {}
        for name, fn in [
            ("Grad-CAM", lambda: self.gradcam_map(x, pred_class)),
            ("Integrated Gradients", lambda: self.ig_map(x, pred_class)),
            ("SHAP", lambda: self.shap_map(x, pred_class)),
            ("Score-CAM", lambda: self.scorecam_map(x, pred_class, batch_size)),
        ]:
            try:
                maps[name] = fn()
            except Exception as e:
                print(f"    map {name} failed: {str(e)[:70]}")
                maps[name] = np.zeros((C.IMG_SIZE, C.IMG_SIZE), np.float32)
        return maps


# --------------------------------------------------------------------------- #
#  Deletion / insertion perturbation and trajectories
# --------------------------------------------------------------------------- #
def deletion_insertion(model, x, b, S, pred_class, steps=C.FAITH_STEPS,
                       batch_size=32):
    """
    Rank pixels by saliency S (desc), progressively delete/insert against
    baseline b, and return the t-grid and confidence trajectories D(t), I(t)
    for the fixed predicted class.
    """
    H, W = S.shape
    order = np.argsort(S.ravel())[::-1]          # most-salient first
    total = H * W
    ts = np.linspace(0.0, 1.0, steps + 1)        # 0%,5%,...,100%

    del_imgs, ins_imgs = [], []
    for t in ts:
        n = int(round(t * total))
        m = np.zeros(total, np.float32)
        if n > 0:
            m[order[:n]] = 1.0
        M = m.reshape(H, W)[:, :, None]
        del_imgs.append(x * (1.0 - M) + b * M)   # remove salient
        ins_imgs.append(b * (1.0 - M) + x * M)   # reveal salient

    probs_del = model_probs(model, np.stack(del_imgs), batch_size)[:, pred_class]
    probs_ins = model_probs(model, np.stack(ins_imgs), batch_size)[:, pred_class]
    return ts, probs_del, probs_ins


def localisation_rho(S, mask):
    """Energy pointing-game ratio: saliency mass inside the lesion mask."""
    s = S.astype(np.float64)
    denom = s.sum()
    if denom <= 0 or mask is None:
        return np.nan
    return float((s * (mask > 0)).sum() / denom)


# --------------------------------------------------------------------------- #
#  Mask loading (BUS-BRA only)
# --------------------------------------------------------------------------- #
def load_mask(img_id):
    fname = "mask_" + str(img_id).split("bus_")[-1] + ".png"
    path = os.path.join(C.DATASET_DIR, "Masks", fname)
    if not os.path.exists(path):
        return None
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, (C.IMG_SIZE, C.IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


# --------------------------------------------------------------------------- #
#  Main per-config routine
# --------------------------------------------------------------------------- #
def run_config(modality, backbone, n_images=C.N_FAITH_IMAGES):
    import pandas as pd
    spec = C.MODALITIES[modality]
    nc = spec["num_classes"]
    tag = backbone + spec["weight_suffix"]
    wpath = C.weights_path(backbone, spec["weight_suffix"])
    if not os.path.exists(wpath):
        print(f"[faith] missing weights for {tag}; skip")
        return

    print(f"\n===== FAITHFULNESS  {modality} / {backbone} =====")
    cache = (D.build_or_load_cache() if modality == "BUS-BRA"
             else D.build_or_load_fetal_cache())
    X_test = cache["X_test"]
    y_test = cache["y_test"].astype(int)
    ids_test = cache["ids_test"] if "ids_test" in cache else None

    tf.keras.backend.clear_session()
    model, _, last_conv = build_model(backbone, num_classes=nc)
    model.load_weights(wpath)
    bs = C.batch_for(backbone)

    # choose a stratified sample of test images
    rng = np.random.RandomState(C.SEED)
    n_images = min(n_images, len(X_test))
    sample = rng.choice(len(X_test), size=n_images, replace=False)

    # SHAP background from the training set
    bg_idx = rng.choice(len(cache["X_train"]),
                        size=min(32, len(cache["X_train"])), replace=False)
    background = cache["X_train"][bg_idx].astype("float32")
    exp = Explainers(model, last_conv, background)

    methods = C.ATTRIBUTION_METHODS
    rows = []
    # accumulate curves: per method -> list of D(t), I(t)
    curves = {m: {"del": [], "ins": []} for m in methods}
    ts_grid = None
    qual = {"idx": [], "img": [], "mask": [], "maps": {m: [] for m in methods}}

    for c, i in enumerate(sample):
        x = X_test[i].astype("float32")
        probs = model_probs(model, x[None], bs)[0]
        pred_class = int(np.argmax(probs))
        conf0 = float(probs[pred_class])
        b = gaussian_baseline(x)
        mask = load_mask(ids_test[i]) if (spec["has_masks"] and ids_test is not None) else None

        maps = exp.all_maps(x, pred_class, bs)
        for m in methods:
            S = maps[m]
            ts, Dt, It = deletion_insertion(model, x, b, S, pred_class,
                                            batch_size=bs)
            ts_grid = ts
            auc_del = float(np.trapz(Dt, ts))
            auc_ins = float(np.trapz(It, ts))
            k_idx = int(round(C.FAITH_K * C.FAITH_STEPS))       # index for 20%
            dfaith = conf0 - float(Dt[k_idx])                   # eq. 3
            rho = localisation_rho(S, mask)
            rows.append({
                "modality": modality, "backbone": backbone, "method": m,
                "img_index": int(i), "pred_class": pred_class, "conf0": conf0,
                "AUC_del": auc_del, "AUC_ins": auc_ins,
                "F": auc_ins - auc_del,
                "delta_faith": dfaith, "DA_hit": int(dfaith > 0),
                "rho": rho,
            })
            curves[m]["del"].append(Dt)
            curves[m]["ins"].append(It)

        # keep first few examples for the qualitative figure
        if len(qual["idx"]) < 3:
            qual["idx"].append(int(i))
            qual["img"].append(x)
            qual["mask"].append(mask if mask is not None
                                else np.zeros((C.IMG_SIZE, C.IMG_SIZE), np.uint8))
            for m in methods:
                qual["maps"][m].append(maps[m])

        if (c + 1) % 20 == 0 or c == 0:
            print(f"  [{modality}/{backbone}] {c+1}/{n_images} images")

    # ---- persist -----------------------------------------------------------
    df = pd.DataFrame(rows)
    out_csv = os.path.join(C.FAITH_DIR, f"faith_{modality}_{backbone}.csv")
    df.to_csv(out_csv, index=False)

    mean_curves = {"ts": ts_grid}
    for m in methods:
        mean_curves[f"{m}__del"] = np.mean(curves[m]["del"], axis=0)
        mean_curves[f"{m}__ins"] = np.mean(curves[m]["ins"], axis=0)
        mean_curves[f"{m}__del_sd"] = np.std(curves[m]["del"], axis=0)
        mean_curves[f"{m}__ins_sd"] = np.std(curves[m]["ins"], axis=0)
    np.savez(os.path.join(C.FAITH_DIR, f"curves_{modality}_{backbone}.npz"),
             **mean_curves)

    qsave = {"idx": np.array(qual["idx"]),
             "img": np.array(qual["img"]),
             "mask": np.array(qual["mask"])}
    for m in methods:
        qsave["map__" + m] = np.array(qual["maps"][m])
    np.savez(os.path.join(C.FAITH_DIR, f"qual_{modality}_{backbone}.npz"), **qsave)

    # console summary
    summ = df.groupby("method")[["AUC_del", "AUC_ins", "F", "DA_hit", "rho"]].mean()
    print(summ.round(3).to_string())
    print(f"[faith] saved -> {os.path.basename(out_csv)}")


if __name__ == "__main__":
    modality = sys.argv[1] if len(sys.argv) > 1 else "BUS-BRA"
    backbone = sys.argv[2] if len(sys.argv) > 2 else "EfficientNetB4"
    run_config(modality, backbone)
