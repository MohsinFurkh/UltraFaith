"""
Data loading and patient-level, stratified train / val / test split for BUS-BRA.

Images are read as grayscale, resized to 128 x 128 and kept in the [0, 255]
float range (the per-model `preprocess_input` is applied *inside* each network,
see models.py).  The split is done at the *patient* (`Case`) level so that the
two views (left / right) of the same patient never leak across sets.

The arrays and the split indices are cached to a single .npz file so that the
training, evaluation and saliency stages all operate on exactly the same data.
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold

import config as C


# --------------------------------------------------------------------------- #
def _read_image(path, size=C.IMG_SIZE):
    """Read one image as grayscale float32 [0,255] of shape (size,size,1)."""
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)          # (H, W) in [0,255]
    return arr[..., np.newaxis]                       # (H, W, 1)


def _load_raw():
    """Load every image + label from the CSV.  Returns X, y, groups, ids."""
    df = pd.read_csv(C.CSV_PATH)
    df = df[df[C.LABEL_COLUMN].isin(C.CLASS_NAMES)].reset_index(drop=True)

    X, y, groups, ids = [], [], [], []
    missing = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Reading images"):
        img_id = str(row["ID"])
        path = os.path.join(C.IMAGES_DIR, img_id + ".png")
        if not os.path.exists(path):
            missing += 1
            continue
        X.append(_read_image(path))
        y.append(1 if row[C.LABEL_COLUMN] == C.POSITIVE_LABEL else 0)
        groups.append(int(row[C.GROUP_COLUMN]))
        ids.append(img_id)

    if missing:
        print(f"[data_loader] WARNING: {missing} image files were not found.")

    X = np.stack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int32)
    groups = np.asarray(groups, dtype=np.int32)
    ids = np.asarray(ids)
    return X, y, groups, ids


def _patient_level_split(y, groups):
    """
    Stratified + grouped split -> train / val / test index arrays.

    Uses StratifiedGroupKFold twice so that (a) no patient appears in more than
    one set and (b) the malignant/benign ratio is preserved as closely as the
    grouping allows.
    """
    idx_all = np.arange(len(y))

    # ---- carve out the test set (~20%) --------------------------------------
    n_splits_test = int(round(1.0 / C.TEST_FRACTION))           # 5 -> 20%
    sgkf = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True,
                                random_state=C.SEED)
    trainval_idx, test_idx = next(iter(sgkf.split(idx_all, y, groups)))

    # ---- carve validation out of the remaining train/val --------------------
    # val fraction expressed relative to the train/val pool
    rel_val = C.VAL_FRACTION / (1.0 - C.TEST_FRACTION)
    n_splits_val = max(2, int(round(1.0 / rel_val)))            # ~5 -> ~19%
    sgkf2 = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True,
                                 random_state=C.SEED)
    sub_train, sub_val = next(iter(
        sgkf2.split(trainval_idx, y[trainval_idx], groups[trainval_idx])))
    train_idx = trainval_idx[sub_train]
    val_idx = trainval_idx[sub_val]

    # sanity: no patient leakage
    assert not (set(groups[train_idx]) & set(groups[test_idx]))
    assert not (set(groups[val_idx]) & set(groups[test_idx]))
    assert not (set(groups[train_idx]) & set(groups[val_idx]))
    return train_idx, val_idx, test_idx


# Frozen BUS-BRA split (image-id -> fold) for cross-machine reproducibility.
BUSBRA_SPLIT_JSON = os.path.join(C.PROJECT_DIR, "busbra_split.json")


def build_or_load_cache(force=False):
    """
    Return a dict with X_train/val/test, y_train/val/test and the raw ids.
    Cached to config.DATA_CACHE so all stages share the identical split.
    Uses busbra_split.json when present (reproducible across machines).
    """
    if os.path.exists(C.DATA_CACHE) and not force:
        d = np.load(C.DATA_CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}

    import json
    print("[data_loader] Building dataset cache ...")
    X, y, groups, ids = _load_raw()

    if os.path.exists(BUSBRA_SPLIT_JSON):
        split = json.load(open(BUSBRA_SPLIT_JSON))
        id_to_i = {n: i for i, n in enumerate(ids)}
        tr = np.array([id_to_i[n] for n in split["train"] if n in id_to_i])
        va = np.array([id_to_i[n] for n in split["val"] if n in id_to_i])
        te = np.array([id_to_i[n] for n in split["test"] if n in id_to_i])
        print(f"[data_loader] using frozen split from {os.path.basename(BUSBRA_SPLIT_JSON)}")
    else:
        tr, va, te = _patient_level_split(y, groups)
        json.dump({"train": ids[tr].tolist(), "val": ids[va].tolist(),
                   "test": ids[te].tolist()}, open(BUSBRA_SPLIT_JSON, "w"))
        print(f"[data_loader] wrote frozen split -> {os.path.basename(BUSBRA_SPLIT_JSON)}")

    cache = dict(
        X_train=X[tr], y_train=y[tr], ids_train=ids[tr],
        X_val=X[va],   y_val=y[va],   ids_val=ids[va],
        X_test=X[te],  y_test=y[te],  ids_test=ids[te],
    )
    np.savez_compressed(C.DATA_CACHE, **cache)
    _report(cache)
    return cache


def _report(cache):
    def dist(y):
        y = np.asarray(y)
        return f"n={len(y)}  benign={int((y==0).sum())}  malignant={int((y==1).sum())}"
    print("  Train :", dist(cache["y_train"]))
    print("  Val   :", dist(cache["y_val"]))
    print("  Test  :", dist(cache["y_test"]))


def compute_class_weights(y_train, num_classes=2):
    """Inverse-frequency class weights for imbalanced tasks (binary or multi)."""
    y = np.asarray(y_train)
    n = len(y)
    classes = np.arange(max(num_classes, int(y.max()) + 1))
    weights = {}
    for c in classes:
        n_c = max(1, int((y == c).sum()))
        weights[int(c)] = n / (len(classes) * n_c)
    return weights


# =========================================================================== #
#  FETAL_PLANES_DB  (second modality: 6-class fetal plane identification)
# =========================================================================== #
FETAL_LABEL_COLUMN = "Plane"
FETAL_GROUP_COLUMN = "Patient_num"
# canonical, sorted class order (kept identical to config.MODALITIES["FETAL"])
FETAL_CLASSES = ["Fetal abdomen", "Fetal brain", "Fetal femur",
                 "Fetal thorax", "Maternal cervix", "Other"]


def _load_fetal_raw():
    df = pd.read_csv(C.FETAL_CSV_PATH, sep=";")
    df.columns = [c.strip() for c in df.columns]
    cls_to_idx = {c: i for i, c in enumerate(FETAL_CLASSES)}

    X, y, groups, names = [], [], [], []
    missing = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Reading fetal images"):
        name = str(row[FETAL_LABEL_COLUMN])
        if name not in cls_to_idx:
            continue
        img_name = str(row["Image_name"])
        path = os.path.join(C.FETAL_IMAGES_DIR, img_name + ".png")
        if not os.path.exists(path):
            missing += 1
            continue
        X.append(_read_image(path))
        y.append(cls_to_idx[name])
        groups.append(int(row[FETAL_GROUP_COLUMN]))
        names.append(img_name)
    if missing:
        print(f"[data_loader] WARNING: {missing} fetal image files not found.")
    return (np.stack(X).astype(np.float32),
            np.asarray(y, dtype=np.int32),
            np.asarray(groups, dtype=np.int32),
            np.asarray(names))


# A frozen split (image-name -> fold) makes the train/val/test partition
# reproducible across machines and scikit-learn versions -- essential when the
# models are trained elsewhere (e.g. on Kaggle) but evaluated here.
FETAL_SPLIT_JSON = os.path.join(C.PROJECT_DIR, "fetal_split.json")


def build_or_load_fetal_cache(force=False):
    """Patient-level, class-stratified split for FETAL_PLANES_DB (6 classes)."""
    if os.path.exists(C.FETAL_CACHE) and not force:
        d = np.load(C.FETAL_CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}

    import json
    print("[data_loader] Building FETAL cache ...")
    X, y, groups, names = _load_fetal_raw()

    if os.path.exists(FETAL_SPLIT_JSON):
        # reproduce a previously frozen split by image name
        split = json.load(open(FETAL_SPLIT_JSON))
        name_to_i = {n: i for i, n in enumerate(names)}
        tr = np.array([name_to_i[n] for n in split["train"] if n in name_to_i])
        va = np.array([name_to_i[n] for n in split["val"] if n in name_to_i])
        te = np.array([name_to_i[n] for n in split["test"] if n in name_to_i])
        print(f"[data_loader] using frozen split from {os.path.basename(FETAL_SPLIT_JSON)}")
    else:
        tr, va, te = _patient_level_split(y, groups)
        json.dump({"train": names[tr].tolist(), "val": names[va].tolist(),
                   "test": names[te].tolist()}, open(FETAL_SPLIT_JSON, "w"))
        print(f"[data_loader] wrote frozen split -> {os.path.basename(FETAL_SPLIT_JSON)}")

    cache = dict(
        X_train=X[tr], y_train=y[tr], ids_train=names[tr],
        X_val=X[va],   y_val=y[va],   ids_val=names[va],
        X_test=X[te],  y_test=y[te],  ids_test=names[te],
    )
    np.savez_compressed(C.FETAL_CACHE, **cache)
    for split_name in ("train", "val", "test"):
        yy = cache["y_" + split_name]
        counts = {FETAL_CLASSES[c]: int((yy == c).sum()) for c in range(6)}
        print(f"  {split_name:5s}: n={len(yy)}  {counts}")
    return cache


if __name__ == "__main__":
    c = build_or_load_cache(force=True)
    _report(c)
    print("class weights:", compute_class_weights(c["y_train"]))
