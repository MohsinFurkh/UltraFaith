"""
Load Keras-3 `.weights.h5` checkpoints (as produced on Kaggle with TF 2.16+)
into the local TF 2.8 / Keras 2 models.

Keras 3 renames layers to generic type-indexed names (batch_normalization_10,
conv2d, dense, ...) and stores weights under /layers/<name>/vars/<i>.  Because
both the Kaggle and the local model come from the *same* build_model code, the
i-th layer of a given type here corresponds to the i-th layer of that type in
the checkpoint.  We match by (type, creation order) and assert every weight
shape -- and callers verify by checking the reconstructed test accuracy.
"""
import re
import h5py
import numpy as np

# local Keras-2 layer class -> Keras-3 generic name prefix
_PREFIX = {
    "Conv2D": "conv2d",
    "DepthwiseConv2D": "depthwise_conv2d",
    "SeparableConv2D": "separable_conv2d",
    "BatchNormalization": "batch_normalization",
    "Dense": "dense",
    "Normalization": "normalization",
    "PReLU": "p_re_lu",
}


def _suffix_num(name, prefix):
    m = re.fullmatch(prefix + r"(?:_(\d+))?", name)
    return int(m.group(1)) if (m and m.group(1)) else 0


def load_keras3_weights(model, h5path):
    """
    Assign Keras-3 checkpoint weights to a Keras-2 `model` in place.

    Layers are matched by (type prefix, weight-shape signature), in creation
    order within each such group.  Using the shape signature disambiguates
    sub-groups whose global creation order differs between Keras 2 and Keras 3
    (e.g. EfficientNet squeeze-excite vs. main-path convolutions), while order
    within one signature -- which follows block order -- is preserved.
    """
    f = h5py.File(h5path, "r")
    layers_grp = f["layers"]

    def _read(gkey):
        v = layers_grp[gkey]["vars"]
        return [np.array(v[str(i)]) for i in range(len(v.keys()))]

    # bucket checkpoint groups by (prefix, shape-signature), in creation order
    ckpt = {}
    for key in layers_grp.keys():
        g = layers_grp[key]
        if "vars" not in g or len(g["vars"].keys()) == 0:
            continue
        prefix = re.sub(r"_\d+$", "", key)
        arrs = _read(key)
        sig = (prefix, tuple(a.shape for a in arrs))
        ckpt.setdefault(sig, []).append((_suffix_num(key, prefix), key))
    for sig in ckpt:
        ckpt[sig].sort()                       # by creation index

    counters, n_assigned = {}, 0
    for layer in model.layers:
        if not layer.weights:
            continue
        cls = type(layer).__name__
        prefix = _PREFIX.get(cls)
        if prefix is None:
            raise ValueError(f"No checkpoint prefix for layer {layer.name} ({cls})")
        cur = layer.get_weights()
        sig = (prefix, tuple(c.shape for c in cur))
        if sig not in ckpt:
            raise ValueError(f"No checkpoint group for {layer.name} sig={sig}")
        idx = counters.get(sig, 0)
        counters[sig] = idx + 1
        if idx >= len(ckpt[sig]):
            raise ValueError(f"Ran out of groups for {layer.name} sig={sig}")
        arrs = _read(ckpt[sig][idx][1])
        layer.set_weights(arrs)
        n_assigned += 1
    f.close()
    return n_assigned
