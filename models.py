"""
Model builders for the four pre-trained backbones.

Design
------
The requested input is 128 x 128 x 1 (grayscale).  ImageNet backbones expect a
3-channel input, so the *first two layers of every network* turn the grayscale
image into a 3-channel, correctly pre-processed tensor **inside the graph**:

    Input(128,128,1) -> Concatenate(x3) -> Lambda(preprocess_input) -> backbone

Because the backbone is attached with `input_tensor=`, the whole thing is a
single *flat* functional model.  That matters for the saliency stage: Grad-CAM /
Score-CAM / Integrated-Gradients / SHAP can all reach the convolutional feature
maps directly (no nested sub-model), and transfer learning still works because
the ImageNet weights are loaded normally.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

# Backbones + their matching preprocessing functions
from tensorflow.keras.applications import (
    EfficientNetB4, MobileNetV2, ResNet50, DenseNet121)
from tensorflow.keras.applications.efficientnet import preprocess_input as pp_eff
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as pp_mob
from tensorflow.keras.applications.resnet50 import preprocess_input as pp_res
from tensorflow.keras.applications.densenet import preprocess_input as pp_dense

import config as C

# name -> (backbone class, preprocess fn)
_BACKBONES = {
    "EfficientNetB4": (EfficientNetB4, pp_eff),
    "MobileNetV2":    (MobileNetV2,    pp_mob),
    "ResNet50":       (ResNet50,       pp_res),
    "DenseNet121":    (DenseNet121,    pp_dense),
}


def _last_conv_layer_name(model):
    """Name of the last 4-D (conv/activation) layer -> used by Grad/Score-CAM."""
    for layer in reversed(model.layers):
        try:                              # Keras 2
            shape = layer.output_shape
        except AttributeError:            # Keras 3
            try:
                shape = tuple(layer.output.shape)
            except Exception:
                continue
        if len(shape) == 4:
            return layer.name
    raise ValueError("No 4-D layer found for CAM.")


def build_model(name, dropout=C.DROPOUT, num_classes=None):
    """
    Build one classifier.  Returns (model, backbone, last_conv_layer_name).
    The backbone is returned separately so the training stage can (un)freeze it.

    num_classes:
        1 (default)  -> single sigmoid unit (binary, e.g. BUS-BRA).  This keeps
                        the exact architecture of the originally trained weights.
        C > 1        -> C-way softmax head (e.g. FETAL, 6 planes).
    """
    if name not in _BACKBONES:
        raise ValueError(f"Unknown model '{name}'. Options: {list(_BACKBONES)}")
    if num_classes is None:
        num_classes = C.NUM_CLASSES
    backbone_cls, preprocess = _BACKBONES[name]

    inputs = layers.Input(shape=(C.IMG_SIZE, C.IMG_SIZE, C.CHANNELS),
                          name="grayscale_input")
    # 1 -> 3 channels (replicate) then apply the network's own preprocessing
    x = layers.Concatenate(name="to_rgb")([inputs, inputs, inputs])
    x = layers.Lambda(preprocess, name="preprocess")(x)

    backbone = backbone_cls(include_top=False, weights="imagenet",
                            input_tensor=x)
    backbone._name = f"{name}_backbone"

    y = layers.GlobalAveragePooling2D(name="gap")(backbone.output)
    y = layers.Dropout(dropout, name="dropout")(y)
    activation = "sigmoid" if num_classes == 1 else "softmax"
    reg = tf.keras.regularizers.l2(C.L2_REG) if C.L2_REG > 0 else None
    outputs = layers.Dense(num_classes, activation=activation,
                           kernel_regularizer=reg, name="predictions")(y)

    model = Model(inputs, outputs, name=name)
    last_conv = _last_conv_layer_name(model)
    return model, backbone, last_conv


def compile_model(model, lr, num_classes=1, weight_decay=1e-5):
    """Compile for binary (sigmoid) or multiclass (softmax) heads."""
    try:                                    # weight decay if the TF build has it
        opt = Adam(learning_rate=lr, weight_decay=weight_decay)
    except TypeError:
        opt = Adam(learning_rate=lr)
    if num_classes == 1:
        loss = tf.keras.losses.BinaryCrossentropy(
            label_smoothing=C.LABEL_SMOOTHING)
        model.compile(optimizer=opt, loss=loss,
                      metrics=["accuracy",
                               tf.keras.metrics.AUC(name="auc"),
                               tf.keras.metrics.Precision(name="precision"),
                               tf.keras.metrics.Recall(name="recall")])
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    return model


def set_backbone_trainable(backbone, trainable, unfreeze_top=None):
    """
    Freeze / unfreeze the backbone.  When `unfreeze_top` is given only that many
    top layers become trainable (BatchNorm layers are always kept frozen to keep
    the running statistics stable during fine-tuning).
    """
    if not trainable:
        backbone.trainable = False
        return
    backbone.trainable = True
    if unfreeze_top is not None:
        for layer in backbone.layers[:-unfreeze_top]:
            layer.trainable = False
    # keep BatchNorm frozen for a stable, low-LR fine-tune
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


if __name__ == "__main__":
    for n in C.MODEL_NAMES:
        m, bb, lc = build_model(n)
        tot = m.count_params()
        print(f"{n:15s}  params={tot:>10,}  last_conv={lc}")
        tf.keras.backend.clear_session()
