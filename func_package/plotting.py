# func_package/plotting.py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from tensorflow.keras import Input, Model

# ---------- Internal helpers ----------

def _functional_clone_from_sequential(seq_model, input_shape):
    """Rebuild a clean Functional model with identical layers & weights."""
    inputs = Input(shape=input_shape, name="gc_input")
    x = inputs
    new_layers = []
    for old in seq_model.layers:
        new = old.__class__.from_config(old.get_config())
        x = new(x)
        new_layers.append(new)
    f_model = Model(inputs, x, name="gc_functional")
    for old, new in zip(seq_model.layers, new_layers):
        if old.weights:
            new.set_weights(old.get_weights())
    return f_model

def _build_grad_model(seq_model, last_conv_layer_name, input_shape):
    f_model = _functional_clone_from_sequential(seq_model, input_shape)
    return Model(
        inputs=f_model.input,
        outputs=[f_model.get_layer(last_conv_layer_name).output, f_model.output],
        name="grad_model_clean"
    )

def _upsample_repeat(arr, target_h, target_w):
    """Integer-repeat upsampling for small CAMs (falls back to original if not divisible)."""
    h, w = arr.shape
    sh = target_h // h
    sw = target_w // w
    if sh * h == target_h and sw * w == target_w:
        return np.repeat(np.repeat(arr, sh, axis=0), sw, axis=1)
    return arr  # caller can still display with imshow(bilinear)

# ---------- Public API ----------

def gradcam_heatmap(model, x_single, last_conv_layer_name="conv2d_1"):
    """
    Compute a Grad-CAM heatmap for a single image.
    Parameters
    ----------
    model : tf.keras.Model
        Trained Sequential model.
    x_single : np.ndarray
        Single image shaped (H, W, C) with values in [0,1].
    last_conv_layer_name : str
        Name of the last conv layer (e.g., 'conv2d_1').

    Returns
    -------
    heatmap : np.ndarray (h, w)
        Grad-CAM heatmap in [0,1] at conv-layer resolution.
    pred_index : int
        Predicted class index (as per model's output ordering).
    pred_probs : np.ndarray (num_classes,)
        Softmax probabilities for the single image.
    """
    H, W, C = x_single.shape
    grad_model = _build_grad_model(model, last_conv_layer_name, input_shape=(H, W, C))
    img = np.expand_dims(x_single, axis=0).astype("float32")  # (1,H,W,C)

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img, training=False)
        pred_index = tf.argmax(preds[0])
        class_score = preds[:, pred_index]

    grads = tape.gradient(class_score, conv_out)      # (1,h,w,k)
    weights = tf.reduce_mean(grads, axis=(1, 2))      # (1,k)
    conv_maps = conv_out[0]                           # (h,w,k)

    cam = tf.tensordot(conv_maps, tf.squeeze(weights), axes=(2, 0))  # (h,w)
    cam = tf.nn.relu(cam)
    heatmap = cam / (tf.reduce_max(cam) + 1e-10)
    return heatmap.numpy(), int(pred_index.numpy()), preds.numpy()[0]

def overlay_heatmap(gray_image, heatmap, alpha=0.45, cmap_name="jet"):
    """
    Upsample heatmap to image size and blend (Matplotlib only).
    gray_image: (H,W) in [0,1]; heatmap: (h,w) in [0,1].
    Returns: (heatmap_up_RGB, overlay_RGB) both in [0,1].
    """
    H, W = gray_image.shape
    h, w = heatmap.shape
    heat_up = _upsample_repeat(heatmap, H, W)

    gray_rgb = np.stack([gray_image, gray_image, gray_image], axis=-1)
    cmap = cm.get_cmap(cmap_name)
    heat_rgb = cmap(heat_up)[..., :3]  # drop alpha
    overlay = (1 - alpha) * gray_rgb + alpha * heat_rgb
    return heat_rgb, np.clip(overlay, 0.0, 1.0)

def plot_gradcam(model, x_data, idx, last_conv_layer_name="conv2d_1",
                 alpha=0.45, cmap_name="jet", show=True):
    """
    End-to-end: compute heatmap for x_data[idx] and plot Original / Heatmap / Overlay.
    Returns dict with prediction and arrays.
    """
    x = x_data[idx]  # (H,W,C) in [0,1]
    heat, pred_idx, pred_probs = gradcam_heatmap(model, x, last_conv_layer_name)
    gray = x[..., 0] if x.shape[-1] == 1 else np.mean(x, axis=-1)
    heat_rgb, overlay = overlay_heatmap(gray, heat, alpha=alpha, cmap_name=cmap_name)

    if show:
        plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1); plt.imshow(gray, cmap="gray"); plt.title("Original"); plt.axis("off")
        plt.subplot(1, 3, 2); plt.imshow(heat_rgb);          plt.title("Grad-CAM"); plt.axis("off")
        plt.subplot(1, 3, 3); plt.imshow(overlay);           plt.title("Overlay");  plt.axis("off")
        plt.tight_layout(); plt.show()

    return {"pred_index": pred_idx, "pred_probs": pred_probs,
            "heatmap": heat, "overlay": overlay}

def misclassified_indices(model, x, y_onehot, limit=None):
    """Return indices where argmax(pred) != argmax(true)."""
    preds = model.predict(x, verbose=0)
    y_true = np.argmax(y_onehot, axis=1)
    y_pred = np.argmax(preds, axis=1)
    idxs = np.where(y_true != y_pred)[0]
    return idxs[:limit] if limit is not None else idxs

def plot_gradcam_grid(model, x, indices, last_conv_layer_name="conv2d_1",
                      cols=5, alpha=0.45, cmap_name="jet"):
    """Plot a grid of overlays for a list of indices."""
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = np.atleast_2d(axes)
    for ax, idx in zip(axes.ravel(), indices):
        out = plot_gradcam(model, x, idx, last_conv_layer_name, alpha, cmap_name, show=False)
        ax.imshow(out["overlay"]); ax.set_title(f"idx={idx}  pred={out['pred_index']}"); ax.axis("off")
    # hide any extras
    for ax in axes.ravel()[len(indices):]:
        ax.axis("off")
    plt.tight_layout(); plt.show()
