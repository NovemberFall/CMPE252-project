# heatmap_analysis.py
# Created by Lorelei English-Webster on 3/14/26
# Generates CNN activation heatmap for ultrasound image

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.preprocessing import image

MODEL_PATH = "best_busi_model.keras"
IMG_SIZE = (224, 224)


def get_img_array(img_path, size):
    img = image.load_img(img_path, target_size=size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model")


def make_activation_heatmap(img_array, model, last_conv_layer_name):
    # Build model first
    _ = model(img_array, training=False)

    # Create a model that outputs the last conv layer activations only
    activation_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer(last_conv_layer_name).output
    )

    conv_output = activation_model(img_array, training=False)[0]  # shape: (H, W, C)

    # Average across channels to get a 2D activation map
    heatmap = tf.reduce_mean(conv_output, axis=-1)

    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def save_heatmap_overlay(img_path, heatmap, save_path="accuracy_plot/activation_heatmap_overlay.png", alpha=0.4):
    img = image.load_img(img_path)
    img = image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)

    jet = plt.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
    superimposed_img = image.array_to_img(superimposed_img)

    Path("accuracy_plot").mkdir(exist_ok=True)
    superimposed_img.save(save_path)

    print("Activation heatmap saved to:", save_path)


if __name__ == "__main__":
    img_path = "dataset/val/benign/benign (11).png"

    model = tf.keras.models.load_model(MODEL_PATH)

    img_array = get_img_array(img_path, IMG_SIZE)

    last_conv_layer = find_last_conv_layer(model)
    print("Using last conv layer:", last_conv_layer)

    heatmap = make_activation_heatmap(img_array, model, last_conv_layer)

    save_heatmap_overlay(img_path, heatmap)
