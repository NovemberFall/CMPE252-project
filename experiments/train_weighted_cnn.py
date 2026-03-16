# This script trains a CNN using class loss weights from config.yaml.
# It loads the dataset, builds the model, trains with weighted loss,
# and prints training progress as part of the experiment pipeline.
import sys
import os

# Add project root to import path so local modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import yaml
import tensorflow as tf
from model import build_cnn

# Load experiment configuration from config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Read training settings and dataset locations
train_dir = config["data"]["train_dir"]
val_dir = config["data"]["val_dir"]
batch_size = config["cnn"]["batch_size"]
epochs = config["cnn"]["epochs"]
class_weights = config["cnn"]["loss_weights"]

# Load training data from disk
train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, image_size=(224, 224), batch_size=batch_size)

# Load validation data for evaluation during training
val_ds = tf.keras.utils.image_dataset_from_directory(val_dir, image_size=(224, 224), batch_size=batch_size)

# Build the CNN model based on the number of classes in the dataset
num_classes = len(train_ds.class_names)

model = build_cnn(num_classes=num_classes)

# Train the model using class weights to address class imbalance
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    class_weight=class_weights,
    verbose=1
)
