import sys
sys.path.append("..")
from model import build_cnn
import yaml
import tensorflow as tf
from pathlib import Path




TRAIN_DIR = Path("dataset/train")
VAL_DIR = Path("dataset/val")

def run_loss_weight_experiment():

    print("\nRunning experiment with different loss weights")

    # load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    epochs = config["cnn"]["epochs"]
    batch_size = config["cnn"]["batch_size"]
    class_weights = config["cnn"]["loss_weights"]    

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(TRAIN_DIR, image_size=(224, 224), batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(VAL_DIR, image_size=(224, 224), batch_size=batch_size)

    num_classes = len(train_ds.class_names)

    model = build_cnn(num_classes=num_classes)

    # # Different loss weights
    # class_weights = {
    #     0: 1.0,
    #     1: 2.0,
    #     2: 2.0
    # }

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights
    )


if __name__ == "__main__":
    run_loss_weight_experiment()