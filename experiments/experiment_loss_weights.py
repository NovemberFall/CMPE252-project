# This script runs one experiment that uses class loss weights.
# It loads the dataset, builds the CNN model, trains with weighted loss,
# and prints training progress for comparison with other experiments.
import sys
sys.path.append("..")
from model import build_cnn
import yaml
import tensorflow as tf
from pathlib import Path

TRAIN_DIR = Path("dataset/train")
VAL_DIR = Path("dataset/val")

# Train the CNN using class loss weights from the config file.
# This experiment tests whether reweighting classes improves
# performance on imbalanced data in the overall experiment pipeline
def run_loss_weight_experiment():
    """
    Para:
        None. 
        The function reads settings from config.yaml and uses
        the dataset folders defined in this script.

    return:
        None. 
        Training metrics are printed by Keras during training.
    """

    print("\nRunning experiment with different loss weights")

    # Load experiment settings from the shared config file
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Read training settings for this run (epochs, batch size, loss weights)
    epochs = config["cnn"]["epochs"]
    batch_size = config["cnn"]["batch_size"]
    class_weights = config["cnn"]["loss_weights"]    

    # Load training data from disk
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(TRAIN_DIR, image_size=(224, 224), batch_size=batch_size)

    # Load validation data for evaluation during training
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(VAL_DIR, image_size=(224, 224), batch_size=batch_size)

    # Build the CNN model based on the number of classes in the dataset
    num_classes = len(train_ds.class_names)

    model = build_cnn(num_classes=num_classes)

    # # Different loss weights
    # class_weights = {
    #     0: 1.0,
    #     1: 2.0,
    #     2: 2.0
    # }

    # Train the model using class weights to reduce class imbalance impact
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        verbose=2
    )


if __name__ == "__main__":
    run_loss_weight_experiment()
