# This script runs the baseline training experiment without extra augmentation.
# It calls the training pipeline with a fixed epoch count to create
# a simple baseline for comparison in the experiment pipeline.
from pathlib import Path
import yaml
import tensorflow as tf

from model import build_cnn
from training_analysis import plot_training_history

# Train the CNN with default settings and no extra augmentation
# Provides a baseline performance reference for other experiments
def run_without_augmentation():
    """
    args:
        None.
        The training settings are read from config.yaml.
    Return:
        None.
        Training logs are printed by Keras during training.
    """

    print("\nRunning baseline experiment (default CNN settings)")

    project_root = Path(__file__).resolve().parents[1]
    with open(project_root / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_dir = config["data"]["train_dir"]
    val_dir = config["data"]["val_dir"]
    batch_size = config["cnn"]["batch_size"]
    epochs = 10

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=(224, 224), batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=(224, 224), batch_size=batch_size
    )

    num_classes = len(train_ds.class_names)
    model = build_cnn(num_classes=num_classes)

    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[callback],
        verbose=1,
    )

    output_dir = Path(__file__).resolve().parent / "accuracy_plot"
    plot_training_history(
        history.history,
        save_dir=output_dir,
        save_name="without_augmentation.png",
    )


if __name__ == "__main__":
    run_without_augmentation()
