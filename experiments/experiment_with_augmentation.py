# This script runs training with augmentation enabled in the CNN pipeline.
# It uses a longer training schedule to test whether augmented data
# improves generalization in the experiment pipeline.
from pathlib import Path
import yaml
import tensorflow as tf

from model import build_cnn
from training_analysis import plot_training_history

#  Train the CNN using the augmentation settings defined in the model
#  Tests whether augmentation improves model performance compared
#  with the baseline experiment
def run_with_augmentation():
    """
    args:
        None
        The training settings are read from config.yaml.

    Return:
        None
        Training logs are printed by Keras during training.
    """

    print("\nRunning augmentation experiment (CNN already includes augmentation)")

    project_root = Path(__file__).resolve().parents[1]
    with open(project_root / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_dir = config["data"]["train_dir"]
    val_dir = config["data"]["val_dir"]
    batch_size = config["cnn"]["batch_size"]
    epochs = 20

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
        save_name="with_augmentation.png",
    )


if __name__ == "__main__":
    run_with_augmentation()
