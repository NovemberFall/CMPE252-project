from pathlib import Path
import tensorflow as tf

MODEL_PATH = Path("best_busi_model.keras")
TEST_DIR = Path("dataset/val")  # 

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def evaluate():

    if not MODEL_PATH.exists():
        print("Model not found")
        return

    model = tf.keras.models.load_model(MODEL_PATH)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    loss, acc = model.evaluate(test_ds)

    print("\nValidation Accuracy:", acc)
    print("Validation Loss:", loss)


if __name__ == "__main__":
    evaluate()