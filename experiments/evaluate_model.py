# Our script evaluates a saved model on the validation dataset.
# It will load the trained model, prepares the validation data, running
# evaluation, and prints the final accuracy and loss.
from pathlib import Path
import tensorflow as tf

# Model file path
MODEL_PATH = Path("best_busi_model.keras")
# Validation data path
TEST_DIR = Path("dataset/val")  

# Image and batch settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


# Run model evaluation
# Provides a final performance check after training experiments finish.
def evaluate():
    """
    args:
        None
        Uses model and dataset paths defined at the top of the file.

    return:
        None 
        Only prints validation accuracy and loss to the console.
    """

    # Check that the model file exists before loading
    if not MODEL_PATH.exists():
        print("Model not found")
        return

    # Load the saved model from disk
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load the validation data without shuffling for stable evaluation
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Evaluate the model on the validation dataset
    loss, acc = model.evaluate(test_ds)

    # Print evaluation results for easy review in logs
    print("\nValidation Accuracy:", acc)
    print("Validation Loss:", loss)


if __name__ == "__main__":
    evaluate()
