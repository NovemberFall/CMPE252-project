# This script runs training with augmentation enabled in the CNN pipeline.
# It uses a longer training schedule to test whether augmented data
# improves generalization in the experiment pipeline.
import subprocess

#  Train the CNN using the augmentation settings defined in the model
#  Tests whether augmentation improves model performance compared
#  with the baseline experiment
def run_with_augmentation():
    """
    args:
        None
        The CLI arguments specify the training duration.

    Return:
        None 
        Training logs are printed by the training script.
    """

    print("\nRunning augmentation experiment (CNN already includes augmentation)")

    # Build the training command for the augmentation run
    cmd = [
        "python3",
        "main.py",
        "train-cnn",
        "--epochs",
        "20"
    ]

    # Run the training script and fail if it exits with an error
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    run_with_augmentation()
