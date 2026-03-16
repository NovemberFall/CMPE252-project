# This script runs the baseline training experiment without extra augmentation.
# It calls the main training command with a fixed epoch count to create
# a simple baseline for comparison in the experiment pipeline.
import subprocess

# Train the CNN with default settings and no extra augmentation
# Provides a baseline performance reference for other experiments
def run_without_augmentation():
    """
    args:
        None. 
        The training settings are passed through the CLI arguments.
    Return:
        None. 
        Training logs are printed by the training script.
    """

    print("\nRunning baseline experiment (default CNN settings)")

    # Build the training command for a simple baseline run
    cmd = [
        "python3",
        "main.py",
        "train-cnn",
        "--epochs",
        "10"
    ]

    # Run the training script and fail if it exits with an error
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    run_without_augmentation()
