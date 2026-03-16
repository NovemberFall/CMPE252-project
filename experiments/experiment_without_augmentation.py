import subprocess

def run_without_augmentation():
    """
    Baseline experiment.
    We keep the default CNN training settings defined in main.py.
    """

    print("\nRunning baseline experiment (default CNN settings)")

    cmd = [
        "python3",
        "main.py",
        "train-cnn",
        "--epochs",
        "10"
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    run_without_augmentation()