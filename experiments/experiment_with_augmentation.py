import subprocess

def run_with_augmentation():

    print("\nRunning augmentation experiment (CNN already includes augmentation)")

    cmd = [
        "python3",
        "main.py",
        "train-cnn",
        "--epochs",
        "20"
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    run_with_augmentation()