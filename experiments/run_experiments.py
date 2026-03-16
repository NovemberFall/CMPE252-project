import subprocess
import json
from pathlib import Path

RESULT_FILE = Path("experiments/results.json")

experiments = [
    {"name": "baseline", "epochs": 10},
    {"name": "more_training", "epochs": 20},
    {"name": "less_training", "epochs": 5},
]

def run_experiment(exp):
    print(f"\nRunning experiment: {exp['name']} (epochs={exp['epochs']})")

    cmd = [
        "python3",
        "main.py",
        "train-cnn",
        "--epochs",
        str(exp["epochs"])
    ]

    subprocess.run(cmd, check=True)


# run without augmentation
def run_without_augmentation():
    print("\nRunning augmentation experiment (without)")

    subprocess.run(
        ["python3", "experiments/experiment_without_augmentation.py"],check=True)

# run with augmentation
def run_with_augmentation():
    print("\nRunning augmentation experiment (with)")

    subprocess.run(
        ["python3", "experiments/experiment_with_augmentation.py"], check=True)

# run loss weights
def run_loss_weights():
    print("\nRunning experiment: different loss weights")

    subprocess.run(
        ["python3", "experiments/experiment_loss_weights.py"], check=True)

def evaluate_model():
    print("\nRunning final evaluation")

    cmd = ["python3", "experiments/evaluate_model.py"]
    subprocess.run(cmd, check=True)


def main():
    results = []

    for exp in experiments:
        run_experiment(exp)
        results.append(exp)

    RESULT_FILE.parent.mkdir(exist_ok=True)
    with open(RESULT_FILE, "w") as f:
        json.dump(results, f, indent=4)

    # run with_augmentation vs without_augmentation, loss weights
    run_without_augmentation()
    run_with_augmentation()
    run_loss_weights()

    print("\nAll experiments finished")
    evaluate_model()

if __name__ == "__main__":
    main()