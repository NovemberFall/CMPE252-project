# This script makes all experiment run and save output into `results.log`
# It executes multiple training setups, captures console output, and
# records which experiment settings were used for later comparison.
import subprocess
import json
import os
import sys
import pty
from pathlib import Path

# Results file paths
RESULT_FILE = Path("experiments/results.json")
RESULT_LOG = Path("experiments/results.log")
# Project root path
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Execute a Python script from the project root with proper PYTHONPATH.
# Running all experiments from the same root keeps imports consistent
# and ensures logs are captured for reproducibility in the pipeline.
def run_script_from_project_root(args):
    """
    args:
        args: List of command arguments (e.g., ["main.py", "train-cnn", "--epochs", "10"]).

    Return:
        None. Output is streamed to the terminal and appended to a log file.
    """
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")

    # Set PYTHONPATH so internal imports work for child scripts
    if not existing_pythonpath:
        env["PYTHONPATH"] = str(PROJECT_ROOT)
    else:
        env["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT), existing_pythonpath])

    RESULT_LOG.parent.mkdir(exist_ok=True)
    with open(RESULT_LOG, "ab") as log_file:
        master_fd, slave_fd = pty.openpty()
        try:
            # Start the child process in the project root
            process = subprocess.Popen(
                [sys.executable, *args],
                cwd=PROJECT_ROOT,
                env=env,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                close_fds=True,
            )
        finally:
            os.close(slave_fd)

        try:
            # Read from PTY and write to both terminal and log file
            while True:
                try:
                    output = os.read(master_fd, 1024)
                except OSError:
                    break
                if not output:
                    break
                sys.stdout.buffer.write(output)
                sys.stdout.buffer.flush()
                log_file.write(output)
                log_file.flush()
        finally:
            os.close(master_fd)

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, [sys.executable, *args])

# Experiments to run
experiments = [
    {"name": "baseline", "epochs": 10},
    {"name": "more_training", "epochs": 20},
    {"name": "less_training", "epochs": 5},
]

# Run a single baseline-style training experiment with a chosen epoch count
# This step compares how different training lengths affect model quality
# in the experiment pipeline.
def run_experiment(exp):
    """
    Inputs:
        exp: Dict containing "name" and "epochs" for this experiment.

    Return:
        None
        Training output is printed and logged by the runner.
    """

    print(f"\nRunning experiment: {exp['name']} (epochs={exp['epochs']})")

    # Build command for the training CLI with the selected epoch count
    cmd = [
        "main.py",
        "train-cnn",
        "--epochs",
        str(exp["epochs"])
    ]

    # Run the training script and capture logs
    run_script_from_project_root(cmd)


# Run the baseline training script without extra augmentation.
# Provides a baseline model to compare against augmented training runs
def run_without_augmentation():
    """
    Inputs:
        None
        Uses default settings inside the script.

    Return:
        None
        Output is logged by the runner.
    """
    print("\nRunning augmentation experiment (without)")

    # Run the script that uses the default (no extra augmentation) setup
    run_script_from_project_root(["experiments/experiment_without_augmentation.py"])

# Run the training script that includes augmentation in the CNN pipeline.
# Tests whether data augmentation improves generalization performance
def run_with_augmentation():
    """
    Inputs:
        None
        Uses augmentation settings already defined in the model code.

    Return:
        None
        Output is logged by the runner.
    """
    print("\nRunning augmentation experiment (with)")

    # Run the script that trains with augmentation-enabled settings
    run_script_from_project_root(["experiments/experiment_with_augmentation.py"])

# Run the experiment that applies class loss weights during training
# Evaluates whether class weighting helps with class imbalance in training
def run_loss_weights():
    """
    Inputs:
        None
        Uses loss weights from config.yaml in the experiment script
    Return:
        None
        Output is logged by the runner
    """
    print("\nRunning experiment: different loss weights")

    # Run the script that trains with weighted loss
    run_script_from_project_root(["experiments/experiment_loss_weights.py"])

# Run the final evaluation script on the best saved model
# Produces a final validation score to summarize experiment results
def evaluate_model():
    """
    Inputs:
        None
        Uses the model and dataset paths defined in the evaluation script.
    Return:
        None
        Evaluation metrics are printed and logged.
    """
    print("\nRunning final evaluation")

    # Run the evaluation script to compute final metrics
    run_script_from_project_root(["experiments/evaluate_model.py"])


# Execute all configured experiments and store their settings
# Centralizes experiment execution so results are reproducible and organized.
def main():
    """
    Inputs:
        None. Uses the experiments list defined in this file.
    Outputs:
        None. Writes results.json and appends to results.log.
    """
    results = []

    # Start a new log section for this run
    RESULT_LOG.parent.mkdir(exist_ok=True)
    with open(RESULT_LOG, "a", encoding="utf-8") as log_file:
        log_file.write("===== NEW RUN =====\n")

    # Run each baseline experiment and record its settings
    for exp in experiments:
        run_experiment(exp)
        results.append(exp)

    # Save experiment configuration list to a JSON file
    RESULT_FILE.parent.mkdir(exist_ok=True)
    with open(RESULT_FILE, "w") as f:
        json.dump(results, f, indent=4)

    # Run additional experiments that have their own scripts
    run_without_augmentation()
    run_with_augmentation()
    run_loss_weights()

    # Run final evaluation to summarize performance
    print("\nAll experiments finished")
    evaluate_model()

if __name__ == "__main__":
    main()
