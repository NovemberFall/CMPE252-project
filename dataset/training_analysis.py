# training_analysis.py
#
# Created by Lorelei English-Webster on 3/14/26.

import matplotlib.pyplot as plt
from pathlib import Path


# This script generates plots of training vs validation accuracy and loss.
# It is separate from the model training pipeline so visualization work
# does not interfere with the team’s training code.

def plot_training_history(history, save_dir="../accuracy_plot", save_name="training_history_analysis.png"):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / save_name

    # Extract metrics from the training history
    train_acc = history.get("accuracy", [])
    val_acc = history.get("val_accuracy", [])
    train_loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])

    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, marker="o", label="Training Accuracy")
    plt.plot(epochs, val_acc, marker="o", label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, marker="o", label="Training Loss")
    plt.plot(epochs, val_loss, marker="o", label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
   
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Training visualization saved to: {save_path}")
