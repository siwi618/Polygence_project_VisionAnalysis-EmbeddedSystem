"""Load training history and plot loss/accuracy curves."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

HISTORY_PATH = Path("training_history.json")
LOSS_PNG = Path("training_loss.png")
ACCURACY_PNG = Path("training_accuracy.png")


def load_history(path: Path = HISTORY_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run museum.py first to train and save history."
        )
    with path.open() as f:
        return json.load(f)


def plot_loss(history: dict, save_path: Path = LOSS_PNG) -> None:
    epochs = range(1, len(history["loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["loss"], marker="o", label="Training loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_accuracy(history: dict, save_path: Path = ACCURACY_PNG) -> None:
    epochs = range(1, len(history["accuracy"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["accuracy"], marker="o", label="Training accuracy")
    plt.plot(epochs, history["val_accuracy"], marker="o", label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def main() -> None:
    history = load_history()
    plot_loss(history)
    plot_accuracy(history)


if __name__ == "__main__":
    main()
