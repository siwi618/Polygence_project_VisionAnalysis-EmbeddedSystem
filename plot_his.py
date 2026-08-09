"""Load training history and plot loss/accuracy curves."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

JOBS = [
    {
        "history": Path("training_history.json"),
        "loss_png": Path("training_loss.png"),
        "acc_png": Path("training_accuracy.png"),
        "title": "Custom CNN",
    },
    {
        "history": Path("mobilenet_training_history.json"),
        "loss_png": Path("mobilenet_training_loss.png"),
        "acc_png": Path("mobilenet_training_accuracy.png"),
        "title": "MobileNet",
    },
]


def load_history(path: Path = HISTORY_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run museum.py first to train and save history."
        )
    with path.open() as f:
        return json.load(f)


def plot_loss(history: dict, save_path: Path, title: str) -> None:
    epochs = range(1, len(history["loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["loss"], marker="o", label="Training loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title}: Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_accuracy(history: dict, save_path: Path, title: str) -> None:
    epochs = range(1, len(history["accuracy"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["accuracy"], marker="o", label="Training accuracy")
    plt.plot(epochs, history["val_accuracy"], marker="o", label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title}: Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def main() -> None:
    plotted = 0
    for job in JOBS:
        if not job["history"].exists():
            print(f"Skip {job['title']}: {job['history']} not found")
            continue
        history = load_history(job["history"])
        plot_loss(history, job["loss_png"], job["title"])
        plot_accuracy(history, job["acc_png"], job["title"])
        plotted += 1
    if plotted == 0:
        raise FileNotFoundError("No training history JSON found.")


if __name__ == "__main__":
    main()
