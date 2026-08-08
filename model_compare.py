"""Compare all four museum models: accuracy, file size, inference time.

Models:
  1. Custom CNN (Keras)         — museum_model.keras
  2. Custom CNN (TFLite fp16)   — museum_fp16.tflite
  3. MobileNet (Keras)          — mobilenet_museum_model.keras
  4. MobileNet (TFLite fp16)    — mobilenet_museum_fp16.tflite
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
RESULTS_PATH = BASE_DIR / "model_compare_results.txt"

# Custom CNN was trained at 180; MobileNet at 160
CNN_SIZE = (180, 180)
MOBILENET_SIZE = (160, 160)

N_BENCH_RUNS = 100  # timed runs for avg ms / image
SEED = 123


#  helpers


def load_val_arrays(img_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Load full validation set as numpy arrays (0–255 float32)."""
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=img_size,
        batch_size=32,
        shuffle=False,
    )
    images, labels = [], []
    for batch_x, batch_y in val_ds:
        images.append(batch_x.numpy())
        labels.append(batch_y.numpy())
    return np.concatenate(images), np.concatenate(labels)


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def make_tflite_interpreter(path: Path, batch_size: int = 1) -> tf.lite.Interpreter:
    """Load TFLite; resize batch dim to 1 when the export allows it."""
    interpreter = tf.lite.Interpreter(model_path=str(path))
    input_details = interpreter.get_input_details()[0]
    shape = list(input_details["shape"])
    if shape[0] != batch_size:
        shape[0] = batch_size
        # strict=False: allow overriding a frozen positive batch dim when possible
        interpreter.resize_tensor_input(input_details["index"], shape, strict=False)
    interpreter.allocate_tensors()
    return interpreter


def tflite_predict_one(
    interpreter: tf.lite.Interpreter, image: np.ndarray
) -> np.ndarray:
    """Run one image through TFLite. image shape: (H, W, 3)."""
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    batch = np.expand_dims(image, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], batch)
    interpreter.invoke()
    return interpreter.get_tensor(output_details["index"])[0]


def tflite_predict_all(
    interpreter: tf.lite.Interpreter, images: np.ndarray
) -> np.ndarray:
    return np.stack([tflite_predict_one(interpreter, img) for img in images], axis=0)


def accuracy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == labels))


# accuracy


def eval_keras_accuracy(
    model: tf.keras.Model, images: np.ndarray, labels: np.ndarray
) -> float:
    logits = model.predict(images, verbose=0)
    return accuracy_from_logits(logits, labels)


def eval_tflite_accuracy(
    interpreter: tf.lite.Interpreter, images: np.ndarray, labels: np.ndarray
) -> float:
    logits = tflite_predict_all(interpreter, images)
    return accuracy_from_logits(logits, labels)


# inference time (ms / image)


def benchmark_keras(
    model: tf.keras.Model, images: np.ndarray, n_runs: int = N_BENCH_RUNS
) -> float:
    """Average ms per single-image Keras predict (after warm-up)."""
    sample = images[:1]
    _ = model.predict(sample, verbose=0)  # warm-up

    start = time.perf_counter()
    for _ in range(n_runs):
        _ = model.predict(sample, verbose=0)
    end = time.perf_counter()
    return (end - start) / n_runs * 1000


def benchmark_tflite(
    interpreter: tf.lite.Interpreter, images: np.ndarray, n_runs: int = N_BENCH_RUNS
) -> float:
    """Average ms per single-image TFLite invoke (after warm-up)."""
    sample = images[0]
    _ = tflite_predict_one(interpreter, sample)  # warm-up

    start = time.perf_counter()
    for _ in range(n_runs):
        _ = tflite_predict_one(interpreter, sample)
    end = time.perf_counter()
    return (end - start) / n_runs * 1000


# main


def main() -> None:
    models = {
        "Custom CNN (Keras)": {
            "path": BASE_DIR / "museum_model.keras",
            "kind": "keras",
            "img_size": CNN_SIZE,
        },
        "Custom CNN (TFLite fp16)": {
            "path": BASE_DIR / "museum_fp16.tflite",
            "kind": "tflite",
            "img_size": CNN_SIZE,
        },
        "MobileNet (Keras)": {
            "path": BASE_DIR / "mobilenet_museum_model.keras",
            "kind": "keras",
            "img_size": MOBILENET_SIZE,
        },
        "MobileNet (TFLite fp16)": {
            "path": BASE_DIR / "mobilenet_museum_fp16.tflite",
            "kind": "tflite",
            "img_size": MOBILENET_SIZE,
        },
    }

    # Cache val arrays per image size
    val_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    rows = []

    for name, cfg in models.items():
        path: Path = cfg["path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing model file: {path}")

        img_size = cfg["img_size"]
        if img_size not in val_cache:
            print(f"Loading validation set at {img_size}...")
            val_cache[img_size] = load_val_arrays(img_size)
            # Debug once per size (arrays, not tf.data.Dataset) Check numpy arrays directly
            print(f"  Val array shape: {images.shape}")  # (N, H, W, 3)
            print(f"  First 5 labels: {labels[:5]}")
            print(f"  Pixel range: {images.min():.2f} to {images.max():.2f}")

        images, labels = val_cache[img_size]
        print(f"\nEvaluating {name} on {len(labels)} val images...")
      
        size_mb = file_size_mb(path)

        if cfg["kind"] == "keras":
            model = tf.keras.models.load_model(path)
            acc = eval_keras_accuracy(model, images, labels)
            ms = benchmark_keras(model, images)
            del model
        else:
            interpreter = make_tflite_interpreter(path)
            acc = eval_tflite_accuracy(interpreter, images, labels)
            ms = benchmark_tflite(interpreter, images)

        rows.append((name, acc, size_mb, ms))
        print(f"  accuracy={acc:.4f}  size={size_mb:.1f} MB  latency={ms:.1f} ms")

    # Print table
    header = (
        f"{'Model':<26}| {'Accuracy':^8} | {'File Size':^9} | {'Inference Time':^14}"
    )
    sep = "-" * 26 + "|" + "-" * 10 + "|" + "-" * 11 + "|" + "-" * 15
    lines = [
        f"Validation images (CNN 180): {len(val_cache[CNN_SIZE][1])}",
        f"Validation images (MobileNet 160): {len(val_cache[MOBILENET_SIZE][1])}",
        f"Benchmark runs per model: {N_BENCH_RUNS} (after 1 warm-up)",
        "",
        header,
        sep,
    ]
    for name, acc, size_mb, ms in rows:
        lines.append(f"{name:<26}|  {acc:.4f}  | {size_mb:>5.1f} MB  | {ms:>8.1f} ms")
    lines.append("")

    table = "\n".join(lines)
    print("\n" + table)
    RESULTS_PATH.write_text(table)
    print(f"Saved table to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
