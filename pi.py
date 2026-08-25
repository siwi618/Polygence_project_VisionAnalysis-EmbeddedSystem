"""Benchmark and evaluate TFLite fp16 models on Raspberry Pi.

Needs tflite_runtime (not full TensorFlow), plus numpy and Pillow.
Both models include a Rescaling layer — feed pixels in [0, 255].
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "dataset"
# True: same val split as museum.py / model_compare.py (seed=123, 20%).
# False: score every image under TEST_DIR (use this for a held-out test folder).
USE_VAL_SPLIT = True
SEED = 123
VALIDATION_SPLIT = 0.2
N_BENCH_RUNS = 100
IMAGE_FORMATS = (".png",)
RESULTS_PATH = BASE_DIR / "pi_results.json"

MODELS = [
    {
        "name": "Custom CNN (TFLite fp16)",
        "path": BASE_DIR / "museum_fp16.tflite",
        "img_size": (180, 180),
    },
    {
        "name": "MobileNet (TFLite fp16)",
        "path": BASE_DIR / "mobilenet_museum_fp16.tflite",
        "img_size": (160, 160),
    },
]


def list_class_names(directory: Path) -> list[str]:
    names = []
    for name in sorted(os.listdir(directory)):
        if name.startswith("."):
            continue
        if (directory / name).is_dir():
            names.append(name.rstrip("/"))
    return names


def collect_labeled_images(
    directory: Path,
) -> tuple[list[Path], np.ndarray, list[str]]:
    """Walk class subfolders. Folder name is the label (Keras alphanumeric order)."""
    class_names = list_class_names(directory)
    class_indices = {name: i for i, name in enumerate(class_names)}
    paths: list[Path] = []
    labels: list[int] = []
    for class_name in class_names:
        class_dir = directory / class_name
        for root, _, files in sorted(os.walk(class_dir), key=lambda item: item[0]):
            for fname in sorted(files):
                if fname.lower().endswith(IMAGE_FORMATS):
                    paths.append(Path(root) / fname)
                    labels.append(class_indices[class_name])
    return paths, np.array(labels, dtype=np.int32), class_names


def keras_val_split(
    paths: list[Path],
    labels: np.ndarray,
    validation_split: float = VALIDATION_SPLIT,
    seed: int = SEED,
) -> tuple[list[Path], np.ndarray]:
    """Match image_dataset_from_directory(..., shuffle=True, subset='validation')."""
    paths = list(paths)
    labels = np.array(labels, dtype=np.int32)
    rng = np.random.RandomState(seed)
    rng.shuffle(paths)
    rng = np.random.RandomState(seed)
    rng.shuffle(labels)
    n_val = int(validation_split * len(paths))
    return paths[-n_val:], labels[-n_val:]


def preprocess_image(image_path: Path, img_size: tuple[int, int], dtype) -> np.ndarray:
    """RGB, bilinear resize, pixels in [0, 255], dtype from the interpreter."""
    img = Image.open(image_path).convert("RGB").resize(img_size, Image.BILINEAR)
    array = np.array(img).astype(dtype)
    return np.expand_dims(array, axis=0)


def load_interpreter(model_path: Path):
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError as exc:
        raise ImportError(
            "pi.py uses tflite_runtime, not TensorFlow. "
            "On the Pi: pip install tflite-runtime numpy Pillow"
        ) from exc

    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def predict_one(interpreter, batch: np.ndarray) -> np.ndarray:
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_details["index"], batch)
    interpreter.invoke()
    return interpreter.get_tensor(output_details["index"])[0]


def benchmark(interpreter, batch: np.ndarray, n_runs: int = N_BENCH_RUNS) -> float:
    predict_one(interpreter, batch)
    start = time.perf_counter()
    for _ in range(n_runs):
        predict_one(interpreter, batch)
    end = time.perf_counter()
    return (end - start) / n_runs * 1000


def accuracy_from_logits(
    logits: np.ndarray, labels: np.ndarray
) -> tuple[float, int, int]:
    preds = np.argmax(logits, axis=1)
    n_correct = int(np.sum(preds == labels))
    n_total = int(len(labels))
    acc = n_correct / n_total if n_total else 0.0
    return acc, n_correct, n_total


def evaluate(
    interpreter,
    paths: list[Path],
    labels: np.ndarray,
    img_size: tuple[int, int],
) -> tuple[float, int, int]:
    input_dtype = interpreter.get_input_details()[0]["dtype"]
    logits = np.stack(
        [
            predict_one(interpreter, preprocess_image(path, img_size, input_dtype))
            for path in paths
        ],
        axis=0,
    )
    return accuracy_from_logits(logits, labels)


def build_results(
    *,
    class_names: list[str],
    n_images: int,
    split_note: str,
    models: list[dict],
) -> dict:
    return {
        "class_names": list(class_names),
        "n_images": int(n_images),
        "split": split_note,
        "use_val_split": USE_VAL_SPLIT,
        "seed": SEED if USE_VAL_SPLIT else None,
        "validation_split": VALIDATION_SPLIT if USE_VAL_SPLIT else None,
        "benchmark_runs": N_BENCH_RUNS,
        "models": models,
    }


def save_results_json(payload: dict, path: Path = RESULTS_PATH) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

    paths, labels, class_names = collect_labeled_images(TEST_DIR)
    if USE_VAL_SPLIT:
        paths, labels = keras_val_split(paths, labels)
        split_note = f"validation split (seed={SEED}, fraction={VALIDATION_SPLIT})"
    else:
        split_note = "all images in TEST_DIR"

    print(f"Classes: {class_names}")
    print(f"Evaluating {len(paths)} images ({split_note})")
    print(f"Benchmark runs per model: {N_BENCH_RUNS} (after 1 warm-up)\n")

    rows = []
    for cfg in MODELS:
        path: Path = cfg["path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing model file: {path}")

        interpreter = load_interpreter(path)
        input_dtype = interpreter.get_input_details()[0]["dtype"]
        print(f"{cfg['name']}")
        print(f"  input dtype: {input_dtype}, size: {cfg['img_size']}")

        acc, n_correct, n_total = evaluate(interpreter, paths, labels, cfg["img_size"])
        sample = preprocess_image(paths[0], cfg["img_size"], input_dtype)
        ms = benchmark(interpreter, sample)

        rows.append(
            {
                "name": cfg["name"],
                "path": path.name,
                "img_size": list(cfg["img_size"]),
                "input_dtype": np.dtype(input_dtype).name,
                "accuracy": float(acc),
                "n_correct": int(n_correct),
                "n_total": int(n_total),
                "ms_per_img": round(float(ms), 2),
            }
        )
        print(f"  accuracy: {acc:.4f}  ({n_correct}/{n_total})")
        print(f"  inference: {ms:.2f} ms/img\n")

    name_w = max(len(r["name"]) for r in rows)
    print(f"{'Model':<{name_w}} | Accuracy | ms/img")
    print("-" * name_w + "-|----------|--------")
    for row in rows:
        print(
            f"{row['name']:<{name_w}} | {row['accuracy']:.4f}   | {row['ms_per_img']:.2f}"
        )

    payload = build_results(
        class_names=class_names,
        n_images=len(paths),
        split_note=split_note,
        models=rows,
    )
    save_results_json(payload)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
