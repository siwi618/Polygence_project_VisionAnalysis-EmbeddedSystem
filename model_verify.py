"""Verify museum TFLite models against their Keras counterparts.

Reports for Custom CNN and MobileNet:
  - Keras accuracy vs TFLite accuracy
  - Agreement rate
  - Max / average probability difference
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
RESULTS_PATH = BASE_DIR / "model_verify_results.txt"

CNN_SIZE = (180, 180)
MOBILENET_SIZE = (160, 160)
SEED = 123


def load_val_arrays(img_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
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


def make_tflite_interpreter(path: Path) -> tf.lite.Interpreter:
    interpreter = tf.lite.Interpreter(model_path=str(path))
    input_details = interpreter.get_input_details()[0]
    shape = list(input_details["shape"])
    if shape[0] != 1:
        shape[0] = 1
        interpreter.resize_tensor_input(input_details["index"], shape, strict=False)
    interpreter.allocate_tensors()
    return interpreter


def tflite_predict_all(interpreter: tf.lite.Interpreter, images: np.ndarray) -> np.ndarray:
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    results = []
    for img in images:
        batch = np.expand_dims(img, axis=0).astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], batch)
        interpreter.invoke()
        results.append(interpreter.get_tensor(output_details["index"])[0])
    return np.stack(results, axis=0)


def verify_pair(
    name: str,
    keras_path: Path,
    tflite_path: Path,
    images: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Compare one Keras model with its TFLite twin on the same images."""
    keras_model = tf.keras.models.load_model(keras_path)
    interpreter = make_tflite_interpreter(tflite_path)

    keras_logits = keras_model.predict(images, verbose=0)
    tflite_logits = tflite_predict_all(interpreter, images)

    keras_preds = np.argmax(keras_logits, axis=1)
    tflite_preds = np.argmax(tflite_logits, axis=1)

    keras_prob = tf.nn.softmax(keras_logits, axis=1).numpy()
    tflite_prob = tf.nn.softmax(tflite_logits, axis=1).numpy()
    prob_diff = np.abs(keras_prob - tflite_prob)

    total = len(labels)
    num_classes = keras_logits.shape[1]
    result = {
        "name": name,
        "total": total,
        "keras_acc": float(np.mean(keras_preds == labels)),
        "tflite_acc": float(np.mean(tflite_preds == labels)),
        "keras_correct": int(np.sum(keras_preds == labels)),
        "tflite_correct": int(np.sum(tflite_preds == labels)),
        "agreement": float(np.mean(keras_preds == tflite_preds)),
        "agree_count": int(np.sum(keras_preds == tflite_preds)),
        "max_prob_diff": float(np.max(prob_diff)),
        "avg_prob_diff": float(np.mean(prob_diff)),  # == sum / (total * num_classes)
        "num_classes": num_classes,
    }
    return result


def format_result(r: dict) -> str:
    return "\n".join(
        [
            f"=== {r['name']} ===",
            f"validation set size: {r['total']}",
            f"Keras accuracy:  {r['keras_acc']:.4f} ({r['keras_correct']}/{r['total']})",
            f"TFLite accuracy: {r['tflite_acc']:.4f} ({r['tflite_correct']}/{r['total']})",
            f"accuracy change: {r['tflite_acc'] - r['keras_acc']:+.4f}",
            f"agreement rate:  {r['agreement']:.4f} ({r['agree_count']}/{r['total']})",
            f"max probability difference:     {r['max_prob_diff']:.6f}",
            f"average probability difference: {r['avg_prob_diff']:.6f}",
            "",
        ]
    )


def format_summary_table(results: list[dict]) -> str:
    header = (
        f"{'Model':<14}| {'Keras Acc':^9} | {'TFLite Acc':^10} | "
        f"{'Agree':^7} | {'Max Δp':^10} | {'Avg Δp':^10}"
    )
    sep = "-" * 14 + "|" + "-" * 11 + "|" + "-" * 12 + "|" + "-" * 9 + "|" + "-" * 12 + "|" + "-" * 12
    lines = ["Summary table (for lab notebook)", header, sep]
    for r in results:
        lines.append(
            f"{r['name']:<14}|  {r['keras_acc']:.4f}  |   {r['tflite_acc']:.4f}  | "
            f"{r['agreement']:.4f} | {r['max_prob_diff']:.6f} | {r['avg_prob_diff']:.6f}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    pairs = [
        {
            "name": "Custom CNN",
            "keras": BASE_DIR / "museum_model.keras",
            "tflite": BASE_DIR / "museum_fp16.tflite",
            "img_size": CNN_SIZE,
        },
        {
            "name": "MobileNet",
            "keras": BASE_DIR / "mobilenet_museum_model.keras",
            "tflite": BASE_DIR / "mobilenet_museum_fp16.tflite",
            "img_size": MOBILENET_SIZE,
        },
    ]

    val_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    results = []
    detail_blocks = []

    for pair in pairs:
        for p in (pair["keras"], pair["tflite"]):
            if not p.exists():
                raise FileNotFoundError(f"Missing: {p}")

        img_size = pair["img_size"]
        if img_size not in val_cache:
            print(f"Loading validation set at {img_size}...")
            val_cache[img_size] = load_val_arrays(img_size)
        images, labels = val_cache[img_size]

        print(f"Verifying {pair['name']}...")
        r = verify_pair(pair["name"], pair["keras"], pair["tflite"], images, labels)
        results.append(r)
        block = format_result(r)
        detail_blocks.append(block)
        print(block)

    summary = format_summary_table(results)
    print(summary)

    text = "\n".join(detail_blocks) + "\n" + summary
    RESULTS_PATH.write_text(text)
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
