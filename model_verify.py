"""Verify museum TFLite models against their Keras counterparts.

Main comparison: Keras vs TFLite (accuracy, agreement, prob diffs).
Runs once for Custom CNN and once for MobileNet.
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
    arch: str,
    keras_path: Path,
    tflite_path: Path,
    images: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Compare Keras vs TFLite on the same validation images."""
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
    return {
        "arch": arch,
        "total": total,
        "keras_acc": float(np.mean(keras_preds == labels)),
        "tflite_acc": float(np.mean(tflite_preds == labels)),
        "keras_correct": int(np.sum(keras_preds == labels)),
        "tflite_correct": int(np.sum(tflite_preds == labels)),
        "agreement": float(np.mean(keras_preds == tflite_preds)),
        "agree_count": int(np.sum(keras_preds == tflite_preds)),
        "max_prob_diff": float(np.max(prob_diff)),
        "avg_prob_diff": float(np.mean(prob_diff)),
    }


def format_result(r: dict) -> str:
    """Keras / TFLite are the Model rows — the main comparison."""
    return "\n".join(
        [
            f"=== {r['arch']} (Keras vs TFLite) ===",
            f"validation set size: {r['total']}",
            "",
            f"{'Model':<10}| {'Accuracy':^10}",
            "-" * 10 + "|" + "-" * 12,
            f"{'Keras':<10}|  {r['keras_acc']:.4f}   ({r['keras_correct']}/{r['total']})",
            f"{'TFLite':<10}|  {r['tflite_acc']:.4f}   ({r['tflite_correct']}/{r['total']})",
            "",
            f"accuracy change:                {r['tflite_acc'] - r['keras_acc']:+.4f}",
            f"agreement rate:                 {r['agreement']:.4f} ({r['agree_count']}/{r['total']})",
            f"max probability difference:     {r['max_prob_diff']:.6f}",
            f"average probability difference: {r['avg_prob_diff']:.6f}",
            "",
        ]
    )


def main() -> None:
    pairs = [
        {
            "arch": "Custom CNN",
            "keras": BASE_DIR / "museum_model.keras",
            "tflite": BASE_DIR / "museum_fp16.tflite",
            "img_size": CNN_SIZE,
        },
        {
            "arch": "MobileNet",
            "keras": BASE_DIR / "mobilenet_museum_model.keras",
            "tflite": BASE_DIR / "mobilenet_museum_fp16.tflite",
            "img_size": MOBILENET_SIZE,
        },
    ]

    val_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    blocks = []

    for pair in pairs:
        for p in (pair["keras"], pair["tflite"]):
            if not p.exists():
                raise FileNotFoundError(f"Missing: {p}")

        img_size = pair["img_size"]
        if img_size not in val_cache:
            print(f"Loading validation set at {img_size}...")
            val_cache[img_size] = load_val_arrays(img_size)
        images, labels = val_cache[img_size]

        print(f"Verifying {pair['arch']}: Keras vs TFLite...")
        r = verify_pair(pair["arch"], pair["keras"], pair["tflite"], images, labels)
        block = format_result(r)
        blocks.append(block)
        print(block)

    text = "\n".join(blocks)
    RESULTS_PATH.write_text(text)
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
