import tensorflow as tf
from pathlib import Path

base_dir = Path(__file__).resolve().parent

model = tf.keras.models.load_model(base_dir / "museum_model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open(base_dir / "museum_fp16.tflite", "wb") as f:
    f.write(tflite_model)
