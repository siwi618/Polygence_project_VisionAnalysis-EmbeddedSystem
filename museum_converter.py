import tensorflow as tf
from pathlib import Path

base_dir = Path(__file__).resolve().parent

model = tf.keras.models.load_model(base_dir / "museum_model.keras")


# Pin batch=1 so TFLite input is [1, H, W, 3] (not frozen batch=4)
concrete = tf.function(lambda x: model(x, training=False))
concrete = concrete.get_concrete_function(tf.TensorSpec([1, 180, 180, 3], tf.float32))
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open(base_dir / "museum_fp16.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved museum_fp16.tflite")
