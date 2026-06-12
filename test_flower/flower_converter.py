# Import libraries
import tensorflow as tf
from pathlib import Path

# Path(__file__).resolve() would get the path of flower_converter.py, .parent woulf get the parent file, like test_flower
base_dir = Path(__file__).resolve().parent

# Load the model, flower_model.keras in test flower
model = tf.keras.models.load_model(base_dir / "flower_model.keras")

# Generate a converter for the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Give parameters, default optimizations and transforming supported types from 32bits(original) to 16bits
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# Start the converter, come out with bytes
tflite_model = converter.convert()

# Create a file called "flower_fp16.tflite", "wb" means write in binary
with open(base_dir / "flower_fp16.tflite", "wb") as f:
    # Write bytes into file
    f.write(tflite_model)
