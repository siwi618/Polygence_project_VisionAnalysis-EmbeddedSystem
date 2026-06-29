import tensorflow as tf
import numpy as np
import pathlib

# Prepare validation set
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file("flower_photos", origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir) / "flower_photos"

base_dir = pathlib.Path(__file__).resolve().parent

batch_size = 32
img_size = (180, 180)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
)

# Load two models
keras_model = tf.keras.models.load_model(base_dir / "flower_model.keras")

interpreter = tf.lite.Interpreter(model_path=str(base_dir / "flower_fp16.tflite"))

# Allocate memory for inference
interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = input_details[0]["index"]
output_index = output_details[0]["index"]
input_dtype = input_details[0]["dtype"]


def run_tflite_batch(images: np.ndarray):
    """run tflite inference for a batch of images"""
    results = []
    for img in images:
        # add batch dimension: (180, 180, 3) -> (1, 180, 180, 3)
        batch_input = np.expand_dims(img, axis=0).astype(input_dtype)
        interpreter.set_tensor(input_index, batch_input)
        interpreter.invoke()
        results.append(interpreter.get_tensor(output_index)[0])  # shape (5,)
    return np.stack(results, axis=0)  # shape (batch_size, 5)


# Evaluate on validation set
keras_correct = 0
tflite_correct = 0
agree_count = 0
total = 0
max_prob_diff = 0.0
prob_mae_sum = 0.0

for images, labels in val_ds:

    images_np = images.numpy()
    labels_np = labels.numpy()

    # Keras valid in bulk
    keras_logits = keras_model.predict(images_np, verbose=0)

    # TFLite valid in bulk
    tflite_logits = run_tflite_batch(images_np)

    # argmax(logits) is same to argmax(softmax(logits)) 
    keras_preds = np.argmax(keras_logits, axis=1)
    tflite_preds = np.argmax(tflite_logits, axis=1)

    # compare probability distribution to reflect the actual impact of quantization
    keras_prob = tf.nn.softmax(keras_logits, axis=1).numpy()
    tflite_prob = tf.nn.softmax(tflite_logits, axis=1).numpy()
    prob_diff = np.abs(keras_prob - tflite_prob)

    batch_size_actual = len(labels_np)
    keras_correct += np.sum(keras_preds == labels_np)
    tflite_correct += np.sum(tflite_preds == labels_np)
    agree_count += np.sum(keras_preds == tflite_preds)
    total += batch_size_actual

    max_prob_diff = max(max_prob_diff, np.max(prob_diff))
    prob_mae_sum += np.sum(prob_diff)

# Output comparison results
keras_acc = keras_correct / total
tflite_acc = tflite_correct / total
agreement = agree_count / total
prob_mae = prob_mae_sum / (total * keras_logits.shape[1])

print(f"validation set size: {total}")
print(f"Keras accuracy: {keras_acc:.4f} ({keras_correct}/{total})")
print(f"TFLite accuracy: {tflite_acc:.4f} ({tflite_correct}/{total})")
print(f"accuracy change:    {tflite_acc - keras_acc:+.4f}")
print(f"agreement rate:    {agreement:.4f} ({agree_count}/{total})")
print(f"max probability difference:    {max_prob_diff:.6f}")
print(f"average probability difference:    {prob_mae:.6f}")
