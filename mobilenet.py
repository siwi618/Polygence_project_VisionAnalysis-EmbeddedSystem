"""MobileNetV2 transfer learning on museum data.

Phase 1: freeze base, train head only.
Phase 2: unfreeze top layers, fine-tune with lower LR.
"""

import json
import pathlib

import tensorflow as tf

# 1. Local dataset (same as museum.py)
data_dir = pathlib.Path("dataset")

# 2. Params — MobileNet prefers 160/224; museum.py uses 180
batch_size = 32
img_size = (160, 160)
IMG_SHAPE = img_size + (3,)

initial_epochs = 10  # Phase 1
fine_tune_epochs = 10  # Phase 2
base_learning_rate = 1e-4
fine_tune_at = 100  # freeze layers before this in Phase 2

# 3. Load data
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)
print("Number of classes:", num_classes)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 4. Augmentation — on at train, off at val/infer
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomBrightness(0.1),
    ],
    name="data_augmentation",
)

# 5. NEW: pretrained ImageNet backbone (no 1000-class top)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # Phase 1: freeze

# 6. Build model — Rescaling maps [0,255] → [-1,1] for MobileNet
#    (same math as mobilenet_v2.preprocess_input; do NOT also use that)
#    0 → -1,  127.5 → 0,  255 → 1
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(x)
x = base_model(x, training=False)  # keep BatchNorm in infer mode
x = tf.keras.layers.GlobalAveragePooling2D()(x)  # NEW: vs Flatten
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes)(x)  # logits
model = tf.keras.Model(inputs, outputs)

model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1,
)

# 7. Phase 1 — train head only
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

print("\nPhase 1: feature extraction (base frozen)")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=initial_epochs,
    callbacks=[early_stop],
)

# 8. Phase 2 — unfreeze top of base, lower LR
base_model.trainable = True
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
print(
    f"\nFine-tuning from layer {fine_tune_at}: {trainable_count} trainable base layers"
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate / 10),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Fresh EarlyStopping for phase 2 (previous callback already used / may have stopped)
early_stop_ft = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1,
)

total_epochs = initial_epochs + fine_tune_epochs
print("\nPhase 2: fine-tuning")
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1] + 1,
    callbacks=[early_stop_ft],
)

# 9. Merge histories for plotting
merged = {}
for key in history.history:
    merged[key] = history.history[key] + history_fine.history[key]

with open("mobilenet_training_history.json", "w") as f:
    json.dump(merged, f, indent=2)
print("Training history saved to mobilenet_training_history.json")
print(
    f"Phase 1 epochs: {len(history.history['loss'])}, "
    f"Phase 2 epochs: {len(history_fine.history['loss'])}"
)

# 10. Save (names differ from museum.py so they don't overwrite)
model.save("mobilenet_museum_model.keras")
with open("mobilenet_class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)
print("Model saved to mobilenet_museum_model.keras")
