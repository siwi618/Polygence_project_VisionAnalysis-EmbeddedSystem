import json
import pathlib

import tensorflow as tf

# 1. Local museum dataset
data_dir = pathlib.Path("dataset")

# 2. Params
batch_size = 32
img_size = (180, 180)

# 3. Load train / val (same seed → same split)
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

# Debug: check one val batch
for images, labels in val_ds.take(1):
    print(f"Batch shape: {images.shape}")
    print(f"First 5 labels: {labels[:5].numpy()}")
    print(
        f"Pixel range: {tf.reduce_min(images).numpy():.2f} to {tf.reduce_max(images).numpy():.2f}"
    )

# 4. Class names (= subfolder names under dataset/)
class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)
print("Number of classes:", num_classes)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Augmentation — on at train, off at val/infer
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomBrightness(0.1),
    ]
)


def debug_visualize_augmentation(
    dataset, augmentation, class_names, num_images=4, save_path="augmentation_debug.png"
):
    """Save a few original vs augmented images to check augmentation works."""
    import matplotlib.pyplot as plt

    images, labels = next(iter(dataset))
    images = images[:num_images]
    labels = labels[:num_images]

    augmented = augmentation(images, training=True)

    fig, axes = plt.subplots(num_images, 2, figsize=(6, 2.5 * num_images))
    if num_images == 1:
        axes = [axes]

    for i, (orig, aug, label) in enumerate(zip(images, augmented, labels)):
        for ax, img, title in zip(
            axes[i],
            (orig, aug),
            ("Original", "Augmented"),
        ):
            display = tf.clip_by_value(img, 0, 255).numpy().astype("uint8")
            ax.imshow(display)
            ax.set_title(f"{title} — {class_names[int(label)]}")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Augmentation debug plot saved to {save_path}")
    print(
        "If augmented images look identical to originals every run, augmentation is not active."
    )


debug_visualize_augmentation(train_ds, data_augmentation, class_names)

# 5. Simple CNN
model = tf.keras.Sequential(
    [
        data_augmentation,
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(16, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes),
    ]
)
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Stop if val_loss does not improve; restore best weights
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1,
)

# 6. Train (epochs is a ceiling; early stopping may finish sooner)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[early_stop],
)

with open("training_history.json", "w") as f:
    json.dump(history.history, f, indent=2)
print("Training history saved to training_history.json")
print(
    f"Stopped at epoch {len(history.history['loss'])} "
    "(best weights restored if early-stopped)"
)

# 7. Save model
model.save("museum_model.keras")
