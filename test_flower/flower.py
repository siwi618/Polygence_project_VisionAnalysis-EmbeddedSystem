import tensorflow as tf # the main libraries to achieve recognition
import pathlib # easier to code the pathway

# 1. Download flower dataset

# source of the flower dataset
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz" 

# download dataset
data_dir = tf.keras.utils.get_file("flower_photos", origin=dataset_url, untar=True) 

# save the dataset into five categories
data_dir = pathlib.Path(data_dir) / "flower_photos"  

# 2. Parameter

# feed 32 images at once
batch_size = 32 

# edit the size of images into the same
img_size = (180, 180) 

# 3. Read data

# training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    
    # import dataset
    data_dir, 
    
    # the ratio of training subset and validation subset is 8:2
    validation_split=0.2, 
    
    # read training subset because it's a train_ds
    subset="training",
    
    # a fixed seed which can generate the same random sequence all the time, so it can make sure that the model always gets same results
    seed=123, 
    
    # import image_size
    image_size=img_size,
    
    # import batch_size
    batch_size=batch_size,
)

# validation dataset, most of the parametres are the same
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    # read validation subset because it's a val_ds
    subset="validation", 
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
)

# 4. Perfeomance optimization

# TensorFlow would choose the optimal numbers of progresses automatically based on the performance of CPU, GPU and Memory
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
"""
cache the dataset
shuffle 1000 images first and pick randomly from it to improve model's performance and avoid overfitting
prefetch let CPU and GPU work at the same time, so there is no need for GPU to wait
"""

val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
#there is no need to shuffle because order does not affect the accuracy

# 5. Build a model
model = tf.keras.Sequential( # Sequential means execute the code following the order in the list
    [
        # let every pixel value divided by (1.0 / 255) to transform the range of them from 0-255 to 0-1 , because a smaller value is better for deep learning.
        tf.keras.layers.Rescaling(1.0 / 255),

        # 16 3*3 filters scan the images, "relu" is used to eliminate noise, highlight features and achieve nonlinearity
        tf.keras.layers.Conv2D(16, 3, activation="relu"),

        # take the maximum number in every 2*2(default) area in order to maintain the features, and at the same time, reduce the size of the images
        tf.keras.layers.MaxPooling2D(),

        # 32 3*3 filters scan the images
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        # 64 3*3 filters scan the images
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        # flatten the latest output in three dimensions into one dimensions, because Dense could only receive one dimension input
        tf.keras.layers.Flatten(),

        # put the number list into 128 neurons to match each features and output similarity
        tf.keras.layers.Dense(128, activation="relu"),

        # put the similarity into 5 neurons which correspond to each class and output which class is the most similar
        tf.keras.layers.Dense(5),
    ]
)

# 6. Compile
model.compile(

    
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# 7. 
model.fit(train_ds, validation_data=val_ds, epochs=5)

# 8. Save
model.save("flower_model.keras")
