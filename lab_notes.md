Flower_fp16.tflite in this file is the model I trained following the instructions in the official tensorflow website: 
[Official intructions](https://www.tensorflow.org/tutorials/images/classification)

## Flower Training 
I first run the flower.py and output the model called flower_model.keras.
As the screenshot shows, the value accuracy increase from 0.52 to 0.63, and the val_loss decrease from 1.1215 to 1.0146.
However, the model experiences a few overfitting problems.
Because flower_model.keras is too big nearly 40MB, I use flower_converter.py to compress it. The new model which is uploaded now is only 6MB
Float16 quantization stores each weight as a 16-bit floating point number instead of 32-bit. All neurons and weights are preserved. Only the numerical precision is reduced.

Later, I use model_compare.py to compare the accuracy of flower_fp16.tflite which is compressed with that of flower.keras. Out of my expectation, TFLite is statistically indistinguishable from Keras on this dataset: they agree on 733 out of 734 images, with sub-millisecond probability differences. The accuracy difference of +0.0014 is within the noise floor of the dataset.

# RUN 1

## Museum CNN Training

**Dataset:** 5 classes (bronze bowl, fish fan, jade figure, pen container, crystal cup), ~40 images per class. fish fan is still 20

**Architecture:** Custom CNN (same as flower.py, output Dense layer = 5)

**Hyperparameters:**
- Image size: 180×180
- Batch size: 32
- Epochs: 30
- Data augmentation: RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness

**Results:**
- Final training accuracy: 0.98
- Final validation accuracy: 0.89
- Best validation accuracy: 0.92 (at epoch 17)
- Loss/accuracy plot: [Graph](https://github.com/siwi618/Polygence_project_VisionAnalysis-EmbeddedSystem/blob/main/Run_1/training_accuracy.png)
- Json history: [File](https://github.com/siwi618/Polygence_project_VisionAnalysis-EmbeddedSystem/blob/main/Run_1/training_history.json)

**Observations:**
[I think with simply expanding dataset and adding argumentation the accuracy of model is already high]

## Museum MobileNet Training

**Dataset:** 5 classes (bronze bowl, fish fan, jade figure, pen container, crystal cup), ~40 images per class. fish fan is still 20

**Architecture:** MobileNet

**Hyperparameters:**
- Image size: 160×160
- Batch size: 32
- initial_epochs = 10 
- fine_tune_epochs = 10 
- base_learning_rate = 1e-4
- fine_tune_at = 100
- Data augmentation: RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness

**Results:**
- Final training accuracy: 0.85
- Final validation accuracy: 0.86
- Best validation accuracy: 0.86 (at epoch 20)
- Loss/accuracy plot: [Graph](https://github.com/siwi618/Polygence_project_VisionAnalysis-EmbeddedSystem/blob/main/Run_1/mobilenet_training_accuracy.png)
- Json history: [File](https://github.com/siwi618/Polygence_project_VisionAnalysis-EmbeddedSystem/blob/main/Run_1/mobilenet_training_history.json)

**Compare all:**

Model                     | Accuracy | File Size | Inference Time
--------------------------|----------|-----------|---------------
Custom CNN (Keras)        |  1.0000  |  37.8 MB  |     27.8 ms
Custom CNN (TFLite fp16)  |  1.0000  |   6.3 MB  |      2.4 ms
MobileNet (Keras)         |  0.7568  |  23.5 MB  |     31.9 ms
MobileNet (TFLite fp16)   |  0.7568  |   4.3 MB  |      5.2 ms

**Compare CNN (Keras vs TFLite):**

Model     |  Accuracy 
----------|------------
Keras     |  1.0000   (37/37)
TFLite    |  1.0000   (37/37)

accuracy change:                +0.0000
agreement rate:                 1.0000 (37/37)
max probability difference:     0.000307
average probability difference: 0.000010

**Compare MobileNet (Keras vs TFLite):**

Model     |  Accuracy 
----------|------------
Keras     |  0.7568   (28/37)
TFLite    |  0.7568   (28/37)

accuracy change:                +0.0000
agreement rate:                 1.0000 (37/37)
max probability difference:     0.008208
average probability difference: 0.001752

# RUN 2

## Museum CNN Training

**Dataset:** 5 classes (bronze bowl, fish fan, jade figure, pen container, crystal cup), ~40 images per class. fish fan is still 20

**Architecture:** Custom CNN

**Hyperparameters:**
- Image size: 180×180
- Batch size: 32
- Epochs: 38
- Data augmentation: RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness

**Results:**
- Final training accuracy: 1
- Final validation accuracy: 0.89
- Best validation accuracy: 0.94 (at epoch 32)
- Loss/accuracy plot: [Graph](https://github.com/siwi618/Polygence_project_VisionAnalysis-EmbeddedSystem/blob/main/Run_2/training_accuracy.png)
- Json history: [File](https://github.com/siwi618/Polygence_project_VisionAnalysis-EmbeddedSystem/blob/main/Run_2/training_history.json)

## Museum MobileNet Training

**Dataset:** 5 classes (bronze bowl, fish fan, jade figure, pen container, crystal cup), ~40 images per class. fish fan is still 20

**Architecture:** MobileNet

**Hyperparameters:**
- Image size: 160×160
- Batch size: 32
- initial_epochs = 10 
- fine_tune_epochs = 10 
- base_learning_rate = 1e-4
- fine_tune_at = 100
- Data augmentation: RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness

**Results:**
- Final training accuracy: 0.88
- Final validation accuracy: 0.65
- Best validation accuracy: 0.65 (at epoch 20)
- Loss/accuracy plot: [Graph](https://github.com/siwi618/Polygence_project_VisionAnalysis-EmbeddedSystem/blob/main/Run_2/mobilenet_training_accuracy.png)
- Json history: [File](https://github.com/siwi618/Polygence_project_VisionAnalysis-EmbeddedSystem/blob/main/Run_2/mobilenet_training_history.json)

**Compare all:**

Model                     | Accuracy | File Size | Inference Time
--------------------------|----------|-----------|---------------
Custom CNN (Keras)        |  0.9189  |  37.8 MB  |     24.9 ms
Custom CNN (TFLite fp16)  |  0.9189  |   6.3 MB  |      2.4 ms
MobileNet (Keras)         |  0.6486  |  23.5 MB  |     31.4 ms
MobileNet (TFLite fp16)   |  0.6486  |   4.3 MB  |      4.7 ms

**Compare CNN (Keras vs TFLite):**

Model     |  Val Accuracy 
----------|----------------
Keras     |    0.9189     (34/37)
TFLite    |    0.9189     (34/37)

val accuracy change (TFLite - Keras): +0.0000
prediction agreement rate (on val):   1.0000 (37/37)
max probability difference (on val):  0.000458
avg probability difference (on val):  0.000018


**Compare MobileNet (Keras vs TFLite):**

Model     |  Val Accuracy 
----------|----------------
Keras     |    0.6486     (24/37)
TFLite    |    0.6486     (24/37)

val accuracy change (TFLite - Keras): +0.0000
prediction agreement rate (on val):   1.0000 (37/37)
max probability difference (on val):  0.007059
avg probability difference (on val):  0.001026

**Observations:**

[Training accuracy of CNN get 100% more frequently, the best validation accuracy remains the same.] It's seem that the validation accuracy of CNN returned to normal (from 1 to 0.9189). However, the overall accuracy of MobileNet is still lower than that of CNN

## Debugging Notes — CNN “100%” validation accuracy

### Symptom
'model_compare' && 'model_verify' reported Custom CNN accuracy **1.0000** (37/37), while training history never sustained that: last 'fit' val accuracy was about **0.89** (peak ~0.92).

### What I tried
- Compare scripts on the same 'seed=123' split.
- Checked 'model(..., training=True)' vs 'training=False' on the val set (augmentation was not the cause of the 1.0 gap).
- Confirmed Keras vs TFLite agreement ≈ 100% with tiny probability diffs -> conversion was fine; the issue was the metric itself.
- Audited train/val construction: 'museum.py' vs 'model_compare.py' / 'model_verify.py' (path, 'validation_split', 'seed', 'shuffle').

### What I found
- Evaluat scripts **rebuilt** the dataset with 'image_dataset_from_directory' instead of reusing training’s 'val_ds'.
- I used **'shuffle=False'** while training used the Keras default **'shuffle=True'**. In Keras, shuffle runs **before** the train/val cut, so this produced a **different validation set**, which can overlap training images → inflated accuracy.
- Dataset is tiny (~187 images, ~37 val). One mistake ≈ 2.7%; near-duplicate shots of the same object make the split easy to 'memorize'. Class sizes are uneven (`fish_fan` smaller).

### Root cause
1. **Primary bug:** mismatched 'shuffle' between train and eval → wrong (over-optimistic / leaky) validation set → reported **100%**.
2. **Underlying ML issue:** small data + overfitting + noisy 37-image val; even on the correct split, high scores are fragile and should not be treated as 'perfect generalization'.

### Fix / takeaway
- Set 'shuffle=True' in 'model_compare.py' and 'model_verify.py' to match 'museum.py' . After the fix, CNN compare accuracy dropped to ~**0.92**, in line with training curves ('Run_2').
- Always report both **fit 'val_accuracy'** and **evaluate accuracy**; do not cite 100% alone.

# RUN 3

## Museum CNN Training

**Dataset:** 5 classes (bronze bowl, fish fan, jade figure, pen container, crystal cup), ~40 images per class.

**Architecture:** Custom CNN (same as flower.py, output Dense layer = 5)

**Hyperparameters:**
- Image size: 180×180
- Batch size: 32
- Epochs: 17
- Data augmentation: RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness

**Results:**
- Final training accuracy: 0.91
- Final validation accuracy: 0.78
- Best validation accuracy: 0.85 (at epoch 9)
- Loss/accuracy plot: [Graph](https://github.com/siwi618/Polygence_project_VisionAnalysis-EmbeddedSystem/blob/main/Run_3/training_accuracy.png)
- Json history: [File](https://github.com/siwi618/Polygence_project_VisionAnalysis-EmbeddedSystem/blob/main/Run_3/training_history.json)

## Museum MobileNet Training

**Dataset:** 5 classes (bronze bowl, fish fan, jade figure, pen container, crystal cup), ~40 images per class.

**Architecture:** MobileNet

**Hyperparameters:**
- Image size: 160×160
- Batch size: 32
- initial_epochs = 10 
- fine_tune_epochs = 10 
- base_learning_rate = 1e-4
- fine_tune_at = 100
- Data augmentation: RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness

**Results:**
- Final training accuracy: 0.90
- Final validation accuracy: 0.90
- Best validation accuracy: 0.90 (at epoch 19)
- Loss/accuracy plot: [Graph](https://github.com/siwi618/Polygence_project_VisionAnalysis-EmbeddedSystem/blob/main/Run_3/mobilenet_training_accuracy.png)
- Json history: [File](https://github.com/siwi618/Polygence_project_VisionAnalysis-EmbeddedSystem/blob/main/Run_3/mobilenet_training_history.json)

**Compare all:**

Model                     | Accuracy | File Size | Inference Time
--------------------------|----------|-----------|---------------
Custom CNN (Keras)        |  0.8537  |  37.8 MB  |     23.5 ms
Custom CNN (TFLite fp16)  |  0.8537  |   6.3 MB  |      2.3 ms
MobileNet (Keras)         |  0.9024  |  23.5 MB  |     29.7 ms
MobileNet (TFLite fp16)   |  0.9024  |   4.3 MB  |      4.5 ms

**Compare CNN (Keras vs TFLite):**

Model     |  Val Accuracy 
----------|----------------
Keras     |    0.8537     (35/41)
TFLite    |    0.8537     (35/41)

val accuracy change (TFLite - Keras): +0.0000
prediction agreement rate (on val):   1.0000 (41/41)
max probability difference (on val):  0.000281
avg probability difference (on val):  0.000023

**Compare MobileNet (Keras vs TFLite):**

Model     |  Val Accuracy 
----------|----------------
Keras     |    0.9024     (37/41)
TFLite    |    0.9024     (37/41)

val accuracy change (TFLite - Keras): +0.0000
prediction agreement rate (on val):   1.0000 (41/41)
max probability difference (on val):  0.010798
avg probability difference (on val):  0.001413

**Observations:**
This run, I added 20 picture of fish fan in another side. Although the pattern is difference, all of these pictures refer to the same class. Comparing to Run_2, the val_accuracy of CNN decrease and become lower than MobileNet, which is I expected in beginning. I guess CNN is more dependent on the dataset, because of its small size.
