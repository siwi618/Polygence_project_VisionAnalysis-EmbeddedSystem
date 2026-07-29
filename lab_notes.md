Flower_fp16.tflite in this file is the model I trained following the instructions in the official tensorflow website: 
[Official intructions](https://www.tensorflow.org/tutorials/images/classification)

## Flower Training 
I first run the flower.py and output the model called flower_model.keras.
As the screenshot shows, the value accuracy increase from 0.52 to 0.63, and the val_loss decrease from 1.1215 to 1.0146.
However, the model experiences a few overfitting problems.
Because flower_model.keras is too big nearly 40MB, I use flower_converter.py to compress it. The new model which is uploaded now is only 6MB
Float16 quantization stores each weight as a 16-bit floating point number instead of 32-bit. All neurons and weights are preserved. Only the numerical precision is reduced.

Later, I use model_compare.py to compare the accuracy of flower_fp16.tflite which is compressed with that of flower.keras. Out of my expectation, TFLite is statistically indistinguishable from Keras on this dataset: they agree on 733 out of 734 images, with sub-millisecond probability differences. The accuracy difference of +0.0014 is within the noise floor of the dataset.

## Museum CNN Training (Run 1)

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
- Loss/accuracy plot: [Graph](https://github.com/siwi618/Polygence_project_VisionAnalysis-EmbeddedSystem/blob/main/training_accuracy.png)
- Json history: [File](https://github.com/siwi618/Polygence_project_VisionAnalysis-EmbeddedSystem/blob/main/training_history.json)

**Observations:**
[I think with simply expanding dataset and adding argumentation the accuracy of model is already high]

## Museum MobileNet Training (Run 1)

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
- Loss/accuracy plot: [Graph](None yet)
- Json history: [File](https://github.com/siwi618/Polygence_project_VisionAnalysis-EmbeddedSystem/blob/main/mobilenet_training_history.json)

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
