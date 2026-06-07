Flower_fp16.tflite in this file is the model I trained following the instructions in the official tensorflow website: 
[Official intructions](https://www.tensorflow.org/tutorials/images/classification)

I first run the flower.py and output the model called flower_model.keras.
As the screenshot shows, the value accuracy increase from 0.52 to 0.63, and the val_loss decrease from 1.1215 to 1.0146.
However, the model experiences a few overfitting problems.
Because flower_model.keras is too big nearly 40MB, I use flower_converter.py to compress it. The new model which is uploaded now is only 6MB
