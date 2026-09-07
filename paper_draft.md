# Lightweight Image Classification for Museum Artifact Recognition on Raspberry Pi 5: Custom CNN vs. MobileNetV2 with TFLite Quantisation

## Abstract
This study compares two image classification models: a custom CNN and MobileNetV2, both converted to TensorFlow Lite (TFLite) with fp16 quantisation, and deploys a TFLite model on an embedded system to personalise museum experiences.
The embedded system that two models are deployed on is a Raspberry Pi 5 single board to evaluate the feasibility for embedded museum application.
In the final comparison, the CNN fp16 achieved an accuracy of 85.37%, compared with 90.24% for MobileNetV2.
The CNN inference time on Pi is 6.79 ms/img, and the MobileNetV2 inference time is 11.65 ms/img.
Moreover, through quantisation, the CNN file size is reduced to 6.3 MB, and the MobileNetV2 file size is 4.3 MB.
Hence, the deployment of CNN on the Pi 5 is feasible, but MobileNetV2 is still the optimal model.
