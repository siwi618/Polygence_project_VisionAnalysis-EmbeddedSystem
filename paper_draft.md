# Lightweight Image Classification for Museum Artifact Recognition on Raspberry Pi 5: Custom CNN vs. MobileNetV2 with TFLite Quantisation

## Abstract
This study compares two image classification models: a custom CNN and MobileNetV2, both converted to TensorFlow Lite (TFLite) with fp16 quantisation, and deploys a TFLite model on an embedded system to personalise museum experiences.
The embedded system that the two models are deployed on is a Raspberry Pi 5 single board to evaluate the feasibility for an embedded museum application.
In the final comparison, the CNN fp16 achieved an accuracy of 85.37%, compared with 90.24% for MobileNetV2.
On the Raspberry Pi 5, the CNN achieved a lower inference time of 6.79 ms/image, compared with 11.65 ms/image for MobileNetV2.
FP16 quantisation reduced the model sizes to 6.3 MB for the CNN and 4.3 MB for MobileNetV2, respectively.
Hence, these results show that the deployment of the CNN on the Raspberry Pi 5 is feasible, but MobileNetV2 provides a better overall trade-off between accuracy and model size.

## Introduction
In traditional museum operations, museums often provide guidance in group for clients. The problem caused by this is that different clients feel interested with different artifacts. Group guides may not have enough energy to serve everyone in the group. As a result, this paper aims to use lightweight image classifications in embedded systems to give all clients a unique museum experience. 
