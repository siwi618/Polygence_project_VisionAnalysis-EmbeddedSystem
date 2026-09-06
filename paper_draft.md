# Lightweight Image Classification for Museum Artifact Recognition on Raspberry Pi 5: Custom CNN vs. MobileNetV2 with TFLite Quantisation

## Abstract
This paper proposes comparing two frameworks: a custom CNN and MobileNetV2, both with TFLite fp16 quantisation, and deploying a TFLite model on an embedded system to personalise museum experiences.
The embedded system this paper chooses is a Raspberry Pi 5 single board.
In the final comparison, the CNN fp16 accuracy is 85.37%, and MobileNetV2 accuracy is 90.24%.
The CNN inference time on Pi is 6.79 ms/img, and the MobileNetV2 inference time is 11.65 ms/img.
Moreover, through quantisation, the CNN file size is reduced to 6.3 MB, and the MobileNetV2 file size is 4.3 MB.
Hence, the deployment of CNN on the Pi 5 is feasible, but MobileNetV2 is still the optimal model.
