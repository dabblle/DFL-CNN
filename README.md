# fine-grained cars classification
This is a simple pytorch re-implementation of CVPR 2018 Learning a Discriminative Filter Bank Within a CNN for Fine-Grained Recognition.
### Introduction: ###
The features are summarized blow：
* Use VGG16 as base Network.
* got best result __Top1__ 94%
* Datasets 35042 images __Train__/ __Test__ ( 60% / 40%)
### Pipeline ###
- process_data
- YOLOV3
- color detector
- DFL-CNN
### Usage ###
> sh run.sh
  
  