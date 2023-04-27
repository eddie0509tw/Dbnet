# CIS 519 Applied Machine Learning Final Project :Manga Text Detection

PyTorch re-implementation of [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947)

<img src="https://github.com/SURFZJY/Real-time-Text-Detection/blob/master/demo/dbnet.png" alt="contour" >

### Difference between previous project and this implementation

1. We try the network on Manga109 dataset for manga text detection
2. Try MobileNetv2 and MobileNetV3(Large) as backbone for dbnet.
3. Concatenate a Unet after FPN head

Note: Manga109 dataset is from University of Tokyo, and we are not allowed to public the dataset. If needed, please request permission from http://www.manga109.org/

### Introduction
Mainly reimplement and inherit from :
- https://github.com/SURFZJY/Real-time-Text-Detection-DBNet

Also thanks to these project:

- https://github.com/WenmuZhou/PAN.pytorch
- https://github.com/d-li14/mobilenetv2.pytorch
- https://github.com/d-li14/mobilenetv3.pytorch

The features are summarized blow:

+ Use **resnet18/resnet50/shufflenetV2/mobilenetV2andV3** as backbone.  
+ Use **Unet** as afterburner.  


### Contents

1. [Installation](#installation)
2. [Download](#download)
3. [Train](#train)
4. [Predict](#predict)
5. [Eval](#eval)
6. [Demo](#demo)


### Installation

1. pytorch 1.1.0
 

### Train

1. Go to manga process to generate txt list file for training/testing data

2. modify config.json

3. run 

```python
python train.py
```

### Predict

1. run 
```python
python predict.py
```

### Eval

run
```python
python eval.py
```
