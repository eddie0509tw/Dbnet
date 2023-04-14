# Real-time-Text-Detection

PyTorch re-implementation of [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947)

<img src="https://github.com/SURFZJY/Real-time-Text-Detection/blob/master/demo/dbnet.png" alt="contour" >

### Difference between thesis and this implementation
//TODO

### Introduction
Mainly reimplement from :
- https://github.com/SURFZJY/Real-time-Text-Detection-DBNet/tree/b76ee8d840b1fcebf7b9545402907416c7daf24e 

Also thanks to these project:

- https://github.com/WenmuZhou/PAN.pytorch
- https://github.com/d-li14/mobilenetv2.pytorch/tree/1733532bd43743442077326e1efc556d7cfd025d

The features are summarized blow:

+ Use **resnet18/resnet50/shufflenetV2/mobilenetV2** as backbone.  

### Contents

1. [Installation](#installation)
2. [Download](#download)
3. [Train](#train)
4. [Predict](#predict)
5. [Eval](#eval)
6. [Demo](#demo)


### Installation

1. pytorch 1.1.0
 
### Download

1. ShuffleNet_V2 Models trained on ICDAR 2013+2015 (training set) 

https://pan.baidu.com/s/1Um0wzbTFjJC0jdJ703GR7Q

or https://mega.nz/#!WdhxXAxT!oGURvmbQFqTHu5hljUPdbDMzI75_UO2iWLaXX5dJrDw

### Train

1. modify genText.py to generate txt list file for training/testing data

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

### Examples



