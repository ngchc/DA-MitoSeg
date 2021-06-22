Uncertainty-Aware Label Rectification for Domain Adaptive Mitochondria Segmentation
====
Siqi Wu, Chang Chen, Zhiwei Xiong, Xuejin Chen, Xiaoyan Sun. Uncertainty-Aware Label Rectification for Domain Adaptive Mitochondria Segmentation. In MICCAI 2021. <br/>

## Requirements
Anaconda>=5.2.0 (Python 3.6) <br/>
PyTorch>=1.1.0 <br/>
One or more GPUs with sufficient memory <br/>
Memory>=128 GB for data caching <br/>

## Datasets
To access MitoEM, download files from [https://mitoem.grand-challenge.org](https://mitoem.grand-challenge.org) and convert to arrays. <br/>
```
cd mitoem
python png2npy.py
python tif2npy.py
```
Otherwise, download .npy files from [https://pan.baidu.com/s/1wt1giVGjreYXuArxfOuo1Q ](https://pan.baidu.com/s/1wt1giVGjreYXuArxfOuo1Q) (key: us4d). <br/>
```
unzip mitoem-train.zip -d mitoem
unzip mitoem-valid.zip -d mitoem
```
To access a part of raw images from FAFB and the corresponding labels.
```
cd fafb-valid/im
cd fafb-valid/seg
```

## Test the pre-trained models
For the reproduction of results listed in Tables 1 and 2.
```
cd <Name-of-Folder>
python inference.py
```
Note that, it may take minutes to calculate all of the four metrics.

## Train the model
Train an uncertainty-aware model with data from source domain (Rat).
```
cd u2d-bc-rat-uc-train
python main.py
```
Generate, rectify, and cache pseudo labels for training.
```
cd u2d-bc-r2h-train
python inference_4train.py
python generate_mask.py
```
Train a model with generated labels on target domain (Human).
```
python main.py
```
