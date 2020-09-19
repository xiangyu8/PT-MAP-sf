Few-Shot Learning by Integreting Spatial and Frequency Representation
=======

A few-shot classification algorithm: [Charting the Right Manifold: Manifold Mixup for Few-shot Learning](https://arxiv.org/pdf/1907.12087.pdf)

Our code is built upon the code base of 

[A Closer Look at Few-shot Classification](https://openreview.net/pdf?id=HkxLXnAcFQ), 

[Charting the Right Manifold: Manifold Mixup for Few-shot Learning](https://github.com/nupurkmr9/S2M2_fewshot.git), 

[Learning in the Frequency Domain](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_Learning_in_the_Frequency_Domain_CVPR_2020_paper.pdf) ([code](https://github.com/calmevtime/DCTNet.git))

[Leveraging the Feature Distribution in Transfer-based Few-Shot Learning](https://arxiv.org/pdf/2006.03806v2.pdf) ([code](https://github.com/yhu01/PT-MAP))

Running the code
------------
**Dataset**: mini-ImageNet, CIFAR-FS, CUB

***Donwloading the dataset***:

CUB

* Change directory to filelists/CUB/
* run 'source ./download_CUB.sh' 

CIFAR-FS
* Change directory to filelists/cifar/
* run 'source ./download_cifar.sh' 

miniImagenet
* Change directory to filelists/miniImagenet/
* run 'source ./download_miniImagenet.sh' 


**Training**

DATASETNAME: miniImagenet/cifar/CUB

METHODNAME: S2M2_R/rotation/manifold_mixup

1) train frequency version:
```
	python train_dct.py --dataset [DATASETNAME] --method [METHODNAME] --model WideResNet28_10 --train_aug --dct_status
```	

2) train spatial version:
```
	python train_dct.py --dataset [DATASETNAME] --method [METHODNAME] --model WideResNet28_10 --train_aug
```
		
All results will be saved in the folder "checkpoints", the best model is "best.tar".

**Saving the features of a checkpoint for checkpoint evalution**
***S2M2_R algorithm***
1) save frequency version:
```
	python save_features.py --dataset [DATASETNAME] --method [METHODNAME] --model WideResNet28_10 --train_aug --dct_status
```
2) save spatial version:
```
	python save_features.py --dataset [DATASETNAME] --method [METHODNAME] --model WideResNet28_10 --train_aug
```
3) save spatial and frequency version:
```
	python save_features_both.py --dataset [DATASETNAME] --method [METHODNAME] --model WideResNet28_10 --train_aug
```
***PT-MAP algorithm***
1) save frequency version:
```
	python save_plk.py --dataset [DATASETNAME] --method [METHODNAME] --model WideResNet28_10 --train_aug --dct_status
```
2) save spatial version:
```
	python save_plk.py --dataset [DATASETNAME] --method [METHODNAME] --model WideResNet28_10 --train_aug
```
3) save spatial and frequency version:
```
	python save_plk_both.py --dataset [DATASETNAME] --method [METHODNAME] --model WideResNet28_10 --train_aug
```

All features will be saved in the folder "features".


**Evaluating the few-shot performance**
***S2M2_R algorithm***
1) test frequency version:
```
	python test_dct.py --dataset [DATASETNAME] --method [METHODNAME] --model WideResNet28_10 --n_shot [1/5] --train_aug --dct_status	
```

2) test spatial version:
```
	python test_dct.py --dataset [DATASETNAME] --method [METHODNAME] --model WideResNet28_10 --n_shot [1/5] --train_aug	
```
3) test spatial and frequency version:
```
	python test_dct_both.py --dataset [DATASETNAME] --method [METHODNAME] --model WideResNet28_10 --n_shot [1/5] --train_aug	
```
***PT-MAP algorithm***
```
	python test_standard.py	
```
Revise the .plk file folder in FSLtask.py for frequency(out.plk), spatial(out.plk), frequency+spatial versions (out_both.plk);
Revise the n_shot in test_standard.py to get result of 5-shot or 1-shot.

Comparison with prior/current state-of-the-art methods on mini-ImageNet, CUB and CIFAR-FS dataset.
Note: We implemented LEO on CUB dataset. Other numbers are reported directly from the paper. 


|      Method    | mini-ImageNet |               |      CUB      |               |   CIFAR-FS     |               |
|:--------------:|:-------------:|:-------------:|:-------------:|:-------------:|:--------------:|:-------------:|
|                |     1-shot    |     5-shot    |     1-shot    |     5-shot    |    1-shot      |     5-shot    |
|   Baseline++   | 57.33 +- 0.10 | 72.99 +- 0.43 |  70.4 +- 0.81 |  82.92 +-0.78 | 67.5 +- 0.64   | 80.08 +- 0.32 |
|       LEO      | 61.76 +- 0.08 | 77.59 +- 0.12 |  68.22+- 0.22 | 78.27 +- 0.16 |       -        |       -       |
|       DCO      | 62.64 +- 0.61 | 78.63 +- 0.46 |       -       |       -       | 72.0 +- 0.7    | 84.2 +- 0.5   |
| Manifold Mixup | 57.6 +- 0.17  | 75.89 +- 0.13 | 73.47 +- 0.89 | 85.42 +- 0.53 | 69.20 +- 0.2   | 83.42 +- 0.15 |               
|    Rotation    | 63.9 +- 0.18  | 81.03 +- 0.11 | 77.61 +- 0.86 | 89.32 +- 0.46 | 70.66 +- 0.2   | 84.15 +- 0.14 |
|     S2M2_R     | 64.93 +- 0.18 | 83.18 +- 0.11 | 80.68 +- 0.81 | 90.85 +- 0.44 | 74.81 +- 0.19  | 87.47 +- 0.13 |


References
------------
[A Closer Look at Few-shot Classification](https://openreview.net/pdf?id=HkxLXnAcFQ)

[Meta-Learning with Latent Embedding Optimization](https://arxiv.org/pdf/1807.05960.pdf)

[Meta Learning with Differentiable Convex Optimization](https://arxiv.org/pdf/1904.03758.pdf)

[Manifold Mixup: Better Representations by Interpolating Hidden States](http://proceedings.mlr.press/v97/verma19a.html)
