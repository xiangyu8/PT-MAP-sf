Few-Shot Learning by Integreting Spatial and Frequency Representation
=======

A few-shot classification algorithm: [Few-Shot Learning by Integreting Spatial and Frequency Representation]

Our code is built upon the code base of 

[A Closer Look at Few-shot Classification](https://openreview.net/pdf?id=HkxLXnAcFQ), 

[Charting the Right Manifold: Manifold Mixup for Few-shot Learning](https://github.com/nupurkmr9/S2M2_fewshot.git), 

[Learning in the Frequency Domain](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_Learning_in_the_Frequency_Domain_CVPR_2020_paper.pdf) ([code](https://github.com/calmevtime/DCTNet.git))

[Leveraging the Feature Distribution in Transfer-based Few-Shot Learning](https://arxiv.org/pdf/2006.03806v2.pdf) ([code](https://github.com/yhu01/PT-MAP))

Running the code
------------
**Dataset**: mini-ImageNet, CIFAR-FS, CUB

============================================================================

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

============================================================================

**Training**

DATASETNAME: miniImagenet/cifar/CUB

METHODNAME: S2M2_R/rotation

1) train frequency version:
```
	python train_dct.py --dataset [DATASETNAME] --method [METHODNAME] --model WideResNet28_10 --train_aug --dct_status
```	

2) train spatial version:
```
	python train_dct.py --dataset [DATASETNAME] --method [METHODNAME] --model WideResNet28_10 --train_aug
```
		
All results will be saved in the folder "checkpoints", the best model is "best.tar".

============================================================================

**Saving the features of a checkpoint for checkpoint evalution**

***S2M2_R algorithm***
1) save frequency version:
```
	python save_features.py --dataset [DATASETNAME] --method S2M2_R --model WideResNet28_10 --train_aug --dct_status
```
2) save spatial version:
```
	python save_features.py --dataset [DATASETNAME] --method S2M2_R --model WideResNet28_10 --train_aug
```
3) save spatial and frequency version:
```
	python save_features_both.py --dataset [DATASETNAME] --method S2M2_R --model WideResNet28_10 --train_aug
```
***PT-MAP algorithm***
1) save frequency version:
```
	python save_plk.py --dataset [DATASETNAME] --method S2M2_R --model WideResNet28_10 --train_aug --dct_status
```
2) save spatial version:
```
	python save_plk.py --dataset [DATASETNAME] --method S2M2_R --model WideResNet28_10 --train_aug
```
3) save spatial and frequency version:
```
	python save_plk_both.py --dataset [DATASETNAME] --method S2M2_R --model WideResNet28_10 --train_aug
```

All features will be saved in the folder "features".

=================================================================================

**Evaluating the few-shot performance**

***S2M2_R algorithm***
1) test frequency version:
```
	python test_dct.py --dataset [DATASETNAME] --method S2M2_R --model WideResNet28_10 --n_shot [1/5] --train_aug --dct_status	
```

2) test spatial version:
```
	python test_dct.py --dataset [DATASETNAME] --method S2M2_R --model WideResNet28_10 --n_shot [1/5] --train_aug	
```
3) test spatial and frequency version:
```
	python test_dct_both.py --dataset [DATASETNAME] --method S2M2_R --model WideResNet28_10 --n_shot [1/5] --train_aug	
```
***PT-MAP algorithm***
```
	python test_standard.py	
```

Revise the .plk file folder in FSLtask.py for frequency(out.plk), spatial(out.plk), frequency+spatial versions (out_both.plk);

Revise the n_shot in test_standard.py to get result of 5-shot or 1-shot.

Comparison with prior/current state-of-the-art methods on mini-ImageNet, CUB and CIFAR-FS dataset.


|      Method    | mini-ImageNet |               |      CUB      |               |   CIFAR-FS     |               |
|:--------------:|:-------------:|:-------------:|:-------------:|:-------------:|:--------------:|:-------------:|
|                |     1-shot    |     5-shot    |     1-shot    |     5-shot    |    1-shot      |     5-shot    |
|   S2M2_R (s)   | 63.51 +- 0.18 | 81.54 +- 0.12 | 80.55 +- 0.78 | 91.52 +- 0.39 | 73.54 +- 0.20  | 86.90 +- 0.13 |
|   S2M2_R (f)   | 63.03 +- 0.18 | 80.80 +- 0.11 | 81.00 +- 0.76 | 91.08 +- 0.40 | 72.21 +- 0.20  | 85.72 +- 0.13 |
|  S2M2_R (s+f)  | 66.96 +- 0.18 | 84.31 +- 0.10 | 84.87 +- 0.72 | 93.52 +- 0.35 | 76.60 +- 0.19  | 88.55 +- 0.13 |
|   PT-MAP(s)    | 81.01 +- 0.25 | 88.07 +- 0.13 | 92.25 +- 0.18 | 94.62 +- 0.09 | 87.82 +- 0.22  | 91.00 +- 0.16 |
|   PT-MAP(f)    | 82.04 +- 0.23 | 88.68 +- 0.12 | 93.18 +- 0.16 | 95.02 +- 0.08 | 86.57 +- 0.23  | 90.28 +- 0.15 |
|   PT-MAP(s+f)  | 85.01 +- 0.22 | 90.72 +- 0.11 | 95.45 +- 0.13 | 96.70 +- 0.07 | 89.39 +- 0.21  | 92.08 +- 0.15 |


References
------------
[A Closer Look at Few-shot Classification](https://openreview.net/pdf?id=HkxLXnAcFQ)

[Meta-Learning with Latent Embedding Optimization](https://arxiv.org/pdf/1807.05960.pdf)

[Meta Learning with Differentiable Convex Optimization](https://arxiv.org/pdf/1904.03758.pdf)

[Manifold Mixup: Better Representations by Interpolating Hidden States](http://proceedings.mlr.press/v97/verma19a.html)
