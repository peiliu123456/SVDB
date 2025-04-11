SVDB: Semantic-Preserving Virtual Decision Boundary Mitigate Semantic Drift in Continual Test-Time Adaptation

This is the official project repository for SVDB: Semantic-Preserving Virtual Decision Boundary Mitigate Semantic Drift in Continual Test-Time Adaptation. This repository is built based on the [SAR 🔗](https://github.com/mr-eggplant/SAR).

<p align="center">
<img src="overall.png" alt="SVDB" width="100%" align=center />
</p>

**Installation**:
This repository contains code for evaluation on ImageNet-C,CIFAR10-C,CIFAR100-C, ImageNet-R, VisDA-2021, ACDC with ViT.
For CIFAR10-C and CIFAR100-C, the pre-trained ViT-B model weights are from [MAE (CVPR 2024) 🔗](https://github.com/RanXu2000/continual-mae?tab=readme-ov-file)
For ImageNet-C,ImageNet-R and  VisDA-2021, the pre-trained ViT-B model weights are from timm .

**Dataset Download**:

[CIFAR100-C 🔗](https://zenodo.org/records/3555552)
[CIFAR10-C 🔗](https://zenodo.org/records/2535967)
[ImageNet-C 🔗](https://zenodo.org/records/2235448#.Yj2RO_co_mF)


**Details of ours code**:
In ./models/svdb\_transformer.py, we implemented Semantic Knowledge Preservation. In vdb\_loss.py, we implemented Virtual Decision Boundary.

**Usage in ImageNet-C**:
```
python main.py --method svdb --test_batch_size 64 --lr 1e-3  --num_classes 1000 --device 0  --data_corruption XX
python main.py --method dct --test_batch_size 64 --lr 1e-3  --num_classes 1000 --device 0  --data_corruption XX
python main.py --method dpal --test_batch_size 64 --lr 1e-3  --num_classes 1000 --device 0  --data_corruption XX
python main.py --method sar --test_batch_size 64 --lr 1e-3  --num_classes 1000 --device 0  --data_corruption XX
python main.py --method tent --test_batch_size 64 --lr 1e-3  --num_classes 1000 --device 0  --data_corruption XX
python main.py --method cotta --test_batch_size 64 --lr 1e-4  --num_classes 1000 --device 0  --data_corruption XX
```
**Usage in ImageNet-C BATCH SIZE=1**:
```
python main.py --method svdb --test_batch_size 1 --lr 1e-4 --num_classes 1000 --device 0  --data_corruption XX
python main.py --method dct --test_batch_size 1 --lr 1e-3  --num_classes 1000 --device 0  --data_corruption XX
python main.py --method dpal --test_batch_size 1 --lr 1e-3  --num_classes 1000 --device 0  --data_corruption XX
python main.py --method sar --test_batch_size 1 --lr 1e-3  --num_classes 1000 --device 0  --data_corruption XX
python main.py --method tent --test_batch_size 1 --lr 1e-3  --num_classes 1000 --device 0  --data_corruption XX
python main.py --method cotta --test_batch_size 1 --lr 1e-4  --num_classes 1000 --device 0  --data_corruption XX
```



**Usage in CIFAR10-C and CIFAR100-C**:
```
python cifar_main.py --method svdb --test_batch_size 32 --lr 1e-3 --num_classes 10 --device 0
python cifar_main.py --method svdb --test_batch_size 32 --lr 1e-3  --num_classes 100 --device 0

python cifar_main.py --method dct --test_batch_size 32 --lr 1e-3  --num_classes 10 --device 0
python cifar_main.py --method dct --test_batch_size 32 --lr 1e-3  --num_classes 100 --device 0

python cifar_main.py --method dpal --test_batch_size 32 --lr 1e-3  --num_classes 10 --device 0
python cifar_main.py --method dpal --test_batch_size 32 --lr 1e-3  --num_classes 100 --device 0

python cifar_main.py --method sar --test_batch_size 32 --lr 1e-3  --num_classes 10 --device 0
python cifar_main.py --method sar --test_batch_size 32 --lr 1e-3  --num_classes 100 --device 0

python cifar_main.py --method tent --test_batch_size 32 --lr 1e-3  --num_classes 10 --device 0
python cifar_main.py --method tent --test_batch_size 32 --lr 1e-3  --num_classes 100 --device 0

python cifar_main.py --method cotta --test_batch_size 32 --lr 1e-4  --num_classes 10 --device 0
python cifar_main.py --method cotta --test_batch_size 32 --lr 1e-4  --num_classes 100 --device 0
```

## Acknowledgment
The code is inspired by the [DCT (MM 2024) 🔗](https://github.com/yushuntang/DCT) and [SAR (ICLR 2023) 🔗](https://github.com/mr-eggplant/SAR).
