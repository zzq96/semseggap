## clone from https://bitbucket.org/JianboJiao/semseggap/src/master/


# Geometry-Aware Distillation for Indoor Semantic Segmentation

By [Jianbo Jiao](https://jianbojiao.com/), [Yunchao Wei](https://weiyc.github.io/), [Zequn Jie](http://jiezequn.me/), Honghui Shi, [Rynson W. H. Lau](http://www.cs.cityu.edu.hk/~rynson/), [Thomas S. Huang](http://ifp-uiuc.github.io/).


### Introduction

This repository contains the codes and models described in the paper "[Geometry-Aware Distillation for Indoor Semantic Segmentation](https://jianbojiao.com/pdfs/cvpr_GAD.pdf)" . This work addresses the problem of semantic segmentation for indoor scenes, by incorporating geometry-aware knowledge implicitly.

### Usage

0. The project was implemented and tested with Python 2.7, [PyTorch](https://pytorch.org) (version 0.3) and [TorchVision](https://pytorch.org/docs/0.2.0/) (version 0.2.0) on Linux with GPUs. Please setup the environment according to the [instructions](https://pytorch.org/get-started/previous-versions/) first.
1. Download the model with the [script](models/download.sh) under folder "models".
2. Run the [main.py](main.py) to evaluate the performance on the NYUD-v2 dataset for semantic segmentation.
3. You may optionally store the network predictions (color-coded results) by uncommenting line 70 in [main.py](main.py). The results will be saved to a folder "outimgs".

### Citation

If you find these codes and models useful in your research, please cite:

	@InProceedings{CVPR19_GAD,
		author = {Jianbo Jiao, Yunchao Wei, Zequn Jie, Honghui Shi, Rynson Lau, Thomas S. Huang},
		title = {Geometry-Aware Distillation for Indoor Semantic Segmentation},
		booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
		year = {2019}
	}
**References**

[1] Nathan Silberman, Derek Hoiem, Pushmeet Kohli and Rob Fergus. [Indoor Segmentation and Support Inference from RGBD Images](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html). In Proceedings of the European Conference on Computer Vision 2012.

[2] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dolla ́r. [Focal Loss for Dense Object Detection](http://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf). In Proceedings of the IEEE International Conference on Computer Vision 2017.

[3] Iro Laina, Christian Rupprecht, Vasileios Belagiannis, Federico Tombari, Nassir Navab.  [Deeper Depth Prediction with Fully Convolutional Residual Networks](https://arxiv.org/pdf/1606.00373.pdf). In Proceedings of the International Conference on 3D Vision 2016.



If you have any questions please email the [authors](mailto:jiaojianbo.i@gmail.com)
