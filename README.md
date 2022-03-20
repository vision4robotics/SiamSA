# SiamPSA: Siamese Object Tracking for Vision-Based UAM Approaching with Pairwise Scale-Channel Attention

:bust_in_silhouette: [Guangze Zheng](https://scholar.google.com/citations?hl=zh-CN&user=-kcZWRQAAAAJ), [Changhong Fu\*](https://scholar.google.com/citations?hl=zh-CN&user=zmbMZ4kAAAAJ), [Junjie Ye](https://scholar.google.com/citations?hl=zh-CN&user=uy-TfXgAAAAJ), [Bowen Li](https://scholar.google.com/citations?hl=zh-CN&user=XjZjcakAAAAJ), [Geng Lu](https://www.au.tsinghua.edu.cn/info/1082/1683.htm), and [Jia Pan](https://scholar.google.com/citations?hl=zh-CN&user=YYT8-7kAAAAJ)

## Abstract

Visual approaching to the target object is crucial to the subsequent manipulating of the **unmanned aerial manipulator (UAM)**. Although the manipulating methods have been widely studied, the **vision-based UAM approaching** generally lacks efficient design. The key to the visual UAM approaching lies in **object tracking**, while current approaching generally relies on costly model-based methods. Besides, UAM approaching often confronts more severe object scale variation issues, which makes it inappropriate to directly employ state-of-the-art model-free Siamese-based methods from the object tracking field. To address the above problems, this work proposes a novel Siamese network with **pairwise scale-channel attention** (SiamPSA) for vision-based UAM approaching. Specifically, SiamPSA consists of a scale attention network (SAN) and a scale-aware anchor proposal network (SA-APN). SAN acquires valuable scale information for feature processing, while SAAPN mainly attaches scale-awareness to anchor proposing. Moreover, a new tracking benchmark for UAM approaching, namely **UAMT100**, is recorded with 35K frames on a flying UAM platform for evaluation. Exhaustive experiments on the benchmark and real-world tests validate the efficiency and practicality of SiamPSA with a promising speed.

<img src="img/main.png" width="800" alt="main" />

## 1. Demo

- 📹 Demo of real-world vision-based UAM approaching tests on [Youtube](https://www.youtube.com/watch?v=CHogCCP-FH4) demonstrates the practicality of SiamPSA.

<img src="img/2022_IROS-16465373740221.gif" width="700" alt="2022_IROS" />

## 2. UAMT100 benchmark

### 2.1 Introduction

- **UAMT100** is a benchmark to evaluate object tracking methods for UAM approaching. 
- **100** image sequences are contained, which are captured from a flying UAM platform.  
- **16** kinds of **objects** are involved, and **11 attributes** are annotated for each sequence.

<img src="img/dataset+attribute.jpg" width="500" alt="dataset+attribute" />

- The first row is from **a fixed camera** as the **third perspective**, while the second and third rows are from the **UAM onboard camera** as the **first perspective**. 
- The red box and green circle denote the object and the onboard camera. 
- Attribute statistics is presented.

### 2.2 Download and evaluation 

- Download UAMT100 from [GoogleDrive](https://drive.google.com/file/d/1WD5sPkwqj7E63bDFQQsmCxLKP_CM_D1I/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1Oes8-s2qce3EXUZ-qfLmLA) (pw: v4ra).

- Download evaluation results (.mat) of SOTA trackers on UAMT100 from [GoogleDrive]() or [BaiduYun](https://pan.baidu.com/s/1OpXzu7FiXi3Mgn4i3K54Qw) (pw: v4rr).

  <img src="img/overall_UAMT100_overlap_OPE.png" width="600" alt="overall_UAMT100_overlap_OPE" />

  <img src="img/overall_UAMT100_np_OPE.png" width="600" alt="overall_UAMT100_np_OPE" />

## 3. Environment setup

This code has been tested on Ubuntu 18.04, Python 3.8.3, Pytorch 0.7.0/1.6.0, CUDA 10.2.
Please install related libraries before running this code: 

```bash
pip install -r requirements.txt
```

## 4. Test
- Download pretrained backbone from [GoogleDrive]() or [BaiduYun](https://pan.baidu.com/s/1qA_cFpzahUahravBrTg-cg) (pw: v4rc) and put it into `pretrained_models` directory.
- Download tracker model from [GoogleDrive](https://drive.google.com/file/d/1CeSwuycTE5LpwUpK4iQVSC8MuxvGiQB-/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1LWtPQYYQoN7_5P9wkCr23w) (pw: v4rb) and put it into `snapshot` directory.
- Download testing datasets and put them into `test_dataset` directory. If you want to test the tracker on a new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to set test_dataset.


```
python tools/test.py 	                    \
	--trackername SiamPSA                   \ # tracker_name
	--dataset UAMT100                       \ # dataset_name
	--snapshot snapshot/model.pth             # model_path
```

The testing result will be saved in the `results/dataset_name/tracker_name` directory.

## 5. Train

### 5.1 Prepare training datasets

Download the datasets：
* [VID](http://image-net.org/challenges/LSVRC/2017/)
* [YOUTUBEBB](https://pan.baidu.com/s/1ZTdfqvhIRneGFXur-sCjgg) (code: t7j8)
* [COCO](http://cocodataset.org)
* [GOT-10K](http://got-10k.aitestunion.com/downloads)

**Note:** `train_dataset/dataset_name/readme.md` has listed detailed operations about how to generate training datasets.


### 5.2 Train a model
To train the SiamPSA model, run `train.py` with the desired configs:

```bash
python tools/train.py 
```

## 6. Evaluation 
If you want to evaluate the tracker mentioned above, please put those results into  `results` directory.
```
python eval.py 	                      \
	--tracker_path ./results          \ # result path
	--dataset UAMT100                 \ # dataset_name
	--tracker_prefix 'model'            # tracker_name
```
## Contact
If you have any questions, please contact me.

Guangze Zheng

Email: [mmlp@tongji.edu.cn](mmlp@tongji.edu.cn)

## Acknowledgement
- The code is implemented based on [pysot](https://github.com/STVIR/pysot), [SiamAPN](https://github.com/vision4robotics/SiamAPN), and [SiamSE](https://github.com/ISosnovik/SiamSE). We would like to express our sincere thanks to the contributors. 
- We would like to thank Ziang Cao for his advice on the code. 
- We appreciate the help from Fuling Lin, Haobo Zuo, and Liangliang Yao. 
- We would like to thank Kunhan Lu for his advice on TensorRT acceleration. 

