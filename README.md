# SiamPSA&SiamSA

*Official code for **SiamPSA** (Siamese Object Tracking for Vision-Based UAM Approaching with Pairwise Scale-Channel Attention) and **SiamSA** (Scale-Aware Siamese Object Tracking for Vision-Based UAM Approaching)*

:bust_in_silhouette: [Guangze Zheng](https://scholar.google.com/citations?hl=zh-CN&user=-kcZWRQAAAAJ), [Changhong Fu\*](https://scholar.google.com/citations?hl=zh-CN&user=zmbMZ4kAAAAJ), [Junjie Ye](https://scholar.google.com/citations?hl=zh-CN&user=uy-TfXgAAAAJ), [Bowen Li](https://scholar.google.com/citations?hl=zh-CN&user=XjZjcakAAAAJ), [Geng Lu](https://www.au.tsinghua.edu.cn/info/1082/1683.htm), and [Jia Pan](https://scholar.google.com/citations?hl=zh-CN&user=YYT8-7kAAAAJ)

## 1. Demo

- 📹 Demo of real-world vision-based UAM approaching tests.

  <img src="/img/UAM_approaching.gif" width="700" alt="" />

- Refer to [SiamPSA](https://youtu.be/rkq188GhJ_A) and [SiamSA](https://youtu.be/Fi6kESBBpnk) on Youtube for more real-world tests.

## 2. UAMT100&UAMT20L benchmark

### 2.1 Introduction

- With 100 image sequences, **UAMT100** is a benchmark to evaluate object tracking methods for UAM approaching, while **UAMT20L** is a subset which contains 20 sequences. All sequences are recorded on a flying UAM platform.  

- **16** kinds of **objects** are involved, and **12 attributes** are annotated for each sequence, including aspect ratio change (ARC), background clutter (BC), fast motion (FM), low illumination (LI), object blur (OB), out-of-view (OV), partial occlusion (POC), similar object (SOB), scale variation (SV), **UAM attack (UAM-A)**, viewpoint change (VC), and **wind disturbance (WD)**.

  <img src="/img/dataset.png" width="700" alt="" />

- Scenes from both **first** (left) and **third** (right) perspectives in UAMT100 and UAMT20L. The first perspective is from the UAM onboard camera, while the third perspective is from a fixed camera. 

- The **red** boxes and the **green** circles denote the tracking objects and the onboard camera, respectively.

- In a), an industrial fan is set to form **wind disturbance**, while in d), a stick is used to cause **UAM attack**.

### 2.2 Scale variation difference between UAV and UAM tracking 

<img src="/img/SV.png" width="580" alt="" />

- $R$ denotes the degree of SV, which is measured by the ratio of the current object ground-truth bounding box area to the initial one. 
- Generally, SV happens when $R$ is outside the range [0.5, 2], i.e., $|log_2(R)|>1$. The percentage of frames whose $|log_2(R)|$ is with a certain section is drawn as the SV plot, with an interval length of 0.1 over a range of 1 to 3. The proportion of $|log_2(R)|>3$ is relatively low and not of reference significance. 
- Therefore, the larger area under the curve means the higher frequency of object SV, and during UAM tracking for approaching, SV is more common and more severe than UAV tracking.

### 2.3 Download and evaluation 

- Download UAMT100 from [GoogleDrive](https://drive.google.com/file/d/1WD5sPkwqj7E63bDFQQsmCxLKP_CM_D1I/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1Oes8-s2qce3EXUZ-qfLmLA) (pw: v4ra).
- Download UAMT20L from [GoogleDrive](https://drive.google.com/file/d/1mDDC8318U3EMD9u54hYoKlVfAc4DiXd9/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1ObtZ2xfiq1MCbHlWYBj8TQ) (pw: v4rf).
- Download evaluation results (.mat) of SOTA trackers on UAMT100 from [GoogleDrive]() or [BaiduYun](https://pan.baidu.com/s/1OpXzu7FiXi3Mgn4i3K54Qw) (pw: v4rr).


## 3. Environment setup

This code has been tested on Ubuntu 18.04, Python 3.8.3, Pytorch 0.7.0/1.6.0, CUDA 10.2.
Please install related libraries before running this code: 

```bash
pip install -r requirements.txt
```

## 4. Test

- Download pretrained backbone from [GoogleDrive]() or [BaiduYun](https://pan.baidu.com/s/1qA_cFpzahUahravBrTg-cg) (pw: v4rc) and put it into `pretrained_models` directory.
- Download SiamPSA model from [GoogleDrive](https://drive.google.com/file/d/1CeSwuycTE5LpwUpK4iQVSC8MuxvGiQB-/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1LWtPQYYQoN7_5P9wkCr23w) (pw: v4rb) and put it into `snapshot` directory.
- Download SiamSA model from [GoogleDrive](https://drive.google.com/file/d/1wVfyp4hUB415hb1d06fea3y2DDHhPD6G/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1FX-6BpvFINUpVOFGUal0Dg) (pw: v4re) and put it into `snapshot` directory.
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

