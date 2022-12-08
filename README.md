<img src="/img/UAM_approaching.gif" width="800" alt="" />

*Official code for our work on UAM object tracking*:

- *[IROS 2022] Siamese Object Tracking for Vision-Based UAM Approaching with Pairwise Scale-Channel Attention*
- *[TII 2022] Scale-Aware Siamese Object Tracking for Vision-Based UAM Approaching*

:bust_in_silhouette: [Guangze Zheng](https://scholar.google.com/citations?hl=zh-CN&user=-kcZWRQAAAAJ), [Changhong Fu\*](https://scholar.google.com/citations?hl=zh-CN&user=zmbMZ4kAAAAJ), [Junjie Ye](https://scholar.google.com/citations?hl=zh-CN&user=uy-TfXgAAAAJ), [Bowen Li](https://scholar.google.com/citations?hl=zh-CN&user=XjZjcakAAAAJ), [Geng Lu](https://www.au.tsinghua.edu.cn/info/1082/1683.htm), and [Jia Pan](https://scholar.google.com/citations?hl=zh-CN&user=YYT8-7kAAAAJ)

## 1. Introduction

**SiamSA** aims to provide a model-free solution for UAM tracking during approaching the object (for manipulation). Since the **Scale Variation** issue has been more crucial than general object-tracking scenes, the novel scale awareness is proposed with powerful attention methods.

Please refer to our project page, papers, dataset, and videos for more details.

:newspaper:[[Project page]](https://george-zhuang.github.io/siamsa/)  :page_facing_up:[[IROS Paper]](https://arxiv.org/abs/2211.14564)  :books:[[UAM Tracking Dataset]](https://george-zhuang.github.io/siamsa/)  :movie_camera: [[TII Demo]](https://www.youtube.com/watch?v=Fi6kESBBpnk)  :movie_camera: [[IROS Presentation]](https://www.youtube.com/watch?v=FS1tJolGGV8)

*Paper of TII-version coming soon...*

## 2. UAMT100&UAMT20L benchmark

### 2.1 Introduction

- With 100 image sequences, **UAMT100** is a benchmark to evaluate object tracking methods for UAM approaching, while **UAMT20L** contains 20 long sequences. All sequences are recorded on a flying UAM platform;

- **16** kinds of **objects** are involved;

- **12 attributes** are annotated for each sequence:

  - aspect ratio change (ARC);background clutter (BC), fast motion (FM), low illumination (LI), object blur (OB), out-of-view (OV), partial occlusion (POC), similar object (SOB), scale variation (SV), **UAM attack (UAM-A)**, viewpoint change (VC), and **wind disturbance (WD)**.

  <img src="/img/UAMT_benchmark.png" width="700" alt="" />

  

### 2.2 Scale variation difference between UAV and UAM tracking 

<img src="/img/SV.png" width="540" alt="" />

*A larger area under the curve means a higher frequency of object SV. It is clear that SV of UAM tracking is much more common and severe than UAV tracking.*

### 2.3 Download and evaluation 

- Please download the dataset from our [project page](https://george-zhuang.github.io/siamsa/).
- You can directly download our evaluation results (.mat) of SOTA trackers on the UAMT benchmark from [GoogleDrive](https://drive.google.com/file/d/18IUT9Yu7Oai62IE6AYSfWOy3XUEvzzqc/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1OpXzu7FiXi3Mgn4i3K54Qw?pwd=v4rr).

## 3. Get started!

### 3.1 Environmental Setup

This code has been tested on Ubuntu 18.04, Python 3.8.3, Pytorch 1.6.0, CUDA 10.2.
Please install related libraries before running this code: 

```bash
git clone https://github.com/vision4robotics/SiamSA
pip install -r requirements.txt
```

### 3.2 Test

- For testing SiamSA_IROS22:
  - Download SiamSA_IROS22 model from [GoogleDrive](https://drive.google.com/file/d/1CeSwuycTE5LpwUpK4iQVSC8MuxvGiQB-/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1LWtPQYYQoN7_5P9wkCr23w?pwd=v4rb) and put it into `snapshot` directory.
- For testing SiamSA_TII22:
  - Download SiamSA_TII22 model from [GoogleDrive](https://drive.google.com/file/d/1wVfyp4hUB415hb1d06fea3y2DDHhPD6G/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1FX-6BpvFINUpVOFGUal0Dg?pwd=v4re) and put it into `snapshot` directory.
- Download testing datasets (UAMT100/UAMT20L/UAV123@10fps) and put them into `test_dataset` directory. If you want to test the tracker on a new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to set test_dataset.


```
python tools/test.py 	                    \
	--trackername SiamSA                   \ # tracker_name
	--dataset UAMT100                       \ # dataset_name
	--snapshot snapshot/model.pth             # model_path
```

The testing result will be saved in the `results/dataset_name/tracker_name` directory.

### 3.3 Evaluate 

If you want to evaluate the tracker mentioned above, please put those results into `results` directory.

```
python eval.py 	                      \
	--tracker_path ./results          \ # result path
	--dataset UAMT100                 \ # dataset_name
	--tracker_prefix 'model'            # tracker_name
```

### 3.4 Train

- Download pretrained backbone from [GoogleDrive](https://drive.google.com/file/d/1Lv9HbABSNYBNetT4_3qSzDlJc0wuiAYm/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1qA_cFpzahUahravBrTg-cg?pwd=v4rc) and put it into `pretrained_models` directory.

- #### Prepare training datasets

  Download the datasets：

  - [VID](http://image-net.org/challenges/LSVRC/2017/)
  - [YOUTUBEBB](https://pan.baidu.com/s/1ZTdfqvhIRneGFXur-sCjgg) (code: t7j8)
  - [COCO](http://cocodataset.org)
  - [GOT-10K](http://got-10k.aitestunion.com/downloads)

  **Note:** `train_dataset/dataset_name/readme.md` has listed detailed operations about how to generate training datasets.

- #### Train a model

  To train the SiamSA model, run `train.py` with the desired configs:

  ```python
  python tools/train.py 
  ```

- #### Test and evaluate

  Once you get a model, you may want to test and evaluate its performance by following the above **3.2** and **3.3** instructions

## Contact

If you have any questions, please contact me.

Guangze Zheng

Email: [mmlp@tongji.edu.cn](mmlp@tongji.edu.cn)

Homepage: [Guangze Zheng (george-zhuang.github.io)](https://george-zhuang.github.io/)

## Acknowledgement

- The code is implemented based on [pysot](https://github.com/STVIR/pysot), [SiamAPN](https://github.com/vision4robotics/SiamAPN), and [SiamSE](https://github.com/ISosnovik/SiamSE). We would like to express our sincere thanks to the contributors. 
- We would like to thank Ziang Cao for his advice on the code. 
- We appreciate the help from Fuling Lin, Haobo Zuo, and Liangliang Yao. 
- We would like to thank Kunhan Lu for his advice on TensorRT acceleration. 

