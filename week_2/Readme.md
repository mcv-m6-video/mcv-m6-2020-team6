### Week 2

The goal of this week 2 is to learn about modelling the background pixels of a video sequence using a simple statistical model to classify the background / foreground.

#### Installation

```bash
pip install -r requirements.txt
```
#### Execution

 All the implemented tasks can be run from ```main_week_2.py``` which is used to collect all work done.

#### Directory structure

```bash
├── datasets
│   ├── kitti
│   ├── train
├── week_2
│   ├── detections_alpha
│   ├── detections_alpha_hsv
│   ├── detections_sota
│   ├── gridsearch
│   ├── gridsearch_hsv
│   ├── gridsearch_yuv
│   ├── metrics
│   │   ├── evaluation_funcs.py
│   │   ├── Optical_flow_metrics.py
│   │   ├── Optical_flow_visualization.py
│   ├── model
│   │   ├── BackgroundSubtraction.py
│   │   ├── BackgroundSubtractionWithColor.py
│   │   ├── bbox.py
│   │   ├── frame.py
│   │   ├── frameExtraction.py
│   │   ├── read_annotation.py
│   │   ├── video.py
│   ├── utils
│   │   ├── iou_over_time_RT.py
│   │   ├── show_bb_singleFrame.py
│   │   ├── show_frame.py
│   │   ├── preprocessing.py
│   ├── week_2_resultats
│   ├── adaptative_optimization.py
│   ├── annotation.txt
│   ├── annotation_fix.txt
│   ├── background_sota.py
│   ├── main_week_2.py
│   ├── map_vs_alpha.py
│   ├── requirements.txt
```

#### Tasks done during this week:

- Task 1.1: Gaussian distribution
<div align="center">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/week_2/week_2_resultats/Task_3_sota_own.gif">
</div>


- Task 1.2 & 1.3: Evaluate results
  
  
<div align="center">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/week_2/week_2_resultats/map_alpha.png">
</div>

- Task 2.1: Recursive Gaussian modeling

<div align="center">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/week_2/week_2_resultats/Task_3_sota_own_adapt.gif">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/week_2/week_2_resultats/Capture.PNG">
</div>



- Task 2.2: Evaluate and compare to non-recursive


| Method | mAP@50 |
|-|-|
|Non-adaptive|0.2906|
|Adaptive|0.3639|



- Task 3: Compare with state-of-the-art

<div align="center">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/week_2/week_2_resultats/Task_3_sota_MOG.gif">
</div>

| Method | mAP@50 |
|-|-|
|MOG|0.5110|
|MOG2|0.3539|
|KNN|0.4549|
|Own adap|0.3639|


- Task 4: Color sequences


